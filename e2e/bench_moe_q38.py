"""MoE block prefill at Qwen3.8-Flash-Next shapes, random weights (phase 1: before downloading the checkpoint).

Qwen3.8-Flash-Next has 512 routed experts of 2560 x 640 with top-10 routing plus a shared expert, so a 2048-token
prompt gives each expert ~40 rows (Qwen3.5-35B-A3B: 256 x 512, top-8, ~64 rows). This times one MoE half-block
(RMSNorm, router, experts, shared expert, combine, residual) at both shapes:
  ours_sc     fused W4A8 block of e2e/fused_qwen35.py (INT8 activations, grouped INT8 matmul2d expert GEMMs with
              in-kernel unpack, shared expert as expert E+1), SC-MXFP4 weights (4.07 bits)
  ours_b16    the same kernels with the integer-exact 16-block format e2m1/B16/M1..10 (4.26 bits)
  mlx_mxfp4   mlx_lm's SparseMoeBlock as in the qwen4_exp PR (SwitchGLU, gather_qmm), MXFP4 W4A16
  mlx_affine4 the same, MLX 4-bit affine g64
  bf16        the same, BF16 gather_mm
Every expert holds the same random matrices (one quantized gate/up/down set, tiled; memory traffic is that of
distinct experts). The router is random, so routing is near uniform; --skew gives experts lognormal popularity.

Usage: python bench_moe_q38.py --check                      (exactness + agreement, development run)
       python bench_moe_q38.py --sweep 512,2048,8192         (grouped-GEMM tile sweep)
       python bench_moe_q38.py --lengths 512,2048,8192 --out results/10_moe_q38.csv
"""
import argparse
import csv
import os
import sys
import time

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_lm.models.switch_layers import SwitchGLU

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import bench_prefill as bp  # noqa: E402
import fused_olmoe as fo  # noqa: E402
import fused_qwen35 as fq  # noqa: E402
import pack_format as pf  # noqa: E402
import sc_format as F  # noqa: E402
import sc_kernels as S  # noqa: E402
import sc_moe  # noqa: E402

SHAPES = {"qwen38": dict(E=512, D=2560, F=640, k=10), "qwen35": dict(E=256, D=2048, F=512, k=8)}
B16 = "e2m1/B16/M1..10"
TILES = [((64, 64, 4), 64), ((64, 32, 4), 128), ((32, 64, 2), 128), ((32, 64, 4), 64), ((32, 128, 4), 64),
         ((64, 128, 8), 128)]


def quantize(w, fmt_name):
    """-> codes, bcodes, wscale, dequantized float32 weights, kernel Format."""
    if fmt_name == "sc":
        fmt = F.Format()
        codes, bc, ws, qw = F.quantize(w, fmt)
    else:
        fmt = pf.sc_format_of(fmt_name)
        codes, bc, ws, st = pf.pack_matrix(w, fmt_name, fmt)
        assert st["exact_q"] and st["exact_w"]
        qw = F.decode_numpy(codes, bc, w.shape[0], w.shape[1], fmt)
    codes = np.asarray(codes).reshape(w.shape[0], -1)  # packers return the code words flat
    return codes, bc, ws, (qw.astype(np.float64) * ws[:, None]).astype(np.float32), qw, fmt


class Weights:
    """One random gate/up/down set, router, shared-expert gate and norm, packed for the kernels and tiled over the
    E routed experts + the shared expert."""

    def __init__(self, shape, fmt_name, seed=0, skew=0.0):
        E, D, Fi = shape["E"], shape["D"], shape["F"]
        self.shape, E1 = shape, E + 1
        rng = np.random.default_rng(seed)
        g, u, d = (quantize(rng.standard_normal(s) * 0.02, fmt_name) for s in ((Fi, D), (Fi, D), (D, Fi)))
        self.fmt = g[5]
        perm = S.interleave_gate_up(Fi)
        cat = lambda i: np.concatenate([g[i], u[i]])[perm]  # noqa: E731
        tile = lambda a: mx.array(np.ascontiguousarray(np.broadcast_to(a, (E1,) + a.shape)))  # noqa: E731
        self.gu = (tile(cat(0).reshape(-1)), tile(cat(1)), tile(cat(2)))
        self.down = (tile(d[0].reshape(-1)), tile(d[1]), tile(d[2]))
        self.deq = dict(gate=g[3], up=u[3], down=d[3])
        self.qw_gu = cat(4)
        rs = np.exp(skew * rng.standard_normal(E))[:, None] if skew else 1.0
        self.gate_w = mx.array((rng.standard_normal((E, D)) * 0.02 * rs).astype(np.float32)).astype(mx.bfloat16)
        self.sg_w = mx.array((rng.standard_normal((1, D)) * 0.02).astype(np.float32)).astype(mx.bfloat16)
        self.pn = mx.array((1 + 0.1 * rng.standard_normal(D)).astype(np.float32)).astype(mx.bfloat16)
        mx.eval(self.gu, self.down, self.gate_w, self.sg_w, self.pn)


def ours_fn(w, L, tile_gu=(64, 64, 4), tile_d=(64, 64, 4), bk=64):
    """The MoE half-block of fused_qwen35.FusedQwen35._moe_fn (in-kernel unpack), compiled."""
    S.set_format(w.fmt)
    a = w.shape
    k, E1, F2 = a["k"], a["E"] + 1, 2 * a["F"]
    tm, R = max(tile_gu[0], tile_d[0]), L * (k + 1)
    eps = mx.array([1e-6], mx.float32)

    def moe(h, pn, gate_w, sg_w, cg, bg, sg, cd, bd, sd):
        S.set_format(w.fmt)  # kernels are fetched while mx.compile traces (first call), so set the format here
        xq, xs, xn = S.rmsnorm_quant_n(h, pn, eps)
        inds, scores = fq.route(fq._Router(gate_w, sg_w), xn[:L], k)
        pos, tok_p, e_sorted, off, tstart, maxt = sc_moe.route_sort(inds, E1, tm)
        xq_s, xs_s = xq[tok_p], xs[tok_p[:R]]
        act = sc_moe.moe_mm(xq_s, R, cg, bg, sg, off, tstart, maxt, F2, 3, xs_s, tile=tile_gu, bk=bk)
        xq2, xs2 = S.act_quant(act, R + tm)
        yd = sc_moe.moe_mm(xq2, R, cd, bd, sd, off, tstart, maxt, a["D"], 1, xs2, tile=tile_d, bk=bk)
        return fo.moe_combine(yd, pos, scores, h, pn, eps, L, quant=False)[0]

    fn = mx.compile(moe)
    args = (w.pn, w.gate_w, w.sg_w, *w.gu, *w.down)
    return lambda h: fn(h, *args)


class MlxBlock(nn.Module):
    """qwen4_exp SparseMoeBlock (PR #1788) behind an RMSNorm, with a residual add."""

    def __init__(self, w):
        super().__init__()
        a = w.shape
        E, D, Fi = a["E"], a["D"], a["F"]
        self.k = a["k"]
        self.norm = nn.RMSNorm(D, eps=1e-6)
        self.gate = nn.Linear(D, E, bias=False)
        self.switch_mlp = SwitchGLU(D, Fi, E)
        self.shared_expert = _MLP(D, Fi)
        self.shared_expert_gate = nn.Linear(D, 1, bias=False)
        bf = lambda x: mx.array(x).astype(mx.bfloat16)  # noqa: E731
        self.norm.weight = w.pn
        self.gate.weight, self.shared_expert_gate.weight = w.gate_w, w.sg_w
        for n in ("gate", "up", "down"):
            m = bf(w.deq[n])
            getattr(self.switch_mlp, n + "_proj").weight = mx.contiguous(mx.repeat(m[None], E, axis=0))  # E real copies (repeat alone is a stride-0 view)
            getattr(self.shared_expert, n + "_proj").weight = m
        self.set_dtype(mx.bfloat16)
        mx.eval(self.parameters())

    def __call__(self, h):
        x = self.norm(h)
        logits = self.gate(x.astype(mx.float32))
        idx = mx.argpartition(-logits, self.k - 1, axis=-1)[..., : self.k]
        sc = mx.softmax(mx.take_along_axis(logits, idx, axis=-1), axis=-1, precise=True)
        out = (self.switch_mlp(x, idx) * sc[..., None]).sum(axis=-2).astype(x.dtype)
        return h + out + mx.sigmoid(self.shared_expert_gate(x)) * self.shared_expert(x)


class _MLP(nn.Module):
    def __init__(self, D, Fi):
        super().__init__()
        self.gate_proj, self.up_proj = nn.Linear(D, Fi, bias=False), nn.Linear(D, Fi, bias=False)
        self.down_proj = nn.Linear(Fi, D, bias=False)

    def __call__(self, x):
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


def mlx_fn(w, kind):
    blk = MlxBlock(w)
    if kind != "bf16":
        gs, mode = (32, "mxfp4") if kind == "mlx_mxfp4" else (64, "affine")
        nn.quantize(blk, group_size=gs, bits=4, mode=mode,
                    class_predicate=lambda p, m: hasattr(m, "to_quantized") and p != "gate")
        mx.eval(blk.parameters())
    return lambda h: blk(h[None])[0]


def bench(fn, x, n, reps):
    """Seconds per call of n chained calls (x <- fn(x), as consecutive layers: no overlap between calls)."""
    mx.eval(fn(x))
    ts = []
    for _ in range(reps):
        mx.synchronize()
        t0 = time.perf_counter()
        y = x
        for _ in range(n):
            y = fn(y)
        mx.eval(y)
        mx.synchronize()
        ts.append((time.perf_counter() - t0) / n)
    return ts


def inputs(shape, L, seed=1):
    rng = np.random.default_rng(seed)
    return mx.array(rng.standard_normal((L, shape["D"])).astype(np.float32)).astype(mx.bfloat16)


def routing_stats(w, h, L):
    k, E = w.shape["k"], w.shape["E"]
    xn = mx.fast.rms_norm(h, w.pn, 1e-6)
    inds, _ = fq.route(fq._Router(w.gate_w, w.sg_w), xn[:L], k)
    c = np.bincount(np.array(inds[:, :k]).ravel(), minlength=E)
    return c.mean(), c.max(), int((c == 0).sum())


def check():
    ok = True
    for name, shape in SHAPES.items():
        for fmt_name in ("sc", B16):
            w = Weights(shape, fmt_name)
            L, k, E1 = 512, shape["k"], shape["E"] + 1
            h = inputs(shape, L)
            # 1. grouped gate/up GEMM: INT32 accumulators vs float64 (exact below 2^53) for every routed row
            S.set_format(w.fmt)
            xq, xs, xn = S.rmsnorm_quant_n(h, w.pn, mx.array([1e-6], mx.float32))
            inds, _ = fq.route(fq._Router(w.gate_w, w.sg_w), xn[:L], k)
            pos, tok_p, e_sorted, off, tstart, maxt = sc_moe.route_sort(inds, E1, 64)
            R = L * (k + 1)
            acc = np.array(sc_moe.moe_mm(xq[tok_p], R, *w.gu, off, tstart, maxt, 2 * shape["F"], 0, bk=64))
            ref = np.array(xq[tok_p][:R]).astype(np.float64) @ w.qw_gu.astype(np.float64).T
            err = np.abs(acc - ref).max()
            # 2. whole block vs MLX BF16 on the same (dequantized) weights: only INT8 activation rounding differs
            y = ours_fn(w, L)(h).astype(mx.float32)
            yb = mlx_fn(w, "bf16")(h).astype(mx.float32)
            rel = float(mx.linalg.norm(y - yb).item() / mx.linalg.norm(yb - h.astype(mx.float32)).item())
            print(f"{name} {fmt_name}: grouped GEMM INT32 max err {err:.0f}; block output vs BF16 (same weights) "
                  f"rel. err of the MoE update {rel:.4f}", flush=True)
            ok &= err == 0 and rel < 0.05
    print("bench_moe_q38 check", "passed" if ok else "FAILED")


def sweep(lengths, n=10, reps=5):
    for name, shape in SHAPES.items():
        for fmt_name in ("sc", B16):
            w = Weights(shape, fmt_name)
            for L in lengths:
                h = inputs(shape, L)
                res = []
                for tile, bk in TILES:
                    fn = ours_fn(w, L, tile, tile, bk)
                    res.append((float(np.median(bench(fn, h, n, reps))) * 1e3, tile, bk))
                res.sort()
                print(f"{name} {fmt_name} L={L}: " + "  ".join(f"{t}/{b}: {ms:.3f}" for ms, t, b in res), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--sweep", default="")
    ap.add_argument("--lengths", default="512,2048,8192")
    ap.add_argument("--methods", default="ours_sc,ours_b16,mlx_mxfp4,mlx_affine4,bf16")
    ap.add_argument("--shapes", default="qwen38,qwen35")
    ap.add_argument("--tile", default="auto", help="ours: BM,BN,SG,BK for both grouped GEMMs, or auto: the best of "
                    "the sweep (results/10_moe_q38_sweep.txt), (32,64,2)/128 below 24 rows per expert, else (64,64,4)/64")
    ap.add_argument("--skew", type=float, default=0.0)
    ap.add_argument("--reps", type=int, default=7)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    if a.check:
        return check()
    if a.sweep:
        return sweep([int(x) for x in a.sweep.split(",")])
    rows = []
    for name in a.shapes.split(","):
        shape = SHAPES[name]
        ws = {"sc": Weights(shape, "sc", skew=a.skew), "b16": Weights(shape, B16, skew=a.skew)}
        for L in [int(x) for x in a.lengths.split(",")]:
            h = inputs(shape, L)
            mean, mx_, empty = routing_stats(ws["sc"], h, L)
            t = ([32, 64, 2, 128] if L * shape["k"] / shape["E"] < 24 else [64, 64, 4, 64]) if a.tile == "auto" \
                else [int(x) for x in a.tile.split(",")]
            tile, bk = tuple(t[:3]), t[3]
            fns = {}
            for m in a.methods.split(","):
                fns[m] = (ours_fn(ws[m[5:]], L, tile, tile, bk) if m.startswith("ours")
                          else mlx_fn(ws["sc"], m))
            # sanity: ours vs BF16 on the same dequantized weights (only INT8 activation rounding differs; catches a
            # wrong kernel format); MLX's re-quantized baselines vs BF16 on the SC-MXFP4 weights (loose bound)
            hf = h.astype(mx.float32)
            upd = lambda fn: fn(h).astype(mx.float32) - hf  # noqa: E731
            yb = {k: upd(mlx_fn(ws[k], "bf16")) for k in ("sc", "b16")}
            err = {m: float(mx.linalg.norm(upd(fn) - yb[m[5:] if m.startswith("ours") else "sc"]).item()
                            / mx.linalg.norm(yb["sc"]).item()) for m, fn in fns.items()}
            assert all(e < (0.05 if m.startswith("ours") else 0.3) for m, e in err.items()), err
            del yb
            n = max(3, int(40 * 512 / L))
            times = {m: [] for m in fns}
            for _ in range(a.reps):  # methods interleaved, one timed burst each per round
                for m, fn in fns.items():
                    times[m] += bench(fn, h, n, 1)
            for m, ts in times.items():
                print(f"# {name} L={L} {m}: " + " ".join(f"{t * 1e3:.2f}" for t in ts), flush=True)
            R = L * (shape["k"] + 1)
            ops = 2 * R * 3 * shape["D"] * shape["F"]
            ref = float(np.median(times["mlx_mxfp4"])) if "mlx_mxfp4" in times else None
            for m, ts in times.items():
                ms = float(np.median(ts)) * 1e3
                row = dict(shape=name, L=L, tile=f"{tile}/{bk}" if m.startswith("ours") else "", rows_per_expert=round(mean, 1), max_rows=int(mx_), empty_experts=empty,
                           method=m, rel_err_vs_bf16=round(err[m], 4), ms=round(ms, 4), expert_tops=round(ops / ms / 1e9, 1),
                           vs_mlx_mxfp4=round(ref * 1e3 / ms, 3) if ref else "")
                rows.append(row)
                print(row, flush=True)
            del fns
    state = bp.power_state()
    print("#", state)
    if a.out:
        with open(a.out, "w", newline="") as f:
            f.write(f"# {state}; skew {a.skew}\n")
            wr = csv.DictWriter(f, fieldnames=list(rows[0]))
            wr.writeheader()
            wr.writerows(rows)
        print("wrote", a.out)


if __name__ == "__main__":
    main()
