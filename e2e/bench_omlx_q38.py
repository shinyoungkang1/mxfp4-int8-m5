"""Prefill kernels at Qwen3.8-Flash-Next shapes: ours vs oMLX's INT8-activation path (concurrent work, oMLX v0.7.1rc1,
PRs 3548 / 4320) vs MLX W4A16, random weights, prompt lengths 512 / 2048 / 8192, latency per call (chained calls).

  dense projection (DeltaNet qkvz 16384x2560, out 2560x6144, attention qkv 13312x2560, o 2560x6144):
    ours        per-token INT8 + unpack the packed SC-MXFP4 weights to INT8 + one INT8 matmul2d GEMM with the scales in
                the epilogue (fused_qwen38._gemm, the deployed dense prefill path)
    omlx_a8     oMLX Stage A (per-token INT8, reorder) + its INT8 GEMM on MLX affine Q4 / GS64 weights with the affine
                correction at every 64-element group (qwen35_oq_a8_qmm_t, metadata pre-transposed as its plan does)
    mlx_mxfp4 / mlx_affine4   MLX quantized_matmul, W4A16
  routed experts (512 experts of 2560 x 640, top-10, tokens sorted by expert; no shared expert, no router):
    ours        grouped INT8 gate|up GEMM with SwiGLU epilogue, per-row INT8, grouped INT8 down GEMM (sc_moe)
    omlx_a8     oMLX's A8 gate+up gather kernel with SwiGLU (sorted_gather_qmm_a8) + A16 down (gather_qmm affine), as
                its Qwen3.8 path ships (PR 4320: "the Down projection stays on the existing A16 path")
    mlx_mxfp4   gather_qmm gate, up, swiglu, gather_qmm down (W4A16, sorted)
Run with .venv-omlx (mlx 0.32.4 = oMLX's pin; oMLX wheel + the NAX metallib of its macOS 26 app) under run_timed.sh.
"""
import os
import sys
import time

import mlx.core as mx
import mlx.nn as nn
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import bench_moe_q38 as BM  # noqa: E402
import sc_format as F  # noqa: E402
import sc_kernels as S  # noqa: E402
import sc_moe  # noqa: E402
from omlx.custom_kernels.qwen35_prefill import fast as OF  # noqa: E402
from omlx.patches import m5_gather_qmm_a8 as OG  # noqa: E402

DENSE = {"qkvz": (16384, 2560), "out": (2560, 6144), "qkv": (13312, 2560), "o": (2560, 6144)}
LENGTHS = (512, 2048, 8192)


def chain(fn, x, n, reps=5):
    """ms per call of n chained calls x <- x + 0 * fn(x)[first element]."""
    def run():
        y = x
        for _ in range(n):
            out = fn(y)
            y = y + (out.reshape(-1)[:1] * 0).astype(y.dtype)
        return y
    mx.eval(run())
    ts = []
    for _ in range(reps):
        mx.synchronize()
        t0 = time.perf_counter()
        mx.eval(run())
        mx.synchronize()
        ts.append((time.perf_counter() - t0) / n)
    return float(np.median(ts)) * 1e3


def rel(a, b):
    a, b = a.astype(mx.float32), b.astype(mx.float32)
    return float((mx.linalg.norm((a - b).reshape(-1)) / mx.linalg.norm(b.reshape(-1))).item())


def dense(rng):
    print("## dense projections (ms per call; speedup vs MLX MXFP4 W4A16)\n")
    print("| layer | tokens | ours | oMLX A8 | MLX MXFP4 | MLX affine4 | ours vs oMLX | ours vs MLX MXFP4 | oMLX vs MLX MXFP4 |")
    print("|---|---|---|---|---|---|---|---|---|")
    S.set_format(F.Format())
    for name, (N, K) in DENSE.items():
        w = rng.standard_normal((N, K)) * 0.02
        codes, bc, ws, _ = F.quantize(w)
        import fused_qwen3 as fq
        pw = fq.PackedWeight([(np.asarray(codes).reshape(-1), bc, ws)])
        wb = mx.array(w.astype(np.float32)).astype(mx.bfloat16)
        mq = mx.quantize(wb, group_size=32, bits=4, mode="mxfp4")
        aq = mx.quantize(wb, group_size=64, bits=4)                      # affine Q4 GS64 (oMLX / MLX 4-bit)
        a_w, a_s, a_b = aq
        o_s, o_b = mx.contiguous(a_s.T), mx.contiguous(a_b.T)             # oMLX plan: metadata transposed once
        mx.eval(pw.codes, mq, aq, o_s, o_b)

        def ours(x):
            M = x.shape[0]
            xq, xs = S.act_quant(x, S.pad_rows(M))
            return S.mm_epi(xq, pw.unpack(), xs, pw.wscale, 1, x.dtype, tile=(64, 128, 4, 8))[:M]

        def omlx(x):
            qa, sa, ra = OF.qwen35_oq_a8_stage_a_v8(x, 0)
            return OF.qwen35_oq_a8_qmm_t(qa, sa, ra, a_w, o_s, o_b, 4, 0, 800)

        mlx_m = lambda x: mx.quantized_matmul(x, *mq, transpose=True, group_size=32, bits=4, mode="mxfp4")  # noqa: E731
        mlx_a = lambda x: mx.quantized_matmul(x, *aq, transpose=True, group_size=64, bits=4)  # noqa: E731
        for L in LENGTHS:
            x = (mx.array(rng.standard_normal((L, K)).astype(np.float32))).astype(mx.bfloat16)
            if L == LENGTHS[0]:  # agreement with the W4A16 results of the same weights (INT8 activations differ)
                print(f"<!-- {name}: rel diff vs MLX on its own weights: ours {rel(ours(x), mlx_m(x)):.4f}, "
                      f"oMLX {rel(omlx(x), mlx_a(x)):.4f} -->")
            n = max(3, int(16 * 512 / L))
            t = {k: chain(f, x, n) for k, f in (("ours", ours), ("omlx", omlx), ("mlx_m", mlx_m), ("mlx_a", mlx_a))}
            print(f"| {name} {N}x{K} | {L} | {t['ours']:.3f} | {t['omlx']:.3f} | {t['mlx_m']:.3f} | {t['mlx_a']:.3f} | "
                  f"{t['omlx'] / t['ours']:.2f}x | {t['mlx_m'] / t['ours']:.2f}x | {t['mlx_m'] / t['omlx']:.2f}x |",
                  flush=True)


def experts(rng):
    print("\n## routed experts, 512 x (2560 x 640), top-10 (ms per call)\n")
    print("| tokens | rows/expert | ours | oMLX (A8 gate+up, A16 down) | MLX MXFP4 | ours vs oMLX | ours vs MLX | oMLX vs MLX |")
    print("|---|---|---|---|---|---|---|---|")
    shape = BM.SHAPES["qwen38"]
    E, D, Fi, k = shape["E"], shape["D"], shape["F"], shape["k"]
    W = BM.Weights(shape, "sc")  # ours: SC-MXFP4 packed experts (E + 1 slots, the last unused here)
    S.set_format(W.fmt)
    bf = lambda a: mx.array(a).astype(mx.bfloat16)  # noqa: E731
    rep = lambda m: mx.contiguous(mx.repeat(bf(m)[None], E, axis=0))  # noqa: E731
    g, u, d = rep(W.deq["gate"]), rep(W.deq["up"]), rep(W.deq["down"])
    gu_aff = mx.quantize(mx.concatenate([g, u], axis=1), group_size=64, bits=4)   # oMLX: gate rows then up rows
    d_aff = mx.quantize(d, group_size=64, bits=4)
    gm, um, dm = (mx.quantize(t, group_size=32, bits=4, mode="mxfp4") for t in (g, u, d))
    del g, u, d
    mx.eval(gu_aff, d_aff, gm, um, dm)
    for L in LENGTHS:
        x = (mx.array(rng.standard_normal((L, D)).astype(np.float32))).astype(mx.bfloat16)
        idx = mx.array(np.stack([rng.choice(E, k, replace=False) for _ in range(L)]).astype(np.int32))
        flat = idx.reshape(-1)
        order = mx.argsort(flat)
        e_sorted = flat[order].astype(mx.uint32)
        row_map = (order // k).astype(mx.uint32)
        R = L * k
        tm = 32 if L * k / E < 24 else 64
        bk = 128 if tm == 32 else 64
        t = (tm, 64, 2 if tm == 32 else 4)
        off, tstart, maxt = sc_moe.segments(e_sorted.astype(mx.int32), E + 1, tm)
        rows = mx.concatenate([row_map.astype(mx.int32), mx.zeros((tm,), mx.int32)])
        mx.eval(e_sorted, row_map, off, tstart, rows)

        def ours(v):
            xq, xs = S.act_quant(v, S.pad_rows(L))
            act = sc_moe.moe_mm(xq[rows], R, *W.gu, off, tstart, maxt, 2 * Fi, 3, xs[rows[:R]], tile=t, bk=bk)
            xq2, xs2 = S.act_quant(act, R + tm)
            return sc_moe.moe_mm(xq2, R, *W.down, off, tstart, maxt, D, 1, xs2, tile=t, bk=bk)

        def omlx(v):
            h = OG.sorted_gather_qmm_a8(v[:, None, :], *gu_aff, e_sorted, row_map, group_size=64, bits=4, swiglu=True)
            return mx.gather_qmm(h, *d_aff, rhs_indices=e_sorted, transpose=True, group_size=64, bits=4,
                                 sorted_indices=True)

        def mlx(v):
            xs_ = v[row_map.astype(mx.int32)][:, None, :]
            gq = lambda a, q: mx.gather_qmm(a, *q, rhs_indices=e_sorted, transpose=True, group_size=32, bits=4,  # noqa: E731
                                            mode="mxfp4", sorted_indices=True)
            h = nn.silu(gq(xs_, gm)) * gq(xs_, um)
            return gq(h, dm)

        if L == LENGTHS[0]:
            assert OG.supports(x[:, None, :], gu_aff[0], gu_aff[1], gu_aff[2], e_sorted, 64, 4, "affine", row_map), \
                "oMLX A8 gather declined these inputs"
            yo, ym = omlx(x).reshape(R, D), mlx(x).reshape(R, D)
            print(f"<!-- experts: rel diff vs MLX MXFP4 W4A16: ours {rel(ours(x), ym):.4f} (SC-MXFP4 W4A8), "
                  f"oMLX {rel(yo, ym):.4f} (affine weights) -->")
        n = max(3, int(16 * 512 / L))
        tt = {nm: chain(f, x, n) for nm, f in (("ours", ours), ("omlx", omlx), ("mlx", mlx))}
        print(f"| {L} | {R / E:.0f} | {tt['ours']:.3f} | {tt['omlx']:.3f} | {tt['mlx']:.3f} | {tt['omlx'] / tt['ours']:.2f}x | "
              f"{tt['mlx'] / tt['ours']:.2f}x | {tt['mlx'] / tt['omlx']:.2f}x |", flush=True)


def main():
    assert OF.oq_a8_available(), "oMLX INT8 kernels unavailable (copy the NAX metallib of the macOS 26 app)"
    rng = np.random.default_rng(0)
    dense(rng)
    experts(rng)


if __name__ == "__main__":
    main()
