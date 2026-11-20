"""Batch-1 decode GEMVs at the six Qwen3.5-35B-A3B shapes, SC-MXFP4 against E2M1 8-wide (PLAN.md step B4; audit
results/AUDIT_2026-11-05.md, M2), on synthetic packs, no model load.

The 35B 8-wide decode step is about 11-12.5% slower than SC-MXFP4 in the same session, while 8-wide reads only 10.7%
more weight bytes (and 7.7% more bytes per step in all). This times each GEMV of a decode step in both formats and
compares the time ratio with the byte ratio of that shape, then tries the audit's two code changes for 8-wide:
  inline   sc_mult(i) = i + 1 as ALU code instead of a lookup in a constant table of 10 values (same outputs)
  tiles    rows per simdgroup R, activations per lane per step V, simdgroups per threadgroup NSG
Shapes (N x K, as the fused 35B path calls them):
  qkvz  12288 x 2048  DeltaNet in_proj_qkv | in_proj_z   (30 layers)
  out    2048 x 4096  DeltaNet out_proj                  (30 layers)
  qkv    9216 x 2048  attention q | k | v                (10 layers)
  o      2048 x 4096  attention o_proj                   (10 layers)
  gu     1024 x 2048  expert gate | up, 8 routed + shared slot, SwiGLU epilogue   (40 layers)
  down   2048 x  512  expert down, 9 slots, per-row INT8 in the kernel           (40 layers)
Weights: random codes, valid random block codes, 40 independent weight sets per shape and format, cycled so that
every call streams its weights from DRAM (each set's 9 expert slots are distinct experts). Calls are chained (each
input depends on the previous output through a one-element op), so the times are latencies, median of 7 x 240 calls.
Usage: python bench_gemv_qwen35.py [--sweep] [--out results/x.csv]   (under run_timed.sh)
"""
import argparse
import csv
import dataclasses
import os
import sys
import time

import mlx.core as mx
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "sim"))
sys.path.insert(0, HERE)
import provenance as prov  # noqa: E402
import bench_prefill as bp  # noqa: E402
import pack_format as pf  # noqa: E402
import sc_decode as G  # noqa: E402
import sc_format as F  # noqa: E402
import sc_kernels as S  # noqa: E402

SHAPES = {"qkvz": (12288, 2048, 30), "out": (2048, 4096, 30), "qkv": (9216, 2048, 10), "o": (2048, 4096, 10),
          "gu": (1024, 2048, 40), "down": (2048, 512, 40)}
NSETS, NEXP, K_ROUTED = 40, 9, 8
TILES = [(4, 16, 2), (2, 16, 2), (8, 16, 2), (4, 8, 2), (4, 32, 2), (8, 8, 2), (8, 32, 2), (4, 16, 4), (8, 16, 1)]


def make_set(rng, fmt, N, K, E=None):
    """Random packed weights in format fmt: codes uint32, bcodes uint8 (each field < len(mult)), wscale f32."""
    lead = () if E is None else (E,)
    codes = rng.integers(0, 2 ** 32, lead + (N * K // 8,), dtype=np.uint64).astype(np.uint32)
    nb = K // fmt.block
    fields = rng.integers(0, len(fmt.mult), lead + (N, nb // fmt.per_byte, fmt.per_byte)).astype(np.uint8)
    bc = np.zeros(lead + (N, nb // fmt.per_byte), np.uint8)
    for i in range(fmt.per_byte):
        bc |= (fields[..., i] << (fmt.bc_bits * i)).astype(np.uint8)
    ws = (rng.random(lead + (N,)) * 1e-3 + 1e-4).astype(np.float32)
    return mx.array(codes), mx.array(bc), mx.array(ws)


def nbytes(s):
    return sum(a.nbytes for a in s)


def chain(fn, sets, x, n=240, reps=7):
    """us per call of n chained calls cycling through the weight sets."""
    def run():
        y = x
        for i in range(n):
            out = fn(sets[i % len(sets)], y)
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
    return float(np.median(ts)) * 1e6


def caller(name, N, K, tile):
    R, V, nsg = tile
    eidx = mx.arange(K_ROUTED, dtype=mx.int32)
    if name == "gu":  # one input row, INT8 outside (as DecodeQwen35's lean MoE), SwiGLU epilogue, 9 slots
        def fn(s, v):
            xq, xs = S.act_quant(v[None], 1)
            return G.gemv(xq, *s[:2], xs, s[2], N, 3, eidx=eidx, xstride=0, shared=NEXP - 1, rows=R, vpl=V, nsg=nsg)
        x = mx.random.normal((K,), key=mx.random.key(1)).astype(mx.bfloat16)
    elif name == "down":  # 9 input rows, per-row INT8 in the kernel
        def fn(s, v):
            return G.gemv(v, *s[:2], None, s[2], N, 1, eidx=eidx, xstride=1, shared=NEXP - 1, pro=1, rows=R, vpl=V, nsg=nsg)
        x = mx.random.normal((NEXP, K), key=mx.random.key(1)).astype(mx.bfloat16)
    else:  # dense: INT8 input row from the preceding norm / quant op, as in DecodeQwen35.step
        def fn(s, v):
            xq, xs = S.act_quant(v[None], 1)
            return G.gemv(xq, *s[:2], xs, s[2], N, 1, rows=R, vpl=V, nsg=nsg)
        x = mx.random.normal((K,), key=mx.random.key(1)).astype(mx.bfloat16)
    return fn, x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", action="store_true", help="also sweep the tiles for both formats")
    ap.add_argument("--shapes", default=",".join(SHAPES))
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    sc = F.Format()
    b8 = pf.sc_format_of("e2m1/B8/M1..10")
    fmts = {"scmxfp4": sc, "b8": b8, "b8_inline": dataclasses.replace(b8, inline_mult=True)}
    state = bp.power_state()
    rng = np.random.default_rng(0)
    rows = []
    for name in a.shapes.split(","):
        N, K, layers = SHAPES[name]
        E = NEXP if name in ("gu", "down") else None
        sets = {}
        for fl, fmt in fmts.items():
            if fl == "b8_inline":
                sets[fl] = sets["b8"]
                continue
            sets[fl] = [make_set(rng, fmt, N, K, E) for _ in range(NSETS)]
        byte = {fl: nbytes(s[0]) for fl, s in sets.items()}  # bytes read per call (experts: the set's 9 slots)
        outs = []
        for fl in ("b8", "b8_inline"):  # the inline multiplier must not change a single output
            S.set_format(fmts[fl])
            fn, x = caller(name, N, K, G.GEMV_TILE)
            outs.append(fn(sets[fl][0], x))
        same = bool(mx.array_equal(outs[0], outs[1]).item())
        print(f"{name}: 8-wide outputs with inline multiplier identical to the table: {same}", flush=True)
        if not same:
            raise SystemExit("inline sc_mult changed the outputs")
        tiles = TILES if a.sweep else [G.GEMV_TILE]
        for fl, fmt in fmts.items():
            S.set_format(fmt)
            for t in tiles:
                R, V, nsg = t
                if K % V or K % 128 or N % (R * nsg) or (name == "gu" and R % 4):
                    continue
                fn, x = caller(name, N, K, t)
                try:
                    us = chain(fn, sets[fl], x)
                except Exception as e:  # a tile this kernel does not take
                    print(f"{name} {fl} {t}: skipped ({str(e).splitlines()[0][:80]})", flush=True)
                    continue
                rows.append(dict(shape=name, N=N, K=K, layers=layers, format=fl, R=R, V=V, NSG=nsg, us=f"{us:.2f}",
                                 bytes=byte[fl], gb_s=f"{byte[fl] / us / 1e3:.0f}"))
                print(f"{name:5s} {N}x{K} {fl:10s} tile {t}: {us:7.2f} us  {byte[fl] / us / 1e3:5.0f} GB/s", flush=True)
        del sets
        mx.clear_cache()
    # summary: default tile and best tile per shape and format, ratio against SC-MXFP4 at its default tile
    print("\n| shape | N x K | bytes 8w/SC | SC us | 8w us | 8w/SC | 8w inline us | inline/SC | best 8w (tile) | best/SC |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    summ = []
    for name in a.shapes.split(","):
        N, K, layers = SHAPES[name]
        rs = [r for r in rows if r["shape"] == name]
        dt = lambda fl: next((float(r["us"]) for r in rs if r["format"] == fl and (r["R"], r["V"], r["NSG"]) == G.GEMV_TILE), np.nan)  # noqa: E731
        best = min((r for r in rs if r["format"].startswith("b8")), key=lambda r: float(r["us"]))
        bsc = next(r["bytes"] for r in rs if r["format"] == "scmxfp4")
        b8b = next(r["bytes"] for r in rs if r["format"] == "b8")
        s0 = dt("scmxfp4")
        summ.append((name, layers, s0, dt("b8"), dt("b8_inline"), float(best["us"])))
        print(f"| {name} | {N}x{K} | {b8b / bsc:.3f} | {s0:.1f} | {dt('b8'):.1f} | {dt('b8') / s0:.3f} | {dt('b8_inline'):.1f} | "
              f"{dt('b8_inline') / s0:.3f} | {float(best['us']):.1f} ({best['format']} {best['R']},{best['V']},{best['NSG']}) | "
              f"{float(best['us']) / s0:.3f} |")
    tot = [sum(s[1] * s[i] for s in summ) / 1e3 for i in (2, 3, 4, 5)]
    print(f"\nGEMV time per decode step (sum over layers, ms): SC {tot[0]:.3f}, 8-wide {tot[1]:.3f} ({tot[1] / tot[0]:.3f}x), "
          f"8-wide inline {tot[2]:.3f} ({tot[2] / tot[0]:.3f}x), 8-wide best tiles {tot[3]:.3f} ({tot[3] / tot[0]:.3f}x)")
    print("#", state)
    if a.out:
        with open(a.out, "w", newline="") as fh:
            fh.write(prov.csv_head(state))
            wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
            wr.writeheader()
            wr.writerows(rows)
        prov.sidecar(a.out, fresh=True, power=state, model="Qwen/Qwen3.5-35B-A3B-Base (shapes only, synthetic weights)",
                     what="gemv", format=list(fmts), step_ms=dict(zip(("sc", "b8", "b8_inline", "b8_best"), tot)))
        print("wrote", a.out)


if __name__ == "__main__":
    main()
