"""Tile sweeps behind the configuration choices of results/07 (run on an idle GPU, through run_timed.sh):
  gemv     decode GEMV (rows per simdgroup, activations per lane, simdgroups) at the Qwen3 / OLMoE shapes vs MLX qmv
  prefill  INT8 GEMM + in-register epilogue (mm_epi) tiles at the Qwen3 shapes for M = 512 / 2048 / 8192
Timings: min over 30 rounds of 30 back-to-back calls (robust to background noise), microseconds.
Usage: python tune_07.py gemv|prefill
"""
import os
import sys
import time

import mlx.core as mx
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sc_decode as G  # noqa: E402
import sc_kernels as S  # noqa: E402


def bench(fn, n=30, reps=30):
    mx.eval(fn())
    ts = []
    for _ in range(reps):
        mx.synchronize()
        t0 = time.perf_counter()
        mx.eval([fn() for _ in range(n)])
        mx.synchronize()
        ts.append((time.perf_counter() - t0) / n)
    return float(np.min(ts)) * 1e6


def gemv_sweep():
    rng = np.random.default_rng(0)
    tiles = [(4, 16, 2), (4, 32, 2), (8, 16, 1), (8, 16, 2), (4, 16, 4), (8, 32, 1), (2, 16, 4), (4, 8, 2)]
    shapes = [("qwen3 qkv", 4096, 2048, 1), ("qwen3 o", 2048, 2048, 1), ("qwen3 gate|up", 12288, 2048, 1),
              ("qwen3 down", 2048, 6144, 1), ("olmoe qkv", 6144, 2048, 1), ("olmoe expert gate|up x8", 2048, 2048, 8),
              ("olmoe expert down x8", 2048, 1024, 8)]
    for name, N, K, slots in shapes:
        codes = mx.array(rng.integers(0, 2 ** 32, (slots, N * K // 8), dtype=np.uint32))
        bc = mx.array(rng.integers(0, 256, (slots, N, K // 128)).astype(np.uint8))
        ws = mx.ones((slots, N), mx.float32)
        xs = mx.ones((slots,), mx.float32)
        xq = mx.array(rng.integers(-127, 128, (slots, K)).astype(np.int8))
        eidx = mx.arange(slots, dtype=mx.int32) if slots > 1 else None
        x = mx.array(rng.standard_normal((1, K)).astype(np.float32)).astype(mx.bfloat16)
        W = mx.array(rng.standard_normal((slots, N, K)).astype(np.float32) * 0.02).astype(mx.bfloat16)
        wm, sm = mx.quantize(W, group_size=32, bits=4, mode="mxfp4")
        if slots == 1:
            t_mlx = bench(lambda: mx.quantized_matmul(x, wm[0], sm[0], transpose=True, group_size=32, bits=4,
                                                      mode="mxfp4"))
        else:
            idx = mx.arange(slots, dtype=mx.uint32)[None]
            t_mlx = bench(lambda: mx.gather_qmm(x[:, None, None], wm, sm, rhs_indices=idx, transpose=True,
                                                group_size=32, bits=4, mode="mxfp4"))
        res = []
        for R, V, nsg in tiles:
            if K % (32 * V) or N % (R * nsg):
                continue
            t = bench(lambda: G.gemv(xq if slots > 1 else xq[0], codes if slots > 1 else codes[0],
                                     bc if slots > 1 else bc[0], xs, ws if slots > 1 else ws[0], N, 1, rows=R, vpl=V,
                                     nsg=nsg, eidx=eidx, xstride=1 if slots > 1 else 0))
            res.append(((R, V, nsg), t))
        best = min(res, key=lambda r: r[1])
        gb = slots * N * K * 4.07 / 8 / 1e9
        print(f"{name:26s} N={N:5d} K={K:4d}: MLX mxfp4 {t_mlx:6.1f} us | " + " ".join(
            f"{t[0]}x{t[1]}x{t[2]}:{v:.1f}" for t, v in res) + f" | best {best[0]} {best[1]:.1f} us "
            f"({gb / best[1] * 1e6:.0f} GB/s, {t_mlx / best[1]:.2f}x MLX)", flush=True)


def prefill_sweep():
    rng = np.random.default_rng(0)
    tiles = [(128, 64, 4, 8), (64, 128, 4, 8), (64, 64, 4, 8), (128, 128, 4, 8)]
    for M in (512, 2048, 8192):
        xs = mx.ones((M,), mx.float32) * 1e-3
        res = mx.array(rng.standard_normal((M, 2048)).astype(np.float32)).astype(mx.bfloat16)
        for name, (N, K, epi) in {"qkv": (4096, 2048, 1), "o": (2048, 2048, 2), "gu": (12288, 2048, 3),
                                  "down": (2048, 6144, 2)}.items():
            xq = mx.array(rng.integers(-127, 128, (M, K)).astype(np.int8))
            wq = mx.array(rng.integers(-96, 97, (N, K)).astype(np.int8))
            ws = mx.ones((N,), mx.float32) * 1e-4
            out = []
            for t in tiles:
                v = bench(lambda: S.mm_epi(xq, wq, xs, ws, epi, mx.bfloat16, res=res if epi == 2 else None, tile=t),
                          n=10, reps=10)
                out.append((t, v))
            best = min(out, key=lambda r: r[1])
            print(f"M={M:5d} {name:5s} N={N:5d} K={K:4d}: " + " ".join(f"{t[0]}x{t[1]}:{v:.0f}" for t, v in out)
                  + f" us | best {best[0][:2]} {2 * M * N * K / best[1] / 1e6:.0f} TOPS", flush=True)


if __name__ == "__main__":
    {"gemv": gemv_sweep, "prefill": prefill_sweep}[sys.argv[1]]()
