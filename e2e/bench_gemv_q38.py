"""Batch-1 decode kernels at Qwen3.8-Flash-Next shapes, ours (sc_decode.gemv on the packed SC-MXFP4 layout, per-token
INT8 inside the kernel) vs MLX MXFP4 (quantized_matmul / gather_qmm, W4A16), random weights, microseconds per call.
Calls are chained (each input depends on the previous output through a one-element op, as consecutive layers), so
the numbers are latencies; GB/s = weight bytes read per call / time.
  dense: DeltaNet qkvz (16384 x 2560), out (2560 x 6144), attention qkv (13312 x 2560), o (2560 x 6144)
  MoE:   ours = gate|up GEMV (10 routed + shared, SwiGLU epilogue) + down GEMV (11 slots);
         MLX  = mlx_lm SwitchGLU decode (gather_qmm gate, up, swiglu, gather_qmm down; 10 experts) + shared expert MLP
Usage (.venv): python bench_gemv_q38.py   (under run_timed.sh)
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
import sc_decode as G  # noqa: E402
import sc_format as F  # noqa: E402
import sc_kernels as S  # noqa: E402

DENSE = {"qkvz": (16384, 2560), "out": (2560, 6144), "qkv": (13312, 2560), "o": (2560, 6144)}


def chain(fn, x, n=200, reps=5):
    """us per call of n chained calls x <- x + 0 * fn(x)[0]."""
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
    return float(np.median(ts)) * 1e6


def main():
    S.set_format(F.Format())
    rng = np.random.default_rng(0)
    print("| kernel | ours (us) | MLX MXFP4 (us) | ours GB/s | MLX GB/s | ours / MLX |\n|---|---|---|---|---|---|")
    for name, (N, K) in DENSE.items():
        w = rng.standard_normal((N, K)) * 0.02
        codes, bc, ws, _ = F.quantize(w)
        codes, bc, ws = mx.array(np.asarray(codes).reshape(-1)), mx.array(bc), mx.array(ws)
        wq, sc = mx.quantize(mx.array(w.astype(np.float32)).astype(mx.bfloat16), group_size=32, bits=4, mode="mxfp4")
        x = mx.array(rng.standard_normal(K).astype(np.float32)).astype(mx.bfloat16)
        ours = chain(lambda v: G.gemv(v, codes, bc, None, ws, N, 1, pro=1), x)
        variants = {}
        for t in ((4, 16, 2), (8, 16, 2), (8, 16, 4), (4, 32, 2), (16, 16, 1)):  # separate per-token INT8 (pro=0)
            def f0(v, t=t):
                xq, xs = S.act_quant(v[None], 1)
                return G.gemv(xq, codes, bc, xs, ws, N, 1, rows=t[0], vpl=t[1], nsg=t[2])
            variants[f"pro0 {t}"] = chain(f0, x)
        for t in ((8, 16, 2), (16, 16, 1), (8, 32, 2)):  # in-kernel INT8 (pro=1), more rows per threadgroup
            variants[f"pro1 {t}"] = chain(lambda v, t=t: G.gemv(v, codes, bc, None, ws, N, 1, pro=1, rows=t[0], vpl=t[1],
                                                                nsg=t[2]), x)
        print(f"<!-- {name}: " + ", ".join(f"{k} {v:.1f}" for k, v in variants.items()) + " -->", flush=True)
        mlx = chain(lambda v: mx.quantized_matmul(v[None], wq, sc, transpose=True, group_size=32, bits=4, mode="mxfp4"), x)
        b_ours = codes.nbytes + bc.nbytes + ws.nbytes
        b_mlx = wq.nbytes + sc.nbytes
        print(f"| {name} {N}x{K} | {ours:.1f} | {mlx:.1f} | {b_ours / ours / 1e3:.0f} | {b_mlx / mlx / 1e3:.0f} | {ours / mlx:.2f} |",
              flush=True)
    # MoE decode at Qwen3.8 shapes
    shape = BM.SHAPES["qwen38"]
    W = BM.Weights(shape, "sc")
    E, D, Fi, k = shape["E"], shape["D"], shape["F"], shape["k"]
    bf = lambda a: mx.array(a).astype(mx.bfloat16)  # noqa: E731
    q = {}
    for nme in ("gate", "up", "down"):
        m = bf(W.deq[nme])
        q[nme] = mx.quantize(mx.contiguous(mx.repeat(m[None], E, axis=0)), group_size=32, bits=4, mode="mxfp4")
        q["s_" + nme] = mx.quantize(m, group_size=32, bits=4, mode="mxfp4")
    mx.eval(q)
    idx = mx.array(rng.choice(E, k, replace=False).astype(np.uint32))
    x = mx.array(rng.standard_normal(D).astype(np.float32)).astype(mx.bfloat16)

    def ours_moe(v):
        act = G.gemv(v, *W.gu[:2], None, W.gu[2], 2 * Fi, 3, eidx=idx, xstride=0, shared=E, pro=1)
        return G.gemv(act, *W.down[:2], None, W.down[2], D, 1, eidx=idx, xstride=1, shared=E, pro=1)

    def gq(a, nme):
        return mx.gather_qmm(a, *q[nme], rhs_indices=idx[None, None], transpose=True, group_size=32, bits=4, mode="mxfp4")

    def qm(a, nme):
        return mx.quantized_matmul(a, *q["s_" + nme], transpose=True, group_size=32, bits=4, mode="mxfp4")

    def mlx_moe(v):
        x5 = v.reshape(1, 1, 1, 1, D)
        h = nn.silu(gq(x5, "gate")) * gq(x5, "up")
        y = gq(h, "down")
        s = qm(nn.silu(qm(v[None], "gate")) * qm(v[None], "up"), "down")
        return y.sum(axis=-3).reshape(-1) + s.reshape(-1)

    ours = chain(ours_moe, x)
    mlx = chain(mlx_moe, x)
    b = (k + 1) * (2 * Fi * D + D * Fi) * 4.07 / 8
    print(f"| MoE block (10 + shared) | {ours:.1f} | {mlx:.1f} | {b / ours / 1e3:.0f} | {b / mlx / 1e3:.0f} | {ours / mlx:.2f} |")


if __name__ == "__main__":
    main()
