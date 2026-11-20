"""What 8-bit dense projections would cost a Qwen3.5-35B-A3B decode step (PLAN.md D6, first experiment).

The 35B simulation rows with 8-bit (per-row INT8) dense projections remove most of the 4-bit loss but read 37-51% more
bytes per decode step. Bytes do not predict time (B4: 8-wide reads 10.7% more bytes than SC-MXFP4 in the same time),
so this times MLX's own batch-1 quantized GEMVs at 4 and 8 bits on the four dense shapes of the 35B decode step, plus
our two 4-bit formats for reference, all in one process on synthetic weights (no model load):
  mlx_mxfp4      MLX MXFP4 (group 32, E8M0 scales)            MLX's 4-bit baseline in Table 3
  mlx_affine4    MLX affine 4-bit, group 64
  mlx_affine8    MLX affine 8-bit, group 64                    MLX's default 8-bit
  mlx_affine8g   MLX affine 8-bit, group 128                   closest MLX layout to per-row INT8 (one scale per row)
  mlx_mxfp8      MLX MXFP8 (group 32), if this MLX has it
  ours_sc        our SC-MXFP4 GEMV (INT8 activations)
  ours_b8        our 8-wide GEMV (INT8 activations)
Shapes (N x K, layers) as bench_gemv_qwen35.py: qkvz 12288x2048 (30), out 2048x4096 (30), qkv 9216x2048 (10),
o 2048x4096 (10). MLX methods get a bf16 input row; ours quantize it to INT8 first (as DecodeQwen35.step).
40 weight sets per shape and method, cycled so that every call streams from DRAM; chained calls (latencies),
median of 7 x 240 calls. The summary sums layers x time per method: the dense-projection GEMV time of one step.
Usage: python bench_gemv_bits_qwen35.py [--out results/x.csv]   (under run_timed.sh, VALIDATE="")
"""
import argparse
import csv
import os
import sys

import mlx.core as mx
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "sim"))
sys.path.insert(0, HERE)
import provenance as prov  # noqa: E402
import bench_prefill as bp  # noqa: E402
import bench_gemv_qwen35 as BG  # noqa: E402
import pack_format as pf  # noqa: E402
import sc_decode as G  # noqa: E402
import sc_format as F  # noqa: E402
import sc_kernels as S  # noqa: E402

DENSE = ["qkvz", "out", "qkv", "o"]
MLX = {"mlx_mxfp4": ("mxfp4", 4, 32), "mlx_affine4": ("affine", 4, 64), "mlx_affine8": ("affine", 8, 64),
       "mlx_affine8g": ("affine", 8, 128), "mlx_mxfp8": ("mxfp8", 8, 32)}


def mlx_set(N, K, mode, bits, gs, seed):
    w = (mx.random.normal((N, K), key=mx.random.key(seed)) * 0.02).astype(mx.bfloat16)
    q = mx.quantize(w, group_size=gs, bits=bits, mode=mode)
    q = tuple(a for a in q)
    if len(q) == 2:
        q = q + (None,)
    mx.eval([a for a in q if a is not None])
    return q


def mlx_fn(mode, bits, gs):
    def fn(s, v):
        w, sc, b = s
        return mx.quantized_matmul(v[None], w, sc, b, transpose=True, group_size=gs, bits=bits, mode=mode)
    return fn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="")
    ap.add_argument("--methods", default=",".join(list(MLX) + ["ours_sc", "ours_b8"]))
    a = ap.parse_args()
    methods = a.methods.split(",")
    ours = {"ours_sc": F.Format(), "ours_b8": pf.sc_format_of("e2m1/B8/M1..10")}
    state = bp.power_state()
    rng = np.random.default_rng(0)
    rows = []
    for name in DENSE:
        N, K, layers = BG.SHAPES[name]
        x = mx.random.normal((K,), key=mx.random.key(1)).astype(mx.bfloat16)
        for m in methods:
            try:
                if m in MLX:
                    mode, bits, gs = MLX[m]
                    sets = [mlx_set(N, K, mode, bits, gs, 1000 + i) for i in range(BG.NSETS)]
                    fn = mlx_fn(mode, bits, gs)
                    xin = x
                else:
                    S.set_format(ours[m])
                    sets = [BG.make_set(rng, ours[m], N, K) for _ in range(BG.NSETS)]
                    fn, xin = BG.caller(name, N, K, G.GEMV_TILE)
                byte = sum(t.nbytes for t in sets[0] if t is not None)
                us = BG.chain(fn, sets, xin)
            except Exception as e:  # e.g. a quantization mode this MLX does not have
                print(f"{name} {m}: skipped ({str(e).splitlines()[0][:100]})", flush=True)
                continue
            finally:
                sets = None
                mx.clear_cache()
            rows.append(dict(shape=name, N=N, K=K, layers=layers, method=m, us=f"{us:.2f}", bytes=byte,
                             bits_w=f"{byte * 8 / (N * K):.3f}", gb_s=f"{byte / us / 1e3:.0f}"))
            print(f"{name:5s} {N}x{K} {m:13s} {us:7.2f} us  {byte * 8 / (N * K):5.2f} bits/w  {byte / us / 1e3:5.0f} GB/s",
                  flush=True)
    got = [m for m in methods if any(r["method"] == m for r in rows)]
    t = {(r["shape"], r["method"]): float(r["us"]) for r in rows}
    print("\n| method | " + " | ".join(f"{s} us" for s in DENSE) + " | step ms | vs mlx_mxfp4 |")
    print("|---|" + "---|" * (len(DENSE) + 2))
    step = {m: sum(BG.SHAPES[s][2] * t.get((s, m), np.nan) for s in DENSE) / 1e3 for m in got}
    for m in got:
        print(f"| {m} | " + " | ".join(f"{t.get((s, m), np.nan):.1f}" for s in DENSE)
              + f" | {step[m]:.3f} | {step[m] - step.get('mlx_mxfp4', np.nan):+.3f} ms |")
    print("#", state)
    if a.out:
        with open(a.out, "w", newline="") as fh:
            fh.write(prov.csv_head(state))
            wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
            wr.writeheader()
            wr.writerows(rows)
        prov.sidecar(a.out, fresh=True, power=state, model="Qwen/Qwen3.5-35B-A3B-Base (shapes only, synthetic weights)",
                     what="gemv_bits", format=got, step_ms=step)
        print("wrote", a.out)


if __name__ == "__main__":
    main()
