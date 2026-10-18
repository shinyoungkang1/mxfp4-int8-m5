"""MoE prefill (time to first token) on OLMoE-1B-7B-0924: fused SC-MXFP4 W4A8 (e2e/fused_olmoe.py) vs MLX's MXFP4
and 4-bit affine checkpoints of the same model (e2e/convert_mlx.py). Protocol of e2e/bench_fused.py.

Methods:
  mlx_mxfp4, mlx_affine4, bf16   mlx_lm (SwitchGLU: tokens sorted by expert, gather_qmm / gather_mm), W4A16 / BF16
  fused                          grouped INT8 expert GEMM, expert weights unpacked in-kernel (4.07 bits/weight);
                                 two-pass (below) once experts average >= 256 rows (2048-token prompts)
  fused_inkernel                 in-kernel unpack at every length (ablation)
  fused_2pass                    all experts unpacked to an INT8 scratch every call, then the grouped GEMM (ablation)

Usage: python bench_moe.py [--lengths 512,2048,4096] [--reps 7] [--out results/07_moe_prefill.csv]
       python bench_moe.py --sweep 512,2048,4096     (grouped-GEMM tile sweep on real routing)
"""
import argparse
import csv
import os
import sys
import time

import mlx.core as mx
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import bench_fused as bf  # noqa: E402
import bench_prefill as bp  # noqa: E402
import fused_olmoe as fo  # noqa: E402
from mlx_lm import load  # noqa: E402

REPO = "allenai/OLMoE-1B-7B-0924"


def loaders(cache_dir):
    base = REPO.split("/")[-1]
    return {
        "bf16": lambda: bf.Plain(load(REPO, model_config=fo.MODEL_CONFIG)[0]),
        "mlx_mxfp4": lambda: bf.Plain(load(os.path.join(cache_dir, base + "-mxfp4"))[0]),
        "mlx_affine4": lambda: bf.Plain(load(os.path.join(cache_dir, base + "-4bit"))[0]),
        "fused": lambda: bf.Fused(fo.load_fused_olmoe(REPO, cache_dir)[0]),
        "fused_inkernel": lambda: bf.Fused(fo.load_fused_olmoe(REPO, cache_dir, two_pass=False)[0]),
        "fused_2pass": lambda: bf.Fused(fo.load_fused_olmoe(REPO, cache_dir, two_pass=True)[0]),
    }


def sweep(cache_dir, lengths):
    """Times the two grouped expert GEMMs of block 0 on real routing of random tokens for several tiles."""
    import sc_kernels as S
    import sc_moe
    f, _ = fo.load_fused_olmoe(REPO, cache_dir)
    f.record, f.record_layers = {}, (0,)
    a = f.args
    ex = f.layers[0][2]
    rng = np.random.default_rng(0)
    for L in lengths:
        ids = mx.array(rng.integers(0, a.vocab_size, (1, L)).astype(np.int32))
        mx.eval(f(ids))
        xq, xq_s, xs_s, act, xq2, xs2, yd, e_sorted, xn = f.record[(0, "experts")]
        R = L * a.num_experts_per_tok
        for tile, bk in (((64, 64, 4), 128), ((64, 64, 4), 64), ((64, 32, 4), 128), ((64, 32, 2), 128),
                         ((128, 64, 4), 128), ((128, 64, 4), 64), ((64, 128, 4), 64), ((64, 64, 2), 128),
                         ((32, 64, 2), 128), ((64, 128, 8), 128)):
            off, tstart, maxt = sc_moe.segments(e_sorted, a.num_experts, tile[0])
            pad = mx.zeros((tile[0], xq_s.shape[1]), mx.int8)
            x1 = mx.concatenate([xq_s[:R], pad])
            x2 = mx.concatenate([xq2[:R], mx.zeros((tile[0], xq2.shape[1]), mx.int8)])
            t1 = bench(lambda: sc_moe.moe_mm(x1, R, *ex["gu"], off, tstart, maxt, 2 * a.intermediate_size, 3, xs_s,
                                              tile=tile, bk=bk))
            t2 = bench(lambda: sc_moe.moe_mm(x2, R, *ex["down"], off, tstart, maxt, a.hidden_size, 1, xs2,
                                              tile=tile, bk=bk))
            ops = 2 * R * 3 * a.intermediate_size * a.hidden_size
            print(f"L={L} tile {tile} bk {bk}: gate/up {t1:.4f} ms  down {t2:.4f} ms  ({ops / (t1 + t2) / 1e9:.0f} TOPS)",
                  flush=True)
        w1 = sc_moe.unpack_experts(ex["gu"][0], ex["gu"][1], 2 * a.intermediate_size, a.hidden_size)
        tu = bench(lambda: sc_moe.unpack_experts(ex["gu"][0], ex["gu"][1], 2 * a.intermediate_size, a.hidden_size))
        off, tstart, maxt = sc_moe.segments(e_sorted, a.num_experts, 64)
        t3 = bench(lambda: sc_moe.moe_mm_i8(xq_s, R, w1, ex["gu"][2], off, tstart, maxt, 2 * a.intermediate_size, 3,
                                             xs_s))
        print(f"L={L} two-pass gate/up: unpack all experts {tu:.4f} ms + grouped GEMM {t3:.4f} ms", flush=True)
        f.record = {}


def bench(fn, n=20, reps=7):
    mx.eval(fn())
    ts = []
    for _ in range(reps):
        mx.synchronize()
        t0 = time.perf_counter()
        mx.eval([fn() for _ in range(n)])
        mx.synchronize()
        ts.append((time.perf_counter() - t0) / n)
    return float(np.median(ts)) * 1e3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", default=os.path.expanduser("~/.cache/mxfp4-int8-m5"))
    ap.add_argument("--methods", default="mlx_mxfp4,mlx_affine4,fused,fused_inkernel,fused_2pass")
    ap.add_argument("--lengths", default="512,2048,4096")
    ap.add_argument("--reps", type=int, default=7)
    ap.add_argument("--regime", default="")
    ap.add_argument("--sweep", default="")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    if a.sweep:
        sweep(a.cache_dir, [int(x) for x in a.sweep.split(",")])
        return
    ld = loaders(a.cache_dir)
    models = {m: ld[m]() for m in a.methods.split(",")}
    rows = bf.run(models, [int(x) for x in a.lengths.split(",")], a.reps, 50304)
    state = bp.power_state()
    print("#", state)
    if a.out:
        with open(a.out, "w", newline="") as f:
            f.write(f"# {state}\n")
            wr = csv.DictWriter(f, fieldnames=["model", "regime"] + list(rows[0]))
            wr.writeheader()
            for r in rows:
                wr.writerow(dict(model=REPO, regime=a.regime, **r))
        print("wrote", a.out)


if __name__ == "__main__":
    main()
