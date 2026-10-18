"""Exponent-spread statistics of Qwen3.8-Flash-Next (official FP8 checkpoint, experts FP8-decoded): the fraction of
weight rows that are INT8-embeddable as standard MXFP4 (OCP exponents with MSE search, spread <= 3), and for the
violating rows whether a large block (outlier) or a small block (near-zero) causes it; the analysis of
sim/spread_analysis.py and sim/analyze_weights.py, over every matrix that runs in W4A8 (all 48 layers, all 512
experts, the shared experts, DeltaNet and attention projections).

Usage (.venv-q38): python q38_spread.py > ../results/10_spread_qwen38.md
"""
import os
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "sim"))
import mx_int8 as q  # noqa: E402


def stats(w):
    """-> rows, violating rows, caused by a large block, caused by a small block."""
    w = np.asarray(w, np.float64)
    e = q.block_exponents(w, "search")
    bad = (q.exponent_spread(e) > 3).ravel()
    up = down = 0
    if bad.any():
        amax = np.abs(q.blocks(w[bad])).max(-1)
        med = np.median(amax, 1)
        u = np.log2(amax.max(1) / med)
        d = np.log2(med / np.maximum(amax.min(1), 1e-300))
        up, down = int((u >= d).sum()), int((u < d).sum())
    return w.shape[0], int(bad.sum()), up, down


def _job(a):
    kind, w = a
    return kind, stats(w)


def main():
    import mlx.core as mx
    import q38_ckpt as C
    import q38_ppl as P
    from gpu_guard import eval_step
    ck = C.Checkpoint()
    args = ck.text_args()
    acc = defaultdict(lambda: np.zeros(4, np.int64))
    per_layer = []
    guard = C.MemoryGuard()
    with ProcessPoolExecutor(14) as pool:
        for i in range(args.num_hidden_layers):
            with eval_step():
                w = ck.layer_weights(i, args.num_experts)
            lay = np.zeros(4, np.int64)

            def run(jobs):
                for kind, s in pool.map(_job, jobs, chunksize=4):
                    acc[kind] += s
                    lay[:] += s
            kinds = P.DENSE[args.layer_types[i]]
            run([(p.split(".")[-1], np.array(w[p + ".weight"].astype(mx.float32))) for p in kinds])
            run([("shared_" + p.split(".")[-1], np.array(w[p + ".weight"].astype(mx.float32))) for p in P.SHARED])
            for p in P.EXPERTS:  # 64 experts at a time (peak memory)
                ew = w[p + ".weight"]
                for e0 in range(0, ew.shape[0], 64):
                    run([("expert_" + p.split(".")[-1], x) for x in np.array(ew[e0:e0 + 64].astype(mx.float32))])
            del w
            guard.check(f"spread layer {i}")
            per_layer.append(lay.copy())
            print(f"layer {i}: {lay[1] / lay[0]:.3%} of rows violate", file=sys.stderr, flush=True)
    tot = sum(acc.values())
    print(f"## {C.REPO}: exponent spread of the W4A8 matrices (FP8-decoded experts)\n")
    print(f"All matrices: {tot[0]:,} rows, **{1 - tot[1] / tot[0]:.3%} INT8-embeddable as standard MXFP4** "
          f"(spread <= 3); violating rows caused by a large block {tot[2]:,}, by a small block {tot[3]:,}.\n")
    print("| matrix | rows | spread > 3 | caused by a large block | caused by a small block |\n|---|---|---|---|---|")
    for kind, (n, v, up, down) in sorted(acc.items()):
        print(f"| {kind} | {n:,} | {v / n:.3%} | {up:,} | {down:,} |")
    print("\nPer layer (fraction of rows with spread > 3): "
          + ", ".join(f"{i}: {l[1] / l[0]:.2%}" for i, l in enumerate(per_layer)))


if __name__ == "__main__":
    main()
