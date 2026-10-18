"""Why do weight rows satisfy the spread <= 3 condition, and what breaks it?

1. Null model: rows of i.i.d. Gaussian (and Student-t) weights. With E8M0 exponents from the OCP rule,
   spread <= ceil(log2(max_b amax_b / min_b amax_b)), so spread > 3 needs a row whose largest block maximum is
   at least 8x its smallest. For i.i.d. rows the block maxima of 32 values concentrate (ratio ~2), so this
   essentially never happens; we estimate the rate by simulation.
2. Real weights: for every row that violates the condition, is the cause a large block (max amax far above the
   row's median block) or a small block (min amax far below it)? Reported per layer type.

Usage: python spread_analysis.py <hf-repo> > ../results/spread_analysis_<model>.md
"""
import glob
import os
import sys
from collections import defaultdict

import mlx.core as mx

mx.set_default_device(mx.cpu)
import numpy as np
from huggingface_hub import snapshot_download

import mx_int8 as q
from gpu_guard import eval_step


def null_model(k_values=(1024, 2048, 4096, 6144), rows=50_000, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for dist in ("gaussian", "student-t (df=4)"):
        for k in k_values:
            viol, ratios = 0, []
            for _ in range(rows // 10_000):
              with eval_step():  # CPU-heavy: yield to timing runs between chunks
                w = rng.standard_normal((10_000, k)) if dist == "gaussian" else rng.standard_t(4, (10_000, k))
                e = q.block_exponents(w, "search")
                viol += int((q.exponent_spread(e) > 3).sum())
                amax = np.abs(q.blocks(w)).max(-1)
                ratios.append(np.log2(amax.max(1) / amax.min(1)))
            r = np.concatenate(ratios)
            out.append((dist, k, viol / rows, float(np.median(r)), float(np.percentile(r, 99.99))))
    return out


def real_model(repo):
    path = snapshot_download(repo, allow_patterns=["*.safetensors"])
    per_kind = defaultdict(lambda: np.zeros(4))  # rows, violating, cause=large block, cause=small block
    for fn in sorted(glob.glob(os.path.join(path, "*.safetensors"))):
        with eval_step():
            tensors = mx.load(fn)
            for name, t in tensors.items():
                if t.ndim != 2 or ".layers." not in name or not name.endswith("proj.weight"):
                    continue
                w = np.array(t.astype(mx.float32), dtype=np.float64)
                e = q.block_exponents(w, "search")
                bad = (q.exponent_spread(e) > 3).ravel()  # one chunk = whole row
                kind = name.split(".")[-2]
                acc = per_kind[kind]
                acc[0] += w.shape[0]; acc[1] += bad.sum()
                if bad.any():
                    amax = np.abs(q.blocks(w[bad])).max(-1)
                    med = np.median(amax, 1)
                    up = np.log2(amax.max(1) / med)            # how far the largest block sits above the median
                    down = np.log2(med / np.maximum(amax.min(1), 1e-300))  # how far the smallest sits below
                    acc[2] += (up >= down).sum(); acc[3] += (up < down).sum()
    return per_kind


if __name__ == "__main__":
    repo = sys.argv[1]
    if "--no-null" not in sys.argv:
        print(f"## Spread analysis\n\n### Null model: i.i.d. rows (50k rows each, MXFP4 exponent search)\n")
        print("| distribution | K | rows with spread > 3 | median log2(max/min block max) | 99.99th pct |\n|---|---|---|---|---|")
        for dist, k, v, med, p in null_model():
            print(f"| {dist} | {k} | {v:.2e} | {med:.2f} | {p:.2f} |", flush=True)
    print(f"\n### {repo}: violating rows by layer type and cause\n")
    print("Cause = whether the row's largest block sits further above the row's median block than its smallest sits")
    print("below it (outlier block) or the other way round (near-zero block).\n")
    print("| layer type | rows | spread > 3 | caused by a large block | caused by a small block |\n|---|---|---|---|---|")
    for kind, (n, v, up, down) in sorted(real_model(repo).items()):
        print(f"| {kind} | {int(n)} | {v / n:.3%} | {int(up)} | {int(down)} |")
