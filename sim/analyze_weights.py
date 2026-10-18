"""Exponent-spread statistics and weight quantization error of MXFP4 variants on a real model.

Usage: python analyze_weights.py <hf-repo-or-path> <out-prefix>
Writes <out-prefix>_spread.csv, <out-prefix>_error.csv and prints markdown tables.
"""
import csv
import glob
import os
import sys
from collections import defaultdict

import mlx.core as mx
import numpy as np
from huggingface_hub import snapshot_download

import mx_int8 as q

CHUNKS = {"row": None, "1024": 32, "512": 16, "256": 8}  # chunk width in K elements -> blocks


def linear_weights(repo):
    path = repo if os.path.isdir(repo) else snapshot_download(repo, allow_patterns=["*.safetensors", "*.json"])
    for f in sorted(glob.glob(os.path.join(path, "*.safetensors"))):
        for name, t in mx.load(f).items():
            # Transformer-block linear layers only (embeddings / lm_head stay in higher precision).
            if t.ndim == 2 and ".layers." in name and name.endswith("proj.weight"):
                yield name, np.array(t.astype(mx.float32), dtype=np.float64)


def kind(name):
    return name.split(".")[-2]  # q_proj, k_proj, ..., down_proj


def main(repo, prefix):
    spread_hist = {c: np.zeros(64, np.int64) for c in CHUNKS}
    err = defaultdict(lambda: [0.0, 0.0])  # (method, kind) -> [sum sq err, sum sq weight]
    n_mats = 0
    for name, w in linear_weights(repo):
        n_mats += 1
        k = kind(name)
        norm = (w ** 2).sum()
        e_search = q.block_exponents(w, "search")

        nb = e_search.shape[1]
        fits = {c: cb for c, cb in CHUNKS.items() if cb is None or nb % cb == 0}  # skip widths not dividing K
        for c, cb in fits.items():
            s = q.exponent_spread(e_search, cb).astype(int).ravel()
            spread_hist[c] += np.bincount(np.minimum(s, 63), minlength=64)

        methods = {
            "MXFP4 spec": q.dequant_mx(w, q.block_exponents(w, "spec"))[0],
            "MXFP4 ceil (BaseRT)": q.dequant_mx(w, q.block_exponents(w, "ceil"))[0],
            "MXFP4 search": q.dequant_mx(w, e_search)[0],
            "NVFP4": q.nvfp4(w),
            "INT4 affine g64 (MLX 4-bit)": q.int4_affine(w, 64),
            "INT4 affine g32": q.int4_affine(w, 32),
        }
        for c, cb in fits.items():
            e_sc = q.constrain_spread(w, e_search, cb)
            wq, v = q.dequant_mx(w, e_sc)
            q.embed_int8(v, e_sc, cb)  # raises if not exactly embeddable
            methods[f"SC-MXFP4 chunk={c}"] = wq
        for m, wq in methods.items():
            if m.startswith("SC-MXFP4 chunk=") and m != "SC-MXFP4 chunk=row":
                continue  # chunked variants are only comparable on matrices where every width applies
            for key in ((m, k), (m, "all")):
                err[key][0] += ((wq - w) ** 2).sum()
                err[key][1] += norm
        print(f"\r{n_mats} matrices", end="", file=sys.stderr)
    print(file=sys.stderr)

    with open(prefix + "_spread.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["chunk", "spread", "count"])
        for c, h in spread_hist.items():
            for s, n in enumerate(h):
                if n:
                    wr.writerow([c, s, int(n)])
    kinds = sorted({k for _, k in err if k != "all"})
    methods = list(dict.fromkeys(m for m, _ in err))
    with open(prefix + "_error.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["method", "kind", "rel_mse"])
        for (m, k), (a, b) in err.items():
            wr.writerow([m, k, a / b])

    print(f"## {repo}: {n_mats} linear matrices\n")
    print("Fraction of (row, chunk) units whose MXFP4 (search) exponent spread is <= 3, i.e. INT8-embeddable "
          "with no change:\n")
    print("| chunk | <=3 | max spread | mean spread |\n|---|---|---|---|")
    for c, h in spread_hist.items():
        tot = h.sum()
        if not tot:
            continue
        print(f"| {c} | {h[:4].sum() / tot:.1%} | {np.nonzero(h)[0].max()} | {(h * np.arange(64)).sum() / tot:.2f} |")
    print("\nRelative weight MSE x1000 (lower is better):\n")
    print("| method | all | " + " | ".join(kinds) + " |\n|---|" + "---|" * (len(kinds) + 1))
    for m in methods:
        if not err[(m, "all")][1]:
            continue
        row = [err[(m, "all")][0] / err[(m, "all")][1]] + [err[(m, k)][0] / err[(m, k)][1] for k in kinds]
        print(f"| {m} | " + " | ".join(f"{1000 * x:.2f}" for x in row) + " |")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
