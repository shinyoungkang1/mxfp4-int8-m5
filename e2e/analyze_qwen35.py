"""sim/analyze_weights.py's statistics (exponent spread per row / chunk, relative weight MSE of the MXFP4 variants,
NVFP4, INT4 affine, SC-MXFP4) for Qwen3.5-35B-A3B-Base, computed with the same sim/mx_int8 functions, in parallel.

Why not sim/analyze_weights.py directly: it selects 2-D checkpoint tensors named *proj.weight, but this checkpoint
stores the 256 routed experts as fused 3-D tensors (experts.gate_up_proj / experts.down_proj), names the DeltaNet
projections in_proj_qkv / in_proj_z, and includes a multi-token-prediction head (mtp.*). So it would skip >95% of
the weights. This script feeds it exactly the matrices that run in W4A8 (e2e/fused_qwen35.py), taken from the
sanitized mlx_lm model (experts split per expert, MTP dropped), one job per matrix.
Usage: python analyze_qwen35.py [out-prefix]   (default results/08_qwen3.5-35b-a3b)
Writes <prefix>_spread.csv, <prefix>_error.csv (formats of analyze_weights.py) and <prefix>_weights.md.
"""
import csv
import os
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor

import mlx.core as mx
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "sim"))
CHUNKS = {"row": None, "1024": 32, "512": 16, "256": 8}  # as analyze_weights.py


def one(args):
    """analyze_weights.main's per-matrix body: (spread histograms per chunk, {(method, kind): [sq err, sq weight]})."""
    import mx_int8 as q
    k, w = args
    w = np.asarray(w, np.float64)
    norm = (w ** 2).sum()
    e_search = q.block_exponents(w, "search")
    nb = e_search.shape[1]
    fits = {c: cb for c, cb in CHUNKS.items() if cb is None or nb % cb == 0}
    hist = {c: np.bincount(np.minimum(q.exponent_spread(e_search, cb).astype(int).ravel(), 63), minlength=64)
            for c, cb in fits.items()}
    methods = {
        "MXFP4 spec": q.dequant_mx(w, q.block_exponents(w, "spec"))[0],
        "MXFP4 ceil (BaseRT)": q.dequant_mx(w, q.block_exponents(w, "ceil"))[0],
        "MXFP4 search": q.dequant_mx(w, e_search)[0],
        "NVFP4": q.nvfp4(w),
        "INT4 affine g64 (MLX 4-bit)": q.int4_affine(w, 64),
        "INT4 affine g32": q.int4_affine(w, 32),
    }
    e_sc = q.constrain_spread(w, e_search, None)
    wq, v = q.dequant_mx(w, e_sc)
    q.embed_int8(v, e_sc, None)  # raises if not exactly embeddable
    methods["SC-MXFP4 chunk=row"] = wq
    err = {(m, k): [float(((x - w) ** 2).sum()), float(norm)] for m, x in methods.items()}
    return hist, err


def jobs(model):
    import fused_qwen35 as fz
    for i, layer in enumerate(model.language_model.model.layers):
        for paths in fz.DENSE[fz.layer_kind(layer)].values():
            for p in paths:
                yield [(p.split(".")[-1], np.array(fz.get(layer, p).weight.astype(mx.float32)))]
        for name in fz.EXPERTS:
            W = np.array(getattr(layer.mlp.switch_mlp, name).weight.astype(mx.float32))
            yield [(f"expert.{name}", W[e]) for e in range(W.shape[0])]
            yield [(f"shared.{name}", np.array(getattr(layer.mlp.shared_expert, name).weight.astype(mx.float32)))]
        print(f"\rlayer {i + 1}/40", end="", file=sys.stderr, flush=True)


def main(prefix):
    import fused_qwen35 as fz
    from mlx_lm import load
    model, _ = load(fz.REPO, lazy=True)
    hist = {c: np.zeros(64, np.int64) for c in CHUNKS}
    err = defaultdict(lambda: [0.0, 0.0])
    n = 0
    with ProcessPoolExecutor(14) as ex:
        for batch in jobs(model):
            for h, e in ex.map(one, batch, chunksize=4):
                n += 1
                for c, v in h.items():
                    hist[c] += v
                for (m, k), (a, b) in e.items():
                    for key in ((m, k), (m, "all")):
                        err[key][0] += a
                        err[key][1] += b
    print(file=sys.stderr)
    with open(prefix + "_spread.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["chunk", "spread", "count"])
        for c, h in hist.items():
            for s, cnt in enumerate(h):
                if cnt:
                    wr.writerow([c, s, int(cnt)])
    with open(prefix + "_error.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["method", "kind", "rel_mse"])
        for (m, k), (a, b) in err.items():
            wr.writerow([m, k, a / b])
    kinds = sorted({k for _, k in err if k != "all"})
    methods = list(dict.fromkeys(m for m, _ in err))
    lines = [f"## {fz.REPO}: {n} linear matrices (the W4A8 matrices of e2e/fused_qwen35.py; e2e/analyze_qwen35.py)\n",
             "Fraction of (row, chunk) units whose MXFP4 (search) exponent spread is <= 3, i.e. INT8-embeddable "
             "with no change:\n", "| chunk | <=3 | max spread | mean spread |", "|---|---|---|---|"]
    for c, h in hist.items():
        tot = h.sum()
        if tot:
            lines.append(f"| {c} | {h[:4].sum() / tot:.1%} | {np.nonzero(h)[0].max()} | "
                         f"{(h * np.arange(64)).sum() / tot:.2f} |")
    lines += ["\nRelative weight MSE x1000 (lower is better):\n",
              "| method | all | " + " | ".join(kinds) + " |", "|---|" + "---|" * (len(kinds) + 1)]
    for m in methods:
        if err[(m, "all")][1]:
            row = [err[(m, "all")][0] / err[(m, "all")][1]] + [err[(m, k)][0] / err[(m, k)][1] for k in kinds]
            lines.append(f"| {m} | " + " | ".join(f"{1000 * x:.2f}" for x in row) + " |")
    open(prefix + "_weights.md", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..", "results", "08_qwen3.5-35b-a3b"))
