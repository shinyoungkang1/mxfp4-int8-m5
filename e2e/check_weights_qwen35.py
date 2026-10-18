"""Weight half of the Qwen3.5-MoE check, in exact arithmetic: for every W4A8 matrix (dense projections, 256 routed +
1 shared expert per layer), the packed weights decoded on the CPU (sc_format.decode_numpy, float64) times the row
scale equal sim/eval_ppl.py's fake_weight(w, "sc_mx") element for element.

Why not on the GPU: some expert rows of Qwen3.5-35B-A3B are numerically dead (|w| ~ 1e-38); their row scale
2^(E_lo - 1) is a float32 subnormal, and Metal flushes subnormal operands to zero, so a GPU decode (and the GPU
epilogue in inference) yields 0 there instead of ~1e-37. The weights themselves are exact; such rows are counted.
Usage: python check_weights_qwen35.py [--out results/08_check_weights_qwen35.json]
"""
import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import mlx.core as mx
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "sim"))

SUBNORMAL = np.float32(2.0 ** -126)


def _check(args):
    import eval_ppl
    import sc_format as F
    w, codes, bc, ws = args
    N, K = w.shape
    fake = eval_ppl.fake_weight(np.asarray(w, np.float64), "sc_mx")
    dec = F.decode_numpy(codes, bc, N, K).astype(np.float64) * ws.astype(np.float64)[:, None]
    return int((dec != fake).sum()), int(w.size), int((ws < SUBNORMAL).sum()), N


def main():
    import fused_qwen35 as fz
    from mlx_lm import load
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", default=os.path.expanduser("~/.cache/mxfp4-int8-m5"))
    ap.add_argument("--out", default=os.path.join(HERE, "..", "results", "08_check_weights_qwen35.json"))
    a = ap.parse_args()
    model, _ = load(fz.REPO, lazy=True)
    flat = mx.load(os.path.join(a.cache_dir, fz.REPO.replace("/", "--") + "-scmxfp4-p2.safetensors"))
    tot = dict(mismatches=0, weights=0, subnormal_scale_rows=0, rows=0, matrices=0)
    by_kind = {}
    with ProcessPoolExecutor(14) as ex:
        for i, layer in enumerate(model.language_model.model.layers):
            jobs, kinds = [], []
            for paths in fz.DENSE[fz.layer_kind(layer)].values():
                for p in paths:
                    k = f"layers.{i}.{p}"
                    jobs.append((np.array(fz.get(layer, p).weight.astype(mx.float32)), np.array(flat[k + ".codes"]),
                                 np.array(flat[k + ".bcodes"]), np.array(flat[k + ".wscale"])))
                    kinds.append(p)
            for name in fz.EXPERTS:
                k = f"layers.{i}.experts.{name}"
                W = np.array(getattr(layer.mlp.switch_mlp, name).weight.astype(mx.float32))
                Ws = np.array(getattr(layer.mlp.shared_expert, name).weight.astype(mx.float32))
                C, B, S_ = (np.array(flat[k + s]) for s in (".codes", ".bcodes", ".wscale"))
                for e in range(W.shape[0] + 1):
                    jobs.append((W[e] if e < W.shape[0] else Ws, C[e], B[e], S_[e]))
                    kinds.append(f"{'shared' if e == W.shape[0] else 'expert'}.{name}")
            for kind, (mm, n, sub, rows) in zip(kinds, ex.map(_check, jobs, chunksize=4)):
                for d in (tot, by_kind.setdefault(kind, dict(mismatches=0, weights=0, subnormal_scale_rows=0,
                                                              rows=0, matrices=0))):
                    d["mismatches"] += mm
                    d["weights"] += n
                    d["subnormal_scale_rows"] += sub
                    d["rows"] += rows
                    d["matrices"] += 1
            print(f"layer {i + 1}/40: {tot}", flush=True)
    rep = {"repo": fz.REPO, "exact_float64_decode_vs_fake_weight": tot, "by_matrix_kind": by_kind,
           "passed": tot["mismatches"] == 0}
    with open(a.out, "w") as f:
        json.dump(rep, f, indent=1)
    print("PASSED" if rep["passed"] else "FAILED", json.dumps(tot))


if __name__ == "__main__":
    main()
