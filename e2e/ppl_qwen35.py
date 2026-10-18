"""WikiText-2 perplexity of Qwen3.5-35B-A3B-Base (full test set, 2048-token windows; protocol of sim/eval_ppl.py)
for the rows of results/04_accuracy.md, plus the real systems:
  bf16                       the checkpoint
  sc_mxfp4_w4a8 / _w4a16     fake-quant SC-MXFP4 (sim/ quantizer) of the W4A8 matrices of e2e/fused_qwen35.py,
                             with / without per-token INT8 activations (eval_ppl.METHODS semantics)
  nvfp4_w4a16, int4_affine_g64_w4a16   the same matrices fake-quantized to NVFP4 / MLX-style affine g64
  ours                       the fused W4A8 model on our kernels
  mlx_mxfp4, mlx_affine4     MLX's checkpoints (these also quantize embeddings, lm_head and routers: mlx_lm default)
  mlx_mxfp4_cov              MLX MXFP4 on exactly the W4A8 matrices (convert_mlx.py ... w4a8cov)
Router, shared_expert_gate, in_proj_b/a, embeddings and lm_head stay BF16 in all fake-quant rows and in ours.
Usage: python ppl_qwen35.py [--methods ...] [--out results/08_ppl_qwen35.csv]   (append mode, like eval_ppl.py)
"""
import argparse
import csv
import gc
import os
import sys
import time

import mlx.core as mx
import mlx.nn as nn
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "sim"))
import eval_ppl  # noqa: E402
import fused_qwen35 as fz  # noqa: E402
import ref_qwen35 as rq  # noqa: E402

CACHE = os.path.expanduser("~/.cache/mxfp4-int8-m5")
FAKE = {"sc_mxfp4_w4a8": ("sc_mx", True), "sc_mxfp4_w4a16": ("sc_mx", False), "nvfp4_w4a16": ("nvfp4", False),
        "int4_affine_g64_w4a16": ("int4_g64", False)}


def main():
    from mlx_lm import load
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", default="bf16,ours,mlx_mxfp4,mlx_affine4,sc_mxfp4_w4a8,sc_mxfp4_w4a16,"
                                         "nvfp4_w4a16,int4_affine_g64_w4a16")
    ap.add_argument("--seq", type=int, default=2048)
    ap.add_argument("--out", default=os.path.join(HERE, "..", "results", "08_ppl_qwen35.csv"))
    a = ap.parse_args()
    new = not os.path.exists(a.out)
    toks = None
    built = {}  # reuse the sc_mx fake-quant model for its W4A16 row
    with open(a.out, "a", newline="") as f:
        wr = csv.writer(f)
        if new:
            wr.writerow(["model", "method", "seq", "chunks", "ppl", "seconds"])
        for m in a.methods.split(","):
            t0 = time.time()
            if m == "ours":
                model, tok = fz.load_fused_qwen35()
            elif m.startswith("mlx_"):
                suffix = {"mlx_mxfp4": "mxfp4", "mlx_affine4": "4bit", "mlx_mxfp4_cov": "mxfp4-w4a8cov"}[m]
                model, tok = load(os.path.join(CACHE, "Qwen3.5-35B-A3B-Base-" + suffix))
            else:
                kind, act8 = FAKE.get(m, (None, None))
                if kind and kind in built:
                    model, tok = built[kind]
                    rq.set_act8(model, act8)
                else:
                    built.clear()
                    gc.collect()
                    mx.clear_cache()
                    model, tok = load(fz.REPO, lazy=m != "bf16")
                    if kind:
                        rq.build_fake(model, kind, act8)
                        built[kind] = (model, tok)
            if toks is None:
                toks = eval_ppl.wikitext_tokens(tok)
            ppl, n = eval_ppl.perplexity(model, toks, a.seq, 0)
            wr.writerow([fz.REPO, m, a.seq, n, f"{ppl:.4f}", f"{time.time() - t0:.0f}"])
            f.flush()
            print(f"{m:24s} ppl {ppl:8.4f}  ({n} x {a.seq} tokens, {time.time() - t0:.0f}s)", flush=True)
            if m not in FAKE or FAKE[m][0] not in built:
                del model
            gc.collect()
            mx.clear_cache()


if __name__ == "__main__":
    main()
