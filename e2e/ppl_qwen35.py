"""WikiText-2 perplexity of Qwen3.5-35B-A3B-Base (full test set, 2048-token windows; protocol of sim/eval_ppl.py)
for the rows of results/04_accuracy.md, plus the real systems:
  bf16                       the checkpoint
  sc_mxfp4_w4a8 / _w4a16     fake-quant SC-MXFP4 (sim/ quantizer) of the W4A8 matrices of e2e/fused_qwen35.py,
                             with / without per-token INT8 activations (eval_ppl.METHODS semantics)
  nvfp4_w4a16, int4_affine_g64_w4a16   the same matrices fake-quantized to NVFP4 / MLX-style affine g64
  ours                       the fused W4A8 model on our kernels
  mlx_mxfp4, mlx_affine4     MLX's checkpoints (these also quantize embeddings, lm_head and routers: mlx_lm default)
  mlx_mxfp4_cov              MLX MXFP4 on exactly the W4A8 matrices (convert_mlx.py ... w4a8cov)
  w=<kind>;a=<8|16>[;wd=<kind|bf16>][;ws=<kind|bf16>][;r=<block>]
                             fake-quant with a weight kind per class of matrix (eval_ppl.parse_scoped): w = the routed
                             experts, wd = the dense projections, ws = the shared expert (both default to w); INT8
                             activations (a=8) at every one of these matrices, whatever its weights. Examples:
                             "w=fmt:e2m1/B8/M1..10;a=8" is b8_w4a8; "...;a=8;wd=bf16" quantizes only the experts'
                             weights; "w=bf16;a=8;wd=fmt:e2m1/B8/M1..10" only the dense projections';
                             "w=sc_mx;a=8;wd=int8_row" = SC-MXFP4 experts with 8-bit dense projections
Router, shared_expert_gate, in_proj_b/a, embeddings and lm_head stay BF16 in all fake-quant rows and in ours.
A method name that is none of these stops the run before any model is loaded.
<out>.prov.jsonl gets one line of provenance per method (sim/provenance.py).
Usage: python ppl_qwen35.py [--methods ...] [--out results/08_ppl_qwen35.csv] [--second-half]   (append mode, like
eval_ppl.py; --second-half: llama-perplexity's protocol, see eval_ppl.py)
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
import provenance as prov  # noqa: E402
import ref_qwen35 as rq  # noqa: E402

CACHE = os.path.expanduser("~/.cache/mxfp4-int8-m5")
FAKE = {"sc_mxfp4_w4a8": ("sc_mx", True), "sc_mxfp4_w4a16": ("sc_mx", False), "nvfp4_w4a16": ("nvfp4", False),
        "int4_affine_g64_w4a16": ("int4_g64", False), "nvfp4_w4a8": ("nvfp4", True),
        "nvfp4_search_w4a8": ("nvfp4_search", True),
        "b16_w4a8": ("fmt:e2m1/B16/M1..10", True), "b8_w4a8": ("fmt:e2m1/B8/M1..10", True)}
# (weight kind or None for BF16, INT8 activations, rotation block, rotate only this expert matrix)
ROT = {"bf16_w4a8": (None, True, 0, None), "bf16_w4a8_r32": (None, True, 32, None), "bf16_w4a8_r32_down": (None, True, 32, "down_proj"),
       "b8_w4a8_base": ("fmt:e2m1/B8/M1..10", True, 0, None), "b8_w4a8_r32": ("fmt:e2m1/B8/M1..10", True, 32, None),
       "b8_w4a8_r32_down": ("fmt:e2m1/B8/M1..10", True, 32, "down_proj")}
MLX = {"mlx_mxfp4": "mxfp4", "mlx_affine4": "4bit", "mlx_mxfp4_cov": "mxfp4-w4a8cov"}


def check(m):
    """Stops on a method this script does not have (it used to be scored as the BF16 model under that name)."""
    if m in ("bf16", "ours", "ours_b16") or m in FAKE or m in ROT or m in MLX:
        return
    if m.startswith(("ours_fmt:", "ours_rot:")):
        import formats
        if m.split(":", 1)[1] not in formats.FORMATS:
            raise SystemExit(f"{m}: no such format in sim/formats.py")
        return
    if "=" in m:
        wk, _, _, wd, ws = eval_ppl.parse_scoped(m)
        for k in (wk, wd, ws):
            eval_ppl.check_kind(k, gptq=False)
        return
    raise SystemExit(f"unknown method {m!r}: bf16, ours, ours_b16, ours_fmt:<format>, ours_rot:<format>, "
                     f"{', '.join(list(MLX) + list(FAKE) + list(ROT))}, or a spec w=...;a=...[;wd=...][;ws=...]")


def main():
    from mlx_lm import load
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", default="bf16,ours,mlx_mxfp4,mlx_affine4,sc_mxfp4_w4a8,sc_mxfp4_w4a16,"
                                         "nvfp4_w4a16,int4_affine_g64_w4a16")
    ap.add_argument("--seq", type=int, default=2048)
    ap.add_argument("--out", default=os.path.join(HERE, "..", "results", "08_ppl_qwen35.csv"))
    ap.add_argument("--second-half", action="store_true",
                    help="llama-perplexity's protocol: disjoint windows, only the second half of each scored")
    ap.add_argument("--windows-dir", default="", help="also save each window's summed NLL and token count (for paired bootstrap)")
    a = ap.parse_args()
    for m in a.methods.split(","):
        check(m)
    new = not os.path.exists(a.out)
    toks = None
    built = {}  # reuse the sc_mx fake-quant model for its W4A16 row
    with open(a.out, "a", newline="") as f:
        wr = csv.writer(f)
        if new:
            wr.writerow(["model", "method", "seq", "chunks", "ppl", "seconds"])
        for m in a.methods.split(","):
            t0 = time.time()
            what = {}
            if m == "ours":
                model, tok = fz.load_fused_qwen35()
            elif m == "ours_b16":
                model, tok = fz.load_fused_qwen35(fmt_name="e2m1/B16/M1..10")
            elif m.startswith("ours_fmt:"):  # any sim/formats.py format on the kernels (packed on first use)
                model, tok = fz.load_fused_qwen35(fmt_name=m[len("ours_fmt:"):])
            elif m.startswith("ours_rot:"):  # the same with every packed matrix rotated (r=32, no GPTQ), against "b8_w4a8_r32"
                model, tok = fz.load_fused_qwen35(fmt_name=m[len("ours_rot:"):], rot=True)
            elif "=" in m:  # a weight kind per class of matrix
                built.clear()
                gc.collect()
                mx.clear_cache()
                wk, act8, had, wd, ws = eval_ppl.parse_scoped(m)
                model, tok = load(fz.REPO, lazy=True)
                rq.build_fake(model, wk, act8, had=had, dense=wd, shared=ws)
                what = dict(weights=dict(experts=wk or "bf16", dense=wd or "bf16", shared=ws or "bf16"), act8=act8, rot=had)
            elif m in ROT:
                built.clear()
                gc.collect()
                mx.clear_cache()
                model, tok = load(fz.REPO, lazy=True)
                kind, act8, had, only = ROT[m]
                rq.build_fake(model, kind, act8, had=had, rot_only=only)
            elif m.startswith("mlx_"):
                what = dict(pack=os.path.join(CACHE, "Qwen3.5-35B-A3B-Base-" + MLX[m]))
                model, tok = load(what["pack"])
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
            if m.startswith("ours"):
                what = dict(format=model.fmt_name + ("+rot" if model.rot else ""), pack=model.pack_path)
            if toks is None:
                toks = eval_ppl.wikitext_tokens(tok)
            pw = [] if a.windows_dir else None
            ppl, n = eval_ppl.perplexity(model, toks, a.seq, 0, a.second_half, per_window=pw)
            if pw is not None:
                os.makedirs(a.windows_dir, exist_ok=True)
                np.savez(os.path.join(a.windows_dir, "Qwen_Qwen3.5-35B-A3B-Base__" + m.replace("/", "_").replace(":", "-").replace(";", "_").replace("=", "-") + ".npz"), nll=np.array([x[0] for x in pw]), count=np.array([x[1] for x in pw]))
            name = m + (" (second half)" if a.second_half else "")
            wr.writerow([fz.REPO, name, a.seq, n, f"{ppl:.4f}", f"{time.time() - t0:.0f}"])
            f.flush()
            prov.sidecar(a.out, model=fz.REPO, method=name, seq=a.seq, chunks=n, **what)
            print(f"{name:24s} ppl {ppl:8.4f}  ({n} x {a.seq} tokens, {time.time() - t0:.0f}s)", flush=True)
            if m not in FAKE or FAKE[m][0] not in built:
                del model
            gc.collect()
            mx.clear_cache()


if __name__ == "__main__":
    main()
