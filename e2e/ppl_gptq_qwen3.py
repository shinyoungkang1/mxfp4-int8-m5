"""WikiText-2 perplexity of GPTQ-rounded SC-MXFP4 on Qwen3-1.7B-Base (full test set, 2048-token windows; protocol of
sim/eval_ppl.py, comparable with results/ppl.csv):
  ours, ours_gptq            the fused W4A8 model on our kernels with round-to-nearest / GPTQ weights (gptq_pack.py)
  fake_gptq_w4a8 / _w4a16    sim/eval_ppl.py's fake-quant method "w=gptq:sc_mx;a=8" / "a=16" (eval_ppl.QLinear, BF16
                             matmul, eval_ppl.act_int8 activations), with the GPTQ weights taken from the packed cache:
                             gptq_pack.py runs the same gptq.gptq call as eval_ppl.fake_weight("gptq:sc_mx") and checks
                             that the packed weights decode to its output exactly (float64, every matrix), so these are
                             eval_ppl's weights bit for bit without recomputing GPTQ serially (~5 CPU-hours here)
  ours_fmt:<name>            the fused W4A8 model in another sim/formats.py format (e2e/pack_format.py)
  w=...                      any other sim/eval_ppl.py method string (recomputes its weights)
Usage: python ppl_gptq_qwen3.py [--methods ...] [--out results/08_gptq_ppl_qwen3.csv]   (append mode)
"""
import argparse
import csv
import gc
import os
import sys
import time

import mlx.core as mx

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "sim"))
import eval_ppl  # noqa: E402
import fused_qwen3 as fq  # noqa: E402

REPO = "Qwen/Qwen3-1.7B-Base"
CACHE = os.path.expanduser("~/.cache/mxfp4-int8-m5")


def fake_from_packed(model, act8):
    """eval_ppl.QLinear on every block linear, weights = the packed GPTQ cache decoded (float64 -> float32 -> BF16,
    as QLinear stores fake_weight's output)."""
    import numpy as np
    import sc_format as F
    from w4a8_mlx import LINEARS
    flat = mx.load(os.path.join(CACHE, REPO.replace("/", "--") + "-scmxfp4-gptq.safetensors"))
    for i, layer in enumerate(model.model.layers):
        for path in LINEARS:
            parent_name, name = path.split(".")
            parent = getattr(layer, parent_name)
            lin = getattr(parent, name)
            N, K = lin.weight.shape
            key = f"layers.{i}.{path}"
            codes, bc, ws = F.from_shift_layout(np.array(flat[key + ".codes"]), np.array(flat[key + ".shifts"]),
                                                np.array(flat[key + ".wscale"]))
            w = F.decode_numpy(codes, bc, N, K).astype(np.float64) * ws.astype(np.float64)[:, None]
            q = eval_ppl.QLinear(lin, None, act8, 0)
            q.weight = mx.array(w.astype(np.float32)).astype(lin.weight.dtype)
            setattr(parent, name, q)
    mx.eval(model.parameters())


def main():
    from mlx_lm import load
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", default="ours,ours_gptq,fake_gptq_w4a8,fake_gptq_w4a16")
    ap.add_argument("--seq", type=int, default=2048)
    ap.add_argument("--out", default=os.path.join(HERE, "..", "results", "08_gptq_ppl_qwen3.csv"))
    a = ap.parse_args()
    new = not os.path.exists(a.out)
    toks = None
    with open(a.out, "a", newline="") as f:
        wr = csv.writer(f)
        if new:
            wr.writerow(["model", "method", "seq", "chunks", "ppl", "seconds"])
        for m in a.methods.split(","):
            t0 = time.time()
            if m.startswith("ours_fmt:"):
                model, tok = fq.load_fused(REPO, CACHE, packed="fmt:" + m[len("ours_fmt:"):])
            elif m in ("ours", "ours_gptq"):
                model, tok = fq.load_fused(REPO, CACHE, packed="gptq" if m == "ours_gptq" else "rtn")
            elif m.startswith("fake_gptq_"):
                model, tok = load(REPO)
                fake_from_packed(model, act8=m.endswith("w4a8"))
            else:
                model, tok = load(REPO)
                eval_ppl.quantize_model(model, m, REPO)
            if toks is None:
                toks = eval_ppl.wikitext_tokens(tok)
            ppl, n = eval_ppl.perplexity(model, toks, a.seq, 0)
            wr.writerow([REPO, m, a.seq, n, f"{ppl:.4f}", f"{time.time() - t0:.0f}"])
            f.flush()
            print(f"{m:20s} ppl {ppl:8.4f}  ({n} x {a.seq} tokens, {time.time() - t0:.0f}s)", flush=True)
            del model
            gc.collect()
            mx.clear_cache()


if __name__ == "__main__":
    main()
