"""Checks the batch-1 decode path (e2e/decode.py) against the fused prefill path, which results/07_check_*.json
check against the fake-quant reference.

1. Teacher-forced decode: prefill P WikiText tokens with the fused prefill path, then decode the next T tokens one
   at a time (feeding the true tokens).
   a. against the prefill kernels run one token at a time on their own cache: expected bit-identical logits (the
      GEMVs compute the GEMMs' integers and epilogues; both use MLX's SDPA vector kernel at L = 1);
   b. against one fused prefill over all P + T tokens (informational: MLX's full SDPA kernel rounds differently,
      and per-token INT8 activations turn that into code flips; cf. the top-1 noise floor in 05_check.json).
2. GEMVs on real activations: for every GEMV of the first block at one decode step, the GEMV output equals the
   INT8 matmul2d GEMM + the prefill epilogue on the same INT8 operands, bit for bit.

Usage: python check_decode.py --model qwen3|olmoe [--prompt 512] [--steps 64] [--out results/07_check_decode_*.json]
"""
import argparse
import json
import os
import sys

import mlx.core as mx
import mlx.nn as nn
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "sim"))
import decode as dec  # noqa: E402
import sc_decode as G  # noqa: E402
import sc_kernels as S  # noqa: E402
import eval_ppl  # noqa: E402
from mlx_lm.models.cache import make_prompt_cache  # noqa: E402


def load(name, cache_dir, packed="rtn"):
    if name == "qwen3":
        import fused_qwen3 as fq
        f, tok = fq.load_fused("Qwen/Qwen3-1.7B-Base", cache_dir, packed=packed)
        return f, dec.DecodeQwen3(f), tok
    if name == "qwen35":
        import fused_qwen35 as fz
        f, tok = fz.load_fused_qwen35(cache_dir=cache_dir)
        return f, dec.DecodeQwen35(f), tok
    import fused_olmoe as fo
    f, tok = fo.load_fused_olmoe("allenai/OLMoE-1B-7B-0924", cache_dir)
    return f, dec.DecodeOlmoe(f), tok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3", choices=["qwen3", "olmoe", "qwen35"])
    ap.add_argument("--packed", default="rtn", help="qwen3 only: rtn | gptq | fmt:<sim/formats.py name>")
    ap.add_argument("--cache-dir", default=os.path.expanduser("~/.cache/mxfp4-int8-m5"))
    ap.add_argument("--prompt", type=int, default=512)
    ap.add_argument("--steps", type=int, default=64)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    out = a.out or os.path.join(HERE, "..", "results", f"07_check_decode_{a.model}.json")
    f, d, tok = load(a.model, a.cache_dir, a.packed)
    toks = eval_ppl.wikitext_tokens(tok)
    P, T = a.prompt, a.steps
    ids = mx.array(toks[:P + T + 1])[None]

    # 1a. teacher-forced decode (GEMV) vs the prefill kernels run one token at a time (same SDPA call at L = 1)
    cache = f.make_cache()
    cache2 = f.make_cache()
    mx.eval(f.hidden(ids[:, :P], cache), f.hidden(ids[:, :P], cache2))
    dl, sl = [], []
    for t in range(T):
        lg = d.step(ids[:, P + t:P + t + 1], cache)
        lg2 = f.logits(f.hidden(ids[:, P + t:P + t + 1], cache2))
        mx.eval(lg, lg2)
        dl.append(lg[0, 0].astype(mx.float32))
        sl.append(lg2[0, 0].astype(mx.float32))
    dl, sl = mx.stack(dl), mx.stack(sl)
    ident = int(mx.sum(dl != sl).item())
    print(f"1a. decode GEMV path vs prefill kernels at L=1: {ident}/{dl.size} logits differ", flush=True)
    # 1b. vs one prefill over all P + T tokens (MLX's full SDPA kernel instead of the vector kernel)
    pl = f(ids[:, :P + T])[0, P:P + T].astype(mx.float32)
    tgt = ids[0, P + 1:P + T + 1]
    ppl = lambda lg: float(np.exp(nn.losses.cross_entropy(lg, tgt, reduction="mean").item()))  # noqa: E731
    rep = {"model": a.model, "prompt": P, "steps": T,
           "decode_vs_stepwise_prefill_logits_differing": ident, "logits_compared": int(dl.size),
           "max_abs_dlogit_decode_vs_prefill": float(mx.max(mx.abs(dl - pl)).item()),
           "mean_abs_dlogit": float(mx.mean(mx.abs(dl - pl)).item()),
           "max_abs_logit": float(mx.max(mx.abs(pl)).item()),
           "top1_agreement": float(mx.mean(mx.argmax(dl, -1) == mx.argmax(pl, -1)).item()),
           "ppl_decode": ppl(dl), "ppl_prefill": ppl(pl)}
    print("1.", json.dumps(rep), flush=True)

    # 2. GEMVs of block 0 on real activations (one decode step)
    layer, ws = f.layers[0][0], f.layers[0][-2 if a.model == "qwen35" else 1]
    h = f.model.model.embed_tokens(ids[:, P + T:P + T + 1])[0]
    xq, xs = S.rmsnorm_quant(h, layer.input_layernorm.weight, f.eps, 1)
    rows = mx.concatenate([xq, mx.zeros((127, xq.shape[1]), mx.int8)])
    nd, checked = {}, 0
    for g in (("qkvz", "out") if a.model == "qwen35" else ("qkv", "o")):
        w = ws[g]
        x = rows if g in ("qkv", "qkvz") else mx.concatenate([mx.array(np.random.default_rng(1).integers(
            -127, 128, (1, w.K)).astype(np.int8)), mx.zeros((127, w.K), mx.int8)])
        y = G.gemv(x[0], w.codes, w.bcodes, xs, w.wscale, w.N, 1)
        acc = S.mm_i8_i32(x, S.unpack_i8(w.codes, w.bcodes, w.N, w.K))
        y_ref = (acc[0].astype(mx.float32) * xs[0] * w.wscale).astype(mx.bfloat16)
        nd[g] = int(mx.sum(y != y_ref).item())
        checked += w.N
    rep["gemv_outputs_differing_vs_gemm_epilogue"] = nd
    rep["passed"] = bool(sum(nd.values()) == 0 and ident == 0)
    print("2.", nd, flush=True)
    with open(out, "w") as fh:
        json.dump(rep, fh, indent=1)
    print("PASSED" if rep["passed"] else "FAILED", "wrote", out)


if __name__ == "__main__":
    main()
