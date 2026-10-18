"""Phase-1 check of mlx-lm PR #1788 (qwen4_exp, Qwen3.8-Flash-Next) before downloading the checkpoint.

Builds the model from the real config.json with random weights, but only NLAYERS layers (3 Gated-DeltaNet + 1
full-attention with the QSA indexer, the PLE / n-gram layer included) and a tiny n-gram table, every other shape
real (hidden 2560, 512 experts x 640, top-10 + shared expert, 4 hyper-connection streams). Checks:
  1. the forward pass runs and is finite (BF16);
  2. prefill + token-by-token decode with the cache reproduces the cache-free forward (logits);
  3. the same with MLX MXFP4 weights (nn.quantize, the PR's quant_predicate), i.e. the baseline path we compare to;
  4. reports parameter counts per component, scaled to 48 layers, and the bytes at 4.26 / 4.25 bits.

Run with the PR environment: .venv-q38/bin/python e2e/q38_smoke.py
"""
import glob
import json
import os
import time

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_flatten
from mlx_lm.models import qwen4_exp

NLAYERS = 4
L0, STEPS = 192, 8


def load_config():
    p = glob.glob(os.path.expanduser(
        "~/.cache/huggingface/hub/models--Qwen--Qwen3.8-Flash-Next/snapshots/*/config.json"))[0]
    return json.load(open(p))


def build(cfg):
    tc = dict(cfg["text_config"])
    full_layers = tc["num_hidden_layers"]
    tc.update(num_hidden_layers=NLAYERS, layer_types=tc["layer_types"][:NLAYERS],
              ngram_vocab_size_base=1009, split_ngram_parts=4)
    model = qwen4_exp.Model(qwen4_exp.ModelArgs(model_type="qwen4_exp", text_config=tc))
    model.set_dtype(mx.bfloat16)
    mx.eval(model.parameters())
    return model, full_layers


def counts(model, full_layers):
    groups = {}
    for k, v in tree_flatten(model.parameters()):
        if ".layers." in k:
            g = ("experts" if "switch_mlp" in k else "shared expert" if "shared_expert." in k
                 else "router" if k.endswith("mlp.gate.weight") else "n-gram table" if "ngram_embedding" in k
                 else "attention / deltanet" if ("linear_attn" in k or "self_attn" in k) else "other per-layer")
            scale = 1 if g == "n-gram table" else full_layers / NLAYERS
        else:
            g, scale = "embeddings / head / mixer", 1
        groups[g] = groups.get(g, 0) + v.size * scale
    return groups


def consistency(model, ids, label):
    """Max |logit difference| between cache-free forward and prefill + STEPS single-token decode steps."""
    full = model(ids)[0].astype(mx.float32)
    cache = model.make_cache()
    out = [model(ids[:, :L0], cache=cache)[0, -1:]]
    for t in range(L0, L0 + STEPS - 1):
        out.append(model(ids[:, t:t + 1], cache=cache)[0, -1:])
    dec = mx.concatenate(out).astype(mx.float32)
    ref = full[L0 - 1:L0 + STEPS - 1]
    mx.eval(full, dec)
    finite = bool(mx.all(mx.isfinite(full)).item())
    d = float(mx.max(mx.abs(dec - ref)).item())
    scale = float(mx.max(mx.abs(ref)).item())
    agree = float(mx.mean((mx.argmax(dec, -1) == mx.argmax(ref, -1)).astype(mx.float32)).item())
    print(f"{label}: finite={finite}  max|logit| {scale:.3f}  decode-vs-forward max diff {d:.4f} "
          f"({d / max(scale, 1e-9):.2%} of max)  argmax agreement {agree:.0%}", flush=True)
    return finite and agree >= 0.75


def main():
    cfg = load_config()
    t0 = time.time()
    model, full_layers = build(cfg)
    print(f"built {NLAYERS}-layer model in {time.time() - t0:.0f} s; layer types "
          f"{[l.layer_type for l in model.layers]}; PLE layers {model.model.ple_layers}")
    g = counts(model, full_layers)
    tot = sum(v for k, v in g.items() if k != "n-gram table")
    print("parameters scaled to", full_layers, "layers (n-gram table: tiny here; real one ~51B):")
    for k, v in sorted(g.items(), key=lambda x: -x[1]):
        print(f"  {k:28s} {v / 1e9:8.2f} B")
    print(f"  {'total w/o n-gram table':28s} {tot / 1e9:8.2f} B  -> {tot * 4.26 / 8 / 2**30:.1f} GiB at 4.26 bits")

    rng = np.random.default_rng(0)
    ids = mx.array(rng.integers(0, 248320, (1, L0 + STEPS)).astype(np.int32))
    ok = consistency(model, ids, "BF16 random weights")

    t0 = time.time()
    pred = model.quant_predicate  # as mlx_lm.convert: only modules with to_quantized, router kept
    nn.quantize(model, group_size=32, bits=4, mode="mxfp4",
                class_predicate=lambda p, m: hasattr(m, "to_quantized") and pred(p, m))
    mx.eval(model.parameters())
    print(f"MLX mxfp4 quantization in {time.time() - t0:.0f} s; router kept: "
          f"{type(model.layers[0].mlp.gate).__name__}, experts: {type(model.layers[0].mlp.switch_mlp.gate_proj).__name__}")
    ok &= consistency(model, ids, "MLX MXFP4 weights")
    print("q38 smoke test", "passed" if ok else "FAILED")


if __name__ == "__main__":
    main()
