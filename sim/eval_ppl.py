"""WikiText-2 perplexity of a model under different weight/activation quantization schemes (fake-quant).

The SC-MXFP4 W4A8 configuration reproduces exactly what kernels/w4a8.metal computes: SC-MXFP4 weights
(bit-exact through the INT8 embedding) and symmetric per-token INT8 activations on the input of every
transformer-block linear layer. Embeddings, norms, attention softmax and lm_head stay in BF16.

Usage: python eval_ppl.py <hf-repo> <out.csv> [--methods a,b,...] [--seq 2048] [--max-chunks N] [--second-half]

--second-half scores as llama.cpp's llama-perplexity does: disjoint seq-token windows, of which only the predictions
for the second half (seq/2 - 1 tokens) count; the method is recorded with the suffix " (second half)".

Methods are either a name from METHODS or a spec "w=<weights>;a=<8|16>;r=<rotation block or 0>", where
<weights> is one of the METHODS weight kinds, "fmt:<name>" (a format from formats.FORMATS, round to nearest),
"gptq:<name>" (same format with GPTQ rounding) or "gptq:sc_mx" (SC-MXFP4 with GPTQ rounding). GPTQ uses
calibration Hessians from calib.py (WikiText-2 train), computed once per model.
Example: --methods "w=gptq:e2m1/B16/M1..8;a=8;r=32"
"""
import argparse
import csv
import json
import os
import time

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from mlx_lm import load

import calib
import formats as F
import gptq as G
import mx_int8 as q
from gpu_guard import eval_step

LINEARS = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
           "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]

# name -> (weight quantizer or None, activation INT8 per token, Hadamard block size)
METHODS = {
    "bf16": (None, False, 0),
    "mxfp4_spec_w4a16": ("mx_spec", False, 0),
    "mxfp4_search_w4a16": ("mx_search", False, 0),
    "sc_mxfp4_w4a16": ("sc_mx", False, 0),
    "sc_mxfp4_w4a8": ("sc_mx", True, 0),
    "sc_mxfp4_w4a8_had128": ("sc_mx", True, 128),
    "nvfp4_w4a16": ("nvfp4", False, 0),
    "int4_affine_g64_w4a16": ("int4_g64", False, 0),
    "bf16_w16a8": (None, True, 0),
    "bf16_had128": (None, False, 128),  # sanity check: rotation alone must not change perplexity
}


def hadamard(n):
    h = np.array([[1.0]])
    while h.shape[0] < n:
        h = np.block([[h, h], [h, -h]])
    return h / np.sqrt(n)


def parse_method(m):
    """-> (weight kind or None, INT8 activations, rotation block size)."""
    if m in METHODS:
        return METHODS[m]
    kv = dict(part.split("=", 1) for part in m.split(";"))
    w = kv.get("w", "none")
    return (None if w in ("none", "bf16") else w, kv.get("a", "16") == "8", int(kv.get("r", "0")))


def fake_weight(w, kind, h=None):
    if kind.startswith("fmt:"):
        return F.quantize_mx(w, F.FORMATS[kind[4:]])[0].astype(np.float64)
    if kind.startswith("gptq:"):
        assert h is not None, "GPTQ needs a calibration Hessian (not available for MoE experts)"
        if kind == "gptq:sc_mx":
            return G.gptq(w, h, G.sc_mxfp4_scales(w), q.E2M1)
        f = F.FORMATS[kind[5:]]
        return G.gptq(w, h, F.quantize_mx(w, f)[3], F.CODEBOOKS[f.codebook])
    if kind == "mx_spec":
        return q.dequant_mx(w, q.block_exponents(w, "spec"))[0]
    if kind == "mx_search":
        return q.dequant_mx(w, q.block_exponents(w, "search"))[0]
    if kind == "sc_mx":
        e = q.constrain_spread(w, q.block_exponents(w, "search"))
        wq, v = q.dequant_mx(w, e)
        q.embed_int8(v, e)  # asserts INT8-embeddability
        return wq
    if kind == "nvfp4":
        return q.nvfp4(w)
    if kind == "nvfp4_search":
        return q.nvfp4_search(w)
    if kind == "int4_g64":
        return q.int4_affine(w, 64)
    raise ValueError(kind)


class QLinear(nn.Module):
    def __init__(self, lin, wkind, act8, had, h=None):
        super().__init__()
        w = np.array(lin.weight.astype(mx.float32), dtype=np.float64)
        self.had = had
        if had:
            # Rotate the input dimension in blocks: y = (x R)(W R)^T, R block-diagonal orthonormal.
            hb = hadamard(had)
            signs = np.random.default_rng(0).choice([-1.0, 1.0], size=w.shape[1])
            self.signs = mx.array(signs.astype(np.float32))
            self.hb = mx.array(hb.astype(np.float32))
            w = ((w * signs).reshape(w.shape[0], -1, had) @ hb).reshape(w.shape)
            if h is not None:  # same rotation on the Hessian: H -> R^T H R
                k = w.shape[1]
                h1 = h.astype(np.float64) * signs[:, None] * signs[None, :]
                t = (h1.reshape(k, -1, had) @ hb).reshape(k, k)
                h = (t.T.reshape(k, -1, had) @ hb).reshape(k, k).T
        if wkind:
            w = fake_weight(w, wkind, h)
        self.weight = mx.array(w.astype(np.float32)).astype(lin.weight.dtype)
        self.bias = lin.bias if "bias" in lin else None
        self.act8 = act8

    def __call__(self, x):
        dt = x.dtype
        if self.had:
            xf = x.astype(mx.float32) * self.signs
            sh = xf.shape
            x = (xf.reshape(*sh[:-1], -1, self.had) @ self.hb).reshape(sh).astype(dt)
        if self.act8:
            x = act_int8(x)
        y = x @ self.weight.T
        return y if self.bias is None else y + self.bias


def act_int8(x):
    """Symmetric per-token INT8 fake quantization (round half to even, like the Metal kernel's rint)."""
    xf = x.astype(mx.float32)
    s = mx.max(mx.abs(xf), axis=-1, keepdims=True) / 127.0
    s = mx.where(s == 0, 1.0, s)
    return (mx.clip(mx.round(xf / s), -127, 127) * s).astype(x.dtype)


class QSwitchLinear(nn.Module):
    """Mixture-of-experts SwitchLinear (weight [experts, out, in]) with each expert's matrix fake-quantized
    by the same quantizer as dense layers, and per-token INT8 activations if requested."""

    def __init__(self, lin, wkind, act8):
        super().__init__()
        w = np.array(lin.weight.astype(mx.float32), dtype=np.float64)
        if wkind:
            w = np.stack([fake_weight(w[e], wkind) for e in range(w.shape[0])])
        self.weight = mx.array(w.astype(np.float32)).astype(lin.weight.dtype)
        self.bias = lin.bias if "bias" in lin else None
        self.act8 = act8

    def __call__(self, x, indices, sorted_indices=False):
        if self.act8:
            x = act_int8(x)
        y = mx.gather_mm(x, self.weight.swapaxes(-1, -2), rhs_indices=indices, sorted_indices=sorted_indices)
        return y if self.bias is None else y + mx.expand_dims(self.bias[indices], -2)


def quantize_model(model, method, repo=None):
    wkind, act8, had = parse_method(method)
    if wkind is None and not act8 and not had:
        return
    need_h = bool(wkind) and wkind.startswith("gptq:")
    for i, layer in enumerate(model.model.layers):
        for path in LINEARS:
            parent_name, name = path.split(".")
            parent = getattr(layer, parent_name)
            if hasattr(parent, name):
                h = calib.hessian(repo, i, path) if need_h else None
                setattr(parent, name, QLinear(getattr(parent, name), wkind, act8, had, h))
        switch = getattr(layer.mlp, "switch_mlp", None)  # MoE experts; the router (mlp.gate) stays BF16
        if switch is not None:
            if had:
                raise ValueError("Hadamard ablation is not implemented for MoE experts")
            for name in ("gate_proj", "up_proj", "down_proj"):
                setattr(switch, name, QSwitchLinear(getattr(switch, name), wkind, act8))
    mx.eval(model.parameters())


def wikitext_tokens(tok):
    f = hf_hub_download("Salesforce/wikitext", "wikitext-2-raw-v1/test-00000-of-00001.parquet", repo_type="dataset")
    text = "\n\n".join(pq.read_table(f).column("text").to_pylist())
    return np.array(tok.encode(text), dtype=np.int32)


def perplexity(model, toks, seq, max_chunks, second_half=False):
    # second_half: llama-perplexity's windows (seq tokens, disjoint) and scored predictions (after the first seq/2)
    n = min((len(toks) - (0 if second_half else 1)) // seq, max_chunks or 10 ** 9)
    first = seq // 2 if second_half else 0
    nll, count = 0.0, 0
    for i in range(n):
        chunk = mx.array(toks[i * seq:(i + 1) * seq + (0 if second_half else 1)])[None]
        logits = model(chunk[:, :-1]).astype(mx.float32)[:, first:]
        loss = nn.losses.cross_entropy(logits, chunk[:, first + 1:], reduction="sum")
        mx.eval(loss)
        nll += loss.item()
        count += chunk.shape[1] - 1 - first
    return float(np.exp(nll / count)), n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("repo")
    ap.add_argument("out")
    ap.add_argument("--methods", default=",".join(METHODS))
    ap.add_argument("--seq", type=int, default=2048)
    ap.add_argument("--max-chunks", type=int, default=0)
    ap.add_argument("--model-config", default="{}",
                    help='JSON overrides for fields missing from config.json, e.g. \'{"rms_norm_eps": 1e-5}\'')
    ap.add_argument("--second-half", action="store_true",
                    help="llama-perplexity's protocol: disjoint windows, only the second half of each scored")
    a = ap.parse_args()
    new = not os.path.exists(a.out)
    with open(a.out, "a", newline="") as f:
        wr = csv.writer(f)
        if new:
            wr.writerow(["model", "method", "seq", "chunks", "ppl", "seconds"])
        toks = None
        if any((parse_method(m)[0] or "").startswith("gptq:") for m in a.methods.split(",")):
            calib.compute(a.repo, model_config=json.loads(a.model_config))  # own GPU step, cached
        for m in a.methods.split(","):
            with eval_step():  # yields the GPU to timing runs (tools/gpu_timing.sh) between methods
                t0 = time.time()
                model, tok = load(a.repo, model_config=json.loads(a.model_config))
                if toks is None:
                    toks = wikitext_tokens(tok)
                quantize_model(model, m, a.repo)
                ppl, n = perplexity(model, toks, a.seq, a.max_chunks, a.second_half)
                name = m + (" (second half)" if a.second_half else "")
                wr.writerow([a.repo, name, a.seq, n, f"{ppl:.4f}", f"{time.time() - t0:.0f}"])
                f.flush()
                print(f"{a.repo} {name:24s} ppl {ppl:8.4f}  ({n} x {a.seq} tokens, {time.time() - t0:.0f}s)", flush=True)
                del model


if __name__ == "__main__":
    main()
