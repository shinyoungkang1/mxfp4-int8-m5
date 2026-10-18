"""Calibration statistics: H = X^T X of every transformer-block linear layer's input.

Used as an output-error proxy for format selection (relative error tr(dW H dW^T) / tr(W H W^T)) and as the
Hessian for GPTQ. Calibration text is the WikiText-2 *train* split (perplexity is measured on test).
Layers sharing an input (q/k/v, gate/up) share one H. Cached as float32 .npy files under
~/.cache/mxfp4-int8-m5/hessians/<model>/ (several GB for 1-4B models).

Usage: python calib.py <hf-repo> [--nseq 128] [--seq 2048]
"""
import argparse
import os

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from mlx_lm import load

from gpu_guard import eval_step

CACHE = os.path.expanduser("~/.cache/mxfp4-int8-m5/hessians")
# input group -> linear layers that read it
GROUPS = {"attn_in": ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"],
          "o_in": ["self_attn.o_proj"],
          "mlp_in": ["mlp.gate_proj", "mlp.up_proj"],
          "down_in": ["mlp.down_proj"]}
LINEAR_TO_GROUP = {lin: g for g, lins in GROUPS.items() for lin in lins}


def cache_dir(repo):
    return os.path.join(CACHE, repo.replace("/", "--"))


class _Capture(nn.Module):
    def __init__(self, lin, key, store):
        super().__init__()
        self.lin, self.key, self.store = lin, key, store

    def __call__(self, x):
        x2 = x.reshape(-1, x.shape[-1]).astype(mx.float32)
        h = x2.T @ x2
        self.store[self.key] = h if self.key not in self.store else self.store[self.key] + h
        return self.lin(x)


def wikitext_train_tokens(tok):
    f = hf_hub_download("Salesforce/wikitext", "wikitext-2-raw-v1/train-00000-of-00001.parquet", repo_type="dataset")
    text = "\n\n".join(pq.read_table(f).column("text").to_pylist())
    return np.array(tok.encode(text), dtype=np.int32)


def compute(repo, nseq=128, seq=2048, model_config=None):
    out = cache_dir(repo)
    if os.path.exists(os.path.join(out, "done")):
        return out
    os.makedirs(out, exist_ok=True)
    with eval_step():
        model, tok = load(repo, model_config=model_config or {})
        toks = wikitext_train_tokens(tok)
        rng = np.random.default_rng(0)
        starts = rng.choice(len(toks) - seq - 1, size=nseq, replace=False)
        store = {}
        for i, layer in enumerate(model.model.layers):
            for g, lins in GROUPS.items():
                parent_name, name = lins[0].split(".")
                parent = getattr(layer, parent_name)
                if hasattr(parent, name):  # capture once per shared input, on the first reader
                    setattr(parent, name, _Capture(getattr(parent, name), (i, g), store))
        for j, s in enumerate(starts):
            model(mx.array(toks[s:s + seq])[None])
            mx.eval(list(store.values()))
        for (i, g), h in store.items():
            np.save(os.path.join(out, f"{i}_{g}.npy"), np.array(h / (nseq * seq)))
        open(os.path.join(out, "done"), "w").write(f"{nseq} x {seq} tokens of WikiText-2 train\n")
    return out


def hessian(repo, layer, linear_path):
    return np.load(os.path.join(cache_dir(repo), f"{layer}_{LINEAR_TO_GROUP[linear_path]}.npy"))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("repo")
    ap.add_argument("--nseq", type=int, default=128)
    ap.add_argument("--seq", type=int, default=2048)
    a = ap.parse_args()
    print(compute(a.repo, a.nseq, a.seq))
