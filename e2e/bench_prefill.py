"""End-to-end prefill (time to first token) of Qwen3 under MLX, with the transformer-block linears run as
  bf16           the unquantized checkpoint (MLX matmul)
  mlx_mxfp4      MLX MXFP4 checkpoint (mlx_lm convert --q-mode mxfp4, group 32): W4A16 quantized_matmul
  mlx_affine4    MLX 4-bit affine checkpoint (mlx_lm convert, group 64): W4A16 quantized_matmul
  sc_w4a8        SC-MXFP4 W4A8 two-pass kernels (e2e/w4a8_mlx.py), unpacking the 4.26-bit weights every call
  sc_w8a8_res    same INT8 GEMM with the unpacked INT8 weights kept resident (8 bits/weight; speed ceiling)
  no_linears     every block linear replaced by a zero fill of the output shape: the time of everything else
                 (attention, norms, RoPE, KV cache, embedding, lm_head), so TTFT - this = time in linears
Everything else (embedding, RMSNorm, RoPE, SDPA attention, KV cache) is MLX's own code and identical.

One prefill = the whole prompt through the model in a single forward pass into a fresh KV cache, then lm_head on
the last position only (what mlx_lm's generate does before the first token), evaluated and synchronized.
Timing: wall clock; every (method, length) is warmed up twice, then `reps` rounds, each round visiting every
length and every method in a rotated order (so drift and heat spread over all methods). Median reported.

Usage: python bench_prefill.py [--lengths 512,2048,8192] [--reps 10] [--out results/05_end_to_end.csv]
       (convert the MLX baselines first; see e2e/run_e2e.sh)
"""
import argparse
import csv
import os
import platform
import subprocess
import sys
import time

import mlx.core as mx
import mlx.nn as nn
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import w4a8_mlx as w  # noqa: E402
from mlx_lm import load  # noqa: E402
from mlx_lm.models.cache import make_prompt_cache  # noqa: E402


def prefill(model, ids):
    cache = make_prompt_cache(model)
    h = model.model(ids, cache=cache)[:, -1:]
    logits = model.model.embed_tokens.as_linear(h) if model.args.tie_word_embeddings else model.lm_head(h)
    mx.eval(logits, [c.state for c in cache])
    return logits


class ZeroLinear(nn.Module):
    def __init__(self, n):
        super().__init__()
        self.n = n

    def __call__(self, x):
        # Must consume x: with a constant output MLX's lazy graph would drop the attention feeding o_proj.
        return mx.zeros(x.shape[:-1] + (self.n,), x.dtype) + 0 * x[..., :1]


def strip_linears(model):
    for layer in model.model.layers:
        for path in w.LINEARS:
            parent_name, name = path.split(".")
            parent = getattr(layer, parent_name)
            setattr(parent, name, ZeroLinear(getattr(parent, name).weight.shape[0]))
    return model


def power_state():
    batt = subprocess.run(["pmset", "-g", "batt"], capture_output=True, text=True).stdout.splitlines()[0]
    mode = [ln.split()[-1] for ln in subprocess.run(["pmset", "-g"], capture_output=True, text=True).stdout.splitlines()
            if "powermode" in ln]
    chip = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip()
    return f"{chip}; macOS {platform.mac_ver()[0]}; MLX {mx.__version__}; {batt.strip()}; powermode={mode[0] if mode else '?'}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="Qwen/Qwen3-1.7B-Base")
    ap.add_argument("--cache-dir", default=os.path.expanduser("~/.cache/mxfp4-int8-m5"))
    ap.add_argument("--methods", default="bf16,mlx_mxfp4,mlx_affine4,sc_w4a8,sc_w8a8_res,no_linears")
    ap.add_argument("--lengths", default="512,2048,8192")
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--out", default=os.path.join(HERE, "..", "results", "05_end_to_end.csv"))
    a = ap.parse_args()
    base = a.repo.split("/")[-1]
    loaders = {
        "bf16": lambda: load(a.repo)[0],
        "mlx_mxfp4": lambda: load(os.path.join(a.cache_dir, base + "-mxfp4"))[0],
        "mlx_affine4": lambda: load(os.path.join(a.cache_dir, base + "-4bit"))[0],
        "sc_w4a8": lambda: w.load_w4a8(a.repo, a.cache_dir)[0],
        "sc_w8a8_res": lambda: w.load_w4a8(a.repo, a.cache_dir, resident_int8=True)[0],
        "no_linears": lambda: strip_linears(load(a.repo)[0]),
    }
    methods = a.methods.split(",")
    lengths = [int(x) for x in a.lengths.split(",")]
    models = {m: loaders[m]() for m in methods}
    rng = np.random.default_rng(0)
    vocab = models[methods[0]].args.vocab_size
    prompts = {L: mx.array(rng.integers(0, vocab, (1, L)).astype(np.int32)) for L in lengths}

    t_end = time.time() + 1.0
    while time.time() < t_end:  # ~1 s GPU warm-up
        prefill(models[methods[0]], prompts[lengths[0]])
    for L in lengths:
        for m in methods:
            for _ in range(2):
                prefill(models[m], prompts[L])

    times = {(m, L): [] for m in methods for L in lengths}
    for r in range(a.reps):
        for L in lengths:
            order = methods[r % len(methods):] + methods[:r % len(methods)]
            for m in order:
                mx.synchronize()
                t0 = time.perf_counter()
                prefill(models[m], prompts[L])
                mx.synchronize()
                times[(m, L)].append(time.perf_counter() - t0)

    state = power_state()
    print("#", state)
    rows = []
    for L in lengths:
        ref = np.median(times[("mlx_mxfp4", L)]) if "mlx_mxfp4" in methods else None
        for m in methods:
            t = np.array(times[(m, L)])
            med = float(np.median(t))
            rows.append([a.repo, m, L, a.reps, f"{med * 1e3:.2f}", f"{t.min() * 1e3:.2f}", f"{t.max() * 1e3:.2f}",
                         f"{L / med:.0f}", f"{ref / med:.3f}" if ref else ""])
            print(f"{m:12s} L={L:5d}  TTFT {med * 1e3:8.2f} ms  ({L / med:7.0f} tok/s)"
                  + (f"  {ref / med:.2f}x vs mlx_mxfp4" if ref else ""), flush=True)
    with open(a.out, "w", newline="") as f:
        f.write(f"# {state}\n")
        wr = csv.writer(f)
        wr.writerow(["model", "method", "prompt_tokens", "reps", "ttft_ms_median", "ttft_ms_min", "ttft_ms_max",
                     "prefill_tok_s", "speedup_vs_mlx_mxfp4"])
        wr.writerows(rows)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
