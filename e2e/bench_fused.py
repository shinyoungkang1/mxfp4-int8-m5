"""Prefill (time to first token) of the fused SC-MXFP4 W4A8 path vs MLX's 4-bit checkpoints and the unfused path.
Same protocol as e2e/bench_prefill.py (results/05_end_to_end.md): one prefill = whole prompt in one forward pass into
a fresh KV cache, then the output head on the last position; 1 s warm-up, 2 warm-up passes per (method, length),
then `reps` rounds visiting every length and every method in rotated order; wall clock, median.

Methods:
  mlx_mxfp4, mlx_affine4, bf16      MLX checkpoints (see e2e/run_e2e.sh), W4A16 quantized_matmul / matmul
  sc_w4a8                          unfused SC-MXFP4 W4A8 of results/05 (4 kernels per linear, 4.26 bits/weight)
  fused                            e2e/fused_qwen3.py, 4.07 bits/weight, scaling/residual/swiglu in the GEMM epilogue
  fused_row                        same kernels but INT32 GEMM outputs and row-wise epilogue kernels
  fused_row_overlap                fused_row with each unpack ordered next to an independent GEMM (ablation)
  fused_res                        fused with INT8 weights kept resident (8 bits/weight; ceiling without unpack)

Usage: python bench_fused.py [--lengths 512,2048,8192] [--reps 10] [--methods ...] [--out results/07_prefill.csv]
"""
import argparse
import csv
import os
import sys
import time

import mlx.core as mx
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import bench_prefill as bp  # noqa: E402
import fused_qwen3 as fq  # noqa: E402
import w4a8_mlx as w  # noqa: E402
from mlx_lm import load  # noqa: E402
from mlx_lm.models.cache import make_prompt_cache  # noqa: E402


class Fused:
    def __init__(self, f):
        self.f = f

    def prefill(self, ids):
        cache = make_prompt_cache(self.f.model)
        logits = self.f.logits(self.f.hidden(ids, cache)[:, -1:])
        mx.eval(logits, [c.state for c in cache])


class Plain:
    def __init__(self, m):
        self.m = m

    def prefill(self, ids):
        bp.prefill(self.m, ids)


def loaders(repo, cache_dir):
    base = repo.split("/")[-1]
    return {
        "bf16": lambda: Plain(load(repo)[0]),
        "mlx_mxfp4": lambda: Plain(load(os.path.join(cache_dir, base + "-mxfp4"))[0]),
        "mlx_affine4": lambda: Plain(load(os.path.join(cache_dir, base + "-4bit"))[0]),
        "sc_w4a8": lambda: Plain(w.load_w4a8(repo, cache_dir)[0]),
        "fused": lambda: Fused(fq.load_fused(repo, cache_dir)[0]),
        "fused_row": lambda: Fused(fq.load_fused(repo, cache_dir, epilogue="row")[0]),
        "fused_row_overlap": lambda: Fused(fq.load_fused(repo, cache_dir, epilogue="row", overlap=True)[0]),
        "fused_res": lambda: Fused(fq.load_fused(repo, cache_dir, resident=True)[0]),
        "no_linears": lambda: Plain(bp.strip_linears(load(repo)[0])),
    }


def run(models, lengths, reps, vocab, ref="mlx_mxfp4"):
    methods = list(models)
    rng = np.random.default_rng(0)
    prompts = {L: mx.array(rng.integers(0, vocab, (1, L)).astype(np.int32)) for L in lengths}
    t_end = time.time() + 1.0
    while time.time() < t_end:
        models[methods[0]].prefill(prompts[lengths[0]])
    for L in lengths:
        for m in methods:
            for _ in range(2):
                models[m].prefill(prompts[L])
    times = {(m, L): [] for m in methods for L in lengths}
    for r in range(reps):
        for L in lengths:
            order = methods[r % len(methods):] + methods[:r % len(methods)]
            for m in order:
                mx.synchronize()
                t0 = time.perf_counter()
                models[m].prefill(prompts[L])
                mx.synchronize()
                times[(m, L)].append(time.perf_counter() - t0)
    rows = []
    for L in lengths:
        rt = np.median(times[(ref, L)]) if ref in methods else None
        for m in methods:
            t = np.array(times[(m, L)])
            med = float(np.median(t))
            rows.append(dict(method=m, prompt_tokens=L, reps=reps, ttft_ms_median=f"{med * 1e3:.2f}",
                             ttft_ms_min=f"{t.min() * 1e3:.2f}", ttft_ms_max=f"{t.max() * 1e3:.2f}",
                             prefill_tok_s=f"{L / med:.0f}", speedup_vs_mlx_mxfp4=f"{rt / med:.3f}" if rt else ""))
            print(f"{m:16s} L={L:5d}  TTFT {med * 1e3:8.2f} ms  ({L / med:7.0f} tok/s)"
                  + (f"  {rt / med:.2f}x vs {ref}" if rt else ""), flush=True)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="Qwen/Qwen3-1.7B-Base")
    ap.add_argument("--cache-dir", default=os.path.expanduser("~/.cache/mxfp4-int8-m5"))
    ap.add_argument("--methods", default="mlx_mxfp4,mlx_affine4,sc_w4a8,fused,fused_row,fused_res")
    ap.add_argument("--lengths", default="512,2048,8192")
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--regime", default="")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    ld = loaders(a.repo, a.cache_dir)
    models = {m: ld[m]() for m in a.methods.split(",")}
    vocab = load(a.repo, lazy=True)[0].args.vocab_size if "bf16" not in models else models["bf16"].m.args.vocab_size
    rows = run(models, [int(x) for x in a.lengths.split(",")], a.reps, vocab)
    state = bp.power_state()
    print("#", state)
    if a.out:
        with open(a.out, "w", newline="") as f:
            f.write(f"# {state}\n")
            wr = csv.DictWriter(f, fieldnames=["model", "regime"] + list(rows[0]))
            wr.writeheader()
            for r in rows:
                wr.writerow(dict(model=a.repo, regime=a.regime, **r))
        print("wrote", a.out)


if __name__ == "__main__":
    main()
