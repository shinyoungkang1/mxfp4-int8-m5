"""Batch-1 decode speed (tokens/s): SC-MXFP4 W4A8 GEMV decode (e2e/decode.py) vs MLX's MXFP4 / 4-bit affine / BF16.

Every method runs the same loop: prefill a `ctx`-token random prompt with its own prefill path, then greedily
generate `gen` tokens, building token t+1's graph while token t runs on the GPU (mx.async_eval, as mlx_lm's
generate_step does). Time = generation of `gen` tokens after the first; median over `reps` rounds, methods visited
in rotated order each round. Ours uses the output projection of the MLX MXFP4 checkpoint (head="mxfp4"), so the
comparison differs only in the transformer blocks; "ours_bf16head" keeps the BF16 head of the accuracy evaluation;
"ours_prologue" (ablation) fuses RMSNorm/INT8 quantization into the GEMV prologue instead of separate kernels
("ours_sepquant" = "ours", the name used in the first runs).
Also reported: CPU time to build one step's graph (no evaluation), the lower bound on time per token when the
GPU work is shorter.

Usage: python bench_decode.py --model qwen3|olmoe [--ctx 512,2048] [--gen 128] [--reps 5] [--out results/07_decode_*.csv]
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
import decode as dec  # noqa: E402
from mlx_lm import load  # noqa: E402
from mlx_lm.models.cache import make_prompt_cache  # noqa: E402

REPOS = {"qwen3": "Qwen/Qwen3-1.7B-Base", "olmoe": "allenai/OLMoE-1B-7B-0924"}


class MLXMethod:
    def __init__(self, model):
        self.model = model

    def cache(self):
        return make_prompt_cache(self.model)

    def prefill(self, ids, cache):
        return self.model(ids, cache=cache)[:, -1:]

    def step(self, y, cache):
        return self.model(y, cache=cache)


class OursMethod:
    def __init__(self, fused, d):
        self.f, self.d = fused, d

    def cache(self):
        return make_prompt_cache(self.f.model)

    def prefill(self, ids, cache):
        return self.d.head(self.f.hidden(ids, cache)[:, -1:])

    def step(self, y, cache):
        return self.d.step(y, cache)


def build(model_name, methods, cache_dir):
    repo = REPOS[model_name]
    base = repo.split("/")[-1]
    cfg = {"rms_norm_eps": 1e-5} if model_name == "olmoe" else None
    out, fused = {}, {}  # the packed model is shared by the three "ours" variants (read-only)
    mlx_mx = None
    for m in methods:
        if m in ("mlx_mxfp4", "ours", "ours_sepquant", "ours_prologue") and mlx_mx is None:
            mlx_mx = load(os.path.join(cache_dir, base + "-mxfp4"))[0]
        if m == "mlx_mxfp4":
            out[m] = MLXMethod(mlx_mx)
        elif m == "mlx_affine4":
            out[m] = MLXMethod(load(os.path.join(cache_dir, base + "-4bit"))[0])
        elif m == "bf16":
            out[m] = MLXMethod(load(repo, model_config=cfg)[0])
        elif m in ("ours", "ours_bf16head", "ours_sepquant", "ours_prologue"):
            head = "model" if m == "ours_bf16head" else mlx_mx
            if model_name == "qwen3":
                import fused_qwen3 as fq
                f = fused.get(m) or fused.setdefault("f", fq.load_fused(repo, cache_dir)[0])
                d = dec.DecodeQwen3(f, head=head, fuse_prologue=m == "ours_prologue")
            else:
                import fused_olmoe as fo
                f = fused.get(m) or fused.setdefault("f", fo.load_fused_olmoe(repo, cache_dir)[0])
                d = dec.DecodeOlmoe(f, head=head, fuse_prologue=m == "ours_prologue")
            out[m] = OursMethod(f, d)
    return out


def generate(meth, prompt, n):
    cache = meth.cache()
    y = mx.argmax(meth.prefill(prompt, cache), -1).astype(mx.int32)
    mx.eval(y)
    mx.synchronize()
    t0 = time.perf_counter()
    nxt = mx.argmax(meth.step(y, cache), -1).astype(mx.int32)
    mx.async_eval(nxt)
    for _ in range(n - 1):
        cur = nxt
        nxt = mx.argmax(meth.step(cur, cache), -1).astype(mx.int32)
        mx.async_eval(nxt)
        cur.item()
    nxt.item()
    return time.perf_counter() - t0


def cpu_graph_ms(meth, prompt, n=20):
    cache = meth.cache()
    y = mx.argmax(meth.prefill(prompt, cache), -1).astype(mx.int32)
    mx.eval(y, [c.state for c in cache])
    t0 = time.perf_counter()
    for _ in range(n):
        out = meth.step(y, cache)  # graph only; the cache offset advances but nothing is evaluated
    dt = (time.perf_counter() - t0) / n
    del out
    return dt * 1e3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3", choices=list(REPOS))
    ap.add_argument("--cache-dir", default=os.path.expanduser("~/.cache/mxfp4-int8-m5"))
    ap.add_argument("--methods", default="mlx_mxfp4,mlx_affine4,ours,ours_prologue,ours_bf16head")
    ap.add_argument("--ctx", default="512,2048")
    ap.add_argument("--gen", type=int, default=128)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    methods = a.methods.split(",")
    ms = build(a.model, methods, a.cache_dir)
    vocab = 151936 if a.model == "qwen3" else 50304
    rng = np.random.default_rng(0)
    ctxs = [int(c) for c in a.ctx.split(",")]
    prompts = {c: mx.array(rng.integers(0, vocab, (1, c)).astype(np.int32)) for c in ctxs}
    for m in methods:  # warm-up (kernel compilation)
        generate(ms[m], prompts[ctxs[0]], 8)
    cpu = {m: cpu_graph_ms(ms[m], prompts[ctxs[0]]) for m in methods}
    times = {(m, c): [] for m in methods for c in ctxs}
    for r in range(a.reps):
        for c in ctxs:
            order = methods[r % len(methods):] + methods[:r % len(methods)]
            for m in order:
                times[(m, c)].append(generate(ms[m], prompts[c], a.gen))
    state = bp.power_state()
    print("#", state)
    rows = []
    for c in ctxs:
        ref = a.gen / np.median(times[("mlx_mxfp4", c)]) if "mlx_mxfp4" in methods else None
        for m in methods:
            t = np.array(times[(m, c)])
            tps = a.gen / np.median(t)
            rows.append(dict(model=REPOS[a.model], method=m, context=c, gen_tokens=a.gen, reps=a.reps,
                             tok_s_median=f"{tps:.1f}", tok_s_min=f"{a.gen / t.max():.1f}",
                             tok_s_max=f"{a.gen / t.min():.1f}", ms_per_token=f"{1e3 / tps:.3f}",
                             cpu_graph_ms_per_step=f"{cpu[m]:.3f}",
                             speedup_vs_mlx_mxfp4=f"{tps / ref:.3f}" if ref else ""))
            print(f"{m:14s} ctx={c:5d}  {tps:7.1f} tok/s  ({1e3 / tps:6.2f} ms/token, graph build {cpu[m]:.2f} ms)"
                  + (f"  {tps / ref:.3f}x vs mlx_mxfp4" if ref else ""), flush=True)
    if a.out:
        with open(a.out, "w", newline="") as f:
            f.write(f"# {state}\n")
            wr = csv.DictWriter(f, fieldnames=list(rows[0]))
            wr.writeheader()
            wr.writerows(rows)
        print("wrote", a.out)


if __name__ == "__main__":
    main()
