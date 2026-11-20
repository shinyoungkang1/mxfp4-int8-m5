"""Qwen3.5-35B-A3B-Base: time of one forward step over T new tokens (T = 1..32) on top of a filled cache, for the
fused W4A8 path against MLX's MXFP4 and 4-bit affine checkpoints (PLAN.md step B3).

A step of T tokens is what a speculative-decoding verify pass, parallel sampling of T continuations of one prompt,
or a small serving batch costs per model call; T = 1 is decode, large T is prefill. So the curve joins the decode
and prefill results and shows from which T the INT8 path wins.

Protocol:
  - one prompt of `ctx` random tokens prefilled once per (method, ctx); the cache is then rewound after every timed
    step (attention caches: offset set back, the buffer is kept; DeltaNet caches: the saved arrays put back), so every
    step sees exactly `ctx` cached tokens;
  - each step computes the logits of all T positions (a verify pass needs them) and is synchronised (mx.eval) before
    the clock stops; there is no pipelining across steps, unlike bench_qwen35.py decode;
  - methods interleaved within each repetition, order rotated per repetition; median of --reps steps;
  - ours: T = 1 is the decode path (DecodeQwen35.step, W4A8 GEMVs); T >= 2 is the prefill path (fused_qwen35.hidden,
    W4A8 GEMMs and the fused MoE). With --both-paths, T = 1 is also timed on the prefill path (method ours_prefill).
    Head: MLX's MXFP4 output projection for both systems, as in the decode timing.
Before timing, a check: ours on T = 8 tokens in one step against the same 8 tokens one at a time through the decode
path (largest logit difference and top-1 agreement), printed and kept in the provenance.
Usage: python bench_tokens_qwen35.py [--format e2m1/B8/M1..10] [--tokens 1,2,4,8,16,32] [--ctx 512,2048] --out x.csv
"""
import argparse
import csv
import os
import sys
import time

import mlx.core as mx
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "sim"))
sys.path.insert(0, HERE)
import provenance as prov  # noqa: E402  (first: its import time is the recorded start of the session)
import bench_prefill as bp  # noqa: E402
import decode as dec  # noqa: E402
import fused_qwen35 as fz  # noqa: E402
import sc_format as S  # noqa: E402
from mlx_lm import load  # noqa: E402
from mlx_lm.models.cache import make_prompt_cache  # noqa: E402

CACHE = os.path.expanduser("~/.cache/mxfp4-int8-m5")
BASE = fz.REPO.split("/")[-1]
MLX = {"mlx_mxfp4": "-mxfp4", "mlx_affine4": "-4bit"}
VOCAB = 248320


def snapshot(cache):
    return [("kv", c.offset) if hasattr(c, "offset") else ("arr", list(c.cache)) for c in cache]


def rewind(cache, snap):
    for c, (kind, v) in zip(cache, snap):
        if kind == "kv":
            c.offset = v
        else:
            c.cache = list(v)


class MLXStep:
    def __init__(self, model):
        self.m = model

    def cache(self):
        return make_prompt_cache(self.m)

    def fill(self, ids, cache):
        mx.eval(self.m(ids, cache=cache)[:, -1:], [c.state for c in cache])

    def step(self, ids, cache):
        return self.m(ids, cache=cache)


class OursStep:
    def __init__(self, f, head_model, gemv_at_1=True):
        self.f, self.gemv_at_1 = f, gemv_at_1
        self.d = dec.DecodeQwen35(f, head=head_model)

    def cache(self):
        return self.f.make_cache()

    def fill(self, ids, cache):
        mx.eval(self.d.head(self.f.hidden(ids, cache)[:, -1:]), [c.state for c in cache])

    def step(self, ids, cache):
        if ids.shape[1] == 1 and self.gemv_at_1:
            return self.d.step(ids, cache)
        self.f.two_pass = "auto"
        return self.d.head(self.f.hidden(ids, cache))


def check(meth, prompt, toks):
    """T tokens in one prefill-path step against the same tokens one by one through the decode path."""
    c1 = meth.cache(); meth.fill(prompt, c1)
    meth.f.two_pass = "auto"
    a = meth.d.head(meth.f.hidden(toks, c1)).reshape(toks.shape[1], -1).astype(mx.float32)
    c2 = meth.cache(); meth.fill(prompt, c2)
    b = mx.concatenate([meth.d.step(toks[:, i:i + 1], c2).reshape(1, -1) for i in range(toks.shape[1])]).astype(mx.float32)
    mx.eval(a, b)
    diff = float(mx.abs(a - b).max())
    agree = float((mx.argmax(a, -1) == mx.argmax(b, -1)).astype(mx.float32).mean())
    scale = float(mx.abs(b).max())
    return dict(tokens=toks.shape[1], max_abs_diff=round(diff, 4), max_abs_logit=round(scale, 3), top1_agree=agree)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--format", default="e2m1/B8/M1..10", help="sim/formats.py name, or scmxfp4")
    ap.add_argument("--tokens", default="1,2,3,4,6,8,12,16,24,32")
    ap.add_argument("--ctx", default="512,2048")
    ap.add_argument("--reps", type=int, default=25)
    ap.add_argument("--methods", default="mlx_mxfp4,mlx_affine4,ours")
    ap.add_argument("--both-paths", action="store_true")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    methods = a.methods.split(",")
    if "mlx_mxfp4" not in methods:
        ap.error("keep mlx_mxfp4: it is the reference")
    Ts = [int(x) for x in a.tokens.split(",")]
    ctxs = [int(x) for x in a.ctx.split(",")]
    fmt = None if a.format == "scmxfp4" else a.format
    mlx_models = {m: load(os.path.join(CACHE, BASE + MLX[m]))[0] for m in methods if m in MLX}
    f, _ = fz.load_fused_qwen35(fmt_name=fmt)
    built = {m: MLXStep(mlx_models[m]) for m in methods if m in MLX}
    if "ours" in methods:
        built["ours"] = OursStep(f, mlx_models["mlx_mxfp4"])
        if a.both_paths:
            built["ours_prefill"] = OursStep(f, mlx_models["mlx_mxfp4"], gemv_at_1=False)
    state = bp.power_state()
    rng = np.random.default_rng(0)
    chk = None
    if "ours" in built:
        p = mx.array(rng.integers(0, VOCAB, (1, ctxs[0])).astype(np.int32))
        t8 = mx.array(rng.integers(0, VOCAB, (1, 8)).astype(np.int32))
        chk = check(built["ours"], p, t8)
        print("check, 8 tokens in one step vs one at a time:", chk, flush=True)
    keys = list(built)
    rows = []
    for c in ctxs:
        prompt = mx.array(rng.integers(0, VOCAB, (1, c)).astype(np.int32))
        steps = {T: mx.array(rng.integers(0, VOCAB, (1, T)).astype(np.int32)) for T in Ts}
        caches, snaps = {}, {}
        for k in keys:
            caches[k] = built[k].cache()
            built[k].fill(prompt, caches[k])
            snaps[k] = snapshot(caches[k])
        times = {(k, T): [] for k in keys for T in Ts}

        def one(k, T):
            mx.synchronize()
            t0 = time.perf_counter()
            mx.eval(built[k].step(steps[T], caches[k]))
            mx.synchronize()
            dt = time.perf_counter() - t0
            rewind(caches[k], snaps[k])
            return dt

        for T in Ts:  # warm-up: kernel compilation per shape
            for k in keys:
                for _ in range(2):
                    one(k, T)
        for r in range(a.reps):
            for T in Ts:
                for k in keys[r % len(keys):] + keys[:r % len(keys)]:
                    times[(k, T)].append(one(k, T))
        for T in Ts:
            ref = np.median(times[("mlx_mxfp4", T)])
            for k in keys:
                t = np.array(times[(k, T)])
                med = np.median(t)
                path = ("decode" if (T == 1 and k == "ours") else "prefill") if k.startswith("ours") else "mlx"
                rows.append(dict(model=fz.REPO, format=a.format if k.startswith("ours") else "", method=k, path=path,
                                 context=c, tokens=T, reps=a.reps, ms_median=f"{1e3 * med:.3f}",
                                 ms_min=f"{1e3 * t.min():.3f}", ms_max=f"{1e3 * t.max():.3f}",
                                 tok_s=f"{T / med:.1f}", speedup_vs_mlx_mxfp4=f"{ref / med:.3f}",
                                 pack=os.path.basename(f.pack_path) if k.startswith("ours") else BASE + MLX[k],
                                 head="mlx_mxfp4" if k.startswith("ours") else "own"))
                print(f"ctx={c:5d} T={T:3d} {k:13s} {1e3 * med:8.2f} ms  {T / med:8.1f} tok/s  {ref / med:.3f}x", flush=True)
        del caches, snaps
    print("#", state)
    if a.out:
        with open(a.out, "w", newline="") as fh:
            fh.write(prov.csv_head(state))
            wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
            wr.writeheader()
            wr.writerows(rows)
        prov.sidecar(a.out, fresh=True, power=state, model=fz.REPO, what="tokens_per_step", methods=keys,
                     format=a.format, pack=f.pack_path, head="mlx_mxfp4", check=chk)
        print("wrote", a.out)


if __name__ == "__main__":
    main()
