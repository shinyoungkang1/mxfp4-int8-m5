"""Prefill (time to first token) of Qwen3.8-Flash-Next: ours (fused W4A8, fused_qwen38.py) vs MLX W4A16 on the same
matrices (q38_mlx.py). The two systems do not fit in memory together (63 + 66 GB of weights), so each run times one
system in its own process (protocol of bench_fused.py otherwise: random prompts, 1 s warm-up, 2 untimed runs per
length, `--reps` timed runs, median), and run_q38_prefill.sh alternates sessions A B A B under run_timed.sh.

Usage (.venv-q38): python bench_q38.py --method fused|fused_b16|mlx_mxfp4|mlx_affine4 [--lengths 512,2048,8192]
                   [--reps 5] [--out results/10_prefill_q38.csv] [--session N]
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


class Fused:
    def __init__(self, fmt):
        import fused_qwen38 as FQ
        self.f = FQ.FusedQwen38(fmt)
        self.weight_bytes = self.f.weight_bytes()

    def prefill(self, ids):
        cache = self.f.make_cache()
        logits = self.f.logits(self.f.hidden(ids, cache)[:, -1:])
        mx.eval(logits, [c.state for c in cache])
        return logits, cache

    def step(self, tok, cache):
        if not hasattr(self, "d"):
            import fused_qwen38 as FQ
            self.d = FQ.DecodeQwen38(self.f)
        return self.d.step(tok, cache)


class Plain:
    def __init__(self, method):
        import q38_mlx
        self.m = q38_mlx.load_mlx(method)
        self.weight_bytes = q38_mlx.weight_bytes(self.m)

    def prefill(self, ids):
        cache = self.m.make_cache()
        logits = self.m.lm_head(self.m.model(ids, cache=cache)[:, -1:])
        mx.eval(logits, [c.state for c in cache])
        return logits, cache

    def step(self, tok, cache):
        return self.m(tok, cache=cache)


def decode(sysm, prompt, n):
    """Greedy decode of n tokens after the prompt; tokens/s over the last n - 8 (first 8 are warm-up)."""
    logits, cache = sysm.prefill(prompt)
    tok = mx.argmax(logits[:, -1], axis=-1)[:, None]
    ts = []
    for i in range(n):
        t0 = time.perf_counter()
        logits = sysm.step(tok, cache)
        tok = mx.argmax(logits[:, -1], axis=-1)[:, None]
        mx.eval(tok)
        ts.append(time.perf_counter() - t0)
    return 1.0 / float(np.median(ts[8:]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True)
    ap.add_argument("--lengths", default="512,2048,8192")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--session", default="")
    ap.add_argument("--decode", type=int, default=0, help="also time greedy decode of N tokens after 512-token prompts")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    t0 = time.time()
    sysm = Fused({"fused": "sc", "fused_b16": "e2m1/B16/M1..10"}[a.method]) if a.method.startswith("fused") \
        else Plain(a.method)
    print(f"{a.method}: loaded in {time.time() - t0:.0f} s, {sysm.weight_bytes / 2**30:.1f} GiB of weights", flush=True)
    lengths = [int(x) for x in a.lengths.split(",")]
    rng = np.random.default_rng(0)
    prompts = {L: mx.array(rng.integers(0, 248320, (1, L)).astype(np.int32)) for L in lengths}
    t_end = time.time() + 1.0
    while time.time() < t_end:
        sysm.prefill(prompts[lengths[0]])
    for L in lengths:
        for _ in range(2):
            sysm.prefill(prompts[L])
    mx.reset_peak_memory()
    rows = []
    for L in lengths:
        ts = []
        for _ in range(a.reps):
            mx.synchronize()
            t1 = time.perf_counter()
            sysm.prefill(prompts[L])
            mx.synchronize()
            ts.append(time.perf_counter() - t1)
        med = float(np.median(ts))
        rows.append(dict(method=a.method, session=a.session, prompt_tokens=L, reps=a.reps,
                         ttft_ms_median=f"{med * 1e3:.1f}", ttft_ms_min=f"{min(ts) * 1e3:.1f}",
                         ttft_ms_max=f"{max(ts) * 1e3:.1f}", prefill_tok_s=f"{L / med:.0f}",
                         weights_gib=f"{sysm.weight_bytes / 2**30:.2f}",
                         peak_gib=f"{mx.get_peak_memory() / 2**30:.2f}"))
        print(rows[-1], flush=True)
    if a.decode:
        rates = [decode(sysm, mx.array(rng.integers(0, 248320, (1, 512)).astype(np.int32)), a.decode) for _ in range(3)]
        for r in rows:
            r["decode_tok_s"] = f"{float(np.median(rates)):.1f}"
        print(f"decode: {np.median(rates):.1f} tok/s (sessions of {a.decode} tokens: {[round(x, 1) for x in rates]})")
    state = bp.power_state()
    print("#", state)
    if a.out:
        new = not os.path.exists(a.out)
        with open(a.out, "a", newline="") as f:
            if new:
                f.write(f"# {state}\n")
            wr = csv.DictWriter(f, fieldnames=list(rows[0]))
            if new:
                wr.writeheader()
            wr.writerows(rows)


if __name__ == "__main__":
    main()
