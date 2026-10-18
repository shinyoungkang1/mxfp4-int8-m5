"""Batch-1 decode of Qwen3.8-Flash-Next on our kernels (fused_qwen38.DecodeQwen38): where the time goes, and GEMV tile
tuning in situ (results/10: our decode was 0.85x MLX MXFP4).

1. Breakdown: ms per decode step for the full step, with the expert GEMVs skipped (MoE output = 0), and with every
   W4A8 GEMV skipped (projection outputs = 0): the remainder is the work outside the W4A8 matrices (hyper-connection
   mixers, DeltaNet core, attention, PLE n-gram lookup, lm_head, Python dispatch).
2. Tile sweep: tokens/s with per-shape GEMV tiles (sc_decode.GEMV_TILES) for the Qwen3.8 shapes.
Every measurement: greedy decode of `--tokens` tokens after a 512-token prompt, median ms/step over the last
tokens - 8, `--reps` sessions. Run with .venv-q38 under run_timed.sh (it loads the 59 GiB model: one job at a time).
"""
import argparse
import os
import sys
import time

import mlx.core as mx
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fused_qwen38 as FQ  # noqa: E402
import sc_decode as G  # noqa: E402

# (N, K): DeltaNet qkvz / out, attention qkv / o, experts gate|up (EPI 3, 11 slots) / down (11 slots)
QKVZ, OUT, QKV, O, GU, DN = (16384, 2560), (2560, 6144), (13312, 2560), (2560, 6144), (1280, 2560), (2560, 640)
CONFIGS = {
    "default": {},
    "dn_4x8x2": {DN: (4, 8, 2)}, "dn_4x8x4": {DN: (4, 8, 4)}, "dn_8x8x2": {DN: (8, 8, 2)}, "dn_2x8x4": {DN: (2, 8, 4)},
    "dn_4x16x4": {DN: (4, 16, 4)}, "dn_8x16x1": {DN: (8, 16, 1)},
    "gu_4x16x4": {GU: (4, 16, 4)}, "gu_8x16x2": {GU: (8, 16, 2)}, "gu_4x32x2": {GU: (4, 32, 2)}, "gu_4x8x4": {GU: (4, 8, 4)},
    "qkvz_4x16x4": {QKVZ: (4, 16, 4)}, "qkvz_8x16x2": {QKVZ: (8, 16, 2)}, "qkvz_4x32x2": {QKVZ: (4, 32, 2)},
    "out_4x16x4": {OUT: (4, 16, 4)}, "out_4x32x2": {OUT: (4, 32, 2)}, "out_8x16x1": {OUT: (8, 16, 1)},
    "qkv_4x16x4": {QKV: (4, 16, 4)}, "qkv_8x16x2": {QKV: (8, 16, 2)},
}


class Skip:
    """Context: replace the expert GEMVs (moe) and/or the dense GEMVs by zeros of the right shape."""

    def __init__(self, d, moe=False, dense=False):
        self.d, self.moe, self.dense = d, moe, dense

    def __enter__(self):
        d = self.d
        self.saved = (d._moe, d._gemv)
        if self.moe:
            d._moe = lambda x, *a: mx.zeros_like(x)
        if self.dense:
            d._gemv = lambda x, pw: mx.zeros((pw.N,), x.dtype)
        return self

    def __exit__(self, *e):
        self.d._moe, self.d._gemv = self.saved


def ms_per_step(f, d, prompt, tokens):
    cache = f.make_cache()
    logits = f.logits(f.hidden(prompt, cache)[:, -1:])
    tok = mx.argmax(logits[:, -1], axis=-1)[:, None]
    mx.eval(tok)
    ts = []
    for _ in range(tokens):
        t0 = time.perf_counter()
        logits = d.step(tok, cache)
        tok = mx.argmax(logits[:, -1], axis=-1)[:, None]
        mx.eval(tok)
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts[8:])) * 1e3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=48)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--configs", default=",".join(CONFIGS))
    ap.add_argument("--dense-only", action="store_true", help="only compare dense='ours' vs dense='mlx' decode")
    a = ap.parse_args()
    f = FQ.FusedQwen38(FQ.SC)
    prompt = mx.array(np.random.default_rng(0).integers(0, 248320, (1, 512)).astype(np.int32))
    if a.dense_only:
        print("## dense projections: our W4A8 GEMV vs MLX's MXFP4 GEMV on the same packed bytes (experts: ours)")
        res = {}
        for _ in range(2):  # alternate the two modes
            for mode in ("ours", "mlx"):
                d = FQ.DecodeQwen38(f, dense=mode)
                ms_per_step(f, d, prompt, 12)
                res.setdefault(mode, []).extend(ms_per_step(f, d, prompt, a.tokens) for _ in range(a.reps))
        for mode, v in res.items():
            print(f"dense={mode:5s} {1e3 / np.median(v):6.1f} tok/s  ({np.median(v):.2f} ms/step; {[round(x, 2) for x in v]})")
        return
    d = FQ.DecodeQwen38(f, dense="ours")
    ms_per_step(f, d, prompt, 16)  # warm-up / compile
    print("## breakdown (ms per decode step, median of sessions)")
    for name, kw in (("full step", {}), ("expert GEMVs skipped", dict(moe=True)),
                     ("all W4A8 GEMVs skipped", dict(moe=True, dense=True))):
        with Skip(d, **kw):
            v = [ms_per_step(f, d, prompt, a.tokens) for _ in range(a.reps)]
        print(f"{name:28s} {np.median(v):7.2f} ms  ({1e3 / np.median(v):.1f} tok/s)  sessions {[round(x, 2) for x in v]}",
              flush=True)
    print("\n## GEMV tile sweep (tok/s; DecodeQwen38 rebuilt per config, its MoE traced with the config's tiles)")
    for name in a.configs.split(","):
        G.GEMV_TILES.clear()
        G.GEMV_TILES.update(CONFIGS[name])
        d = FQ.DecodeQwen38(f, dense="ours")
        ms_per_step(f, d, prompt, 12)
        v = [ms_per_step(f, d, prompt, a.tokens) for _ in range(a.reps)]
        print(f"{name:16s} {1e3 / np.median(v):6.1f} tok/s  ({np.median(v):.2f} ms/step)", flush=True)
    G.GEMV_TILES.clear()


if __name__ == "__main__":
    main()
