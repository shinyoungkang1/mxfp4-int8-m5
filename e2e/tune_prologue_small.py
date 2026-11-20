"""Qwen3-1.7B and OLMoE-1B-7B decode with the per-token INT8 of the o (and Qwen3: down) projection inputs inside the
GEMV prologue at 16 simdgroups per threadgroup, against the default separate act_quant kernels (cf. tune_dense35.py
for the 35B). Micro-benchmarks put the gain at 0.02-0.03 ms per token, about 1%. Per model: logits of each
configuration over 16 greedy steps must equal the default decoder's bit for bit; then 5 interleaved reps of 128
tokens at a 512-token context with MLX MXFP4 (the anchor; Qwen3-1.7B has a clean reference, OLMoE none).
Usage: python tune_prologue_small.py --model qwen3|olmoe [--out results/x.csv]   (under run_timed.sh)
"""
import argparse
import csv
import os
import sys

import mlx.core as mx
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "sim"))
sys.path.insert(0, HERE)
import provenance as prov  # noqa: E402
import bench_decode as bd  # noqa: E402
import bench_prefill as bp  # noqa: E402
import decode as dec  # noqa: E402

GEN, REPS, CTX = 128, 5, 512
CONFIGS = {"qwen3": {"default": dict(fuse_prologue=False), "act_4x16x16": dict(fuse_prologue="act", act_tile=(4, 16, 16, 1)),
                     "act_4x16x8": dict(fuse_prologue="act", act_tile=(4, 16, 8, 1))},
           "olmoe": {"default": {}, "o_4x16x16": dict(fuse_prologue="o", o_tile=(4, 16, 16, 1)),
                     "o_4x16x8": dict(fuse_prologue="o", o_tile=(4, 16, 8, 1))}}
VOCAB = {"qwen3": 151936, "olmoe": 50304}


def logits_run(meth, prompt, n=16):
    cache = meth.cache()
    y = mx.argmax(meth.prefill(prompt, cache), -1).astype(mx.int32)
    out = []
    for _ in range(n):
        lg = meth.step(y, cache)
        mx.eval(lg)
        out.append(np.array(lg.astype(mx.float32)))
        y = mx.argmax(lg, -1).astype(mx.int32)
    return np.stack(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=list(CONFIGS), required=True)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    ms = bd.build(a.model, ["mlx_mxfp4", "ours_sepquant"], os.path.expanduser("~/.cache/mxfp4-int8-m5"))
    f, head = ms["ours_sepquant"].f, ms["mlx_mxfp4"].model
    Dec = dec.DecodeQwen3 if a.model == "qwen3" else dec.DecodeOlmoe
    state = bp.power_state()
    prompt = mx.array(np.random.default_rng(0).integers(0, VOCAB[a.model], (1, CTX)).astype(np.int32))
    meths, same = {"mlx_mxfp4": ms["mlx_mxfp4"]}, {"mlx_mxfp4": ""}
    ref = None
    for name, kw in CONFIGS[a.model].items():
        m = bd.OursMethod(f, Dec(f, head=head, **kw))
        lg = logits_run(m, prompt[:, :64])
        ref = lg if ref is None else ref
        same[name] = bool(np.array_equal(lg, ref))
        print(f"{a.model} {name}: logits over 16 steps identical to default: {same[name]}", flush=True)
        if not same[name]:
            raise SystemExit(f"{name}: logits differ from the default decoder")
        meths[name] = m
    for m in meths.values():
        bd.generate(m, prompt, 8)
    keys = list(meths)
    times = {k: [] for k in keys}
    for r in range(REPS):
        for k in keys[r % len(keys):] + keys[:r % len(keys)]:
            times[k].append(bd.generate(meths[k], prompt, GEN))
    ref_tps = GEN / np.median(times["mlx_mxfp4"])
    rows = []
    for k in keys:
        t = np.array(times[k])
        tps = GEN / np.median(t)
        rows.append(dict(model=bd.REPOS[a.model], method="mlx_mxfp4" if k == "mlx_mxfp4" else "ours", config=k, context=CTX,
                         gen_tokens=GEN, reps=REPS, tok_s_median=f"{tps:.1f}", tok_s_min=f"{GEN / t.max():.1f}",
                         tok_s_max=f"{GEN / t.min():.1f}", ms_per_token=f"{1e3 / tps:.3f}",
                         speedup_vs_mlx_mxfp4=f"{tps / ref_tps:.3f}", logits_identical=same[k]))
        print(f"{a.model} {k:14s} {tps:7.1f} tok/s  {1e3 / tps:.3f} ms  {tps / ref_tps:.3f}x vs mlx_mxfp4", flush=True)
    print("#", state)
    if a.out:
        with open(a.out, "w", newline="") as fh:
            fh.write(prov.csv_head(state))
            wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
            wr.writeheader()
            wr.writerows(rows)
        prov.sidecar(a.out, fresh=True, power=state, model=bd.REPOS[a.model], what="decode",
                     configs={k: {kk: list(v) if isinstance(v, tuple) else v for kk, v in kw.items()}
                              for k, kw in CONFIGS[a.model].items()})
        print("wrote", a.out)


if __name__ == "__main__":
    main()
