"""Another INT8-exact weight format (sim/formats.py, packed by e2e/pack_format.py) against SC-MXFP4 on the same fused
kernels, Qwen3-1.7B: prefill TTFT and batch-1 decode, interleaved with MLX MXFP4 in one process. Each fused model
installs its own format at every call (FusedQwen3.fmt), so the two formats alternate safely.

Usage: python bench_format.py prefill|decode --format e2m1/B16/M1..10 [--lengths 512,2048] [--out ...]
queue: run_timed.sh <log> zsh -c "bench_format.py prefill --lengths 512,2048 ...; ... --lengths 8192 ...;
        bench_format.py decode ..." (one GPU lock: burst, sustained and decode back to back, as results/07 does).
"""
import argparse
import csv
import os
import sys

import mlx.core as mx
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import bench_decode as bd  # noqa: E402
import bench_fused as bf  # noqa: E402
import bench_prefill as bp  # noqa: E402
import decode as dec  # noqa: E402
import fused_qwen3 as fq  # noqa: E402

REPO = "Qwen/Qwen3-1.7B-Base"
CACHE = os.path.expanduser("~/.cache/mxfp4-int8-m5")


def main():
    from mlx_lm import load
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["prefill", "decode"])
    ap.add_argument("--format", default="e2m1/B16/M1..10")
    ap.add_argument("--lengths", default="512,2048")
    ap.add_argument("--ctx", default="512,2048")
    ap.add_argument("--reps", type=int, default=7)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    mxm = load(os.path.join(CACHE, REPO.split("/")[-1] + "-mxfp4"))[0]
    f_sc = fq.load_fused(REPO, CACHE)[0]
    f_new = fq.load_fused(REPO, CACHE, packed="fmt:" + a.format)[0]
    names = {"fused_scmxfp4": f_sc, "fused_" + a.format: f_new}
    state = bp.power_state()
    if a.what == "prefill":
        models = {"mlx_mxfp4": bf.Plain(mxm), **{k: bf.Fused(v) for k, v in names.items()}}
        rows = bf.run(models, [int(x) for x in a.lengths.split(",")], a.reps, 151936)
    else:
        ms = {"mlx_mxfp4": bd.MLXMethod(mxm),
              **{k: bd.OursMethod(v, dec.DecodeQwen3(v, head=mxm)) for k, v in names.items()}}
        ctxs = [int(c) for c in a.ctx.split(",")]
        rng = np.random.default_rng(0)
        prompts = {c: mx.array(rng.integers(0, 151936, (1, c)).astype(np.int32)) for c in ctxs}
        for m in ms.values():
            bd.generate(m, prompts[ctxs[0]], 8)
        times = {(m, c): [] for m in ms for c in ctxs}
        keys = list(ms)
        for r in range(5):
            for c in ctxs:
                for m in keys[r % len(keys):] + keys[:r % len(keys)]:
                    times[(m, c)].append(bd.generate(ms[m], prompts[c], 128))
        rows = []
        for c in ctxs:
            ref = 128 / np.median(times[("mlx_mxfp4", c)])
            for m in keys:
                tps = 128 / np.median(times[(m, c)])
                rows.append(dict(method=m, context=c, gen_tokens=128, reps=5, tok_s_median=f"{tps:.1f}",
                                 speedup_vs_mlx_mxfp4=f"{tps / ref:.3f}"))
                print(f"{m:26s} ctx={c:5d} {tps:7.1f} tok/s  {tps / ref:.3f}x vs mlx_mxfp4", flush=True)
    rows = [dict(format=a.format, bits_new=f"{f_new.weight_bytes() * 8 / sum(w.N * w.K for _, ws in f_new.layers for w in ws.values()):.3f}",
                 bits_scmxfp4=f"{f_sc.weight_bytes() * 8 / sum(w.N * w.K for _, ws in f_sc.layers for w in ws.values()):.3f}",
                 **r) for r in rows]
    print("#", state)
    if a.out:
        with open(a.out, "w", newline="") as fh:
            fh.write(f"# {state}\n")
            wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
            wr.writeheader()
            wr.writerows(rows)
        print("wrote", a.out)


if __name__ == "__main__":
    main()
