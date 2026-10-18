"""Qwen3.5-35B-A3B-Base (MoE, hybrid DeltaNet / attention): prefill TTFT and batch-1 decode tokens/s of the fused
SC-MXFP4 W4A8 path (e2e/fused_qwen35.py, decode.DecodeQwen35) vs MLX's MXFP4 and 4-bit affine checkpoints of the same
model (e2e/convert_mlx.py, routers at 8-bit as mlx_lm's quant_predicate asks). Protocols of bench_fused.py /
bench_decode.py. The three W4A8 prefill variants share one copy of the packed weights:
  fused          in-kernel expert unpack, two-pass once experts average >= 256 rows
  fused_inkernel / fused_2pass   the two expert-weight paths forced at every length (ablations)
Usage: python bench_qwen35.py prefill|decode [--lengths 512,2048,8192] [--out results/08_qwen35_<what>.csv]
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
import bench_fused as bf  # noqa: E402
import bench_decode as bd  # noqa: E402
import bench_prefill as bp  # noqa: E402
import decode as dec  # noqa: E402
import fused_qwen35 as fz  # noqa: E402
from mlx_lm import load  # noqa: E402

CACHE = os.path.expanduser("~/.cache/mxfp4-int8-m5")
BASE = fz.REPO.split("/")[-1]


class Variant(bf.Fused):
    def __init__(self, f, two_pass):
        super().__init__(f)
        self.two_pass = two_pass

    def prefill(self, ids):
        self.f.two_pass = self.two_pass
        cache = self.f.make_cache()
        logits = self.f.logits(self.f.hidden(ids, cache)[:, -1:])
        mx.eval(logits, [c.state for c in cache])


class MLXQ35(bf.Plain):
    def prefill(self, ids):
        from mlx_lm.models.cache import make_prompt_cache
        cache = make_prompt_cache(self.m)
        lm = self.m.language_model
        h = lm.model(ids, cache=cache)[:, -1:]
        mx.eval(lm.lm_head(h), [c.state for c in cache])


class OursDecode(bd.OursMethod):
    def cache(self):
        return self.f.make_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["prefill", "decode"])
    ap.add_argument("--lengths", default="512,2048,8192")
    ap.add_argument("--ctx", default="512,2048")
    ap.add_argument("--reps", type=int, default=7)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    mxm = load(os.path.join(CACHE, BASE + "-mxfp4"))[0]
    afm = load(os.path.join(CACHE, BASE + "-4bit"))[0]
    f, _ = fz.load_fused_qwen35()
    state = bp.power_state()
    if a.what == "prefill":
        models = {"mlx_mxfp4": MLXQ35(mxm), "mlx_affine4": MLXQ35(afm), "fused": Variant(f, "auto"),
                  "fused_inkernel": Variant(f, False), "fused_2pass": Variant(f, True)}
        rows = bf.run(models, [int(x) for x in a.lengths.split(",")], a.reps, 248320)
        fields = ["model", "regime"] + list(rows[0])
        rows = [dict(model=fz.REPO, regime="", **r) for r in rows]
    else:
        f.two_pass = "auto"
        ms = {"mlx_mxfp4": bd.MLXMethod(mxm), "mlx_affine4": bd.MLXMethod(afm),
              "ours": OursDecode(f, dec.DecodeQwen35(f, head=mxm))}
        rng = np.random.default_rng(0)
        ctxs = [int(c) for c in a.ctx.split(",")]
        prompts = {c: mx.array(rng.integers(0, 248320, (1, c)).astype(np.int32)) for c in ctxs}
        for m in ms:
            bd.generate(ms[m], prompts[ctxs[0]], 8)
        times = {(m, c): [] for m in ms for c in ctxs}
        for r in range(5):
            for c in ctxs:
                order = list(ms)[r % len(ms):] + list(ms)[:r % len(ms)]
                for m in order:
                    times[(m, c)].append(bd.generate(ms[m], prompts[c], 128))
        rows = []
        for c in ctxs:
            ref = 128 / np.median(times[("mlx_mxfp4", c)])
            for m in ms:
                tps = 128 / np.median(times[(m, c)])
                rows.append(dict(model=fz.REPO, method=m, context=c, gen_tokens=128, reps=5, tok_s_median=f"{tps:.1f}",
                                 ms_per_token=f"{1e3 / tps:.3f}", speedup_vs_mlx_mxfp4=f"{tps / ref:.3f}"))
                print(f"{m:12s} ctx={c:5d} {tps:7.1f} tok/s  {tps / ref:.3f}x vs mlx_mxfp4", flush=True)
        fields = list(rows[0])
    print("#", state)
    if a.out:
        with open(a.out, "w", newline="") as fh:
            fh.write(f"# {state}\n")
            wr = csv.DictWriter(fh, fieldnames=fields)
            wr.writeheader()
            wr.writerows(rows)
        print("wrote", a.out)


if __name__ == "__main__":
    main()
