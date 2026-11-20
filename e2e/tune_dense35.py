"""Qwen3.5-35B-A3B decode with the dense output projections (DeltaNet out_proj, attention o_proj, N x K = 2048 x 4096)
quantizing their own input in the GEMV prologue (DecodeQwen35 out_prologue, PRO 1) at larger threadgroups.

Run 42 / the follow-up micro-benchmarks: the separate act_quant kernel before these GEMVs costs about 3 us per call
against MLX's GEMV + residual add; the fused prologue at the default tile (2 simdgroups, 256 threadgroups) is slower
still (every threadgroup re-quantizes the 4096-wide row), but at 16 simdgroups per threadgroup (64 threadgroups) it
is 2-2.4 us faster than the separate kernel. This times whole decode steps per configuration, both formats in one
process, interleaved with MLX MXFP4 (ctx 512 and 2048, 128 generated tokens, 5 reps, median), and first checks that
every configuration's logits are bit-identical to the default decoder's over 16 greedy steps.
Usage: python tune_dense35.py [--out results/x.csv]   (under run_timed.sh; the CSV goes through anchor_check.py)
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
import fused_qwen35 as fz  # noqa: E402
import sc_decode as G  # noqa: E402

OUT = (2048, 4096)
# name -> (out_prologue, GEMV_TILES)
CONFIGS = {"default": (False, {}), "op_4x16x16": (True, {OUT: (4, 16, 16, 1)}), "op_4x16x8": (True, {OUT: (4, 16, 8, 1)}),
           "op_4x16x16_sk2": (True, {OUT: (4, 16, 16, 2)})}
FORMATS = {"scmxfp4": None, "e2m1/B8/M1..10": "e2m1/B8/M1..10"}
GEN, REPS, VOCAB = 128, 5, 248320


class Ours(bd.OursMethod):
    def __init__(self, f, d, tiles):
        super().__init__(f, d)
        self.tiles = tiles

    def cache(self):
        return self.f.make_cache()

    def step(self, y, cache):
        G.GEMV_TILES.clear()
        G.GEMV_TILES.update(self.tiles)
        return self.d.step(y, cache)


def logits_run(meth, prompt, n=16):
    cache = meth.cache()
    G.GEMV_TILES.clear()
    G.GEMV_TILES.update(getattr(meth, "tiles", {}))
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
    ap.add_argument("--ctx", default="512,2048")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    from mlx_lm import load
    cache_dir = os.path.expanduser("~/.cache/mxfp4-int8-m5")
    mxm = load(os.path.join(cache_dir, fz.REPO.split("/")[-1] + "-mxfp4"))[0]
    state = bp.power_state()
    meths, info = {"mlx_mxfp4": bd.MLXMethod(mxm)}, {"mlx_mxfp4": dict(format="", config="", identical="")}
    rng = np.random.default_rng(0)
    ctxs = [int(c) for c in a.ctx.split(",")]
    prompts = {c: mx.array(rng.integers(0, VOCAB, (1, c)).astype(np.int32)) for c in ctxs}
    for label, name in FORMATS.items():
        f = fz.load_fused_qwen35(fmt_name=name)[0]
        f.two_pass = "auto"
        ref = None
        for cname, (op, tiles) in CONFIGS.items():
            G.GEMV_TILES.clear()
            G.GEMV_TILES.update(tiles)  # the MoE half is traced with the tile set active (unchanged here)
            m = Ours(f, dec.DecodeQwen35(f, head=mxm, out_prologue=op, out_tile=None), dict(tiles))
            lg = logits_run(m, prompts[ctxs[0]][:, :64])
            if ref is None:
                ref = lg
            same = bool(np.array_equal(lg, ref))
            print(f"{label} {cname}: logits over 16 steps identical to default: {same}", flush=True)
            if not same:
                raise SystemExit(f"{label} {cname}: logits differ from the default decoder")
            k = f"{label}:{cname}"
            meths[k], info[k] = m, dict(format=label, config=cname, identical=same)
            bd.generate(m, prompts[ctxs[0]], 8)
    G.GEMV_TILES.clear()
    bd.generate(meths["mlx_mxfp4"], prompts[ctxs[0]], 8)
    keys = list(meths)
    times = {(k, c): [] for k in keys for c in ctxs}
    for r in range(REPS):
        for c in ctxs:
            for k in keys[r % len(keys):] + keys[:r % len(keys)]:
                times[(k, c)].append(bd.generate(meths[k], prompts[c], GEN))
    G.GEMV_TILES.clear()
    rows = []
    for c in ctxs:
        refs = GEN / np.median(times[("mlx_mxfp4", c)])
        for k in keys:
            t = np.array(times[(k, c)])
            tps = GEN / np.median(t)
            i = info[k]
            rows.append(dict(model=fz.REPO, format=i["format"], method="mlx_mxfp4" if k == "mlx_mxfp4" else "ours",
                             config=i["config"], context=c, gen_tokens=GEN, reps=REPS, tok_s_median=f"{tps:.1f}",
                             tok_s_min=f"{GEN / t.max():.1f}", tok_s_max=f"{GEN / t.min():.1f}", ms_per_token=f"{1e3 / tps:.3f}",
                             speedup_vs_mlx_mxfp4=f"{tps / refs:.3f}", logits_identical=i["identical"]))
            print(f"{k:32s} ctx={c:5d} {tps:7.1f} tok/s  {1e3 / tps:.3f} ms  {tps / refs:.3f}x vs mlx_mxfp4", flush=True)
    print("#", state)
    if a.out:
        with open(a.out, "w", newline="") as fh:
            fh.write(prov.csv_head(state))
            wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
            wr.writeheader()
            wr.writerows(rows)
        prov.sidecar(a.out, fresh=True, power=state, model=fz.REPO, what="decode", format=list(FORMATS),
                     configs={k: [v[0], {str(s): t for s, t in v[1].items()}] for k, v in CONFIGS.items()})
        print("wrote", a.out)


if __name__ == "__main__":
    main()
