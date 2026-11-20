"""Qwen3.5-35B-A3B-Base (MoE, hybrid DeltaNet / attention): prefill TTFT and batch-1 decode tokens/s of the fused
W4A8 path (e2e/fused_qwen35.py, decode.DecodeQwen35) vs MLX's MXFP4 and 4-bit affine checkpoints of the same
model (e2e/convert_mlx.py, routers at 8-bit as mlx_lm's quant_predicate asks). Protocols of bench_fused.py /
bench_decode.py.

--methods (the same default for every weight format, so that sessions of different formats have the same composition):
  prefill  mlx_mxfp4,mlx_affine4,fused      fused = in-kernel expert unpack, two-pass once experts average >= 256 rows
           also: fused_inkernel, fused_2pass  the two expert-weight paths forced at every length (ablations; they
                                              share the packed weights with fused)
  decode   mlx_mxfp4,mlx_affine4,ours       ours = W4A8 GEMVs with MLX's MXFP4 output head (the timing configuration)
           also: ours_bf16head              the BF16 output head of the accuracy runs
mlx_mxfp4 must stay in the list: it is the validity anchor (anchor_check.py) and the speed-up reference.
--format: SC-MXFP4 by default ("scmxfp4"), or a sim/formats.py name; "+rot" after a name = the pack in the rotated
basis. Several formats, separated by commas, are timed interleaved in one process (each model installs its format at
every call); every format keeps its own packed weights (about 18 GB each) in memory.
Usage: python bench_qwen35.py prefill|decode [--format F[,F2]] [--methods ...] [--lengths 512,2048,8192] [--out x.csv]
The CSV carries format, pack and head per row; <out>.prov.jsonl has the provenance (sim/provenance.py).
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
import provenance as prov  # noqa: E402  (first: its import time is the recorded start of the session)
import bench_fused as bf  # noqa: E402
import bench_decode as bd  # noqa: E402
import bench_prefill as bp  # noqa: E402
import decode as dec  # noqa: E402
import fused_qwen35 as fz  # noqa: E402
from mlx_lm import load  # noqa: E402

CACHE = os.path.expanduser("~/.cache/mxfp4-int8-m5")
BASE = fz.REPO.split("/")[-1]
SC = "scmxfp4"
MLX = {"mlx_mxfp4": "-mxfp4", "mlx_affine4": "-4bit"}
TWO_PASS = {"fused": "auto", "fused_inkernel": False, "fused_2pass": True}
HEAD = {"ours": "mlx_mxfp4", "ours_bf16head": "bf16"}
DEFAULT = {"prefill": "mlx_mxfp4,mlx_affine4,fused", "decode": "mlx_mxfp4,mlx_affine4,ours"}
KNOWN = {"prefill": list(MLX) + list(TWO_PASS), "decode": list(MLX) + list(HEAD)}
GEN, DECODE_REPS, VOCAB = 128, 5, 248320


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


def formats_of(spec, rot_flag):
    """--format / --rot -> [(label, sim/formats.py name or None for SC-MXFP4, rot)]."""
    out = []
    for s in (spec or SC).split(","):
        rot = s.endswith("+rot")
        name = s[:-4] if rot else s
        rot = rot or rot_flag
        if name == SC and rot:
            raise SystemExit("rotation needs a sim/formats.py format, not scmxfp4")
        out.append((name + ("+rot" if rot else ""), None if name == SC else name, rot))
    if len({x[0] for x in out}) != len(out):
        raise SystemExit(f"--format names a format twice: {spec}")
    if rot_flag and len(out) > 1:
        raise SystemExit("with several formats, write +rot after the name instead of --rot")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["prefill", "decode"])
    ap.add_argument("--lengths", default="512,2048,8192")
    ap.add_argument("--ctx", default="512,2048")
    ap.add_argument("--reps", type=int, default=7, help="prefill repetitions (decode: 5 x 128 tokens)")
    ap.add_argument("--methods", default="", help="comma-separated; default " + " / ".join(f"{k}: {v}" for k, v in DEFAULT.items()))
    ap.add_argument("--format", default="", help="scmxfp4 (default) or sim/formats.py names, comma-separated; +rot after a name = rotated pack")
    ap.add_argument("--rot", action="store_true", help="with one --format: weights packed in the rotated basis, rotated inputs")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    methods = (a.methods or DEFAULT[a.what]).split(",")
    bad = [m for m in methods if m not in KNOWN[a.what]]
    if bad or len(set(methods)) != len(methods):
        ap.error(f"--methods for {a.what}: each once, from {KNOWN[a.what]} (got {a.methods})")
    if "mlx_mxfp4" not in methods:
        ap.error("mlx_mxfp4 is the validity anchor (anchor_check.py) and the speed-up reference: keep it in --methods")
    fmts = formats_of(a.format, a.rot)
    mlx_models = {m: load(os.path.join(CACHE, BASE + MLX[m]))[0] for m in methods if m in MLX}
    mxm = mlx_models["mlx_mxfp4"]
    fused = {label: fz.load_fused_qwen35(fmt_name=name, rot=rot)[0] for label, name, rot in fmts}
    one = len(fused) == 1
    state = bp.power_state()
    built, info = {}, {}  # key (method, or method@format with several formats) -> timed object / its row fields
    for m in methods:
        if m in MLX:
            built[m] = MLXQ35(mlx_models[m]) if a.what == "prefill" else bd.MLXMethod(mlx_models[m])
            info[m] = dict(method=m, format=fmts[0][0] if one else "", pack=BASE + MLX[m], head="own")
            continue
        for label, f in fused.items():
            k = m if one else f"{m}@{label}"
            if a.what == "prefill":
                built[k], head = Variant(f, TWO_PASS[m]), "bf16"
            else:
                f.two_pass = "auto"
                built[k] = OursDecode(f, dec.DecodeQwen35(f, head=mxm if HEAD[m] == "mlx_mxfp4" else "model"))
                head = HEAD[m]
            info[k] = dict(method=m, format=label, pack=os.path.basename(f.pack_path), head=head)
    if a.what == "prefill":
        rows = bf.run(built, [int(x) for x in a.lengths.split(",")], a.reps, VOCAB)
        rows = [dict(model=fz.REPO, regime="", format=info[r["method"]]["format"], **{**r, "method": info[r["method"]]["method"]},
                     pack=info[r["method"]]["pack"], head=info[r["method"]]["head"]) for r in rows]
    else:
        keys = list(built)
        rng = np.random.default_rng(0)
        ctxs = [int(c) for c in a.ctx.split(",")]
        prompts = {c: mx.array(rng.integers(0, VOCAB, (1, c)).astype(np.int32)) for c in ctxs}
        for k in keys:  # warm-up (kernel compilation)
            bd.generate(built[k], prompts[ctxs[0]], 8)
        times = {(k, c): [] for k in keys for c in ctxs}
        for r in range(DECODE_REPS):
            for c in ctxs:
                for k in keys[r % len(keys):] + keys[:r % len(keys)]:
                    times[(k, c)].append(bd.generate(built[k], prompts[c], GEN))
        rows = []
        for c in ctxs:
            ref = GEN / np.median(times[("mlx_mxfp4", c)])
            for k in keys:
                t = np.array(times[(k, c)])
                tps = GEN / np.median(t)
                i = info[k]
                rows.append(dict(model=fz.REPO, format=i["format"], method=i["method"], context=c, gen_tokens=GEN, reps=DECODE_REPS,
                                 tok_s_median=f"{tps:.1f}", tok_s_min=f"{GEN / t.max():.1f}", tok_s_max=f"{GEN / t.min():.1f}",
                                 ms_per_token=f"{1e3 / tps:.3f}", speedup_vs_mlx_mxfp4=f"{tps / ref:.3f}", pack=i["pack"], head=i["head"]))
                print(f"{k:12s} ctx={c:5d} {tps:7.1f} tok/s (min {GEN / t.max():.1f}, max {GEN / t.min():.1f})  {tps / ref:.3f}x vs mlx_mxfp4", flush=True)
    print("#", state)
    if a.out:
        with open(a.out, "w", newline="") as fh:
            fh.write(prov.csv_head(state))
            wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
            wr.writeheader()
            wr.writerows(rows)
        prov.sidecar(a.out, fresh=True, power=state, model=fz.REPO, what=a.what, methods=methods,
                     format=[x[0] for x in fmts], pack={label: f.pack_path for label, f in fused.items()},
                     head=sorted({i["head"] for i in info.values()}))
        print("wrote", a.out)


if __name__ == "__main__":
    main()
