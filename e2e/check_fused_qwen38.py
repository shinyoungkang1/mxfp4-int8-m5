"""Checks of the fused Qwen3.8-Flash-Next W4A8 model (fused_qwen38.py) against the fake-quant reference (q38_ppl.py).

  --layers i,j   per layer: (1) the packed weights decode (x row scale, float64) to exactly the reference's
                 fake-quantized weights (eval_ppl.fake_weight 'sc_mx'), dense matrices and a sample of experts;
                 (2) the decoder layer on our kernels vs the reference layer (FP8-decoded weights fake-quantized,
                 per-token INT8 activations) on the same input: relative difference of the layer's update.
Usage (.venv-q38): python check_fused_qwen38.py --layers 0,1,3 [--experts 16]
"""
import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import mlx.core as mx
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "sim"))
import fused_qwen38 as FQ  # noqa: E402
import q38_ppl as P  # noqa: E402
import sc_format as F  # noqa: E402
import sc_kernels as S  # noqa: E402


def decoded(codes, bc, ws, n, k, fmt):
    return F.decode_numpy(np.asarray(codes).reshape(-1), np.asarray(bc), n, k, fmt).astype(np.float64) * \
        np.asarray(ws, np.float64)[:, None]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--layers", default="0,1,3")
    ap.add_argument("--experts", type=int, default=16)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    layers = [int(x) for x in a.layers.split(",")]
    model = FQ.FusedQwen38(layers=max(layers) + 1, load_top=False)
    ck, args, fmt = model.ck, model.args, model.fmt
    rng = np.random.default_rng(0)
    report = {}
    with ProcessPoolExecutor(14) as pool:
        for i in layers:
            ref = P.load_layer(ck, args, i)
            kind = args.layer_types[i]
            paths = P.DENSE[kind] + P.SHARED + P.EXPERTS
            fakes = P.fake_all({p: P.get_mod(ref, p).weight for p in paths}, "sc_mx", pool)
            # (1) weights
            flat = mx.load(os.path.join(FQ.pack_dir(FQ.SC, os.path.expanduser("~/.cache/mxfp4-int8-m5")),
                                        f"layer_{i:02d}.safetensors"))
            mism, total = 0, 0
            for p in P.DENSE[kind]:
                w = np.array(fakes[p].astype(mx.float32), np.float64)
                d = decoded(flat[p + ".codes"], flat[p + ".bcodes"], flat[p + ".wscale"], *w.shape, fmt)
                mism += int((np.abs(d.astype(np.float32) - w) > 0).sum()); total += w.size
            sample = list(rng.choice(args.num_experts, a.experts, replace=False)) + [args.num_experts]  # + shared
            for name in FQ.EXPERTS:
                for e in sample:
                    src = fakes["mlp.shared_expert." + name] if e == args.num_experts else fakes["mlp.switch_mlp." + name][e]
                    w = np.array(src.astype(mx.float32), np.float64)
                    d = decoded(flat[f"experts.{name}.codes"][e], flat[f"experts.{name}.bcodes"][e],
                                flat[f"experts.{name}.wscale"][e], *w.shape, fmt)
                    mism += int((np.abs(d.astype(np.float32) - w) > 0).sum()); total += w.size
            # (2) the layer on the same input
            for p in paths:
                P.set_mod(ref, p, (P.FQSwitch if p in P.EXPERTS else P.FQLinear)(fakes[p], True))
            L = 512
            ids = mx.array(rng.integers(0, args.vocab_size, (1, L)).astype(np.int32))
            h = mx.array((rng.standard_normal((1, L, args.hc_count * args.hidden_size)) * 0.5).astype(np.float32))
            h = h.astype(mx.bfloat16)
            pc = mx.full((1, args.ngram_size - 1), model.eos, ids.dtype)
            y_ref = ref(h, model.rope, "causal", None, None, None, ids, pc)
            y = model.block(i, h, ids, pc, "causal", None, S.pad_rows(L))
            upd = (y_ref - h).astype(mx.float32)
            rel = float(mx.linalg.norm((y - y_ref).astype(mx.float32).reshape(-1)).item()
                        / mx.linalg.norm(upd.reshape(-1)).item())
            report[i] = dict(kind=kind, weight_mismatches=mism, weights_checked=total, layer_update_rel_diff=rel)
            print(i, report[i], flush=True)
            del ref, fakes, flat
    if a.out:
        json.dump(report, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
