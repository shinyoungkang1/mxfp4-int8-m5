"""Checks the fused W4A8 prefill path (e2e/fused_qwen3.py) against the fake-quant reference of sim/eval_ppl.py
(method sc_mxfp4_w4a8) and against the unfused path of results/05 (e2e/w4a8_mlx.py), as results/05_check.json does.

1. Weights: every block linear, decoded from the 2-bit packed, row-concatenated layout (q|k|v, gate|up), times its
   row scale equals the reference's fake-quantized BF16 weights exactly.
2. GEMMs on real activations: for the 4 GEMMs of the first and last block on a WikiText window, INT32 accumulators
   equal a float64 integer GEMM of the same operands; INT8 activation codes vs the reference quantizer applied to
   the same activations (MLX rms_norm / swiglu of our own residual stream; differences are rounding ties).
3. Logits on WikiText-2 windows: fused vs unfused (expected bit-identical), and perplexity of fused, unfused and
   the fake-quant reference.

Usage: python check_fused.py [--windows 8] [--seq 2048] [--out results/07_check_fused.json]
       python check_fused.py --packed fmt:e2m1/B16/M1..10 --out results/08_check_format_<name>.json
--packed fmt:<name> / gptq: weights of e2e/pack_format.py / gptq_pack.py; the reference is eval_ppl's
"w=fmt:<name>;a=8" / "w=gptq:sc_mx;a=8" and the unfused SC-MXFP4 path of results/05 is skipped.
"""
import argparse
import json
import os
import sys

import mlx.core as mx
import mlx.nn as nn
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "sim"))
import fused_qwen3 as fq  # noqa: E402
import sc_kernels as S  # noqa: E402
import w4a8_mlx as w  # noqa: E402
import eval_ppl  # noqa: E402
from mlx_lm import load  # noqa: E402
from mlx_lm.models.activations import swiglu  # noqa: E402


def get(layer, path):
    for p in path.split("."):
        layer = getattr(layer, p)
    return layer


def ref_codes(x):
    """eval_ppl.act_int8's quantizer (division by s, float32), returning the integer codes."""
    xf = x.astype(mx.float32)
    s = mx.max(mx.abs(xf), axis=-1, keepdims=True) / 127.0
    s = mx.where(s == 0, 1.0, s)
    return mx.clip(mx.round(xf / s), -127, 127)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="Qwen/Qwen3-1.7B-Base")
    ap.add_argument("--cache-dir", default=os.path.expanduser("~/.cache/mxfp4-int8-m5"))
    ap.add_argument("--windows", type=int, default=8)
    ap.add_argument("--seq", type=int, default=2048)
    ap.add_argument("--epilogue", default="gemm", choices=["gemm", "row"])
    ap.add_argument("--packed", default="rtn")
    ap.add_argument("--out", default=os.path.join(HERE, "..", "results", "07_check_fused.json"))
    a = ap.parse_args()
    rep = {"repo": a.repo, "mlx": mx.__version__,
           "path": f"e2e/fused_qwen3.py (fused, epilogue={a.epilogue})"}

    fused, tok = fq.load_fused(a.repo, a.cache_dir, epilogue=a.epilogue, packed=a.packed)
    rep["packed"], rep["format"] = a.packed, str(fused.fmt)
    unf = w.load_w4a8(a.repo, a.cache_dir)[0] if a.packed == "rtn" else None
    ref, _ = load(a.repo)
    ref_method = {"rtn": "sc_mxfp4_w4a8", "gptq": "w=gptq:sc_mx;a=8"}.get(a.packed, f"w={a.packed};a=8")
    rep["reference_method"] = ref_method
    eval_ppl.quantize_model(ref, ref_method, a.repo)
    S.set_format(fused.fmt)
    rep["bits_per_weight_block_linears"] = fused.weight_bytes() * 8 / sum(
        lw.N * lw.K for _, ws in fused.layers for lw in ws.values())

    # 1. weights
    mism, total = 0, 0
    for (_, ws), lr in zip(fused.layers, ref.model.layers):
        for g, paths in fq.GROUPS.items():
            pw = ws[g]
            dec = (S.unpack_i8(pw.codes, pw.bcodes, pw.N, pw.K).astype(mx.float32) * pw.wscale[:, None])
            if pw.perm is not None:  # gate/up rows interleaved for the swiglu epilogue
                dec = dec[mx.array(np.argsort(pw.perm))]
            r0 = 0
            for p, n in zip(paths, pw.splits):
                wr = get(lr, p).weight
                mism += int(mx.sum(dec[r0:r0 + n].astype(wr.dtype) != wr).item())
                total += wr.size
                r0 += n
    rep["weight_mismatches_vs_fakequant"] = mism
    rep["weights_checked"] = total
    print(f"1. weights: {mism}/{total} elements differ from the fake-quant reference", flush=True)

    toks = eval_ppl.wikitext_tokens(tok)

    # 2. GEMMs of the first and last block on real activations
    nl = len(fused.layers)
    fused.record, fused.record_layers = {}, (0, nl - 1)
    L = a.seq
    mx.eval(fused(mx.array(toks[:L])[None]))
    gemm = {}
    for (i, g), (x, xq, wq, acc) in sorted(fused.record.items()):
        layer = fused.layers[i][0]
        if g == "qkv":
            xr = mx.fast.rms_norm(x[:L], layer.input_layernorm.weight, fused.args.rms_norm_eps)
        elif g == "gu":
            xr = mx.fast.rms_norm(x[:L], layer.post_attention_layernorm.weight, fused.args.rms_norm_eps)
        elif g == "o" or not isinstance(x, tuple):
            xr = x[:L]
        else:
            gacc, gws, gxs = x
            y = (gacc.astype(mx.float32) * gxs[:, None] * gws[None, :]).astype(mx.bfloat16)[:L]
            Fd = y.shape[1] // 2
            xr = swiglu(y[:, :Fd], y[:, Fd:])
        if acc is None:  # epilogue="gemm": the GEMM writes BF16; check its INT8 operands with the INT32 GEMM
            acc = S.mm_i8_i32(xq, wq)
        acc_np = np.array(acc[:L]).astype(np.float64)
        acc_ref = np.array(xq[:L]).astype(np.float64) @ np.array(wq).astype(np.float64).T
        nd = int(mx.sum(ref_codes(xr) != xq[:L].astype(mx.float32)).item())
        gemm[f"{i}.{g}"] = {"M": L, "N": int(wq.shape[0]), "K": int(wq.shape[1]),
                            "int32_max_abs_err": float(np.abs(acc_np - acc_ref).max()),
                            "act_codes_differing_vs_ref_quantizer": nd, "act_codes_total": int(xr.size)}
        print(f"2. {i}.{g:5s} {gemm[f'{i}.{g}']}", flush=True)
    rep["gemms"] = gemm
    fused.record = None

    # 3. logits
    nll = {"fused": 0.0, "unfused": 0.0, "ref": 0.0}
    n_diff, n_tot, max_fu, max_fr, agree_fr = 0, 0, 0.0, 0.0, 0
    for j in range(a.windows):
        chunk = mx.array(toks[j * a.seq:(j + 1) * a.seq + 1])[None]
        lf = fused(chunk[:, :-1]).astype(mx.float32)
        lu = unf(chunk[:, :-1]).astype(mx.float32) if unf is not None else lf
        lr = ref(chunk[:, :-1]).astype(mx.float32)
        for k, lg in (("fused", lf), ("unfused", lu), ("ref", lr)):
            nll[k] += nn.losses.cross_entropy(lg, chunk[:, 1:], reduction="sum").item()
        n_diff += int(mx.sum(lf != lu).item())
        n_tot += lf.size
        max_fu = max(max_fu, mx.max(mx.abs(lf - lu)).item())
        max_fr = max(max_fr, mx.max(mx.abs(lf - lr)).item())
        agree_fr += int(mx.sum(mx.argmax(lf, -1) == mx.argmax(lr, -1)).item())
    ntok = a.windows * a.seq
    rep["logits"] = {"windows": a.windows, "seq": a.seq, "fused_vs_unfused_logits_differing": n_diff,
                     "logits_total": n_tot, "fused_vs_unfused_max_abs_diff": max_fu,
                     "fused_vs_ref_max_abs_diff": max_fr, "fused_vs_ref_top1_agreement": agree_fr / ntok,
                     **{f"ppl_{k}": float(np.exp(v / ntok)) for k, v in nll.items()}}
    print("3.", json.dumps(rep["logits"]), flush=True)
    if unf is None:
        rep["logits"].pop("fused_vs_unfused_logits_differing"), rep["logits"].pop("fused_vs_unfused_max_abs_diff")
        rep["logits"].pop("ppl_unfused")
    rep["passed"] = bool(mism == 0 and all(v["int32_max_abs_err"] == 0 for v in gemm.values()) and n_diff == 0)
    with open(a.out, "w") as f:
        json.dump(rep, f, indent=1)
    print("PASSED" if rep["passed"] else "FAILED", "wrote", a.out)


if __name__ == "__main__":
    main()
