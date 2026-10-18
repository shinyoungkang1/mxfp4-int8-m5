"""Checks the fused MoE prefill path (e2e/fused_olmoe.py) on OLMoE-1B-7B-0924 against the fake-quant reference of
sim/eval_ppl.py (method sc_mxfp4_w4a8), as results/05_check.json does for the dense model.

1. Weights: every attention linear and every expert matrix (16 layers x (4 + 64 x 3)), decoded from the packed
   2-bit layout (gate/up rows interleaved), times its row scale equals the reference's fake-quantized weights.
2. Grouped expert GEMMs on real activations (first and last block, a WikiText window): the INT32 accumulators of the
   grouped kernel equal per-expert float64 integer GEMMs; the kernel's scaled/SwiGLU epilogues equal MLX ops applied
   to those INT32 results; INT8 activation codes vs the reference quantizer on the same activations.
3. Perplexity on WikiText-2 windows: fused vs reference (and BF16 for scale).

Usage: python check_fused_olmoe.py [--windows 4] [--seq 2048] [--out results/07_check_moe.json]
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
import fused_olmoe as fo  # noqa: E402
import sc_kernels as S  # noqa: E402
import sc_moe  # noqa: E402
import eval_ppl  # noqa: E402
from check_fused import ref_codes  # noqa: E402
from mlx_lm import load  # noqa: E402
from mlx_lm.models.activations import swiglu  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="allenai/OLMoE-1B-7B-0924")
    ap.add_argument("--cache-dir", default=os.path.expanduser("~/.cache/mxfp4-int8-m5"))
    ap.add_argument("--windows", type=int, default=4)
    ap.add_argument("--seq", type=int, default=2048)
    ap.add_argument("--out", default=os.path.join(HERE, "..", "results", "07_check_moe.json"))
    a = ap.parse_args()
    rep = {"repo": a.repo, "mlx": mx.__version__, "path": "e2e/fused_olmoe.py"}
    fused, tok = fo.load_fused_olmoe(a.repo, a.cache_dir)
    args = fused.args
    E, Fi, D = args.num_experts, args.intermediate_size, args.hidden_size
    ref, _ = load(a.repo, model_config=fo.MODEL_CONFIG)
    eval_ppl.quantize_model(ref, "sc_mxfp4_w4a8")
    n_lin = sum(4 * D * D + E * 3 * Fi * D for _ in fused.layers)
    rep["bits_per_weight_block_linears"] = fused.weight_bytes() * 8 / n_lin

    # 1. weights
    inv = mx.array(np.argsort(fused.perm_gu))
    mism, total = 0, 0
    for (_, ws, ex), lr in zip(fused.layers, ref.model.layers):
        for g, paths in (("qkv", fo.ATTN[:3]), ("o", fo.ATTN[3:])):
            pw = ws[g]
            dec = S.unpack_i8(pw.codes, pw.bcodes, pw.N, pw.K).astype(mx.float32) * pw.wscale[:, None]
            r0 = 0
            for p, n in zip(paths, pw.splits):
                wr = getattr(lr.self_attn, p.split(".")[1]).weight
                mism += int(mx.sum(dec[r0:r0 + n].astype(wr.dtype) != wr).item())
                total += wr.size
                r0 += n
        sw = lr.mlp.switch_mlp
        for e in range(E):
            for key, refw in (("gu", mx.concatenate([sw.gate_proj.weight[e], sw.up_proj.weight[e]])),
                              ("down", sw.down_proj.weight[e])):
                c, b, s = (t[e] for t in ex[key])
                N, K = refw.shape
                dec = S.unpack_i8(c, b, N, K).astype(mx.float32) * s[:, None]
                if key == "gu":
                    dec = dec[inv]
                mism += int(mx.sum(dec.astype(refw.dtype) != refw).item())
                total += refw.size
    rep["weight_mismatches_vs_fakequant"] = mism
    rep["weights_checked"] = total
    print(f"1. weights: {mism}/{total} elements differ from the fake-quant reference", flush=True)

    # 2. grouped expert GEMMs on real activations
    toks = eval_ppl.wikitext_tokens(tok)
    nl = len(fused.layers)
    fused.record, fused.record_layers = {}, (0, nl - 1)
    L = a.seq
    mx.eval(fused(mx.array(toks[:L])[None]))
    gem = {}
    for (i, _), (xq, xq_s, xs_s, act, xq2, xs2, yd, e_sorted, xn) in sorted(fused.record.items()):
        ex = fused.layers[i][2]
        R = L * args.num_experts_per_tok
        tm = max(fused.tile_gu[0], fused.tile_d[0])
        off, tstart, maxt = sc_moe.segments(e_sorted, E, tm)
        acc1 = sc_moe.moe_mm(xq_s, R, *ex["gu"], off, tstart, maxt, 2 * Fi, 0, tile=fused.tile_gu, bk=fused.bk)
        acc2 = sc_moe.moe_mm(xq2, R, *ex["down"], off, tstart, maxt, D, 0, tile=fused.tile_d, bk=fused.bk)
        es = np.array(e_sorted)
        err1 = err2 = 0.0
        x1, x2 = np.array(xq_s[:R]).astype(np.float64), np.array(xq2[:R]).astype(np.float64)
        a1, a2 = np.array(acc1).astype(np.float64), np.array(acc2).astype(np.float64)
        for e in range(E):
            rows = es == e
            if not rows.any():
                continue
            w1 = np.array(S.unpack_i8(ex["gu"][0][e], ex["gu"][1][e], 2 * Fi, D)).astype(np.float64)
            w2 = np.array(S.unpack_i8(ex["down"][0][e], ex["down"][1][e], D, Fi)).astype(np.float64)
            err1 = max(err1, np.abs(a1[rows] - x1[rows] @ w1.T).max())
            err2 = max(err2, np.abs(a2[rows] - x2[rows] @ w2.T).max())
        eidx = e_sorted
        g = (acc1.astype(mx.float32) * xs_s[:, None] * ex["gu"][2][eidx]).astype(mx.bfloat16)
        act_ref = swiglu(g[:, 0::4], g[:, 2::4])  # interleaved rows g_2p, g_2p+1, u_2p, u_2p+1
        act_ref = mx.stack([act_ref, swiglu(g[:, 1::4], g[:, 3::4])], -1).reshape(R, Fi)
        y_ref = (acc2.astype(mx.float32) * xs2[:R, None] * ex["down"][2][eidx]).astype(mx.bfloat16)
        gem[f"{i}.experts"] = {
            "assignments": R, "experts_used": int(len(np.unique(es))),
            "gate_up_int32_max_abs_err": float(err1), "down_int32_max_abs_err": float(err2),
            "swiglu_epilogue_differing": int(mx.sum(act != act_ref).item()),
            "down_epilogue_differing": int(mx.sum(yd != y_ref).item()),
            "expert_input_codes_differing_vs_ref_quantizer":
                int(mx.sum(ref_codes(xn[:L]) != xq[:L].astype(mx.float32)).item()),
            "down_input_codes_differing_vs_ref_quantizer":
                int(mx.sum(ref_codes(act) != xq2[:R].astype(mx.float32)).item()),
            "codes_total": [int(L * D), int(act.size)]}
        print(f"2. layer {i}: {gem[f'{i}.experts']}", flush=True)
    rep["expert_gemms"] = gem
    fused.record = None

    # 3. perplexity
    bf, _ = load(a.repo, model_config=fo.MODEL_CONFIG)
    nll = {"fused": 0.0, "ref": 0.0, "bf16": 0.0}
    agree = 0
    for j in range(a.windows):
        chunk = mx.array(toks[j * a.seq:(j + 1) * a.seq + 1])[None]
        lg = {"fused": fused(chunk[:, :-1]), "ref": ref(chunk[:, :-1]), "bf16": bf(chunk[:, :-1])}
        for k, v in lg.items():
            nll[k] += nn.losses.cross_entropy(v.astype(mx.float32), chunk[:, 1:], reduction="sum").item()
        agree += int(mx.sum(mx.argmax(lg["fused"], -1) == mx.argmax(lg["ref"], -1)).item())
    ntok = a.windows * a.seq
    rep["logits"] = {"windows": a.windows, "seq": a.seq, "fused_vs_ref_top1_agreement": agree / ntok,
                     **{f"ppl_{k}": float(np.exp(v / ntok)) for k, v in nll.items()}}
    print("3.", json.dumps(rep["logits"]), flush=True)
    rep["passed"] = bool(mism == 0 and all(v["gate_up_int32_max_abs_err"] == 0 and v["down_int32_max_abs_err"] == 0
                                           and v["swiglu_epilogue_differing"] == 0
                                           and v["down_epilogue_differing"] == 0 for v in gem.values()))
    with open(a.out, "w") as f:
        json.dump(rep, f, indent=1)
    print("PASSED" if rep["passed"] else "FAILED", "wrote", a.out)


if __name__ == "__main__":
    main()
