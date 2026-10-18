"""Checks the fused W4A8 prefill of Qwen3.5-35B-A3B-Base (e2e/fused_qwen35.py) against the fake-quant reference of
e2e/ref_qwen35.py (sim/eval_ppl.py's sc_mxfp4_w4a8 applied to the same matrices; run it first, it writes the weight
check and the reference's per-window NLL and top-1 tokens to results/08_ref_qwen35.npz).

1. Weights (from ref_qwen35.py): decoded packed weights x row scale == fake-quantized BF16 weights, every matrix.
2. GEMMs on real activations (a WikiText window; a linear-attention block, a full-attention block, the last block):
   INT32 accumulations of the dense GEMMs and of the grouped expert GEMMs (257 experts incl. the shared one) equal
   float64 integer GEMMs; activation codes vs the reference quantizer on the same activations.
3. Perplexity / top-1 agreement on the same windows as the reference.
Usage: python check_fused_qwen35.py [--ref results/08_ref_qwen35.npz] [--out results/08_check_qwen35.json]
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
import fused_qwen35 as fz  # noqa: E402
import sc_kernels as S  # noqa: E402
import sc_moe  # noqa: E402
import eval_ppl  # noqa: E402
from check_fused import ref_codes  # noqa: E402
from mlx_lm.models.activations import swiglu  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default=os.path.join(HERE, "..", "results", "08_ref_qwen35.npz"))
    ap.add_argument("--out", default=os.path.join(HERE, "..", "results", "08_check_qwen35.json"))
    ap.add_argument("--layers", default="0,3,39")
    a = ap.parse_args()
    ref = np.load(a.ref)
    meta = json.loads(str(ref["meta"]))
    W, seq = meta["windows"], meta["seq"]
    f, tok = fz.load_fused_qwen35()
    # weights: exact float64 comparison of check_weights_qwen35.py (the GPU comparison in ref_qwen35.py flushes the
    # float32-subnormal row scales of dead expert rows to zero; reported separately)
    wj = json.load(open(os.path.join(os.path.dirname(a.out), "08_check_weights_qwen35.json")))
    wt = wj["exact_float64_decode_vs_fake_weight"]
    rep = {"repo": fz.REPO, "mlx": mx.__version__, "path": "e2e/fused_qwen35.py",
           "weight_mismatches_vs_fakequant": wt["mismatches"], "weights_checked": wt["weights"],
           "rows_with_float32_subnormal_scale": wt["subnormal_scale_rows"], "rows": wt["rows"],
           "gpu_decode_mismatches_from_subnormal_flush": meta["weight_mismatches_vs_fakequant"]}
    n_w = sum(lw.N * lw.K for _, _, ws, _ in f.layers for lw in ws.values())
    n_w += sum(int(np.prod(ex["gu"][2].shape)) * f.args.hidden_size + int(np.prod(ex["down"][2].shape)) * f.Fi
               for _, _, _, ex in f.layers)
    rep["bits_per_weight_w4a8_matrices"] = f.weight_bytes() * 8 / n_w
    toks = eval_ppl.wikitext_tokens(tok)

    # 2. GEMMs on real activations
    f.record, f.record_layers = {}, tuple(int(x) for x in a.layers.split(","))
    L = seq
    mx.eval(f(mx.array(toks[:L])[None]))
    gem = {}
    D, E1, Fi = f.args.hidden_size, f.E + 1, f.Fi
    for (i, g), v in sorted(f.record.items()):
        layer = f.layers[i][0]
        if g == "experts":
            xq, xq_s, xs_s, act, xq2, xs2, yd, e_sorted, xn = v
            ex = f.layers[i][3]
            R = L * (f.args.num_experts_per_tok + 1)
            off, tstart, maxt = sc_moe.segments(e_sorted, E1, 64)
            pad = mx.zeros((64, D), mx.int8)
            a1 = np.array(sc_moe.moe_mm(mx.concatenate([xq_s[:R], pad]), R, *ex["gu"], off, tstart, maxt, 2 * Fi, 0))
            a2 = np.array(sc_moe.moe_mm(mx.concatenate([xq2[:R], mx.zeros((64, Fi), mx.int8)]), R, *ex["down"], off,
                                        tstart, maxt, D, 0))
            es = np.array(e_sorted)
            x1, x2 = np.array(xq_s[:R]).astype(np.float64), np.array(xq2[:R]).astype(np.float64)
            e1 = e2 = 0.0
            for e in np.unique(es):
                rows = es == e
                w1 = np.array(S.unpack_i8(ex["gu"][0][e], ex["gu"][1][e], 2 * Fi, D)).astype(np.float64)
                w2 = np.array(S.unpack_i8(ex["down"][0][e], ex["down"][1][e], D, Fi)).astype(np.float64)
                e1 = max(e1, float(np.abs(a1[rows] - x1[rows] @ w1.T).max()))
                e2 = max(e2, float(np.abs(a2[rows] - x2[rows] @ w2.T).max()))
            gq = (mx.array(a1).astype(mx.float32) * xs_s[:, None] * ex["gu"][2][e_sorted]).astype(mx.bfloat16)
            act_ref = mx.stack([swiglu(gq[:, 0::4], gq[:, 2::4]), swiglu(gq[:, 1::4], gq[:, 3::4])], -1).reshape(R, Fi)
            gem[f"{i}.experts"] = {
                "assignments": R, "experts_used": int(len(np.unique(es))), "gate_up_int32_max_abs_err": e1,
                "down_int32_max_abs_err": e2, "swiglu_epilogue_differing": int(mx.sum(act != act_ref).item()),
                "expert_input_codes_differing_vs_ref_quantizer":
                    int(mx.sum(ref_codes(xn[:L]) != xq[:L].astype(mx.float32)).item()),
                "down_input_codes_differing_vs_ref_quantizer":
                    int(mx.sum(ref_codes(act) != xq2[:R].astype(mx.float32)).item()), "codes_total": [L * D, R * Fi]}
        else:
            x, xq, wq = v
            if g in ("qkvz", "qkv"):
                xr = mx.fast.rms_norm(x[:L], layer.input_layernorm.weight, f.args.rms_norm_eps)
            else:
                xr = x[:L]
            acc = np.array(S.mm_i8_i32(xq, wq)[:L]).astype(np.float64)
            acc_ref = np.array(xq[:L]).astype(np.float64) @ np.array(wq).astype(np.float64).T
            gem[f"{i}.{g}"] = {"M": L, "N": int(wq.shape[0]), "K": int(wq.shape[1]),
                               "int32_max_abs_err": float(np.abs(acc - acc_ref).max()),
                               "act_codes_differing_vs_ref_quantizer":
                                   int(mx.sum(ref_codes(xr) != xq[:L].astype(mx.float32)).item()),
                               "act_codes_total": int(xr.size)}
        print(f"2. {i}.{g}: {gem[f'{i}.{g}']}", flush=True)
    rep["gemms"] = gem
    f.record = None

    # 3. perplexity on the reference's windows
    nll, agree = [], 0
    for j in range(W):
        chunk = mx.array(toks[j * seq:(j + 1) * seq + 1])[None]
        lg = f(chunk[:, :-1]).astype(mx.float32)
        nll.append(nn.losses.cross_entropy(lg, chunk[:, 1:], reduction="sum").item())
        agree += int((np.array(mx.argmax(lg, -1)[0]) == ref["top1"][j]).sum())
    ntok = W * seq
    rep["logits"] = {"windows": W, "seq": seq, "ppl_fused": float(np.exp(sum(nll) / ntok)), "ppl_ref": meta["ppl_ref"],
                     "fused_vs_ref_top1_agreement": agree / ntok}
    print("3.", json.dumps(rep["logits"]), flush=True)
    ok = rep["weight_mismatches_vs_fakequant"] == 0
    for k, v in gem.items():
        ok &= all(v[x] == 0 for x in v if x.endswith("int32_max_abs_err")) and v.get("swiglu_epilogue_differing", 0) == 0
    rep["passed"] = bool(ok)
    with open(a.out, "w") as fh:
        json.dump(rep, fh, indent=1)
    print("PASSED" if ok else "FAILED", "wrote", a.out)


if __name__ == "__main__":
    main()
