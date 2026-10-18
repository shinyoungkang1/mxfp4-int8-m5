"""Checks the end-to-end W4A8 model (e2e/w4a8_mlx.py) against the fake-quant reference of sim/eval_ppl.py
(method sc_mxfp4_w4a8).

1. Weights: for every block linear, the kernel's unpacked INT8 weights times the per-row scale equal the
   reference's fake-quantized BF16 weights exactly.
2. Linears on real activations: for the 7 linears of the first and last block, the kernel's INT32 accumulators
   equal a float64 integer GEMM of the same INT8 operands (zero error), and the INT8 activation codes match the
   reference quantizer except at rounding ties.
3. Logits on WikiText-2 windows: differences vs the reference model, top-1 agreement, and perplexity of both.
   (The reference rounds dequantized activations to BF16 before a BF16 GEMM, while the kernel applies the scales
   to the exact integer sum in FP32, so logits agree to BF16-level noise rather than bitwise.)

Usage: python check_outputs.py [--repo Qwen/Qwen3-1.7B-Base] [--windows 8] [--seq 2048] [--out results/05_check.json]
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
import w4a8_mlx as w  # noqa: E402
import eval_ppl  # noqa: E402
from mlx_lm import load  # noqa: E402


def get(layer, path):
    for p in path.split("."):
        layer = getattr(layer, p)
    return layer


class Recorder(nn.Module):
    def __init__(self, inner, store, key):
        super().__init__()
        self.inner, self.store, self.key = inner, store, key

    def __call__(self, x):
        self.store[self.key] = x
        return self.inner(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="Qwen/Qwen3-1.7B-Base")
    ap.add_argument("--cache-dir", default=os.path.expanduser("~/.cache/mxfp4-int8-m5"))
    ap.add_argument("--windows", type=int, default=8)
    ap.add_argument("--seq", type=int, default=2048)
    ap.add_argument("--out", default=os.path.join(HERE, "..", "results", "05_check.json"))
    a = ap.parse_args()
    report = {"repo": a.repo, "mlx": mx.__version__}

    ours, tok = w.load_w4a8(a.repo, a.cache_dir)
    ref, _ = load(a.repo)
    eval_ppl.quantize_model(ref, "sc_mxfp4_w4a8")

    # 1. weights
    mism = 0
    for lo, lr in zip(ours.model.layers, ref.model.layers):
        for path in w.LINEARS:
            qo, qr = get(lo, path), get(lr, path)
            dec = (qo.int8_weight().astype(mx.float32) * qo.wscale[:, None]).astype(qr.weight.dtype)
            mism += int(mx.sum(dec != qr.weight).item())
    report["weight_mismatches_vs_fakequant"] = mism
    print(f"1. weights: {mism} elements differ from the fake-quant reference", flush=True)

    toks = eval_ppl.wikitext_tokens(tok)

    # 2. linears of the first and last block on real activations
    store = {}
    nl = len(ours.model.layers)
    for i in (0, nl - 1):
        for path in w.LINEARS:
            parent_name, name = path.split(".")
            parent = getattr(ours.model.layers[i], parent_name)
            setattr(parent, name, Recorder(getattr(parent, name), store, f"{i}.{path}"))
    mx.eval(ours(mx.array(toks[:a.seq])[None]))
    lin_report = {}
    for key, x in store.items():
        i, path = key.split(".", 1)
        lin = get(ours.model.layers[int(i)], path).inner
        x2 = x.reshape(-1, x.shape[-1])
        xq, xs = w.act_quant(x2)
        wq = lin.int8_weight()
        acc = np.array(w.mm_i8_i32(xq, wq)).astype(np.float64)
        acc_ref = np.array(xq).astype(np.float64) @ np.array(wq).astype(np.float64).T
        # reference per-token quantizer of eval_ppl.QLinear (float32 math)
        xf = x2.astype(mx.float32)
        s = mx.max(mx.abs(xf), axis=-1, keepdims=True) / 127.0
        s = mx.where(s == 0, 1.0, s)
        codes_ref = mx.clip(mx.round(xf / s), -127, 127)
        nd = int(mx.sum(codes_ref != xq.astype(mx.float32)).item())
        lin_report[key] = {"M": x2.shape[0], "N": lin.N, "K": lin.K, "int32_max_abs_err": float(np.abs(acc - acc_ref).max()),
                           "act_codes_differing": nd, "act_codes_total": x2.size}
        print(f"2. {key:24s} INT32 max err {lin_report[key]['int32_max_abs_err']:.0f}, act codes differing {nd}/{x2.size}")
    report["linears"] = lin_report
    for i in (0, nl - 1):
        for path in w.LINEARS:
            parent_name, name = path.split(".")
            parent = getattr(ours.model.layers[i], parent_name)
            setattr(parent, name, getattr(parent, name).inner)

    # 3. logits
    nll = {"ours": 0.0, "ref": 0.0}
    agree, count, max_abs, sum_abs, max_ref = 0, 0, 0.0, 0.0, 0.0
    for j in range(a.windows):
        chunk = mx.array(toks[j * a.seq:(j + 1) * a.seq + 1])[None]
        lo = ours(chunk[:, :-1]).astype(mx.float32)
        lr = ref(chunk[:, :-1]).astype(mx.float32)
        d = mx.abs(lo - lr)
        for k, lg in (("ours", lo), ("ref", lr)):
            nll[k] += nn.losses.cross_entropy(lg, chunk[:, 1:], reduction="sum").item()
        agree += int(mx.sum(mx.argmax(lo, -1) == mx.argmax(lr, -1)).item())
        max_abs = max(max_abs, mx.max(d).item())
        sum_abs += mx.sum(d).item()
        max_ref = max(max_ref, mx.max(mx.abs(lr)).item())
        count += a.seq
    n_logits = count * lo.shape[-1]
    report["logits"] = {"windows": a.windows, "seq": a.seq, "tokens": count,
                        "top1_agreement": agree / count, "max_abs_diff": max_abs, "mean_abs_diff": sum_abs / n_logits,
                        "max_abs_ref_logit": max_ref,
                        "ppl_ours": float(np.exp(nll["ours"] / count)), "ppl_ref": float(np.exp(nll["ref"] / count))}
    print("3.", json.dumps(report["logits"]), flush=True)

    # 4. noise floor for top-1 agreement: the same models recomputed in FP32 (first window). Per-token INT8
    # activation rounding turns any FP-level perturbation into whole-code flips, so even the reference disagrees
    # with its own FP32 evaluation on a sizeable fraction of tokens.
    x = mx.array(toks[:a.seq])[None]
    top1 = {"ours_bf16": ours, "ref_bf16": ref}
    bf, _ = load(a.repo)
    top1["unquant_bf16"] = bf
    top1 = {k: mx.argmax(m(x).astype(mx.float32), -1) for k, m in top1.items()}
    for m in (ours, ref, bf):
        m.set_dtype(mx.float32)
    top1.update({k: mx.argmax(m(x), -1) for k, m in (("ours_f32", ours), ("ref_f32", ref), ("unquant_f32", bf))})
    agree = lambda p, q: mx.mean(top1[p] == top1[q]).item()  # noqa: E731
    report["top1_noise_floor"] = {
        "tokens": a.seq,
        "ours_bf16_vs_ref_bf16": agree("ours_bf16", "ref_bf16"),
        "ref_bf16_vs_ref_f32": agree("ref_bf16", "ref_f32"),
        "ours_bf16_vs_ref_f32": agree("ours_bf16", "ref_f32"),
        "ours_f32_vs_ref_f32": agree("ours_f32", "ref_f32"),
        "unquant_bf16_vs_unquant_f32": agree("unquant_bf16", "unquant_f32"),
    }
    print("4.", json.dumps(report["top1_noise_floor"]), flush=True)
    with open(a.out, "w") as f:
        json.dump(report, f, indent=1)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
