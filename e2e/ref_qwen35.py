"""Fake-quant SC-MXFP4 W4A8 reference for Qwen3.5-MoE, and the weight half of its correctness check.

The reference mirrors sim/eval_ppl.py's method sc_mxfp4_w4a8 on the matrices that e2e/fused_qwen35.py runs in W4A8
(eval_ppl.QLinear for the dense ones and the shared expert; eval_ppl.fake_weight + per-token INT8 activations for
the routed experts, as eval_ppl.QSwitchLinear, with the expert matrices fake-quantized in parallel processes).
Router, shared_expert_gate, in_proj_b / in_proj_a, embeddings and lm_head stay BF16, as in the fused model.

Run in its own process (the BF16 reference holds ~70 GB):
  1. weights, on the GPU: fake-quantized matrices vs the packed weights decoded x row scale. Rows whose row scale is a
     float32 subnormal (dead expert rows, |w| ~ 1e-38) are flushed to zero by the GPU and show up as mismatches here;
     the exact comparison is check_weights_qwen35.py (float64 on the CPU);
  2. WikiText-2 windows: per-window NLL and top-1 tokens of the reference, saved for check_fused_qwen35.py.
Usage: python ref_qwen35.py [--windows 4] [--seq 2048] [--out results/08_ref_qwen35.npz]
"""
import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import mlx.core as mx
import mlx.nn as nn
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "sim"))
import eval_ppl  # noqa: E402
import fused_qwen35 as fz  # noqa: E402
import sc_kernels as S  # noqa: E402


def _fake(w, kind="sc_mx"):
    return eval_ppl.fake_weight(np.asarray(w, np.float64), kind).astype(np.float32)


def _fake_kind(args):
    return _fake(*args)


def build_fake(model, wkind, act8, workers=14):
    """Fake-quantizes (sim/eval_ppl.fake_weight `wkind`) the same matrices that the fused model runs in W4A8, with
    per-token INT8 activations if act8 (eval_ppl.METHODS semantics), in place."""
    layers = model.language_model.model.layers
    with ProcessPoolExecutor(workers) as ex:
        for i, layer in enumerate(layers):
            for paths in fz.DENSE[fz.layer_kind(layer)].values():
                for p in paths:
                    parent, name = p.rsplit(".", 1)
                    setattr(fz.get(layer, parent), name, eval_ppl.QLinear(fz.get(layer, p), wkind, act8, 0))
            for name in fz.EXPERTS:
                w = getattr(layer.mlp.switch_mlp, name).weight
                ws = list(np.array(w.astype(mx.float32)))
                fake = np.stack(list(ex.map(_fake_kind, [(x, wkind) for x in ws], chunksize=8)))
                setattr(layer.mlp.switch_mlp, name, RefSwitchLinear(mx.array(fake).astype(w.dtype), act8))
                setattr(layer.mlp.shared_expert, name,
                        eval_ppl.QLinear(getattr(layer.mlp.shared_expert, name), wkind, act8, 0))
                del fake, ws
            mx.eval(layer.parameters())
            print(f"\r{wkind} layer {i + 1}/{len(layers)}", end="", flush=True)
    print()


def set_act8(model, act8):
    for layer in model.language_model.model.layers:
        for paths in fz.DENSE[fz.layer_kind(layer)].values():
            for p in paths:
                fz.get(layer, p).act8 = act8
        for name in fz.EXPERTS:
            getattr(layer.mlp.switch_mlp, name).act8 = act8
            getattr(layer.mlp.shared_expert, name).act8 = act8


class RefSwitchLinear(nn.Module):
    """eval_ppl.QSwitchLinear with precomputed fake-quantized expert weights."""

    def __init__(self, weight, act8=True):
        super().__init__()
        self.weight = weight
        self.act8 = act8

    def __call__(self, x, indices, sorted_indices=False):
        if self.act8:
            x = eval_ppl.act_int8(x)
        return mx.gather_mm(x, self.weight.swapaxes(-1, -2), rhs_indices=indices, sorted_indices=sorted_indices)


def decoded(c, b, s, N, K):
    return S.unpack_i8(c, b, N, K).astype(mx.float32) * s[:, None]


def build(model, flat, workers=14):
    """Turns the BF16 model into the fake-quant reference in place; returns weight-mismatch counts vs `flat`."""
    mism, total = 0, 0
    layers = model.language_model.model.layers
    with ProcessPoolExecutor(workers) as ex:
        for i, layer in enumerate(layers):
            kind = fz.layer_kind(layer)
            for paths in fz.DENSE[kind].values():
                for p in paths:
                    parent, name = p.rsplit(".", 1)
                    q = eval_ppl.QLinear(fz.get(layer, p), "sc_mx", True, 0)
                    setattr(fz.get(layer, parent), name, q)
                    k = f"layers.{i}.{p}"
                    N, K = q.weight.shape
                    dec = decoded(flat[k + ".codes"], flat[k + ".bcodes"], flat[k + ".wscale"], N, K)
                    mism += int(mx.sum(dec.astype(q.weight.dtype) != q.weight).item())
                    total += q.weight.size
            E = layer.mlp.gate.weight.shape[0]
            for name in fz.EXPERTS:
                w = getattr(layer.mlp.switch_mlp, name).weight
                dt = w.dtype
                fake = np.stack(list(ex.map(_fake, list(np.array(w.astype(mx.float32))), chunksize=8)))
                wq = mx.array(fake).astype(dt)
                setattr(layer.mlp.switch_mlp, name, RefSwitchLinear(wq))
                se = getattr(layer.mlp.shared_expert, name)
                qs = eval_ppl.QLinear(se, "sc_mx", True, 0)
                setattr(layer.mlp.shared_expert, name, qs)
                k = f"layers.{i}.experts.{name}"
                c, b, s = flat[k + ".codes"], flat[k + ".bcodes"], flat[k + ".wscale"]
                N, K = qs.weight.shape
                for e in range(E + 1):
                    ref = wq[e] if e < E else qs.weight
                    mism += int(mx.sum(decoded(c[e], b[e], s[e], N, K).astype(dt) != ref).item())
                    total += ref.size
                del fake
            mx.eval(model.language_model.model.layers[i].parameters())
            print(f"reference layer {i + 1}/{len(layers)}: {mism} mismatches so far", flush=True)
    return mism, total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default=fz.REPO)
    ap.add_argument("--cache-dir", default=os.path.expanduser("~/.cache/mxfp4-int8-m5"))
    ap.add_argument("--windows", type=int, default=4)
    ap.add_argument("--seq", type=int, default=2048)
    ap.add_argument("--out", default=os.path.join(HERE, "..", "results", "08_ref_qwen35.npz"))
    a = ap.parse_args()
    from mlx_lm import load
    model, tok = load(a.repo, lazy=True)
    flat = mx.load(os.path.join(a.cache_dir, a.repo.replace("/", "--") + "-scmxfp4-p2.safetensors"))
    mism, total = build(model, flat)
    del flat
    toks = eval_ppl.wikitext_tokens(tok)
    nll, top1 = [], []
    for j in range(a.windows):
        chunk = mx.array(toks[j * a.seq:(j + 1) * a.seq + 1])[None]
        lg = model(chunk[:, :-1]).astype(mx.float32)
        nll.append(nn.losses.cross_entropy(lg, chunk[:, 1:], reduction="sum").item())
        top1.append(np.array(mx.argmax(lg, -1)[0]))
        print(f"window {j}: nll/token {nll[-1] / a.seq:.4f}", flush=True)
    rep = {"weight_mismatches_vs_fakequant": mism, "weights_checked": total, "windows": a.windows, "seq": a.seq,
           "ppl_ref": float(np.exp(sum(nll) / (a.windows * a.seq)))}
    np.savez(a.out, nll=np.array(nll), top1=np.stack(top1), meta=json.dumps(rep))
    print(json.dumps(rep))


if __name__ == "__main__":
    main()
