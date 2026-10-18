"""Layer-streamed WikiText-2 perplexity of Qwen3.8-Flash-Next (official FP8 checkpoint) under weight / activation
quantization (fake quant, the semantics of sim/eval_ppl.py), for several methods in one pass over the layers.

The model (173 GB FP8, ~250 GB as BF16) does not fit in memory, so the hidden states of every evaluation window are
kept for every method (4 hyper-connection streams x 2560, BF16: ~6 GB per method for the full test set) and the
layers are applied one at a time: layer i is read from the checkpoint once (q38_ckpt), its quantized matrices are
fake-quantized once per weight kind, and every method's states go through it. Windows are independent 2048-token
sequences, as in sim/eval_ppl.py (full test set, no overlap); at 2048 tokens the QSA indexer keeps every token
(its budget is 2048), so attention is dense.

Quantized (the W4A8 set, as e2e/fused_qwen35.py for Qwen3.5): Gated-DeltaNet in_proj_qkv / in_proj_z / out_proj,
attention q / k / v / o, the routed experts and the shared expert. BF16 (FP8-decoded for the experts in the
reference): router, shared_expert_gate, in_proj_a / b, the QSA indexer projection, hyper-connection mixers, PLE
projections, n-gram table (FP8, a lookup), embeddings, lm_head.

Methods: names or specs of sim/eval_ppl.py ("sc_mxfp4_w4a8", "w=fmt:e2m1/B16/M1..10;a=8", "nvfp4_w4a16", ...);
"fp8" is the unquantized checkpoint; a spec may add ";s=experts" to quantize only the routed experts (the scope of
the public NVFP4 / FP8 recipes for this model).
Usage (.venv-q38): python q38_ppl.py --methods fp8,sc_mxfp4_w4a8,... --out ../results/10_ppl_qwen38.csv
                   [--windows N] [--layers N (debug: stop after N layers, report nothing)]
"""
import argparse
import csv
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_flatten, tree_unflatten

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "sim"))
import eval_ppl as E  # noqa: E402
import formats as F  # noqa: E402
import q38_ckpt as C  # noqa: E402
from gpu_guard import eval_step  # noqa: E402
from mlx_lm.models import qwen4_exp  # noqa: E402

DENSE = {"linear_attention": ["linear_attn.in_proj_qkv", "linear_attn.in_proj_z", "linear_attn.out_proj"],
         "full_attention": ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj"]}
SHARED = ["mlp.shared_expert.gate_proj", "mlp.shared_expert.up_proj", "mlp.shared_expert.down_proj"]
EXPERTS = ["mlp.switch_mlp.gate_proj", "mlp.switch_mlp.up_proj", "mlp.switch_mlp.down_proj"]


def parse(m):
    """-> (weight kind or None, INT8 activations, scope). Specs may add ";s=experts" to quantize only the routed
    experts (the scope of the public NVFP4 / FP8 recipes for this model); default scope: the whole W4A8 set."""
    if m == "fp8":
        return None, False, "all"
    kv = dict(part.split("=", 1) for part in m.split(";")) if "=" in m else {}
    scope = kv.pop("s", "all")
    assert scope in ("all", "experts"), scope
    wk, a8, had = E.parse_method(";".join(f"{k}={v}" for k, v in kv.items()) if kv else m)
    assert not had, "rotations are not supported here"
    return wk, a8, scope


def _fake(args):
    w, kind = args
    return E.fake_weight(np.asarray(w, np.float64), kind).astype(np.float32)


def fake_all(mats, kind, pool, chunk=64):
    """{path: mx weight [N, K] or [E, N, K]} -> same with fake-quantized weights (each expert separately: NVFP4's
    per-tensor scale is per matrix). Format-family kinds run on the GPU (rows are independent there). Experts are
    processed `chunk` at a time, so that the float32 copies stay small (peak memory)."""
    out = {}
    f = F.FORMATS[kind[4:]] if kind.startswith("fmt:") else None
    for p, w in mats.items():
        w3 = w if w.ndim == 3 else w[None]
        res = []
        for e0 in range(0, w3.shape[0], chunk):
            a = np.array(w3[e0:e0 + chunk].astype(mx.float32))
            if f is not None:
                deq = F.quantize_mx(a.reshape(-1, a.shape[-1]), f)[0].reshape(a.shape)
            else:
                deq = np.stack(list(pool.map(_fake, [(x, kind) for x in a], chunksize=4)))
            res.append(mx.array(deq).astype(w.dtype))
            mx.eval(res[-1])
            del a, deq
        out[p] = mx.concatenate(res) if w.ndim == 3 else res[0][0]
        mx.eval(out[p])
        del res
    return out


class FQLinear(nn.Module):
    def __init__(self, weight, act8):
        super().__init__()
        self.weight, self.act8 = weight, act8

    def __call__(self, x):
        return (E.act_int8(x) if self.act8 else x) @ self.weight.T


class FQSwitch(nn.Module):
    def __init__(self, weight, act8):
        super().__init__()
        self.weight, self.act8 = weight, act8

    def __call__(self, x, indices, sorted_indices=False):
        x = E.act_int8(x) if self.act8 else x
        return mx.gather_mm(x, self.weight.swapaxes(-1, -2), rhs_indices=indices, sorted_indices=sorted_indices)


def get_mod(root, path):
    for p in path.split("."):
        root = getattr(root, p)
    return root


def set_mod(root, path, mod):
    parent, name = path.rsplit(".", 1)
    setattr(get_mod(root, parent), name, mod)


DELTANET_KERNEL = [False]  # --deltanet-kernel


def load_layer(ck, args, i):
    layer = qwen4_exp.DecoderLayer(args, i)
    if layer.ple is not None:  # swap the (lazy, 51B-parameter) table module for the memory-mapped FP8 gather
        emb = layer.ple.ple_embedding
        table = C.NgramTable(ck, i, emb, memo=True)
        emb.pop("ngram_embedding")
        object.__setattr__(emb, "ngram_embedding", table)
    w = ck.layer_weights(i, args.num_experts)
    mine = {k for k, _ in tree_flatten(layer.parameters()) if "ple_embedding" not in k}
    assert set(w) == mine, (sorted(set(w) - mine)[:5], sorted(mine - set(w))[:5])
    for k, v in tree_flatten(layer.parameters()):
        if k in w:
            assert w[k].shape == v.shape, (k, w[k].shape, v.shape)
    layer.update(tree_unflatten(list(w.items())))
    # Training mode (MLX's default) runs the PR's op-by-op reference recurrence for Gated DeltaNet, as passes A / B of
    # results/10; eval mode runs its Metal kernel (less precise: results/10_qwen38.md section 3).
    layer.train(not DELTANET_KERNEL[0])
    mx.eval([v for k, v in tree_flatten(layer.parameters()) if k in w])
    return layer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", default="fp8,sc_mxfp4_w4a8,sc_mxfp4_w4a16,w=fmt:e2m1/B16/M1..10;a=8,"
                    "w=fmt:e2m1/B16/M1..10;a=16")
    ap.add_argument("--seq", type=int, default=2048)
    ap.add_argument("--windows", type=int, default=0)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--layers", type=int, default=0)
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--deltanet-kernel", action="store_true", help="DeltaNet Metal kernel instead of the reference path")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    DELTANET_KERNEL[0] = a.deltanet_kernel
    methods = a.methods.split(",")
    parsed = {m: parse(m) for m in methods}

    from transformers import AutoTokenizer
    ck = C.Checkpoint()
    args = ck.text_args()
    tok = AutoTokenizer.from_pretrained(ck.path)
    toks = E.wikitext_tokens(tok)
    n = (len(toks) - 1) // a.seq
    n = min(n, a.windows) if a.windows else n
    X = mx.array(toks[:n * a.seq].reshape(n, a.seq))
    Y = mx.array(toks[1:n * a.seq + 1].reshape(n, a.seq))
    print(f"{len(toks)} tokens, {n} windows of {a.seq}; methods {methods}", flush=True)

    top = ck.top_weights()
    hc = args.hc_count
    # states: per method, a list of batch arrays [batch, seq, 4 D], updated batch by batch (no whole-state copies:
    # the peak memory of pass B with 6 methods stays far from the 128 GB)
    h0 = [mx.tile(top["embed_tokens.weight"][X[b:b + a.batch]], (1, 1, hc)) for b in range(0, n, a.batch)]
    mx.eval(h0)
    states = {m: list(h0) for m in methods}
    del h0
    rope = qwen4_exp.RotaryEmbedding(int(args.head_dim * args.partial_rotary_factor), args.rope_theta)
    eos = args.eos_token_id[0] if isinstance(args.eos_token_id, list) else args.eos_token_id
    prev_ctx = mx.full((a.batch, args.ngram_size - 1), eos, X.dtype)
    kinds = sorted({wk for wk, _, _ in parsed.values() if wk})
    t_start = time.time()
    guard = C.MemoryGuard()
    with ProcessPoolExecutor(a.workers) as pool:
        for i in range(a.layers or args.num_hidden_layers):
            t0 = time.time()
            with eval_step():
                layer = load_layer(ck, args, i)
                paths = DENSE[layer.layer_type] + SHARED + EXPERTS
                orig = {p: get_mod(layer, p) for p in paths}
                t1 = time.time()
                tq = 0.0
                for kind in [None] + kinds:  # one fake-quantized copy of the layer at a time (peak memory)
                    ms = [m for m in methods if parsed[m][0] == kind]
                    if not ms:
                        continue
                    tk = time.time()
                    fakes = fake_all({p: orig[p].weight for p in paths}, kind, pool) if kind else None
                    tq += time.time() - tk
                    for m in ms:
                        wk, a8, scope = parsed[m]
                        for p in paths:
                            if (wk or a8) and (scope == "all" or p in EXPERTS):
                                w = fakes[p] if wk else orig[p].weight
                                set_mod(layer, p, FQSwitch(w, a8) if p in EXPERTS else FQLinear(w, a8))
                            else:
                                set_mod(layer, p, orig[p])
                        for bi, b in enumerate(range(0, n, a.batch)):
                            hb = states[m][bi]
                            o = layer(hb, rope, "causal", None, None, None, X[b:b + a.batch], prev_ctx[:hb.shape[0]])
                            mx.eval(o)
                            states[m][bi] = o
                            del hb, o
                        guard.check(f"layer {i}, {m}")
                    del fakes
                    mx.clear_cache()
                t2 = t1 + tq
                del layer, orig
                mx.clear_cache()
            print(f"layer {i} ({'DeltaNet' if i % 4 != 3 else 'attention'}): load {t1 - t0:.0f} s, quantize "
                  f"{t2 - t1:.0f} s, forward {time.time() - t2:.0f} s; total {(time.time() - t_start) / 60:.1f} min; "
                  f"|h| " + " ".join(f"{float(mx.sqrt(mx.mean(states[m][0][:1].astype(mx.float32) ** 2)).item()):.3g}"
                                     for m in methods), flush=True)
    if a.layers:
        return

    mixer = qwen4_exp.GatedResidual(args, use_combine=False)
    mixer.eval()
    mixer.update(tree_unflatten([(k[len("hyper_connection_mixer."):], v) for k, v in top.items()
                                 if k.startswith("hyper_connection_mixer.")]))
    head = top["lm_head.weight"]
    rows, per_window = [], {}
    logits_of = lambda m, b: (mixer(states[m][b // a.batch][b % a.batch:b % a.batch + 1]) @ head.T).astype(mx.float32)  # noqa: E731
    kl = {m: [0.0, 0.0] for m in methods if m != "fp8"} if "fp8" in methods else {}
    for m in methods:
        nll = 0.0
        per_window[m] = []
        for b in range(n):
            logits = logits_of(m, b)
            loss = nn.losses.cross_entropy(logits, Y[b:b + 1], reduction="sum")
            mx.eval(loss)
            nll += loss.item()
            per_window[m].append(loss.item())
        ppl = float(np.exp(nll / (n * a.seq)))
        rows.append(dict(model=C.REPO, method=m, ppl=round(ppl, 4), windows=n, seq=a.seq))
    if kl:  # distance to the FP8 model: mean per-token KL(p_fp8 || p_method) and top-1 disagreement
        for b in range(n):
            ref = logits_of("fp8", b)
            lref = ref - mx.logsumexp(ref, axis=-1, keepdims=True)
            pref, aref = mx.exp(lref), mx.argmax(ref, axis=-1)
            for m in kl:
                lg = logits_of(m, b)
                lm = lg - mx.logsumexp(lg, axis=-1, keepdims=True)
                k = mx.sum(pref * (lref - lm), axis=-1).sum()
                f = mx.sum((mx.argmax(lg, axis=-1) != aref).astype(mx.float32))
                mx.eval(k, f)
                kl[m][0] += k.item()
                kl[m][1] += f.item()
        for r in rows:
            if r["method"] in kl:
                r["kl_vs_fp8"] = round(kl[r["method"]][0] / (n * a.seq), 5)
                r["top1_flip_vs_fp8"] = round(kl[r["method"]][1] / (n * a.seq), 5)
    for r in rows:
        print(r, flush=True)
    if a.out:
        new = not os.path.exists(a.out)
        with open(a.out, "a", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=list(rows[-1]))
            if new:
                wr.writeheader()
            wr.writerows(rows)
        print("appended to", a.out)
        # per-window summed NLLs, for paired bootstrap confidence intervals between methods
        np.savez(os.path.splitext(a.out)[0] + f"_windows_{int(time.time())}.npz",
                 **{m.replace("/", "_"): np.array(v) for m, v in per_window.items()})


if __name__ == "__main__":
    main()
