"""Where does Qwen3.5-35B-A3B decode lose time against MLX MXFP4? Times the batch-1 decode step of the whole 40-block
model with each half of every block (DeltaNet / attention half, MoE half) taken from ours, from MLX's MXFP4 model, or
left out, so that every variant streams its weights from memory as the real step does (results/08: the halves timed
in isolation, on one block's cached weights, were within 2-4% of MLX's).

All variants start from the same MLX MXFP4 prefill of a 512-token prompt (the caches have the same layout; MLX's own
step after our prefill runs at 1.00x) and use the MXFP4 head. Ablated variants compute nonsense tokens; only their
time matters. Usage: python diag_decode35.py [--gen 64] [--reps 5] [--out results/08_diag_decode35.csv]
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
import bench_prefill as bp  # noqa: E402
import decode as dec  # noqa: E402
import fused_qwen35 as fz  # noqa: E402
import sc_decode as G  # noqa: E402
import sc_kernels as S  # noqa: E402

CACHE = os.path.expanduser("~/.cache/mxfp4-int8-m5")
PREFILL = []  # default prefill: MLX MXFP4


class Hybrid:
    """DecodeQwen35.step with the attention half `attn` and the MoE half `moe` each "ours", "mlx" or "none"."""

    def __init__(self, d, mxm, attn, moe, embed="mlx"):
        self.d, self.mxm, self.attn, self.moe = d, mxm, attn, moe
        self.emb_model = mxm if embed == "mlx" else d.f.model
        self.mlayers = mxm.language_model.model.layers
        self.D = d.args.hidden_size

    def _fixed(self, i, h, layer, ex):
        import fused_olmoe as fo
        d, f = self.d, self.d.f
        if not hasattr(self, "_fx"):
            k = d.args.num_experts_per_tok
            rng = np.random.default_rng(1)
            self._fx = []
            for _ in f.layers:
                e = np.concatenate([rng.choice(d.args.num_experts, k, replace=False), [d.args.num_experts]])
                s9 = np.concatenate([np.full(k, 1.0 / k), [0.5]]).astype(np.float32)
                self._fx.append((mx.array(e.astype(np.int32)), mx.array(s9[None]).astype(mx.bfloat16)))
            mx.eval(self._fx)
            self._pos = mx.arange(k + 1, dtype=mx.int32)
            F2, D = 2 * f.Fi, d.args.hidden_size
            gemvs = self.moe == "ours_gemvs"

            def fn(h, pn, e, s9, cg, bg, sg, cd, bd, sd):
                xq, xs, xn = S.rmsnorm_quant_n(h, pn, f.eps, 1)
                act = G.gemv(xq, cg, bg, xs, sg, F2, 3, eidx=e, xstride=0)
                xq2, xs2 = S.act_quant(act, k + 1)
                yd = G.gemv(xq2, cd, bd, xs2, sd, D, 1, eidx=e, xstride=1)
                if gemvs:
                    return h + yd[:1]
                return fo.moe_combine(yd, self._pos, s9, h, pn, f.eps, 1, quant=False)[0]

            self._fn = mx.compile(fn)
        e, s9 = self._fx[i]
        return self._fn(h, layer.post_attention_layernorm.weight, e, s9, *ex["gu"], *ex["down"])

    def _rvar(self, i, h, layer, ex):
        import fused_olmoe as fo
        d, f = self.d, self.d.f
        k, E, F2, D = d.args.num_experts_per_tok, d.args.num_experts, 2 * f.Fi, d.args.hidden_size
        mode = self.moe[len("ours_r_"):]
        if not hasattr(self, "_rfn"):
            pos = mx.arange(k + 1, dtype=mx.int32)
            self._rw = []
            for li, (lay, _, _, _) in enumerate(f.layers):
                if mode == "catT":
                    w = mx.contiguous(mx.concatenate([lay.mlp.gate.weight, lay.mlp.shared_expert_gate.weight]).T)
                    self._rw.append((w,))
                else:
                    g, sgm = self.mlayers[li].mlp.gate, self.mlayers[li].mlp.shared_expert_gate
                    self._rw.append((g.weight, g.scales, g.biases, sgm.weight, sgm.scales, sgm.biases))
            mx.eval(self._rw)

            def logits(xn, w):
                if mode == "catT":
                    lg = xn @ w[0]
                    return lg[:, :E], lg[:, E:]
                q = lambda a, b, c: mx.quantized_matmul(xn, a, b, c, transpose=True, group_size=64, bits=8)  # noqa
                return q(*w[:3]), q(*w[3:])

            def fn(h, pn, cg, bg, sg, cd, bd, sd, *w):
                xq, xs, xn = S.rmsnorm_quant_n(h, pn, f.eps, 1)
                lg, slg = logits(xn, w)
                gates = mx.softmax(lg, axis=-1, precise=True)
                inds = mx.stop_gradient(mx.argpartition(gates, kth=-k, axis=-1)[..., -k:])
                scores = mx.take_along_axis(gates, inds, axis=-1)
                scores = scores / scores.sum(axis=-1, keepdims=True)
                sig = mx.sigmoid(slg)
                e = mx.concatenate([inds.astype(mx.int32), mx.full((1, 1), E, mx.int32)], axis=-1).reshape(-1)
                s9 = mx.concatenate([scores, sig.astype(scores.dtype)], axis=-1)
                act = G.gemv(xq, cg, bg, xs, sg, F2, 3, eidx=e, xstride=0)
                xq2, xs2 = S.act_quant(act, k + 1)
                yd = G.gemv(xq2, cd, bd, xs2, sd, D, 1, eidx=e, xstride=1)
                return fo.moe_combine(yd, pos, s9, h, pn, f.eps, 1, quant=False)[0]

            self._rfn = mx.compile(fn)
        return self._rfn(h, layer.post_attention_layernorm.weight, *ex["gu"], *ex["down"], *self._rw[i])

    def step(self, tok, cache):
        d, f, fz_ = self.d, self.d.f, self.d.fz
        D = self.D
        h = self.emb_model.language_model.model.embed_tokens(tok)  # [1, 1, D]
        for i, (layer, kind, ws, ex) in enumerate(f.layers):
            c, ml = cache[i], self.mlayers[i]
            if self.attn == "mlx":
                x = h.reshape(1, 1, D)
                mod = ml.linear_attn if kind == "linear" else ml.self_attn
                h = x + mod(ml.input_layernorm(x), None, c)
            elif self.attn == "ours":
                h = h.reshape(1, D)
                if kind == "linear":
                    m = layer.linear_attn
                    xq, xs, xn = S.rmsnorm_quant_n(h, layer.input_layernorm.weight, f.eps, 1)
                    w = ws["qkvz"]
                    y = G.gemv(xq, w.codes, w.bcodes, xs, w.wscale, w.N, 1)
                    nqkv = 2 * m.key_dim + m.value_dim
                    bb = (xn @ m.in_proj_b.weight.T)[None]
                    aa = (xn @ m.in_proj_a.weight.T)[None]
                    o = fz_.deltanet_core(m, y[:nqkv][None, None], y[nqkv:][None, None], bb, aa, None, c)[0]
                    wo = ws["out"]
                else:
                    m = layer.self_attn
                    xq, xs = S.rmsnorm_quant(h, layer.input_layernorm.weight, f.eps, 1)
                    w = ws["qkv"]
                    y = G.gemv(xq, w.codes, w.bcodes, xs, w.wscale, w.N, 1)
                    o = fz_.attention_core(m, y[None, None], None, c)[0]
                    wo = ws["o"]
                xq, xs = S.act_quant(o, 1)
                h = G.gemv(xq, wo.codes, wo.bcodes, xs, wo.wscale, wo.N, 2, res=h)[None]
            if self.moe == "mlx":
                x = h.reshape(1, 1, D)
                h = x + ml.mlp(ml.post_attention_layernorm(x))
            elif self.moe == "mlx_fixroute":  # MLX's MoE block with the routing replaced by fixed experts / scores
                if not hasattr(self, "_mfx"):
                    rng = np.random.default_rng(1)
                    kk, E = d.args.num_experts_per_tok, d.args.num_experts
                    self._mfx = [mx.array(rng.choice(E, kk, replace=False).astype(np.int32)).reshape(1, 1, kk)
                                 for _ in f.layers]
                    self._msc = mx.full((1, 1, kk), 1.0 / kk, mx.bfloat16)
                    mx.eval(self._mfx, self._msc)
                x = h.reshape(1, 1, D)
                xn_ = ml.post_attention_layernorm(x)
                mlp = ml.mlp
                y = (mlp.switch_mlp(xn_, self._mfx[i]) * self._msc[..., None]).sum(axis=-2)
                h = x + (y + mx.sigmoid(mlp.shared_expert_gate(xn_)) * mlp.shared_expert(xn_))
            elif self.moe == "ours":
                h = d._moe(h.reshape(1, D), layer.post_attention_layernorm.weight, layer.mlp.gate.weight,
                           layer.mlp.shared_expert_gate.weight, *ex["gu"], *ex["down"])
            elif self.moe.startswith("ours_r_"):  # alternative router computations, same GEMVs + combine
                h = self._rvar(i, h.reshape(1, D), layer, ex)
            elif self.moe in ("ours_fixroute", "ours_gemvs"):
                # routing replaced by fixed expert slots / scores (computed once): GEMVs (+ combine)
                h = self._fixed(i, h.reshape(1, D), layer, ex)
        lm = self.mxm.language_model
        return lm.lm_head(lm.model.norm(h.reshape(1, 1, D)))


def generate(v, n):
    step, prefill_cache = v if isinstance(v, tuple) else (v, PREFILL[0])
    cache, y = prefill_cache()
    mx.synchronize()
    t0 = time.perf_counter()
    nxt = mx.argmax(step(y, cache), -1).astype(mx.int32).reshape(1, 1)
    mx.async_eval(nxt)
    for _ in range(n - 1):
        cur = nxt
        nxt = mx.argmax(step(cur, cache), -1).astype(mx.int32).reshape(1, 1)
        mx.async_eval(nxt)
        cur.item()
    nxt.item()
    return (time.perf_counter() - t0) / n


def main():
    from mlx_lm import load
    from mlx_lm.models.cache import make_prompt_cache
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctx", type=int, default=512)
    ap.add_argument("--gen", type=int, default=64)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--variants", default="")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    mxm = load(os.path.join(CACHE, fz.REPO.split("/")[-1] + "-mxfp4"))[0]
    f, _ = fz.load_fused_qwen35()
    f.two_pass = "auto"
    d = dec.DecodeQwen35(f, head=mxm)
    d_nc = dec.DecodeQwen35(f, head=mxm, compiled=False)
    d_old = dec.DecodeQwen35(f, head=mxm, lean=False)
    d_op = dec.DecodeQwen35(f, head=mxm, out_prologue=True)
    prompt = mx.array(np.random.default_rng(0).integers(0, 248320, (1, a.ctx)).astype(np.int32))

    def prefill_cache():
        cache = make_prompt_cache(mxm)
        y = mx.argmax(mxm(prompt, cache=cache)[:, -1:], -1).astype(mx.int32)
        mx.eval(y, [c.state for c in cache])
        return cache, y

    def our_prefill_cache():
        cache = f.make_cache()
        y = mx.argmax(d.head(f.hidden(prompt, cache)[:, -1:]), -1).astype(mx.int32)
        mx.eval(y, [c.state for c in cache])
        return cache, y

    PREFILL.append(prefill_cache)
    variants = {"mlx": lambda t, c: mxm(t, cache=c), "ours": d.step, "ours_uncompiled_moe": d_nc.step,
                "ours_before_lean": d_old.step, "ours_lean_out_prologue": d_op.step,
                "ours_own_prefill": (d.step, our_prefill_cache),
                "attn=ours,moe=ours,bf16_embed": Hybrid(d, mxm, "ours", "ours", embed="bf16").step}
    for at in ("ours", "mlx", "none"):
        for mo in ("ours", "mlx", "none", "ours_fixroute", "ours_gemvs", "ours_r_catT", "ours_r_mlxgate", "mlx_fixroute"):
            if (at, mo) not in (("none", "none"),):
                variants[f"attn={at},moe={mo}"] = Hybrid(d, mxm, at, mo).step
    if a.variants:
        variants = {k: v for k, v in variants.items() if k in a.variants.split(";")}
    for v in variants.values():
        generate(v, 8)
    times = {k: [] for k in variants}
    names = list(variants)
    for r in range(a.reps):
        for k in names[r % len(names):] + names[:r % len(names)]:
            times[k].append(generate(variants[k], a.gen))
    state = bp.power_state()
    ref = np.median(times["mlx"]) if "mlx" in times else None
    rows = []
    for k in names:
        ms = 1e3 * np.median(times[k])
        rows.append(dict(variant=k, ctx=a.ctx, gen=a.gen, reps=a.reps, ms_per_token=f"{ms:.3f}",
                         spread_ms=f"{1e3 * (max(times[k]) - min(times[k])):.3f}",
                         vs_mlx_ms=f"{ms - 1e3 * ref:+.3f}" if ref else ""))
        print(f"{k:28s} {ms:7.3f} ms/token  (spread {rows[-1]['spread_ms']} ms)  {rows[-1]['vs_mlx_ms']}", flush=True)
    print("#", state)
    if a.out:
        with open(a.out, "w", newline="") as fh:
            fh.write(f"# {state}\n")
            wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
            wr.writeheader()
            wr.writerows(rows)


if __name__ == "__main__":
    main()
