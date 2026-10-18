"""Batch-1 decode (token generation) with SC-MXFP4 W4A8 GEMVs that read the packed 4.07-bit weights directly
(e2e/sc_decode.py), for Qwen3 (dense) and OLMoE (MoE: the top-8 experts' rows are gathered inside the GEMV).

The decode step computes the same function as the prefill path for the linears (same INT8 activations, exact
integer accumulations, same epilogues); attention is MLX's SDPA (vector kernel at L = 1), as for every method.

Output head: `head="model"` uses the checkpoint's own BF16 output projection (as in the accuracy evaluation);
`head=<mlx model>` borrows that model's (quantized) output projection, so a decode timing against MLX's 4-bit
checkpoints differs only in the transformer blocks (the BF16 head alone reads 0.6 GB per token on Qwen3-1.7B).
"""
import os
import sys

import mlx.core as mx

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import sc_decode as G  # noqa: E402
import sc_kernels as S  # noqa: E402


def _head_fn(model, head):
    src = model if head == "model" else head
    if src.args.tie_word_embeddings:
        return src.model.embed_tokens.as_linear
    return src.lm_head


class DecodeQwen3:
    """Built from a FusedQwen3 with epilogue="gemm" (gate/up rows interleaved for the swiglu epilogue).
    The fixed-shape parts of a step (before / after attention) run as two mx.compile'd functions, so their ~10
    custom kernels per layer replay without Python overhead; RoPE, the KV cache and SDPA are MLX calls."""

    def __init__(self, fused, head="model", compiled=True, fuse_prologue=False):
        assert fused.epilogue == "gemm"
        self.f, self.model, self.args = fused, fused.model, fused.args
        self.head = _head_fn(fused.model, head)
        a, eps = self.args, fused.eps
        hq, hk, d = a.num_attention_heads, a.num_key_value_heads, a.head_dim

        if fuse_prologue is True:
            # (ablation) RMSNorm + INT8 of each GEMV input and the per-token INT8 of o / act run in the GEMV prologue:
            # one dispatch less per quantization, but every threadgroup recomputes the row statistics; 0.61x of MLX
            # MXFP4 vs 0.98x with separate kernels on an idle GPU (results/07_decode_qwen3.csv)
            def pre(h, c, b, s, inn, qn, kn):
                y = G.gemv(h, c, b, None, s, (hq + 2 * hk) * d, 1, pro=2, nw=inn, eps=eps)
                return S.qkv_headnorm_y(y[None], qn, kn, eps, 1, hq, hk, d)

            def post(o, h, co, bo, so, pn, cg, bg, sg, cd, bd, sd):
                h = G.gemv(o, co, bo, None, so, so.size, 2, res=h, pro=1)[None]
                act = G.gemv(h, cg, bg, None, sg, sg.size, 3, pro=2, nw=pn, eps=eps)
                return G.gemv(act, cd, bd, None, sd, sd.size, 2, res=h, pro=1)[None]
        else:
            # separate rmsnorm_quant kernels; fuse_prologue="act": the per-token INT8 of o and act inside the o / down
            # GEMV prologues (256 threadgroups each), otherwise separate act_quant kernels (default)
            fa = fuse_prologue == "act"

            def pre(h, c, b, s, inn, qn, kn):
                xq, xs = S.rmsnorm_quant(h, inn, eps, 1)
                y = G.gemv(xq, c, b, xs, s, (hq + 2 * hk) * d, 1)
                return S.qkv_headnorm_y(y[None], qn, kn, eps, 1, hq, hk, d)

            def post(o, h, co, bo, so, pn, cg, bg, sg, cd, bd, sd):
                if fa:
                    h = G.gemv(o, co, bo, None, so, so.size, 2, res=h, pro=1)[None]
                else:
                    xq, xs = S.act_quant(o, 1)
                    h = G.gemv(xq, co, bo, xs, so, so.size, 2, res=h)[None]
                xq, xs = S.rmsnorm_quant(h, pn, eps, 1)
                act = G.gemv(xq, cg, bg, xs, sg, sg.size, 3)
                if fa:
                    return G.gemv(act, cd, bd, None, sd, sd.size, 2, res=h, pro=1)[None]
                xq, xs = S.act_quant(act[None], 1)
                return G.gemv(xq, cd, bd, xs, sd, sd.size, 2, res=h)[None]

        self._pre, self._post = (mx.compile(pre), mx.compile(post)) if compiled else (pre, post)

    def step(self, tok, cache):
        """tok int32 [1, 1] -> logits [1, 1, V]."""
        S.set_format(self.f.fmt)
        f = self.f
        h = self.model.model.embed_tokens(tok)[0]  # [1, D]
        for i, (layer, ws) in enumerate(f.layers):
            at = layer.self_attn
            w = ws["qkv"]
            q, k, v = self._pre(h, w.codes, w.bcodes, w.wscale, layer.input_layernorm.weight, at.q_norm.weight,
                                at.k_norm.weight)
            o = f._attention(at, cache[i], q, k, v, 1, None)
            wo, wg, wd = ws["o"], ws["gu"], ws["down"]
            h = self._post(o, h, wo.codes, wo.bcodes, wo.wscale, layer.post_attention_layernorm.weight,
                           wg.codes, wg.bcodes, wg.wscale, wd.codes, wd.bcodes, wd.wscale)
        return self.head(self.model.model.norm(h[None]))


class DecodeOlmoe:
    def __init__(self, fused, head="model", compiled=True, fuse_prologue=False):
        import fused_olmoe as fo
        self.f, self.model, self.args = fused, fused.model, fused.args
        self.head = _head_fn(fused.model, head)
        a, eps = self.args, fused.eps
        H, d = a.num_attention_heads, a.hidden_size // a.num_attention_heads
        F2, D, k = 2 * a.intermediate_size, a.hidden_size, a.num_experts_per_tok
        pos = mx.arange(k, dtype=mx.int32)

        fp = fuse_prologue

        def pre(h, c, b, s, inn, qn, kn):
            if fp:
                y = G.gemv(h, c, b, None, s, 3 * H * d, 1, pro=2, nw=inn, eps=eps)
            else:
                xq, xs = S.rmsnorm_quant(h, inn, eps, 1)
                y = G.gemv(xq, c, b, xs, s, 3 * H * d, 1)
            return S.qkv_fullnorm_y(y[None], qn, kn, eps, 1, H, d)

        def post(o, h, co, bo, so, pn, gate_w, cg, bg, sg, cd, bd, sd):
            if fp:
                h = G.gemv(o, co, bo, None, so, D, 2, res=h, pro=1)[None]
            else:
                xq, xs = S.act_quant(o, 1)
                h = G.gemv(xq, co, bo, xs, so, D, 2, res=h)[None]
            xq, xs, xn = S.rmsnorm_quant_n(h, pn, eps, 1)  # the router needs the BF16 normalized row too
            rw = mx.softmax(xn @ gate_w.T, axis=1, precise=True)  # mlx_lm OlmoeSparseMoeBlock routing
            inds = mx.argpartition(-rw, kth=k - 1, axis=-1)[..., :k]
            scores = mx.take_along_axis(rw, inds, axis=-1)
            e = inds.reshape(-1).astype(mx.int32)
            act = G.gemv(xq, cg, bg, xs, sg, F2, 3, eidx=e, xstride=0)  # [k, F]
            if fp:
                yd = G.gemv(act, cd, bd, None, sd, D, 1, eidx=e, xstride=1, pro=1)  # [k, D], per-row INT8 in prologue
            else:
                xq2, xs2 = S.act_quant(act, k)
                yd = G.gemv(xq2, cd, bd, xs2, sd, D, 1, eidx=e, xstride=1)
            return fo.moe_combine(yd, pos, scores, h, pn, eps, 1, quant=False)[0]

        assert not a.norm_topk_prob
        self._pre, self._post = (mx.compile(pre), mx.compile(post)) if compiled else (pre, post)

    def step(self, tok, cache):
        f = self.f
        h = self.model.model.embed_tokens(tok)[0]
        for i, (layer, ws, ex) in enumerate(f.layers):
            at, c = layer.self_attn, cache[i]
            w = ws["qkv"]
            q, k, v = self._pre(h, w.codes, w.bcodes, w.wscale, layer.input_layernorm.weight, at.q_norm.weight,
                                at.k_norm.weight)
            q, k, v = q[None], k[None], v[None]
            q = at.rope(q, offset=c.offset)
            k = at.rope(k, offset=c.offset)
            k, v = c.update_and_fetch(k, v)
            o = mx.fast.scaled_dot_product_attention(q, k, v, scale=at.scale, mask=None)
            o = o.transpose(0, 2, 1, 3).reshape(1, -1)
            wo = ws["o"]
            h = self._post(o, h, wo.codes, wo.bcodes, wo.wscale, layer.post_attention_layernorm.weight,
                           layer.mlp.gate.weight, *ex["gu"], *ex["down"])
        return self.head(self.model.model.norm(h[None]))


class DecodeQwen35:
    """Qwen3.5-MoE batch-1 decode from a FusedQwen35: GEMVs for the DeltaNet / attention projections, the expert
    GEMVs gather the 8 routed experts and the shared expert (9 slots) themselves; the MoE half of a step is
    mx.compile'd; the DeltaNet recurrence, attention and caches are mlx_lm's code (fused_qwen35.*_core)."""

    def __init__(self, fused, head="model", compiled=True, lean=True, out_prologue=False):
        """lean: the routed experts' indices go to the GEMVs as argpartition returns them, the shared expert is
        the GEMVs' implicit 9th slot and its gate a separate combine input (no casts / concatenations), and the
        down GEMV quantizes its input rows itself (PRO 1); outputs are identical to lean=False. out_prologue: the
        attention output projection also quantizes its input itself."""
        import fused_olmoe as fo
        import fused_qwen35 as fz
        self.f, self.model, self.args = fused, fused.model, fused.args
        self.out_prologue = out_prologue
        self.head = head.language_model.lm_head if head != "model" else fused.lm.lm_head
        self.fz = fz
        a, eps = self.args, fused.eps
        k, F2, D = a.num_experts_per_tok, 2 * fused.Fi, a.hidden_size
        pos = mx.arange(k + 1, dtype=mx.int32)

        def moe(h, pn, gate_w, sg_w, cg, bg, sg, cd, bd, sd):
            xq, xs, xn = S.rmsnorm_quant_n(h, pn, eps, 1)
            inds9, scores9 = fz.route(fz._Router(gate_w, sg_w), xn, k)
            e = inds9.reshape(-1)
            act = G.gemv(xq, cg, bg, xs, sg, F2, 3, eidx=e, xstride=0)  # [9, F]
            xq2, xs2 = S.act_quant(act, k + 1)
            yd = G.gemv(xq2, cd, bd, xs2, sd, D, 1, eidx=e, xstride=1)  # [9, D]
            return fo.moe_combine(yd, pos, scores9, h, pn, eps, 1, quant=False)[0]

        E = a.num_experts

        def moe_lean(h, pn, gate_w, sg_w, cg, bg, sg, cd, bd, sd):
            xq, xs, xn = S.rmsnorm_quant_n(h, pn, eps, 1)
            gates = mx.softmax(xn @ gate_w.T, axis=-1, precise=True)  # = fz.route, without the 9-slot arrays
            inds = mx.stop_gradient(mx.argpartition(gates, kth=-k, axis=-1)[..., -k:])
            scores = mx.take_along_axis(gates, inds, axis=-1)
            scores = scores / scores.sum(axis=-1, keepdims=True)
            sig = mx.sigmoid(xn @ sg_w.T)
            e = inds.reshape(-1)
            act = G.gemv(xq, cg, bg, xs, sg, F2, 3, eidx=e, xstride=0, shared=E)  # [9, F]
            yd = G.gemv(act, cd, bd, None, sd, D, 1, eidx=e, xstride=1, shared=E, pro=1)  # [9, D]
            return fo.moe_combine_split(yd, pos, scores, sig.reshape(-1).astype(scores.dtype), h, pn, eps, 1)[0]

        fn = moe_lean if lean else moe
        self._moe = mx.compile(fn) if compiled else fn

    def step(self, tok, cache):
        f, fz = self.f, self.fz
        h = self.model.language_model.model.embed_tokens(tok)[0]  # [1, D]
        for i, (layer, kind, ws, ex) in enumerate(f.layers):
            c = cache[i]
            if kind == "linear":
                m = layer.linear_attn
                xq, xs, xn = S.rmsnorm_quant_n(h, layer.input_layernorm.weight, f.eps, 1)
                w = ws["qkvz"]
                y = G.gemv(xq, w.codes, w.bcodes, xs, w.wscale, w.N, 1)
                nqkv = 2 * m.key_dim + m.value_dim
                bb = (xn @ m.in_proj_b.weight.T)[None]
                aa = (xn @ m.in_proj_a.weight.T)[None]
                o = fz.deltanet_core(m, y[:nqkv][None, None], y[nqkv:][None, None], bb, aa, None, c)[0]
                wo = ws["out"]
            else:
                m = layer.self_attn
                xq, xs = S.rmsnorm_quant(h, layer.input_layernorm.weight, f.eps, 1)
                w = ws["qkv"]
                y = G.gemv(xq, w.codes, w.bcodes, xs, w.wscale, w.N, 1)
                o = fz.attention_core(m, y[None, None], None, c)[0]
                wo = ws["o"]
            if self.out_prologue:
                h = G.gemv(o, wo.codes, wo.bcodes, None, wo.wscale, wo.N, 2, res=h, pro=1)[None]
            else:
                xq, xs = S.act_quant(o, 1)
                h = G.gemv(xq, wo.codes, wo.bcodes, xs, wo.wscale, wo.N, 2, res=h)[None]
            h = self._moe(h, layer.post_attention_layernorm.weight, layer.mlp.gate.weight,
                          layer.mlp.shared_expert_gate.weight, *ex["gu"], *ex["down"])
        return self.head(self.model.language_model.model.norm(h[None]))
