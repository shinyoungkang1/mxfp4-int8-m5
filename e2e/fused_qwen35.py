"""Fused SC-MXFP4 W4A8 prefill for Qwen3.5-MoE (Qwen3.5-35B-A3B; same architecture as Qwen3.6-35B-A3B): hybrid
Gated-DeltaNet / gated full attention, 256 routed experts (top-8) + 1 shared expert.

W4A8 (packed 4.08-bit SC-MXFP4, INT8 matmul2d, exact integer accumulation) for every large block linear:
  linear-attention layers:  in_proj_qkv | in_proj_z (one GEMM, row-concatenated), out_proj (+ residual epilogue)
  full-attention layers:    q_proj (query | output gate) | k_proj | v_proj (one GEMM), o_proj (+ residual)
  MoE:                      routed experts gate|up (SwiGLU epilogue) and down, the shared expert as a 257th expert
                            that every token visits (same shapes), one grouped GEMM dispatch each
BF16 (as mlx_lm and the fake-quant reference of e2e/ref_qwen35.py): the router, shared_expert_gate, the 32-wide
in_proj_b / in_proj_a, embeddings and lm_head. Everything between the GEMMs (conv1d, L2 norms, the gated delta rule,
RMSNormGated, q/k norm, RoPE, SDPA, the attention output gate) is mlx_lm's own code, replicated op for op.

Combining the routed experts and the shared expert: mlx_lm computes ((y * scores).sum(-2)) + sigmoid(g) * y_shared in
BF16, i.e. a sequential BF16 sum over the 8 experts and then one more BF16 add; with the shared expert as slot 9 and
sigmoid(g) as its score, fused_olmoe.moe_combine's sequential sum computes exactly that.
"""
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import mlx.core as mx
import mlx.nn as nn
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import fused_olmoe as fo  # noqa: E402
import fused_qwen3 as fq  # noqa: E402
import sc_kernels as S  # noqa: E402
import sc_moe  # noqa: E402

REPO = "Qwen/Qwen3.5-35B-A3B-Base"
DENSE = {"linear": {"qkvz": ["linear_attn.in_proj_qkv", "linear_attn.in_proj_z"], "out": ["linear_attn.out_proj"]},
         "full": {"qkv": ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"], "o": ["self_attn.o_proj"]}}
EXPERTS = ["gate_proj", "up_proj", "down_proj"]


def get(mod, path):
    for p in path.split("."):
        mod = getattr(mod, p)
    return mod


def layer_kind(layer):
    return "linear" if layer.is_linear else "full"


def _quantize_one(w):
    import sc_format as F
    return F.quantize(np.asarray(w, np.float64))[:3]


def pack_qwen35(model, cache_path, workers=14):
    """SC-MXFP4 (sim/ reference quantizer) of every W4A8 matrix, 2-bit block codes; experts stacked per layer with the
    shared expert appended as expert E. Cached (about 18 GB)."""
    if os.path.exists(cache_path):
        return mx.load(cache_path)
    layers = model.language_model.model.layers
    out = {}
    with ProcessPoolExecutor(workers) as ex:
        for i, layer in enumerate(layers):
            jobs, keys = [], []
            for paths in DENSE[layer_kind(layer)].values():
                for p in paths:
                    jobs.append(np.array(get(layer, p).weight.astype(mx.float32)))
                    keys.append(f"layers.{i}.{p}")
            res = list(ex.map(_quantize_one, jobs))
            for key, (c, b, s) in zip(keys, res):
                out[key + ".codes"], out[key + ".bcodes"], out[key + ".wscale"] = c, b, s
            for name in EXPERTS:
                w = np.array(getattr(layer.mlp.switch_mlp, name).weight.astype(mx.float32))
                ws = np.array(getattr(layer.mlp.shared_expert, name).weight.astype(mx.float32))
                mats = list(w) + [ws]
                res = list(ex.map(_quantize_one, mats, chunksize=8))
                key = f"layers.{i}.experts.{name}"
                out[key + ".codes"] = np.stack([r[0] for r in res])
                out[key + ".bcodes"] = np.stack([r[1] for r in res])
                out[key + ".wscale"] = np.stack([r[2] for r in res])
                del w, mats, res
            print(f"packed layer {i + 1}/{len(layers)}", flush=True)
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    mx.save_safetensors(cache_path, {k: mx.array(v) for k, v in out.items()})
    del out
    return mx.load(cache_path)


# ---- mlx_lm's attention / DeltaNet code after the projections (op for op, see qwen3_5.py / qwen3_next.py) ----------
def deltanet_core(m, qkv, z, b, a, mask, cache):
    from mlx_lm.models.gated_delta import gated_delta_update, normalize_qk
    B, S_, _ = qkv.shape
    z = z.reshape(B, S_, m.num_v_heads, m.head_v_dim)
    if cache is not None and cache[0] is not None:
        conv_state = cache[0]
    else:
        conv_state = mx.zeros((B, m.conv_kernel_size - 1, m.conv_dim), dtype=qkv.dtype)
    if mask is not None:
        qkv = mx.where(mask[..., None], qkv, 0)
    conv_input = mx.concatenate([conv_state, qkv], axis=1)
    if cache is not None:
        n_keep = m.conv_kernel_size - 1
        if cache.lengths is not None:
            ends = mx.clip(cache.lengths, 0, S_)
            positions = (ends[:, None] + mx.arange(n_keep))[..., None]
            cache[0] = mx.take_along_axis(conv_input, positions, axis=1)
        else:
            cache[0] = mx.contiguous(conv_input[:, -n_keep:, :])
    conv_out = nn.silu(m.conv1d(conv_input))
    q, k, v = [t.reshape(B, S_, h, d) for t, h, d in zip(
        mx.split(conv_out, [m.key_dim, 2 * m.key_dim], -1), [m.num_k_heads, m.num_k_heads, m.num_v_heads],
        [m.head_k_dim, m.head_k_dim, m.head_v_dim])]
    state = cache[1] if cache else None
    q, k = normalize_qk(q, k, inv_scale=m.head_k_dim ** -0.5, eps=1e-6)
    out, state = gated_delta_update(q, k, v, a, b, m.A_log, m.dt_bias, state, mask, use_kernel=True)
    if cache is not None:
        cache[1] = state
        cache.advance(S_)
    out = m.norm(out, z)
    return out.reshape(B, S_, -1)


def attention_core(m, y, mask, cache):
    """y: [1, L, Nq + 2 Nkv] = q_proj | k_proj | v_proj outputs -> gated attention output [1, L, H * D]."""
    from mlx_lm.models.base import scaled_dot_product_attention
    B, L, _ = y.shape
    H, Hk, D = m.num_attention_heads, m.num_key_value_heads, m.head_dim
    nq, nk = 2 * H * D, Hk * D
    queries, gate = mx.split(y[..., :nq].reshape(B, L, H, -1), 2, axis=-1)
    gate = gate.reshape(B, L, -1)
    keys, values = y[..., nq:nq + nk], y[..., nq + nk:]
    queries = m.q_norm(queries).transpose(0, 2, 1, 3)
    keys = m.k_norm(keys.reshape(B, L, Hk, -1)).transpose(0, 2, 1, 3)
    values = values.reshape(B, L, Hk, -1).transpose(0, 2, 1, 3)
    if cache is not None:
        queries = m.rope(queries, offset=cache.offset)
        keys = m.rope(keys, offset=cache.offset)
        keys, values = cache.update_and_fetch(keys, values)
    else:
        queries, keys = m.rope(queries), m.rope(keys)
    output = scaled_dot_product_attention(queries, keys, values, cache=cache, scale=m.scale, mask=mask)
    output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
    return output * mx.sigmoid(gate)


def route(mlp, xn, k):
    """mlx_lm Qwen3NextSparseMoeBlock routing (BF16 router) + the shared expert's sigmoid gate as a 9th slot."""
    gates = mx.softmax(xn @ mlp.gate.weight.T, axis=-1, precise=True)
    inds = mx.stop_gradient(mx.argpartition(gates, kth=-k, axis=-1)[..., -k:])
    scores = mx.take_along_axis(gates, inds, axis=-1)
    scores = scores / scores.sum(axis=-1, keepdims=True)
    sig = mx.sigmoid(xn @ mlp.shared_expert_gate.weight.T)
    E = mlp.gate.weight.shape[0]
    inds9 = mx.concatenate([inds.astype(mx.int32), mx.full((inds.shape[0], 1), E, mx.int32)], axis=-1)
    return inds9, mx.concatenate([scores, sig.astype(scores.dtype)], axis=-1)


class FusedQwen35:
    def __init__(self, model, flat, tile_gu=(64, 64, 4), tile_d=(64, 64, 4), bk=64, two_pass="auto",
                 compiled=True):
        self.model, self.lm = model, model.language_model
        a = self.args = self.lm.args
        self.eps = mx.array([a.rms_norm_eps], mx.float32)
        self.tile_gu, self.tile_d, self.bk, self.two_pass = tile_gu, tile_d, bk, two_pass
        self.compiled, self._fns = compiled, {}
        self.E, self.Fi = a.num_experts, a.moe_intermediate_size
        perm = S.interleave_gate_up(self.Fi)
        npy = lambda k: np.array(flat[k])  # noqa: E731
        self.layers = []
        for i, layer in enumerate(self.lm.model.layers):
            kind = layer_kind(layer)
            parts = lambda p: (npy(f"layers.{i}.{p}.codes"), npy(f"layers.{i}.{p}.bcodes"), npy(f"layers.{i}.{p}.wscale"))  # noqa: E731,E501
            ws = {g: fq.PackedWeight([parts(p) for p in paths]) for g, paths in DENSE[kind].items()}
            ek = lambda n, s: npy(f"layers.{i}.experts.{n}.{s}")  # noqa: E731
            E1 = self.E + 1
            g_c = ek("gate_proj", "codes").reshape(E1, self.Fi, -1)
            u_c = ek("up_proj", "codes").reshape(E1, self.Fi, -1)
            gu_c = np.concatenate([g_c, u_c], 1)[:, perm].reshape(E1, -1)
            gu_b = np.concatenate([ek("gate_proj", "bcodes"), ek("up_proj", "bcodes")], 1)[:, perm]
            gu_s = np.concatenate([ek("gate_proj", "wscale"), ek("up_proj", "wscale")], 1)[:, perm]
            ex = dict(gu=(mx.array(gu_c), mx.array(gu_b), mx.array(gu_s)),
                      down=tuple(mx.array(ek("down_proj", s)) for s in ("codes", "bcodes", "wscale")))
            self.layers.append((layer, kind, ws, ex))
        self.perm_gu = perm
        for layer in self.lm.model.layers:  # the BF16 originals of the packed matrices are never loaded
            if layer.is_linear:
                for n in ("in_proj_qkv", "in_proj_z", "out_proj"):
                    setattr(layer.linear_attn, n, None)
            else:
                for n in ("q_proj", "k_proj", "v_proj", "o_proj"):
                    setattr(layer.self_attn, n, None)
            layer.mlp.switch_mlp = None
            layer.mlp.shared_expert = None
        mx.eval([w for _, _, ws, ex in self.layers for w in [*(p.codes for p in ws.values()), *ex["gu"], *ex["down"]]])
        self.record, self.record_layers = None, ()

    def weight_bytes(self):
        n = 0
        for _, _, ws, ex in self.layers:
            n += sum(p.nbytes() for p in ws.values()) + sum(t.nbytes for t in ex["gu"] + ex["down"])
        return n

    def _moe_fn(self, L, Mp, dtype, compiled, rec):
        """MoE half of a block (pure function of arrays, compiled per prompt length): post-attention RMSNorm + INT8,
        routing, counting sort, grouped expert GEMMs (incl. the shared expert), combine + residual."""
        key = (L, Mp, dtype, compiled, rec, self.two_pass, self.tile_gu, self.tile_d, self.bk)
        if key in self._fns:
            return self._fns[key]
        a, eps = self.args, self.eps
        k, E1, F2, D = a.num_experts_per_tok, self.E + 1, 2 * self.Fi, a.hidden_size
        tm, R = max(self.tile_gu[0], self.tile_d[0]), L * (k + 1)
        two_pass = self.two_pass if self.two_pass != "auto" else R // E1 >= 256

        def moe(h, pn, gate_w, sg_w, cg, bg, sg, cd, bd, sd):
            xq, xs, xn = S.rmsnorm_quant_n(h, pn, eps)
            mlp = _Router(gate_w, sg_w)
            inds9, scores9 = route(mlp, xn[:L], k)
            pos, tok_p, e_sorted, off, tstart, maxt = sc_moe.route_sort(inds9, E1, tm)
            xq_s, xs_s = xq[tok_p], xs[tok_p[:R]]
            if two_pass:
                act = sc_moe.moe_mm_i8(xq_s, R, sc_moe.unpack_experts(cg, bg, F2, D), sg, off, tstart, maxt, F2, 3,
                                       xs_s, tile=self.tile_gu)
            else:
                act = sc_moe.moe_mm(xq_s, R, cg, bg, sg, off, tstart, maxt, F2, 3, xs_s, tile=self.tile_gu, bk=self.bk)
            xq2, xs2 = S.act_quant(act, R + tm)
            if two_pass:
                yd = sc_moe.moe_mm_i8(xq2, R, sc_moe.unpack_experts(cd, bd, D, F2 // 2), sd, off, tstart, maxt, D, 1,
                                      xs2, tile=self.tile_d)
            else:
                yd = sc_moe.moe_mm(xq2, R, cd, bd, sd, off, tstart, maxt, D, 1, xs2, tile=self.tile_d, bk=self.bk)
            h = fo.moe_combine(yd, pos, scores9, h, pn, eps, L, quant=False)[0]
            if rec:
                return h, (xq, xq_s, xs_s, act, xq2, xs2, yd, e_sorted, xn)
            return h

        fn = mx.compile(moe) if compiled else moe
        self._fns[key] = fn
        return fn

    def hidden(self, ids, cache):
        from mlx_lm.models.base import create_attention_mask, create_ssm_mask
        a = self.args
        h = self.lm.model.embed_tokens(ids)[0]
        L = h.shape[0]
        Mp = S.pad_rows(L)
        fa_mask = create_attention_mask(h[None], cache[self.lm.model.fa_idx])
        ssm_mask = create_ssm_mask(h[None], cache[self.lm.model.ssm_idx])
        h = mx.pad(h, [(0, Mp - L), (0, 0)]) if Mp != L else h
        rec = self.record is not None
        moe = self._moe_fn(L, Mp, h.dtype, self.compiled and not rec, rec)
        tq = (64, 128, 4, 8)
        for i, (layer, kind, ws, ex) in enumerate(self.layers):
            c = cache[i]
            if kind == "linear":
                m = layer.linear_attn
                xq, xs, xn = S.rmsnorm_quant_n(h, layer.input_layernorm.weight, self.eps)
                wq = ws["qkvz"].unpack()
                y = S.mm_epi(xq, wq, xs, ws["qkvz"].wscale, 1, h.dtype, tile=tq)
                self._rec(i, "qkvz", h, xq, wq)
                nqkv = 2 * m.key_dim + m.value_dim
                bb = (xn[:L] @ m.in_proj_b.weight.T)[None]
                aa = (xn[:L] @ m.in_proj_a.weight.T)[None]
                o = deltanet_core(m, y[:L, :nqkv][None], y[:L, nqkv:][None], bb, aa, ssm_mask, c)[0]
                wo = ws["out"]
            else:
                m = layer.self_attn
                xq, xs = S.rmsnorm_quant(h, layer.input_layernorm.weight, self.eps)
                wq = ws["qkv"].unpack()
                y = S.mm_epi(xq, wq, xs, ws["qkv"].wscale, 1, h.dtype, tile=tq)
                self._rec(i, "qkv", h, xq, wq)
                o = attention_core(m, y[:L][None], fa_mask, c)[0]
                wo = ws["o"]
            xq, xs = S.act_quant(o, Mp)
            w_o = wo.unpack()
            self._rec(i, "o", o, xq, w_o)
            h = S.mm_epi(xq, w_o, xs, wo.wscale, 2, h.dtype, res=h, tile=tq)
            out = moe(h, layer.post_attention_layernorm.weight, layer.mlp.gate.weight,
                      layer.mlp.shared_expert_gate.weight, *ex["gu"], *ex["down"])
            if rec:
                h = out[0]
                if i in self.record_layers:
                    self.record[(i, "experts")] = out[1]
            else:
                h = out
        return self.lm.model.norm(h[:L][None])

    def _rec(self, i, g, *v):
        if self.record is not None and i in self.record_layers:
            self.record[(i, g)] = v

    def logits(self, h):
        return self.lm.lm_head(h)

    def make_cache(self):
        return self.lm.make_cache()

    def __call__(self, ids, cache=None):
        return self.logits(self.hidden(ids, cache if cache is not None else self.make_cache()))


class _Router:
    """Stand-in for the router modules inside the compiled function (route() reads .gate.weight and
    .shared_expert_gate.weight)."""

    def __init__(self, gate_w, sg_w):
        self.gate = _W(gate_w)
        self.shared_expert_gate = _W(sg_w)


class _W:
    def __init__(self, w):
        self.weight = w


def load_fused_qwen35(repo=REPO, cache_dir=os.path.expanduser("~/.cache/mxfp4-int8-m5"), **kw):
    from mlx_lm import load
    model, tok = load(repo, lazy=True)
    flat = pack_qwen35(model, os.path.join(cache_dir, repo.replace("/", "--") + "-scmxfp4-p2.safetensors"))
    return FusedQwen35(model, flat, **kw), tok
