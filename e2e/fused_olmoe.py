"""Fused SC-MXFP4 W4A8 prefill for OLMoE (mixture of experts: 64 experts, top-8), built on e2e/sc_kernels.py and
the grouped expert GEMM of e2e/sc_moe.py.

Per transformer block:
  y = mm_epi(xq, Wq|Wk|Wv)  ->  q, k, v = qkv_fullnorm_y(y)   (OLMoE's q/k RMSNorm over the full width)
  rope, KV cache, sdpa (MLX)
  h = mm_epi(act_quant(o), Wo, + h)                              (residual add in the GEMM epilogue)
  xq, xs, xn = rmsnorm_quant_n(h)                                (INT8 for the experts, BF16 for the router)
  router (BF16 linear, softmax, top-8: MLX ops, as in mlx_lm and the fake-quant reference)
  assignments sorted by expert (one counting-sort kernel, sc_moe.route_sort); xq rows gathered per assignment
  act = moe_mm(xq_sorted, Wg|Wu, EPI swiglu)                    grouped INT8 GEMM, weights unpacked in-kernel
  y = moe_mm(act_quant(act), Wd, EPI scale)
  h, xq, xs = moe_combine(y, scores, + h) + next block's RMSNorm + INT8   (one row kernel)
Expert weights stay packed at 4.07 bits/weight; the grouped kernel unpacks each expert's weight tile into
threadgroup memory (no INT8 scratch). The router stays BF16 (as in sim/eval_ppl.py's sc_mxfp4_w4a8).
"""
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import mlx.core as mx
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import sc_format as F  # noqa: E402
import sc_kernels as S  # noqa: E402
import sc_moe  # noqa: E402

ATTN = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj"]
EXPERTS = ["gate_proj", "up_proj", "down_proj"]
MODEL_CONFIG = {"rms_norm_eps": 1e-5}  # missing from OLMoE-1B-7B-0924's config.json

_RBF = """
inline float sc_rbf(float x) {  // round to the nearest BF16 (ties to even), as a float
    uint u = as_type<uint>(x);
    u += 0x7FFFu + ((u >> 16) & 1u);
    return as_type<float>(u & 0xFFFF0000u);
}
"""

# Weighted sum of the top-k expert outputs as mlx_lm computes it, (y * scores[..., None]).sum(-2) in BF16 (MLX's
# column reduction accumulates sequentially in the output type), plus the residual, then optionally the next
# block's RMSNorm + INT8 (rmsnorm_quant body).
_COMBINE = """
    const uint row = threadgroup_position_in_grid.x, lid = thread_position_in_threadgroup.x;
    const uint lane = thread_index_in_simdgroup, sg = simdgroup_index_in_threadgroup;
    threadgroup float red[32];
    threadgroup float shared_inv;
    const size_t o = size_t(row) * K + 4 * lid;
    T xv[4];
    if (row < M) {
        // explicit BF16 rounding of every partial sum: the JIT compiler may otherwise keep them in FP32
        float acc[4] = {0, 0, 0, 0};
        for (int j = 0; j < TOPK; ++j) {
            const uint p = pos[row * TOPK + j];
            const float sc = float(scores[row * TOPK + j]);
            for (int i = 0; i < 4; ++i) acc[i] = sc_rbf(sc_rbf(float(y[size_t(p) * K + 4 * lid + i]) * sc) + acc[i]);
        }
        for (int i = 0; i < 4; ++i) xv[i] = static_cast<T>(float(res[o + i]) + acc[i]);
    } else {
        for (int i = 0; i < 4; ++i) xv[i] = res[o + i];
    }
    for (int i = 0; i < 4; ++i) h[o + i] = xv[i];
    if (!QUANT) return;
""" + S._RMS_QUANT_BODY


_COMBINE_SPLIT = _COMBINE.replace(
    "const float sc = float(scores[row * TOPK + j]);",
    "const float sc = j == TOPK - 1 ? float(sscore[row]) : float(scores[row * (TOPK - 1) + j]);")


def moe_combine_split(y, pos, scores, sscore, res, w, eps, M):
    """moe_combine (quant=False) whose last slot's score is given separately: scores BF16 [M, topk - 1], sscore
    BF16 [M] (the shared expert's gate), so that decode needs no concatenation of the two."""
    Mp, K = res.shape
    topk = scores.shape[1] + 1
    k = S._kernel("moe_combine_split", ["y", "pos", "scores", "sscore", "res", "nw", "eps"], ["h", "xq", "xs"],
                  _COMBINE_SPLIT, header_extra=_RBF)
    return k(inputs=[y, pos, scores, sscore, res, w, eps],
             template=[("T", res.dtype), ("K", K), ("M", M), ("TOPK", topk), ("QUANT", False)],
             grid=(Mp * K // 4, 1, 1), threadgroup=(K // 4, 1, 1), output_shapes=[(Mp, K), (Mp, K), (Mp,)],
             output_dtypes=[res.dtype, mx.int8, mx.float32])


def moe_combine(y, pos, scores, res, w, eps, M, quant=True):
    """y BF16 [R, K] (sorted assignments), pos int32 [M*topk] (sorted position of assignment (t, j)), scores BF16
    [M, topk], res [Mp, K] -> (h [Mp, K], xq int8 [Mp, K], xscale [Mp]) with xq = INT8(rms_norm(h, w))."""
    Mp, K = res.shape
    topk = scores.shape[1]
    k = S._kernel("moe_combine", ["y", "pos", "scores", "res", "nw", "eps"], ["h", "xq", "xs"], _COMBINE,
                  header_extra=_RBF)
    return k(inputs=[y, pos, scores, res, w, eps], template=[("T", res.dtype), ("K", K), ("M", M), ("TOPK", topk),
                                                              ("QUANT", quant)],
             grid=(Mp * K // 4, 1, 1), threadgroup=(K // 4, 1, 1), output_shapes=[(Mp, K), (Mp, K), (Mp,)],
             output_dtypes=[res.dtype, mx.int8, mx.float32])


def _quantize_one(w):
    return F.quantize(np.asarray(w, np.float64))[:3]


def pack_olmoe(model, cache_path, workers=14):
    """SC-MXFP4 (sim/ reference quantizer) of every block linear and expert matrix, 2-bit block codes, cached."""
    if os.path.exists(cache_path):
        return mx.load(cache_path)
    flat = {}
    with ProcessPoolExecutor(workers) as ex:
        for i, layer in enumerate(model.model.layers):
            jobs, keys = [], []
            for path in ATTN:
                lin = layer
                for p in path.split("."):
                    lin = getattr(lin, p)
                jobs.append(np.array(lin.weight.astype(mx.float32)))
                keys.append(f"layers.{i}.{path}")
            sw = layer.mlp.switch_mlp
            for name in EXPERTS:
                wt = np.array(getattr(sw, name).weight.astype(mx.float32))
                for e in range(wt.shape[0]):
                    jobs.append(wt[e])
                    keys.append(f"layers.{i}.experts.{name}.{e}")
            for key, (codes, bc, ws) in zip(keys, ex.map(_quantize_one, jobs, chunksize=4)):
                flat[key + ".codes"], flat[key + ".bcodes"], flat[key + ".wscale"] = codes, bc, ws
            print(f"packed layer {i + 1}/{len(model.model.layers)}", flush=True)
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    mx.save_safetensors(cache_path, {k: mx.array(v) for k, v in flat.items()})
    return mx.load(cache_path)


class FusedOlmoe:
    def __init__(self, model, flat, tile_gu=(64, 64, 4), tile_d=(64, 64, 4), bk=64, two_pass="auto", sort="count",
                 compiled=True):
        """Tiles from the sweep in results/07_tune_moe.txt (BK = 64 beats 128 at every length: half the threadgroup
        memory per threadgroup). two_pass=True: unpack all experts to an INT8 scratch per call, then the grouped GEMM
        reads it; "auto": two-pass once the average expert has >= 256 rows (L >= 2048 tokens on OLMoE): in
        results/07_moe_prefill.csv the in-kernel unpack wins at 512 tokens (1.23x vs 0.83x MLX MXFP4), the two are
        even at 2048 and two-pass wins at 4096 (1.47x vs 1.36x)."""
        import fused_qwen3 as fq
        self.model, self.args = model, model.args
        a = self.args
        self.eps = mx.array([a.rms_norm_eps], mx.float32)
        self.tile_gu, self.tile_d, self.bk, self.two_pass, self.sort = tile_gu, tile_d, bk, two_pass, sort
        self.compiled, self._fns = compiled, {}
        E, Fi = a.num_experts, a.intermediate_size
        perm = S.interleave_gate_up(Fi)
        npy = lambda k: np.array(flat[k])  # noqa: E731
        self.layers = []
        for i, layer in enumerate(model.model.layers):
            parts = lambda p: (npy(f"layers.{i}.{p}.codes"), npy(f"layers.{i}.{p}.bcodes"), npy(f"layers.{i}.{p}.wscale"))  # noqa: E731
            qkv = fq.PackedWeight([parts(p) for p in ATTN[:3]])
            o = fq.PackedWeight([parts(ATTN[3])])
            gu_c, gu_b, gu_s, d_c, d_b, d_s = [], [], [], [], [], []
            for e in range(E):
                g, u, d = (parts(f"experts.{n}.{e}") for n in EXPERTS)
                c = np.concatenate([g[0].reshape(Fi, -1), u[0].reshape(Fi, -1)])[perm]
                gu_c.append(c.reshape(-1))
                gu_b.append(np.concatenate([g[1], u[1]])[perm])
                gu_s.append(np.concatenate([g[2], u[2]])[perm])
                d_c.append(d[0]), d_b.append(d[1]), d_s.append(d[2])
            st = lambda xs: mx.array(np.stack(xs))  # noqa: E731
            ex = dict(gu=(st(gu_c), st(gu_b), st(gu_s)), down=(st(d_c), st(d_b), st(d_s)))
            self.layers.append((layer, dict(qkv=qkv, o=o), ex))
        self.perm_gu = perm
        for layer in model.model.layers:  # drop the BF16 weights that the packed ones replace
            for path in ATTN:
                setattr(layer.self_attn, path.split(".")[1], None)
            layer.mlp.switch_mlp = None
        mx.eval([w for _, ws, ex in self.layers for w in [ws["qkv"].codes, ws["o"].codes, *ex["gu"], *ex["down"]]])
        self.record, self.record_layers = None, ()

    def weight_bytes(self):
        n = 0
        for _, ws, ex in self.layers:
            n += ws["qkv"].nbytes() + ws["o"].nbytes() + sum(a.nbytes for a in ex["gu"] + ex["down"])
        return n

    def _rec(self, i, g, *v):
        if self.record is not None and i in self.record_layers:
            self.record[(i, g)] = v

    def route(self, layer, xn, M):
        """mlx_lm OlmoeSparseMoeBlock routing on the BF16 normalized input (identical ops)."""
        k = self.args.num_experts_per_tok
        logits = layer.mlp.gate(xn[:M])
        rw = mx.softmax(logits, axis=1, precise=True)
        inds = mx.stop_gradient(mx.argpartition(-rw, kth=k - 1, axis=-1)[..., :k])
        scores = mx.take_along_axis(rw, inds, axis=-1)
        if self.args.norm_topk_prob:
            scores = scores / scores.sum(axis=-1, keepdims=True)
        return inds, scores

    def _block_fns(self, L, Mp, dtype, compiled, rec=False):
        """The halves of a block before / after attention as pure functions of arrays (mx.compile'd per prompt
        length unless recording): pre = q/k/v GEMM + q/k RMSNorm; post = o GEMM (+residual), RMSNorm + INT8,
        routing (mlx_lm's ops), sort, grouped expert GEMMs, combine (+residual, next RMSNorm + INT8)."""
        key = (L, Mp, dtype, compiled, rec, self.two_pass, self.sort, self.tile_gu, self.tile_d, self.bk)
        if key in self._fns:
            return self._fns[key]
        a, eps = self.args, self.eps
        H, d, D = a.num_attention_heads, a.hidden_size // a.num_attention_heads, a.hidden_size
        k, E, F2 = a.num_experts_per_tok, a.num_experts, 2 * a.intermediate_size
        tq, tm, R = (64, 128, 4, 8), max(self.tile_gu[0], self.tile_d[0]), L * k
        two_pass = self.two_pass if self.two_pass != "auto" else R // E >= 256

        def pre(xq, xs, c, b, s, qn, kn):
            y = S.mm_epi(xq, S.unpack_i8(c, b, s.size, D), xs, s, 1, dtype, tile=tq)
            return S.qkv_fullnorm_y(y, qn, kn, eps, L, H, d)

        def post(o, h, co, bo, so, pn, gate_w, cg, bg, sg, cd, bd, sd, nn):
            xq, xs = S.act_quant(o, Mp)
            h = S.mm_epi(xq, S.unpack_i8(co, bo, D, D), xs, so, 2, dtype, res=h, tile=tq)
            xq, xs, xn = S.rmsnorm_quant_n(h, pn, eps)
            # routing exactly as mlx_lm's OlmoeSparseMoeBlock (gate is a bias-free BF16 nn.Linear)
            rw = mx.softmax(xn[:L] @ gate_w.T, axis=1, precise=True)
            inds = mx.stop_gradient(mx.argpartition(-rw, kth=k - 1, axis=-1)[..., :k])
            scores = mx.take_along_axis(rw, inds, axis=-1)
            if a.norm_topk_prob:
                scores = scores / scores.sum(axis=-1, keepdims=True)
            if self.sort == "count":  # one counting-sort kernel (row order within an expert is irrelevant)
                pos, tok_p, e_sorted, off, tstart, maxt = sc_moe.route_sort(inds, E, tm)
            else:  # mlx_lm's _gather_sort
                flat_i = inds.flatten()
                order = mx.argsort(flat_i)
                pos = mx.argsort(order).astype(mx.int32)
                e_sorted = flat_i[order].astype(mx.int32)
                off, tstart, maxt = sc_moe.segments(e_sorted, E, tm)
                tok_p = mx.concatenate([(order // k).astype(mx.int32), mx.zeros((tm,), mx.int32)])
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
            h, xq_n, xs_n = moe_combine(yd, pos, scores, h, nn, eps, L)
            if rec:
                return h, xq_n, xs_n, (xq, xq_s, xs_s, act, xq2, xs2, yd, e_sorted, xn)
            return h, xq_n, xs_n

        fns = (mx.compile(pre), mx.compile(post)) if compiled else (pre, post)
        self._fns[key] = fns
        return fns

    def hidden(self, ids, cache):
        h = self.model.model.embed_tokens(ids)[0]
        L = h.shape[0]
        Mp = S.pad_rows(L)
        h = mx.pad(h, [(0, Mp - L), (0, 0)]) if Mp != L else h
        mask = "causal" if L > 1 else None
        rec = self.record is not None
        pre, post = self._block_fns(L, Mp, h.dtype, self.compiled and not rec, rec)
        nl = len(self.layers)
        xq, xs = S.rmsnorm_quant(h, self.layers[0][0].input_layernorm.weight, self.eps, Mp)
        for i, (layer, ws, ex) in enumerate(self.layers):
            at, c = layer.self_attn, cache[i]
            wq, wo = ws["qkv"], ws["o"]
            q, kk, v = pre(xq, xs, wq.codes, wq.bcodes, wq.wscale, at.q_norm.weight, at.k_norm.weight)
            q, kk, v = q[None], kk[None], v[None]
            q = at.rope(q, offset=c.offset)
            kk = at.rope(kk, offset=c.offset)
            kk, v = c.update_and_fetch(kk, v)
            o = mx.fast.scaled_dot_product_attention(q, kk, v, scale=at.scale, mask=mask)
            o = o.transpose(0, 2, 1, 3).reshape(L, -1)
            nn = self.layers[i + 1][0].input_layernorm.weight if i + 1 < nl else self.model.model.norm.weight
            out = post(o, h, wo.codes, wo.bcodes, wo.wscale, layer.post_attention_layernorm.weight,
                       layer.mlp.gate.weight, *ex["gu"], *ex["down"], nn)
            h, xq, xs = out[:3]
            if rec:
                self._rec(i, "experts", *out[3])
        return self.model.model.norm(h[:L][None])

    def make_cache(self):
        from mlx_lm.models.cache import make_prompt_cache
        return make_prompt_cache(self.model)

    def logits(self, h):
        m = self.model
        return m.model.embed_tokens.as_linear(h) if self.args.tie_word_embeddings else m.lm_head(h)

    def __call__(self, ids, cache=None):
        from mlx_lm.models.cache import make_prompt_cache
        cache = cache if cache is not None else make_prompt_cache(self.model)
        return self.logits(self.hidden(ids, cache))


def load_fused_olmoe(repo, cache_dir, **kw):
    from mlx_lm import load
    model, tok = load(repo, model_config=MODEL_CONFIG)
    flat = pack_olmoe(model, os.path.join(cache_dir, repo.replace("/", "--") + "-scmxfp4-p2.safetensors"))
    return FusedOlmoe(model, flat, **kw), tok
