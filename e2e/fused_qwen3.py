"""Fused SC-MXFP4 W4A8 prefill for Qwen3 dense models, built from the kernels of e2e/sc_kernels.py.

Per transformer block (13 custom dispatches instead of 28 plus MLX's norms, adds and swiglu in the unfused path):
  [xq, xs = rmsnorm_quant(h)]  (fused into the previous block's last epilogue)
  wqkv = unpack(Wq|Wk|Wv)                    one GEMM for q/k/v on row-concatenated weights
  acc  = mm(xq, wqkv);  q, k, v = qkv_epilogue_headnorm(acc)   (scales + q/k per-head RMSNorm, head-major out)
  q, k = rope(q), rope(k);  KV cache;  o = sdpa(q, k, v)        (MLX's own code, as in every method)
  xq, xs = act_quant(o);  acc = mm(xq, unpack(Wo))
  h, xq, xs = res_norm_quant(acc, h)         residual add + post-attention RMSNorm + INT8 quantization
  acc = mm(xq, unpack(Wg|Wu));  xq, xs = swiglu_quant(acc)
  acc = mm(xq, unpack(Wd));  h, xq, xs = res_norm_quant(acc, h)  (next block's input RMSNorm; last block: add only)
Weights are stored as 4-bit codes + 2-bit block codes + FP32 row scale (e2e/sc_format.py, 4.07 bits/weight) and
unpacked to INT8 on every call, like the unfused path; with overlap=True each unpack is ordered in MLX's graph to
be dispatched right after an independent GEMM, so the memory-bound unpack runs concurrently with the
accelerator-bound GEMM instead of between two barriers.

Numerically the fused model is the unfused W4A8Linear model of e2e/w4a8_mlx.py: every non-GEMM step replicates
MLX's rounding (checked bit for bit in e2e/test_sc_kernels.py) and the INT8 GEMMs are exact.
"""
import os
import sys

import mlx.core as mx
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import sc_format as F  # noqa: E402
import sc_kernels as S  # noqa: E402

GROUPS = {"qkv": ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"], "o": ["self_attn.o_proj"],
          "gu": ["mlp.gate_proj", "mlp.up_proj"], "down": ["mlp.down_proj"]}


class PackedWeight:
    """Row-concatenated packed SC-MXFP4 weights of one or more linears sharing an input."""

    def __init__(self, parts, resident=False, perm=None):
        # parts: list of (codes uint32 [N*K/8], bcodes uint8 [N, K/(BLOCK*per_byte)], wscale f32 [N]) numpy;
        # perm: optional permutation of the concatenated rows (gate/up interleaving for the swiglu epilogue)
        self.splits = [p[2].shape[0] for p in parts]
        self.N = sum(self.splits)
        self.K = parts[0][0].size * 8 // parts[0][2].shape[0]
        codes = np.concatenate([p[0].reshape(p[2].shape[0], -1) for p in parts])
        bcodes = np.concatenate([p[1] for p in parts])
        wscale = np.concatenate([p[2] for p in parts])
        self.perm = perm
        if perm is not None:
            codes, bcodes, wscale = codes[perm], bcodes[perm], wscale[perm]
        self.codes = mx.array(codes.reshape(-1))
        self.bcodes = mx.array(bcodes)
        self.wscale = mx.array(wscale)
        self.wq = S.unpack_i8(self.codes, self.bcodes, self.N, self.K) if resident else None

    def unpack(self, deps=()):
        return self.wq if self.wq is not None else S.unpack_i8(self.codes, self.bcodes, self.N, self.K, deps=deps)

    def nbytes(self):
        return self.codes.nbytes + self.bcodes.nbytes + self.wscale.nbytes


def load_packed(cache_path, n_layers, fmt=F.DEFAULT):
    """From the byte-shift cache of e2e/w4a8_mlx.pack_model (same quantizer) to 2-bit block codes, per linear."""
    flat = mx.load(cache_path)
    out = {}
    for i in range(n_layers):
        for g in GROUPS.values():
            for path in g:
                key = f"layers.{i}.{path}"
                out[key] = F.from_shift_layout(np.array(flat[key + ".codes"]), np.array(flat[key + ".shifts"]),
                                               np.array(flat[key + ".wscale"]), fmt)
    return out


def tiles_for(M):
    """GEMM tile (TM, TN, simdgroups, GROUP_M) per weight group: 64 x 128 was best or within 2% at every shape and
    M = 512 / 2048 / 8192 in the sweep of e2e/tune_07.py (results/07_tune_prefill.txt)."""
    t = (64, 128, 4, 8)
    return {"qkv": t, "o": t, "gu": t, "down": t}


class FusedQwen3:
    """epilogue="gemm": scales, residual adds and swiglu applied in the GEMM's registers (BF16 out);
    epilogue="row": INT32 GEMM out, then row kernels (res_norm_quant, swiglu_quant, qkv_epilogue_headnorm).
    compiled=True (gemm path): each block's halves before / after attention run as mx.compile'd functions."""

    def __init__(self, model, packed, resident=False, overlap=False, epilogue="gemm", compiled=True, fmt=None):
        # fmt: the packed weights' sc_format.Format, installed (S.set_format) at every call, so that models in
        # different formats can run in one process
        self.fmt = fmt if fmt is not None else F.Format()
        S.set_format(self.fmt)
        self.model, self.args = model, model.args
        self.compiled, self._fns = compiled, {}
        self.overlap = overlap and not resident
        self.epilogue = epilogue
        self.eps = mx.array([self.args.rms_norm_eps], mx.float32)
        F_ = self.args.intermediate_size
        perm = {"gu": S.interleave_gate_up(F_)} if epilogue == "gemm" else {}
        self.layers = []
        for i, layer in enumerate(model.model.layers):
            ws = {g: PackedWeight([packed[f"layers.{i}.{p}"] for p in paths], resident, perm.get(g))
                  for g, paths in GROUPS.items()}
            self.layers.append((layer, ws))
        # The block linears are replaced; drop the BF16 weights so they are not resident.
        for layer in model.model.layers:
            for g in GROUPS.values():
                for path in g:
                    parent, name = path.split(".")
                    setattr(getattr(layer, parent), name, None)
        mx.eval([w.codes for _, ws in self.layers for w in ws.values()])
        self.record, self.record_layers = None, ()  # check_fused.py: {(layer, group): (input, xq, wq, acc)}

    def _rec(self, i, g, x, xq, wq, acc):
        if self.record is not None and i in self.record_layers:
            self.record[(i, g)] = (x, xq, wq, acc)

    def weight_bytes(self):
        return sum(w.nbytes() for _, ws in self.layers for w in ws.values())

    def hidden(self, ids, cache):
        """Final normed hidden states [1, L, D] for prompt ids [1, L] (prefill into a fresh or offset cache)."""
        S.set_format(self.fmt)
        if self.epilogue == "gemm":
            if self.compiled and self.record is None:
                return self._hidden_compiled(ids, cache)
            return self._hidden_gemm_epi(ids, cache)
        return self._hidden_row_epi(ids, cache)

    def _block_fns(self, L, Mp, dtype):
        """mx.compile'd fixed-shape halves of a block (before / after attention) for prompt length L: the same
        kernels as _hidden_gemm_epi, but traced once per length so that building the graph of a prefill costs
        ~2 compiled calls + the attention ops per block instead of ~13 Python-side kernel calls."""
        key = (L, Mp, dtype)
        if key in self._fns:
            return self._fns[key]
        a, eps, T = self.args, self.eps, tiles_for(Mp)
        hq, hk, d, D = a.num_attention_heads, a.num_key_value_heads, a.head_dim, a.hidden_size

        def W(w, N, K):  # (codes, bcodes) -> unpack to INT8; (wq,) -> resident INT8
            return w[0] if len(w) == 1 else S.unpack_i8(w[0], w[1], N, K)

        def pre(xq, xs, wqkv, sqkv, qn, kn):
            y = S.mm_epi(xq, W(wqkv, sqkv.size, D), xs, sqkv, 1, dtype, tile=T["qkv"])
            return S.qkv_headnorm_y(y, qn, kn, eps, L, hq, hk, d)

        def post(o, h, wo, so, pn, wgu, sgu, wd, sd, nn):
            xq, xs = S.act_quant(o, Mp)
            h = S.mm_epi(xq, W(wo, so.size, hq * d), xs, so, 2, dtype, res=h, tile=T["o"])
            xq, xs = S.rmsnorm_quant(h, pn, eps)
            act = S.mm_epi(xq, W(wgu, sgu.size, D), xs, sgu, 3, dtype, tile=T["gu"])
            xq, xs = S.act_quant(act, nt=512)
            h = S.mm_epi(xq, W(wd, sd.size, sgu.size // 2), xs, sd, 2, dtype, res=h, tile=T["down"])
            xq, xs = S.rmsnorm_quant(h, nn, eps)
            return h, xq, xs

        self._fns[key] = (mx.compile(pre), mx.compile(post))
        return self._fns[key]

    def _hidden_compiled(self, ids, cache):
        h = self.model.model.embed_tokens(ids)[0]
        L = h.shape[0]
        Mp = S.pad_rows(L)
        h = mx.pad(h, [(0, Mp - L), (0, 0)]) if Mp != L else h
        mask = "causal" if L > 1 else None
        pre, post = self._block_fns(L, Mp, h.dtype)
        wt = lambda pw: (pw.wq,) if pw.wq is not None else (pw.codes, pw.bcodes)  # noqa: E731
        nl = len(self.layers)
        xq, xs = S.rmsnorm_quant(h, self.layers[0][0].input_layernorm.weight, self.eps, Mp)
        for i, (layer, ws) in enumerate(self.layers):
            at = layer.self_attn
            q, k, v = pre(xq, xs, wt(ws["qkv"]), ws["qkv"].wscale, at.q_norm.weight, at.k_norm.weight)
            o = self._attention(at, cache[i], q, k, v, L, mask)
            nn = self.layers[i + 1][0].input_layernorm.weight if i + 1 < nl else self.model.model.norm.weight
            h, xq, xs = post(o, h, wt(ws["o"]), ws["o"].wscale, layer.post_attention_layernorm.weight,
                             wt(ws["gu"]), ws["gu"].wscale, wt(ws["down"]), ws["down"].wscale, nn)
        return self.model.model.norm(h[:L][None])

    def _attention(self, at, c, q, k, v, L, mask):
        q, k, v = q[None], k[None], v[None]
        q = at.rope(q, offset=c.offset)
        k = at.rope(k, offset=c.offset)
        k, v = c.update_and_fetch(k, v)
        o = mx.fast.scaled_dot_product_attention(q, k, v, scale=at.scale, mask=mask)
        return o.transpose(0, 2, 1, 3).reshape(L, -1)

    def _hidden_gemm_epi(self, ids, cache):
        a = self.args
        hq, hk, d = a.num_attention_heads, a.num_key_value_heads, a.head_dim
        h = self.model.model.embed_tokens(ids)[0]
        L = h.shape[0]
        Mp = S.pad_rows(L)
        h = mx.pad(h, [(0, Mp - L), (0, 0)]) if Mp != L else h
        mask = "causal" if L > 1 else None
        T = tiles_for(Mp)
        nl = len(self.layers)
        xq, xs = S.rmsnorm_quant(h, self.layers[0][0].input_layernorm.weight, self.eps, Mp)
        for i, (layer, ws) in enumerate(self.layers):
            at = layer.self_attn
            w_qkv = ws["qkv"].unpack()
            y = S.mm_epi(xq, w_qkv, xs, ws["qkv"].wscale, 1, h.dtype, tile=T["qkv"])
            self._rec(i, "qkv", h, xq, w_qkv, None)
            q, k, v = S.qkv_headnorm_y(y, at.q_norm.weight, at.k_norm.weight, self.eps, L, hq, hk, d)
            o = self._attention(at, cache[i], q, k, v, L, mask)
            xq, xs = S.act_quant(o, Mp)
            w_o = ws["o"].unpack()
            self._rec(i, "o", o, xq, w_o, None)
            h = S.mm_epi(xq, w_o, xs, ws["o"].wscale, 2, h.dtype, res=h, tile=T["o"])
            xq, xs = S.rmsnorm_quant(h, layer.post_attention_layernorm.weight, self.eps)
            w_gu = ws["gu"].unpack()
            self._rec(i, "gu", h, xq, w_gu, None)
            act = S.mm_epi(xq, w_gu, xs, ws["gu"].wscale, 3, h.dtype, tile=T["gu"])
            xq, xs = S.act_quant(act, nt=512)
            w_d = ws["down"].unpack()
            self._rec(i, "down", act, xq, w_d, None)
            h = S.mm_epi(xq, w_d, xs, ws["down"].wscale, 2, h.dtype, res=h, tile=T["down"])
            if i + 1 < nl:
                xq, xs = S.rmsnorm_quant(h, self.layers[i + 1][0].input_layernorm.weight, self.eps)
        return self.model.model.norm(h[:L][None])

    def _hidden_row_epi(self, ids, cache):
        a = self.args
        hq, hk, d = a.num_attention_heads, a.num_key_value_heads, a.head_dim
        h = self.model.model.embed_tokens(ids)[0]
        L = h.shape[0]
        Mp = S.pad_rows(L)
        h = mx.pad(h, [(0, Mp - L), (0, 0)]) if Mp != L else h
        mask = "causal" if L > 1 else None
        nl = len(self.layers)
        ov = self.overlap
        first = self.layers[0]
        xq, xs = S.rmsnorm_quant(h, first[0].input_layernorm.weight, self.eps, Mp)
        w_qkv = first[1]["qkv"].unpack()
        for i, (layer, ws) in enumerate(self.layers):
            at, c = layer.self_attn, cache[i]
            # unpack Wo and Wg|Wu while the q/k/v GEMM runs
            w_o = ws["o"].unpack()
            w_gu = ws["gu"].unpack()
            acc = S.mm_i8_i32(xq, w_qkv)
            self._rec(i, "qkv", h, xq, w_qkv, acc)
            q, k, v = S.qkv_epilogue_headnorm(acc, xs, ws["qkv"].wscale, at.q_norm.weight, at.k_norm.weight, self.eps,
                                              L, hq, hk, d, h.dtype, deps=(w_o, w_gu) if ov else ())
            q, k, v = q[None], k[None], v[None]
            q = at.rope(q, offset=c.offset)
            k = at.rope(k, offset=c.offset)
            k, v = c.update_and_fetch(k, v)
            o = mx.fast.scaled_dot_product_attention(q, k, v, scale=at.scale, mask=mask)
            o = o.transpose(0, 2, 1, 3).reshape(L, hq * d)
            xq, xs = S.act_quant(o, Mp)
            acc = S.mm_i8_i32(xq, w_o)
            self._rec(i, "o", o, xq, w_o, acc)
            h, xq, xs = S.res_norm_quant(acc, xs, ws["o"].wscale, h, layer.post_attention_layernorm.weight, self.eps)
            # unpack Wd (and the next block's Wq|Wk|Wv) while the gate/up GEMM runs
            w_d = ws["down"].unpack()
            w_qkv = self.layers[i + 1][1]["qkv"].unpack() if i + 1 < nl else None
            acc = S.mm_i8_i32(xq, w_gu)
            self._rec(i, "gu", h, xq, w_gu, acc)
            dd = tuple(w for w in (w_d, w_qkv) if w is not None) if ov else ()
            xs_gu = xs
            xq, xs = S.swiglu_quant(acc, xs, ws["gu"].wscale, h.dtype, deps=dd)
            gu_acc = acc
            acc = S.mm_i8_i32(xq, w_d)
            self._rec(i, "down", (gu_acc, ws["gu"].wscale, xs_gu), xq, w_d, acc)
            if i + 1 < nl:
                h, xq, xs = S.res_norm_quant(acc, xs, ws["down"].wscale, h,
                                             self.layers[i + 1][0].input_layernorm.weight, self.eps)
            else:
                h = S.res_epilogue(acc, xs, ws["down"].wscale, h)
        return self.model.model.norm(h[:L][None])

    def make_cache(self):
        from mlx_lm.models.cache import make_prompt_cache
        return make_prompt_cache(self.model)

    def logits(self, h):
        m = self.model
        return m.model.embed_tokens.as_linear(h) if self.args.tie_word_embeddings else m.lm_head(h)

    def __call__(self, ids, cache=None):
        """Logits for every position (like mlx_lm's Model.__call__)."""
        from mlx_lm.models.cache import make_prompt_cache
        cache = cache if cache is not None else make_prompt_cache(self.model)
        return self.logits(self.hidden(ids, cache))


def load_fused(repo, cache_dir, resident=False, overlap=False, epilogue="gemm", compiled=True, packed="rtn"):
    """packed="rtn": round-to-nearest SC-MXFP4 (packed on first use); "gptq": GPTQ-rounded SC-MXFP4 of
    e2e/gptq_pack.py (same format and kernels); "fmt:<name>": a sim/formats.py format packed by e2e/pack_format.py."""
    import w4a8_mlx as w
    from mlx_lm import load
    model, tok = load(repo)
    if packed.startswith("fmt:"):  # another INT8-exact format of sim/formats.py, packed by e2e/pack_format.py
        import pack_format as pf
        cache = pf.cache_path(repo, packed[4:], cache_dir)
        assert os.path.exists(cache), f"{cache} missing: run e2e/pack_format.py --repo {repo} --format {packed[4:]}"
        flat = mx.load(cache)
        pk = {k[:-6]: (np.array(flat[k]), np.array(flat[k[:-6] + ".bcodes"]), np.array(flat[k[:-6] + ".wscale"]))
              for k in flat if k.endswith(".codes")}
        return FusedQwen3(model, pk, resident=resident, overlap=overlap, epilogue=epilogue, compiled=compiled,
                          fmt=pf.sc_format_of(packed[4:])), tok
    suffix = {"rtn": "-scmxfp4.safetensors", "gptq": "-scmxfp4-gptq.safetensors"}[packed]
    cache = os.path.join(cache_dir, repo.replace("/", "--") + suffix)
    if not os.path.exists(cache):
        assert packed == "rtn", f"{cache} missing: run e2e/gptq_pack.py --repo {repo}"
        w.pack_model(model, cache)
    packed = load_packed(cache, len(model.model.layers))
    return FusedQwen3(model, packed, resident=resident, overlap=overlap, epilogue=epilogue, compiled=compiled), tok
