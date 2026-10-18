"""Fused W4A8 prefill for Qwen3.8-Flash-Next (qwen4_exp, mlx-lm PR #1788; run with .venv-q38).

Same kernels as e2e/fused_qwen35.py (packed weights, INT8 x INT8 matmul2d with exact integer accumulation, grouped
expert GEMM with the shared expert as expert E). What is new in this architecture is handled by running the PR's own
module code with the quantized projections swapped out:
  - every sub-block's input is the hyper-connection mix of 4 residual streams (GatedResidual, BF16, PR code); its
    output is injected into all 4 streams (PR's DecoderLayer arithmetic, replicated op for op);
  - DeltaNet / attention: the projections that run in W4A8 (in_proj_qkv | in_proj_z, or q | k | v; one GEMM each) are
    computed by our kernels and handed to the PR's GatedDeltaNet / Attention through stub modules that return them,
    with out_proj / o_proj replaced by the identity (their input is then quantized and multiplied by our kernel), so
    everything between the GEMMs (conv1d, gated delta rule, gated RMSNorm, q/k norms, partial RoPE, the QSA indexer,
    SDPA, output gate) is the PR's code;
  - MoE: per-token INT8 of the mixed input, the PR's routing (FP32 logits, top-10, softmax over the selected) and its
    combine ((routed * w).sum in FP32 -> BF16, + sigmoid(shared gate) * shared expert), our grouped GEMMs between;
  - the PLE n-gram layer is the PR's code, with the FP8 table memory-mapped (q38_ckpt.NgramTable: rows gathered on
    the host, the 51B-parameter table is never resident).
Weights not in W4A8 (as the fake-quant reference e2e/q38_ppl.py): router, shared_expert_gate, in_proj_a / b, QSA
indexer projection, hyper-connection mixers, PLE projections, embeddings, lm_head: BF16 from the checkpoint.

Packing (pack_qwen38): every W4A8 matrix of the FP8 checkpoint (experts FP8-decoded to BF16, as the reference), in
SC-MXFP4 (sim/ reference quantizer, sc_format) or any INT8-exact format of sim/formats.py (pack_format), one
safetensors file per layer in ~/.cache/mxfp4-int8-m5/<model>-<format>/.
"""
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
import fused_qwen3 as fq  # noqa: E402
import q38_ckpt as C  # noqa: E402
import sc_format as F  # noqa: E402
import sc_kernels as S  # noqa: E402
import sc_moe  # noqa: E402
from mlx_lm.models import qwen4_exp  # noqa: E402

DENSE = {"linear_attention": {"qkvz": ["linear_attn.in_proj_qkv", "linear_attn.in_proj_z"], "o": ["linear_attn.out_proj"]},
         "full_attention": {"qkvz": ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"], "o": ["self_attn.o_proj"]}}
EXPERTS = ["gate_proj", "up_proj", "down_proj"]
SC = "sc"  # format name of the published SC-MXFP4 layout


# ---------------------------------------------------------------------------------------------------- packing
def _pack_sc(w):
    c, b, s, _ = F.quantize(np.asarray(w, np.float64))
    return np.asarray(c).reshape(np.shape(w)[0], -1), b, s


def kernel_format(fmt_name):
    if fmt_name == SC:
        return F.Format()
    import pack_format as pf
    return pf.sc_format_of(fmt_name)


def pack_dir(fmt_name, cache_dir):
    import pack_format as pf
    return os.path.join(cache_dir, "Qwen3.8-Flash-Next-" + ("scmxfp4" if fmt_name == SC else "fmt-" + pf.slug(fmt_name)))


def pack_qwen38(fmt_name=SC, cache_dir=os.path.expanduser("~/.cache/mxfp4-int8-m5"), workers=14, layers=None):
    """Packs every W4A8 matrix, one file per layer (resumable). Keys: '<path>.codes|bcodes|wscale' for the dense
    matrices, 'experts.<proj>.codes|...' stacked over the 512 routed experts + the shared expert (expert 512)."""
    out_dir = pack_dir(fmt_name, cache_dir)
    os.makedirs(out_dir, exist_ok=True)
    ck = C.Checkpoint()
    args = ck.text_args()
    fmt = kernel_format(fmt_name)
    guard = C.MemoryGuard()
    if fmt_name != SC:
        import pack_format as pf

        def pack_many(mats):
            res = []
            for w in mats:
                c, b, s, st = pf.pack_matrix(w, fmt_name, fmt)
                assert st["exact_q"] and st["exact_w"]
                res.append((np.asarray(c).reshape(w.shape[0], -1), b, s))
            return res
    with ProcessPoolExecutor(workers) as ex:
        if fmt_name == SC:
            def pack_many(mats):  # noqa: F811
                return list(ex.map(_pack_sc, mats, chunksize=8))
        for i in layers if layers is not None else range(args.num_hidden_layers):
            fn = os.path.join(out_dir, f"layer_{i:02d}.safetensors")
            if os.path.exists(fn):
                continue
            t0 = time.time()
            kind = args.layer_types[i]
            paths = [p for ps in DENSE[kind].values() for p in ps]
            w = ck.layer_weights(i, args.num_experts, skip=())
            out = {}
            for p, (c, b, s) in zip(paths, pack_many([np.array(w[p + ".weight"].astype(mx.float32)) for p in paths])):
                out[p + ".codes"], out[p + ".bcodes"], out[p + ".wscale"] = c.reshape(-1), b, s
            for name in EXPERTS:  # 64 experts at a time (float32 copies stay small; peak memory)
                ew, sw = w[f"mlp.switch_mlp.{name}.weight"], w[f"mlp.shared_expert.{name}.weight"]
                res = []
                for e0 in range(0, ew.shape[0], 64):
                    res += pack_many(list(np.array(ew[e0:e0 + 64].astype(mx.float32))))
                res += pack_many([np.array(sw.astype(mx.float32))])
                out[f"experts.{name}.codes"] = np.stack([r[0].reshape(-1) for r in res])
                out[f"experts.{name}.bcodes"] = np.stack([r[1] for r in res])
                out[f"experts.{name}.wscale"] = np.stack([r[2] for r in res])
                del res
                guard.check(f"packing layer {i}")
            del w
            mx.save_safetensors(fn + ".tmp.safetensors", {k: mx.array(v) for k, v in out.items()})
            with open(fn + ".tmp.safetensors", "rb") as fh:  # on disk before it gets its final name
                os.fsync(fh.fileno())
            os.replace(fn + ".tmp.safetensors", fn)
            print(f"packed layer {i} ({kind}) in {time.time() - t0:.0f} s", flush=True)
    return out_dir


# ---------------------------------------------------------------------------------------------------- model
class Given(nn.Module):
    """Stub for a projection computed by our kernels: returns the precomputed output."""

    def __init__(self):
        super().__init__()
        self.y = None

    def __call__(self, x):
        return self.y


class Identity(nn.Module):
    def __call__(self, x):
        return x


class FusedQwen38:
    def __init__(self, fmt_name=SC, cache_dir=os.path.expanduser("~/.cache/mxfp4-int8-m5"), tile=(64, 128, 4, 8),
                 compiled=True, layers=None, load_top=True):
        self.ck = ck = C.Checkpoint()
        self.args = a = ck.text_args()
        self.fmt = kernel_format(fmt_name)
        S.set_format(self.fmt)
        self.tile, self.compiled, self._fns = tile, compiled, {}
        self.E, self.Fi, self.k = a.num_experts, a.moe_intermediate_size, a.num_experts_per_tok
        d = pack_dir(fmt_name, cache_dir)
        perm = S.interleave_gate_up(self.Fi)
        n_layers = layers or a.num_hidden_layers
        self.layers = []
        for i in range(n_layers):
            kind = a.layer_types[i]
            flat = mx.load(os.path.join(d, f"layer_{i:02d}.safetensors"))
            npy = lambda key: np.array(flat[key])  # noqa: E731
            parts = lambda p: (npy(p + ".codes"), npy(p + ".bcodes"), npy(p + ".wscale"))  # noqa: E731
            ws = {g: fq.PackedWeight([parts(p) for p in paths]) for g, paths in DENSE[kind].items()}
            E1 = self.E + 1
            g_c, u_c = npy("experts.gate_proj.codes").reshape(E1, self.Fi, -1), npy("experts.up_proj.codes").reshape(E1, self.Fi, -1)
            gu_c = np.concatenate([g_c, u_c], 1)[:, perm].reshape(E1, -1)
            gu_b = np.concatenate([npy("experts.gate_proj.bcodes"), npy("experts.up_proj.bcodes")], 1)[:, perm]
            gu_s = np.concatenate([npy("experts.gate_proj.wscale"), npy("experts.up_proj.wscale")], 1)[:, perm]
            ex = dict(gu=(mx.array(gu_c), mx.array(gu_b), mx.array(gu_s)),
                      down=tuple(flat[f"experts.down_proj.{s}"] for s in ("codes", "bcodes", "wscale")))
            del flat, g_c, u_c, gu_c, gu_b, gu_s
            layer = qwen4_exp.DecoderLayer(a, i)
            if layer.ple is not None:
                emb = layer.ple.ple_embedding
                table = C.NgramTable(ck, i, emb)
                emb.pop("ngram_embedding")
                object.__setattr__(emb, "ngram_embedding", table)
            packed = [p for ps in DENSE[kind].values() for p in ps]
            skip = packed + ["mlp.switch_mlp", "mlp.shared_expert"]
            w = ck.layer_weights(i, a.num_experts, skip=skip)
            mine = {k for k, _ in tree_flatten(layer.parameters()) if "ple_embedding" not in k
                    and not any(k.startswith(s + ".") for s in skip)}
            assert set(w) == mine, (sorted(set(w) - mine)[:5], sorted(mine - set(w))[:5])
            layer.update(tree_unflatten(list(w.items())))
            # the quantized modules: stubs (their lazy random weights are dropped, never evaluated)
            sub = layer.linear_attn if kind == "linear_attention" else layer.self_attn
            stubs = {}
            for p in DENSE[kind]["qkvz"]:
                stubs[p] = Given()
                setattr(sub, p.split(".")[1], stubs[p])
            setattr(sub, DENSE[kind]["o"][0].split(".")[1], Identity())
            layer.mlp.switch_mlp = None
            layer.mlp.shared_expert = None
            layer.eval()  # inference mode: the PR's DeltaNet uses its Metal kernel only when not training
            mx.eval(layer.parameters(), [p.codes for p in ws.values()], *ex["gu"], *ex["down"])
            self.layers.append((layer, kind, ws, ex, stubs))
        self.rope = qwen4_exp.RotaryEmbedding(int(a.head_dim * a.partial_rotary_factor), a.rope_theta)
        eos = a.eos_token_id
        self.eos = eos[0] if isinstance(eos, list) else eos
        if not load_top:
            return
        top = ck.top_weights()
        self.embed = top["embed_tokens.weight"]
        self.lm_head = top["lm_head.weight"]
        self.mixer = qwen4_exp.GatedResidual(a, use_combine=False)
        self.mixer.update(tree_unflatten([(k[len("hyper_connection_mixer."):], v) for k, v in top.items()
                                          if k.startswith("hyper_connection_mixer.")]))
        self.mixer.eval()
        mx.eval(self.embed, self.lm_head, self.mixer.parameters())

    def weight_bytes(self):
        n = 0
        for _, _, ws, ex, _ in self.layers:
            n += sum(p.nbytes() for p in ws.values()) + sum(t.nbytes for t in ex["gu"] + ex["down"])
        return n

    def _moe_fn(self, L, dtype):
        """MoE sub-block on the mixed input x [L, D] -> BF16 [L, D] (pure function, compiled per prompt length)."""
        key = (L, dtype)
        if key in self._fns:
            return self._fns[key]
        k, E1, F2, D = self.k, self.E + 1, 2 * self.Fi, self.args.hidden_size
        R = L * (k + 1)
        t = (32, 64, 2) if L * k / self.E < 24 else (64, 64, 4)   # results/10_moe_q38_sweep.txt
        bk = 128 if t[0] == 32 else 64
        tm = t[0]
        fmt = self.fmt

        def moe(x, gate_w, sg_w, cg, bg, sg, cd, bd, sd):
            S.set_format(fmt)
            xq, xs = S.act_quant(x, S.pad_rows(L))
            logits = x.astype(mx.float32) @ gate_w.T.astype(mx.float32)      # PR: gate(x.astype(f32))
            idx = mx.argpartition(-logits, k - 1, axis=-1)[..., :k]
            w = mx.softmax(mx.take_along_axis(logits, idx, axis=-1), axis=-1, precise=True)
            inds = mx.concatenate([idx.astype(mx.int32), mx.full((L, 1), self.E, mx.int32)], axis=-1)
            pos, tok_p, e_sorted, off, tstart, maxt = sc_moe.route_sort(inds, E1, tm)
            xq_s, xs_s = xq[tok_p], xs[tok_p[:R]]
            act = sc_moe.moe_mm(xq_s, R, cg, bg, sg, off, tstart, maxt, F2, 3, xs_s, dtype=dtype, tile=t, bk=bk)
            xq2, xs2 = S.act_quant(act, R + tm)
            yd = sc_moe.moe_mm(xq2, R, cd, bd, sd, off, tstart, maxt, D, 1, xs2, dtype=dtype, tile=t, bk=bk)
            y = yd[pos].reshape(L, k + 1, D)
            routed = (y[:, :k] * w[..., None]).sum(axis=-2).astype(dtype)
            return routed + mx.sigmoid(x @ sg_w.T) * y[:, k]

        fn = mx.compile(moe) if self.compiled else moe
        self._fns[key] = fn
        return fn

    def _gemm(self, x, pw, Mp):
        """x BF16 [L, K] -> per-token INT8 -> INT8 GEMM with the packed weights -> BF16 [L, N]."""
        L = x.shape[0]
        xq, xs = S.act_quant(x, Mp)
        return S.mm_epi(xq, pw.unpack(), xs, pw.wscale, 1, x.dtype, tile=self.tile)[:L]

    def make_cache(self):
        """As the PR's Model.make_cache: KV + indexer cache for attention layers, 4 slots for DeltaNet layers (conv,
        SSM state, PLE conv, n-gram context)."""
        return [qwen4_exp._AttnCache() if t == "full_attention" else qwen4_exp._LayerCache(4)
                for t in self.args.layer_types[:len(self.layers)]]

    def hidden(self, ids, cache=None):
        """ids [1, L] -> final hidden [1, L, D] (the mixer carries the final norm). Masks and the n-gram context are
        handled as in the PR's Qwen4ExpModel.__call__."""
        from mlx_lm.models.base import create_attention_mask
        a = self.args
        L = ids.shape[1]
        Mp = S.pad_rows(L)
        h = mx.tile(self.embed[ids], (1, 1, a.hc_count))
        cache = cache if cache is not None else [None] * len(self.layers)
        types = a.layer_types[:len(self.layers)]
        full = [i for i, t in enumerate(types) if t == "full_attention"]
        mask = create_attention_mask(h, cache[full[0]] if full else None)
        ctx_len = a.ngram_size - 1
        ple = [i for i, (layer, *_) in enumerate(self.layers) if layer.ple is not None]
        pc = cache[ple[0]] if ple else None
        prev = pc[3] if pc is not None else None
        prev_ctx = prev if prev is not None else mx.full((1, ctx_len), self.eos, ids.dtype)
        if pc is not None:
            pc[3] = mx.concatenate([prev_ctx, ids], axis=1)[:, -ctx_len:]
        for i in range(len(self.layers)):
            h = self.block(i, h, ids, prev_ctx, mask, cache[i], Mp)
        return self.mixer(h)

    def block(self, i, h, ids, prev_ctx, mask, c, Mp):
        """Decoder layer i on the hyper-connection state h [1, L, 4 D] (the PR's DecoderLayer with our GEMMs)."""
        layer, kind, ws, ex, stubs = self.layers[i]
        L = h.shape[1]
        if layer.ple is not None:
            h = h + layer.ple(h, ids, prev_ctx, c)
        x, hyper, inject = layer.attn_hyper_connection(h)
        y = self._gemm(x[0], ws["qkvz"], Mp)
        o0 = 0
        for p, n in zip(DENSE[kind]["qkvz"], ws["qkvz"].splits):
            stubs[p].y = y[:, o0:o0 + n][None]
            o0 += n
        if kind == "linear_attention":
            o = layer.linear_attn(x, None, c)
        else:
            o = layer.self_attn(x, self.rope, mask, c, c.indexer if c is not None else None)
        x = self._gemm(o[0], ws["o"], Mp)[None]
        h = hyper + (x[..., None, :] * inject[..., None]).reshape(*x.shape[:-1], -1)
        x, hyper, inject = layer.mlp_hyper_connection(h)
        x = self._moe_fn(L, h.dtype)(x[0], layer.mlp.gate.weight, layer.mlp.shared_expert_gate.weight,
                                     *ex["gu"], *ex["down"])[None]
        return hyper + (x[..., None, :] * inject[..., None]).reshape(*x.shape[:-1], -1)

    def logits(self, hn):
        return hn @ self.lm_head.T

    def __call__(self, ids, cache=None):
        return self.logits(self.hidden(ids, cache))


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", default="", help="format to pack: sc or a sim/formats.py name")
    ap.add_argument("--layers", default="")
    a = ap.parse_args()
    if a.pack:
        pack_qwen38(a.pack, layers=[int(x) for x in a.layers.split(",")] if a.layers else None)


def mxfp4_scales(pw, fmt):
    """Our packed SC-MXFP4 matrix (PackedWeight) -> uint8 E8M0 scales [N, K / 32] such that MLX's mode="mxfp4" kernels
    read our code words unchanged as the same weights: a code nibble is sign << 3 | index into 2 x E2M1 magnitudes,
    the same bits as an E2M1 nibble, and w = wscale * 2 * e2m1 * 2^shift = e2m1 * 2^(log2(wscale) + 1 + shift).
    (Checked bit-exact against our decoder; valid only for the SC-MXFP4 format: power-of-two row scale, shifts.)"""
    assert fmt.block == 32 and tuple(fmt.mult) == (1, 2, 4, 8) and tuple(fmt.code_mag) == (0, 1, 2, 3, 4, 6, 8, 12)
    bc = np.array(pw.bcodes)
    nb = pw.K // 32
    b = np.arange(nb)
    shifts = (bc[:, b // fmt.per_byte] >> (fmt.bc_bits * (b % fmt.per_byte))) & ((1 << fmt.bc_bits) - 1)
    e = np.log2(np.array(pw.wscale).astype(np.float64))[:, None] + 1 + shifts
    ei = np.rint(e)
    assert np.array_equal(e, ei), "row scale is not a power of two"
    return mx.array((ei + 127).astype(np.uint8))


class DecodeQwen38:
    """Batch-1 decode from a FusedQwen38: W4A8 GEMVs on the packed weights (sc_decode.gemv, per-token INT8 inside
    the kernel), the expert GEMVs gather the 10 routed experts and the shared expert (11 slots) themselves; routing
    and combine as the prefill path (PR ops); DeltaNet / attention / PLE / hyper-connections are the PR's code with
    the projections swapped (as FusedQwen38.block)."""

    def __init__(self, fused, compiled=True, dense="mlx"):
        """dense="mlx": the dense projections (DeltaNet / attention) decode with MLX's MXFP4 GEMV reading our packed
        bytes unchanged (SC-MXFP4 files are standard MXFP4; 1.4-2.2x faster than our GEMV at these shapes, W4A16);
        dense="ours": our W4A8 GEMV. The experts always use our gather GEMV (faster than MLX's gather_qmm here)."""
        import sc_decode as G
        self.f, self.G = fused, G
        self.dense = dense if fused.fmt == F.Format() else "ours"
        self._mx = {}
        if self.dense == "mlx":
            for _, _, ws, _, _ in fused.layers:
                for pw in ws.values():
                    self._mx[id(pw)] = (pw.codes.reshape(pw.N, pw.K // 8), mxfp4_scales(pw, fused.fmt))
            mx.eval(list(self._mx.values()))
        a = fused.args
        k, E, F2, D = a.num_experts_per_tok, a.num_experts, 2 * fused.Fi, a.hidden_size
        fmt = fused.fmt

        def moe(x, gate_w, sg_w, cg, bg, sg, cd, bd, sd):
            S.set_format(fmt)
            logits = x.astype(mx.float32) @ gate_w.T.astype(mx.float32)
            idx = mx.argpartition(-logits, k - 1, axis=-1)[..., :k]
            w = mx.softmax(mx.take_along_axis(logits, idx, axis=-1), axis=-1, precise=True)
            e = idx.reshape(-1)
            act = G.gemv(x[0], cg, bg, None, sg, F2, 3, eidx=e, xstride=0, shared=E, pro=1)   # [k + 1, F]
            yd = G.gemv(act, cd, bd, None, sd, D, 1, eidx=e, xstride=1, shared=E, pro=1)       # [k + 1, D]
            routed = (yd[None, :k] * w[..., None]).sum(axis=-2).astype(x.dtype)
            return routed + mx.sigmoid(x @ sg_w.T) * yd[k]

        self._moe = mx.compile(moe) if compiled else moe

    def _gemv(self, x, pw):
        if self.dense == "mlx":
            codes, scales = self._mx[id(pw)]
            return mx.quantized_matmul(x[None], codes, scales, transpose=True, group_size=32, bits=4, mode="mxfp4")[0]
        return self.G.gemv(x, pw.codes, pw.bcodes, None, pw.wscale, pw.N, 1, pro=1)

    def step(self, tok, cache):
        """tok [1, 1] -> logits [1, 1, V]; cache from FusedQwen38.make_cache (shared with the prefill path)."""
        f, a = self.f, self.f.args
        h = mx.tile(f.embed[tok], (1, 1, a.hc_count))
        ctx_len = a.ngram_size - 1
        ple = [i for i, (layer, *_) in enumerate(f.layers) if layer.ple is not None]
        pc = cache[ple[0]] if ple else None
        prev_ctx = pc[3] if pc is not None and pc[3] is not None else mx.full((1, ctx_len), f.eos, tok.dtype)
        if pc is not None:
            pc[3] = mx.concatenate([prev_ctx, tok], axis=1)[:, -ctx_len:]
        for i, (layer, kind, ws, ex, stubs) in enumerate(f.layers):
            c = cache[i]
            if layer.ple is not None:
                h = h + layer.ple(h, tok, prev_ctx, c)
            x, hyper, inject = layer.attn_hyper_connection(h)
            y = self._gemv(x[0, 0], ws["qkvz"])
            o0 = 0
            for p, n in zip(DENSE[kind]["qkvz"], ws["qkvz"].splits):
                stubs[p].y = y[o0:o0 + n][None, None]
                o0 += n
            if kind == "linear_attention":
                o = layer.linear_attn(x, None, c)
            else:
                o = layer.self_attn(x, f.rope, None, c, c.indexer)
            x = self._gemv(o[0, 0], ws["o"])[None, None]
            h = hyper + (x[..., None, :] * inject[..., None]).reshape(*x.shape[:-1], -1)
            x, hyper, inject = layer.mlp_hyper_connection(h)
            x = self._moe(x[0], layer.mlp.gate.weight, layer.mlp.shared_expert_gate.weight, *ex["gu"], *ex["down"])
            x = x[None]
            h = hyper + (x[..., None, :] * inject[..., None]).reshape(*x.shape[:-1], -1)
        return f.logits(f.mixer(h))
