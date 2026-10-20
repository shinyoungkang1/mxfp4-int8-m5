"""MLX baselines for Qwen3.8-Flash-Next, built in memory from the official FP8 checkpoint (no converted copy on disk):
the PR's model (mlx-lm #1788) with exactly the W4A8 matrices of fused_qwen38.py quantized by MLX (nn.quantize:
MXFP4, group 32, or 4-bit affine, group 64; W4A16 gather_qmm / qmm kernels), everything else BF16, and the same
memory-mapped FP8 n-gram table (q38_ckpt.NgramTable), so the systems differ only in the quantized matrices and
their kernels. Run with .venv-q38.
method "mlx_sc_mxfp4": MLX's MXFP4 W4A16 kernels on our packed SC-MXFP4 weights (the same bits our W4A8 kernels read,
with the E8M0 block scales derived from our row scales and block shifts, fused_qwen38.mxfp4_scales), which separates
the effect of INT8 activations from that of the weight quantizer.
"""
import os
import sys
import types

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_flatten, tree_unflatten

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import q38_ckpt as C  # noqa: E402
from mlx_lm.models import qwen4_exp  # noqa: E402

QUANT = ["linear_attn.in_proj_qkv", "linear_attn.in_proj_z", "linear_attn.out_proj", "self_attn.q_proj",
         "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj", "mlp.switch_mlp.gate_proj",
         "mlp.switch_mlp.up_proj", "mlp.switch_mlp.down_proj", "mlp.shared_expert.gate_proj",
         "mlp.shared_expert.up_proj", "mlp.shared_expert.down_proj"]
MODES = {"mlx_mxfp4": dict(group_size=32, bits=4, mode="mxfp4"), "mlx_affine4": dict(group_size=64, bits=4, mode="affine"),
         "mlx_sc_mxfp4": dict(group_size=32, bits=4, mode="mxfp4")}


def _module(root, path):
    for part in path.split("."):
        root = getattr(root, part)
    return root


def use_sc_weights(layer, i, check=False):
    """Replaces the MXFP4 weights MLX quantized itself in layer i by our packed SC-MXFP4 weights (fused_qwen38's
    files: dense matrices, the 512 routed experts and the shared expert as expert 512). The code words are the same
    bits; only the E8M0 scales are derived. check: compare MLX's dequantization with our INT8 unpack times the row
    scales (every dense matrix, every 64th expert); they must be identical."""
    import fused_qwen38 as FQ
    import sc_kernels as S
    fmt = FQ.kernel_format(FQ.SC)
    S.set_format(fmt)
    flat = mx.load(os.path.join(FQ.pack_dir(FQ.SC, os.path.expanduser("~/.cache/mxfp4-int8-m5")),
                                f"layer_{i:02d}.safetensors"))

    def scales(bcodes, wscale, K):
        return FQ.mxfp4_scales(types.SimpleNamespace(bcodes=bcodes, wscale=wscale, K=K), fmt)

    def same(w, s, codes, bcodes, wscale, N, K):
        ours = S.unpack_i8(codes.reshape(-1), bcodes, N, K).astype(mx.float32) * wscale[:, None]
        theirs = mx.dequantize(w, s, group_size=32, bits=4, mode="mxfp4").astype(mx.float32)
        return bool(mx.array_equal(ours, theirs).item())

    dense = sorted(k[:-len(".codes")] for k in flat if k.endswith(".codes") and not k.startswith("experts."))
    for p in dense:
        m = _module(layer, p)
        N = flat[p + ".wscale"].shape[0]
        K = flat[p + ".codes"].size * 8 // N
        w, s = flat[p + ".codes"].reshape(N, K // 8), scales(flat[p + ".bcodes"], flat[p + ".wscale"], K)
        assert m.weight.shape == w.shape and m.scales.shape == s.shape, (p, m.weight.shape, w.shape)
        if check:
            assert same(w, s, flat[p + ".codes"], flat[p + ".bcodes"], flat[p + ".wscale"], N, K), p
        m.weight, m.scales = w, s
    for name in ("gate_proj", "up_proj", "down_proj"):
        c, b, ws = flat[f"experts.{name}.codes"], flat[f"experts.{name}.bcodes"], flat[f"experts.{name}.wscale"]
        E1, N = ws.shape
        K = c.shape[1] * 8 // N
        s = scales(np.array(b).reshape(E1 * N, -1), np.array(ws).reshape(E1 * N), K).reshape(E1, N, K // 32)
        w = c.reshape(E1, N, K // 8)
        if check:
            for e in list(range(0, E1, 64)) + [E1 - 1]:
                assert same(w[e], s[e], c[e], b[e], ws[e], N, K), (name, e)
        sw, sh = _module(layer, f"mlp.switch_mlp.{name}"), _module(layer, f"mlp.shared_expert.{name}")
        assert sw.weight.shape == w[:E1 - 1].shape and sh.weight.shape == w[E1 - 1].shape, (name, sw.weight.shape)
        sw.weight, sw.scales = w[:E1 - 1], s[:E1 - 1]
        sh.weight, sh.scales = w[E1 - 1], s[E1 - 1]
    return len(dense) + 3


def load_mlx(method="mlx_mxfp4", layers=None):
    # caps MLX's buffer cache (otherwise the weights replaced by use_sc_weights, ~65 GB, stay cached) and stops the
    # load if swap grows
    guard = C.MemoryGuard()
    ck = C.Checkpoint()
    tc = dict(ck.config["text_config"])
    if layers:
        tc.update(num_hidden_layers=layers, layer_types=tc["layer_types"][:layers])
    model = qwen4_exp.Model(qwen4_exp.ModelArgs(model_type="qwen4_exp", text_config=tc))
    a = model.args.text
    for i, layer in enumerate(model.layers):
        if layer.ple is not None:
            emb = layer.ple.ple_embedding
            table = C.NgramTable(ck, i, emb)
            emb.pop("ngram_embedding")
            object.__setattr__(emb, "ngram_embedding", table)
        w = ck.layer_weights(i, a.num_experts)
        layer.update(tree_unflatten(list(w.items())))
        del w
        nn.quantize(layer, class_predicate=lambda p, m: hasattr(m, "to_quantized") and p in QUANT, **MODES[method])
        if method == "mlx_sc_mxfp4":
            use_sc_weights(layer, i, check=(i == 0))
        mx.eval([v for k, v in tree_flatten(layer.parameters()) if "ple_embedding" not in k])
        mx.clear_cache()
        guard.check(f"loading layer {i}")
    top = ck.top_weights()
    model.model.embed_tokens.weight = top["embed_tokens.weight"]
    model.lm_head.weight = top["lm_head.weight"]
    model.model.hyper_connection_mixer.update(tree_unflatten(
        [(k[len("hyper_connection_mixer."):], v) for k, v in top.items() if k.startswith("hyper_connection_mixer.")]))
    mx.eval(model.model.embed_tokens.parameters(), model.lm_head.parameters(),
            model.model.hyper_connection_mixer.parameters())
    model.eval()  # as mlx_lm.load: the DeltaNet Metal kernel is used only when not training
    return model


def weight_bytes(model):
    return sum(v.nbytes for k, v in tree_flatten(model.parameters()) if "ple_embedding" not in k)
