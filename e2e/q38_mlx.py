"""MLX baselines for Qwen3.8-Flash-Next, built in memory from the official FP8 checkpoint (no converted copy on disk):
the PR's model (mlx-lm #1788) with exactly the W4A8 matrices of fused_qwen38.py quantized by MLX (nn.quantize:
MXFP4, group 32, or 4-bit affine, group 64; W4A16 gather_qmm / qmm kernels), everything else BF16, and the same
memory-mapped FP8 n-gram table (q38_ckpt.NgramTable), so the systems differ only in the quantized matrices and
their kernels. Run with .venv-q38.
"""
import os
import sys

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten, tree_unflatten

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import q38_ckpt as C  # noqa: E402
from mlx_lm.models import qwen4_exp  # noqa: E402

QUANT = ["linear_attn.in_proj_qkv", "linear_attn.in_proj_z", "linear_attn.out_proj", "self_attn.q_proj",
         "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj", "mlp.switch_mlp.gate_proj",
         "mlp.switch_mlp.up_proj", "mlp.switch_mlp.down_proj", "mlp.shared_expert.gate_proj",
         "mlp.shared_expert.up_proj", "mlp.shared_expert.down_proj"]
MODES = {"mlx_mxfp4": dict(group_size=32, bits=4, mode="mxfp4"), "mlx_affine4": dict(group_size=64, bits=4, mode="affine")}


def load_mlx(method="mlx_mxfp4", layers=None):
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
        mx.eval([v for k, v in tree_flatten(layer.parameters()) if "ple_embedding" not in k])
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
