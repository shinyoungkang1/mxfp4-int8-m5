"""Coverage of the W4A8 path on Qwen3.5-35B-A3B-Base: fraction of block-linear parameters and of prefill FLOPs that
run in W4A8 (e2e/fused_qwen35.py), from the model's module shapes.

Block linears = every Linear / expert matrix inside the 40 blocks: DeltaNet in_proj_qkv / in_proj_z / in_proj_b /
in_proj_a / out_proj, attention q (query + output gate) / k / v / o, router (mlp.gate), shared_expert_gate, routed
experts (256 x gate/up/down), shared expert (gate/up/down). W4A8: all except router, shared_expert_gate,
in_proj_b / in_proj_a. FLOPs: 2 per multiply-add; per token the routed experts count 8 of 256. Attention FLOPs of
the 10 full-attention blocks: QK^T and AV over a causal context (mean L/2 keys); the gated-delta recurrence of the
30 DeltaNet blocks ~ 6 * heads * d_k * d_v per token; conv1d 2 * k * channels. lm_head runs on the last position
only in prefill. Usage: python coverage_qwen35.py [--out results/08_coverage_qwen35.json]
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def main():
    import fused_qwen35 as fz
    from mlx_lm import load
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(HERE, "..", "results", "08_coverage_qwen35.json"))
    a = ap.parse_args()
    model, _ = load(fz.REPO, lazy=True)
    args = model.language_model.args
    k_act = args.num_experts_per_tok
    par = {"w4a8": 0, "bf16": 0}
    flop = {"w4a8": 0, "bf16": 0}  # per token, block linears
    other = {"attention_qk_av_coeff": 0, "deltanet_recurrence": 0, "conv1d": 0}
    for layer in model.language_model.model.layers:
        kind = fz.layer_kind(layer)
        w4 = [p for paths in fz.DENSE[kind].values() for p in paths]
        for p in w4:
            n = fz.get(layer, p).weight.size
            par["w4a8"] += n
            flop["w4a8"] += 2 * n
        if kind == "linear":
            m = layer.linear_attn
            for p in ("in_proj_b", "in_proj_a"):
                n = getattr(m, p).weight.size
                par["bf16"] += n
                flop["bf16"] += 2 * n
            other["deltanet_recurrence"] += 6 * m.num_v_heads * m.head_k_dim * m.head_v_dim
            other["conv1d"] += 2 * m.conv_kernel_size * m.conv_dim
        else:
            m = layer.self_attn
            other["attention_qk_av_coeff"] += 4 * m.num_attention_heads * m.head_dim  # x context length
        mlp = layer.mlp
        for name in fz.EXPERTS:
            w = getattr(mlp.switch_mlp, name).weight
            E = w.shape[0]
            n = w.size
            par["w4a8"] += n
            flop["w4a8"] += 2 * n * k_act // E
            ns = getattr(mlp.shared_expert, name).weight.size
            par["w4a8"] += ns
            flop["w4a8"] += 2 * ns
        for t in (mlp.gate.weight, mlp.shared_expert_gate.weight):
            par["bf16"] += t.size
            flop["bf16"] += 2 * t.size
    emb = model.language_model.model.embed_tokens.weight.size
    head = model.language_model.lm_head.weight.size
    total_block = par["w4a8"] + par["bf16"]
    rep = {"block_linear_params": total_block, "w4a8_params": par["w4a8"], "bf16_block_linear_params": par["bf16"],
           "w4a8_fraction_of_block_linear_params": par["w4a8"] / total_block,
           "w4a8_fraction_of_all_params_incl_embeddings_head": par["w4a8"] / (total_block + emb + head),
           "embedding_params": emb, "lm_head_params": head,
           "per_token_block_linear_flops": flop["w4a8"] + flop["bf16"],
           "w4a8_fraction_of_block_linear_flops": flop["w4a8"] / (flop["w4a8"] + flop["bf16"])}
    for L in (512, 2048, 8192):
        att = other["attention_qk_av_coeff"] * L / 2
        tot = flop["w4a8"] + flop["bf16"] + att + other["deltanet_recurrence"] + other["conv1d"] + 2 * head / L
        rep[f"w4a8_fraction_of_prefill_flops_L{L}"] = flop["w4a8"] / tot
    with open(a.out, "w") as f:
        json.dump(rep, f, indent=1)
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
