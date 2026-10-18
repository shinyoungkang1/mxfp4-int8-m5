"""mlx_lm convert with a model_config override (OLMoE-1B-7B-0924's config.json lacks rms_norm_eps, which mlx_lm
requires; the override is saved into the converted config). Otherwise identical to
`python -m mlx_lm convert --hf-path REPO --mlx-path OUT -q --q-mode MODE --q-group-size G --q-bits 4`.

Usage: python convert_mlx.py REPO OUT MODE GROUP '{"rms_norm_eps": 1e-5}' [router8|w4a8cov]
  router8: MoE routers (mlp.gate, shared_expert_gate) at 8-bit affine group 64, everything else MODE/GROUP 4-bit.
  This is what the model's own quant_predicate asks for (Qwen3.5-MoE), with the mode spelled out so that it also
  works for MODE = mxfp4 (whose 4-bit/group-32 parameters the routers' 8-bit request would otherwise inherit).
  w4a8cov: quantize exactly the matrices that our fused models run in W4A8; embeddings, lm_head, MoE routers,
  shared_expert_gate and the DeltaNet in_proj_b / in_proj_a stay BF16 (a coverage-matched W4A16 baseline).
"""
import json
import sys

import mlx.core as mx
from mlx.utils import tree_map_with_path
from mlx_lm.utils import load, quantize_model, save


def main(repo, out, mode, group, model_config, variant=""):
    model, tok, config = load(repo, model_config=json.loads(model_config), return_config=True, lazy=True)
    cast = getattr(model, "cast_predicate", lambda _: True)
    model.update(tree_map_with_path(
        lambda k, v: v.astype(mx.bfloat16) if cast(k) and mx.issubdtype(v.dtype, mx.floating) else v,
        model.parameters()))
    pred = None
    if variant == "router8":
        def pred(path, _):
            if path.endswith("mlp.gate") or path.endswith("shared_expert_gate"):
                return {"group_size": 64, "bits": 8, "mode": "affine"}
            return True
    elif variant == "w4a8cov":
        keep = ("embed_tokens", "lm_head", "mlp.gate", "shared_expert_gate", "in_proj_b", "in_proj_a")

        def pred(path, _):
            return not path.endswith(keep)
    model, config = quantize_model(model, config, int(group), 4, mode=mode, quant_predicate=pred)
    save(out, repo, model, tok, config)
    print("wrote", out)


if __name__ == "__main__":
    main(*sys.argv[1:7])
