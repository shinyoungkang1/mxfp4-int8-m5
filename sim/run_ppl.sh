#!/bin/zsh
# Reproduces results/ppl.csv: WikiText-2 perplexity (2048-token windows, full test set) for each model and method.
# Usage: run_ppl.sh <python-with-mlx-lm>
set -e
cd "${0:A:h}"
for m in Qwen/Qwen3-0.6B-Base Qwen/Qwen3-1.7B-Base Qwen/Qwen3-4B-Base; do
  $1 eval_ppl.py $m ../results/ppl.csv
done
# Other families (no Hadamard ablation). OLMoE's config.json omits rms_norm_eps (HF default 1e-5), and its
# ~3k expert matrices make each quantized method take ~7 min, so only the methods the paper cites are run.
$1 eval_ppl.py HuggingFaceTB/SmolLM3-3B-Base ../results/ppl.csv \
  --methods bf16,bf16_w16a8,int4_affine_g64_w4a16,nvfp4_w4a16,mxfp4_spec_w4a16,mxfp4_search_w4a16,sc_mxfp4_w4a16,sc_mxfp4_w4a8
$1 eval_ppl.py allenai/OLMoE-1B-7B-0924 ../results/ppl.csv --model-config '{"rms_norm_eps": 1e-5}' \
  --methods bf16,bf16_w16a8,int4_affine_g64_w4a16,nvfp4_w4a16,sc_mxfp4_w4a16,sc_mxfp4_w4a8
