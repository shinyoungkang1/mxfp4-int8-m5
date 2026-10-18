#!/bin/zsh
# Perplexity of the INT8-exact format family (results/ppl.csv), W4A8 unless noted.
set -e
cd "${0:A:h}"
PY=$1
F16="w=fmt:e2m1/B16/M1..10;a=8"; F32="w=fmt:e2m1/B32/M1..8;a=8"; NV="w=nvfp4;a=8"; ROT="w=sc_mx;a=8;r=32"
$PY eval_ppl.py Qwen/Qwen3-1.7B-Base ../results/ppl.csv --methods "$F16,$F32,$NV,$ROT,w=fmt:e2m1/B16/M1..10;a=16"
$PY eval_ppl.py Qwen/Qwen3-4B-Base ../results/ppl.csv --methods "$F16,$F32,$NV,$ROT"
$PY eval_ppl.py HuggingFaceTB/SmolLM3-3B-Base ../results/ppl.csv --methods "$F16,$NV"
# Fill the remaining models of the main table (run after the block above).
NVS="w=nvfp4_search;a=8"
$PY eval_ppl.py HuggingFaceTB/SmolLM3-3B-Base ../results/ppl.csv --methods "$NVS"
$PY eval_ppl.py Qwen/Qwen3-0.6B-Base ../results/ppl.csv --methods "$F16,$NV,$NVS"
$PY eval_ppl.py allenai/OLMoE-1B-7B-0924 ../results/ppl.csv --model-config '{"rms_norm_eps": 1e-5}' --methods "$F16,$NV"
