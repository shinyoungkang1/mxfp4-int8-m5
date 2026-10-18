#!/bin/zsh
# Reproduces results/07_*: fused dense prefill (Qwen3-1.7B), MoE prefill (OLMoE-1B-7B), batch-1 decode (both).
# Run on AC power with High Power mode. Every timing goes through run_timed.sh (tools/gpu_timing.sh lock + a check
# that no other application is using the GPU); correctness checks don't need the lock.
# Usage: run_07.sh <python-with-mlx-lm> [model-cache-dir]
set -e
cd "${0:A:h}"
PY=$1; D=${2:-$HOME/.cache/mxfp4-int8-m5}; R=../results
mkdir -p $D
Q=Qwen/Qwen3-1.7B-Base; O=allenai/OLMoE-1B-7B-0924
# MLX's own 4-bit checkpoints of both models (OLMoE needs rms_norm_eps, missing from its config.json)
[[ -d $D/Qwen3-1.7B-Base-mxfp4 ]] || $PY -m mlx_lm convert --hf-path $Q --mlx-path $D/Qwen3-1.7B-Base-mxfp4 -q --q-mode mxfp4 --q-group-size 32 --q-bits 4
[[ -d $D/Qwen3-1.7B-Base-4bit ]] || $PY -m mlx_lm convert --hf-path $Q --mlx-path $D/Qwen3-1.7B-Base-4bit -q --q-mode affine --q-group-size 64 --q-bits 4
[[ -d $D/OLMoE-1B-7B-0924-mxfp4 ]] || $PY convert_mlx.py $O $D/OLMoE-1B-7B-0924-mxfp4 mxfp4 32 '{"rms_norm_eps": 1e-5}'
[[ -d $D/OLMoE-1B-7B-0924-4bit ]] || $PY convert_mlx.py $O $D/OLMoE-1B-7B-0924-4bit affine 64 '{"rms_norm_eps": 1e-5}'

# ---- correctness (packs the SC-MXFP4 weights on first use, cached in $D)
$PY sc_format.py
$PY test_sc_kernels.py
$PY sc_moe.py
$PY sc_decode.py
$PY test_formats.py
$PY check_fused.py --cache-dir $D --out $R/07_check_fused.json
$PY check_fused_olmoe.py --cache-dir $D --out $R/07_check_moe.json
$PY check_decode.py --model qwen3 --cache-dir $D --out $R/07_check_decode_qwen3.json
$PY check_decode.py --model olmoe --cache-dir $D --out $R/07_check_decode_olmoe.json

# ---- timing (tile sweeps + all benchmarks; each step waits for an idle GPU): e2e/queue_07.sh
./run_timed.sh $R/07_tune_gemv.txt $PY tune_07.py gemv; ./run_timed.sh $R/07_tune_prefill.txt $PY tune_07.py prefill
./queue_07.sh $PY $D
$PY make_07_tables.py $R
