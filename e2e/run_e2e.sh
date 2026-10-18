#!/bin/zsh
# Reproduces results/05_end_to_end.csv, 05_end_to_end_burst.csv and 05_check.json. Run on AC power with High Power mode, with
# nothing else heavy running (other GPU or memory-bandwidth-bound jobs skew the baselines).
# Usage: run_e2e.sh <python-with-mlx-lm> [model-cache-dir]
set -e
cd "${0:A:h}"
PY=$1; D=${2:-$HOME/.cache/mxfp4-int8-m5}
REPO=Qwen/Qwen3-1.7B-Base; B=${REPO:t}
mkdir -p $D
# MLX's own 4-bit checkpoints of the same model (embeddings and lm_head are quantized too, as mlx_lm does it).
[[ -d $D/$B-mxfp4 ]] || $PY -m mlx_lm convert --hf-path $REPO --mlx-path $D/$B-mxfp4 -q --q-mode mxfp4 --q-group-size 32 --q-bits 4
[[ -d $D/$B-4bit ]] || $PY -m mlx_lm convert --hf-path $REPO --mlx-path $D/$B-4bit -q --q-mode affine --q-group-size 64 --q-bits 4
$PY w4a8_mlx.py                                   # kernel self-test (exact INT8 GEMM, unpack, padding)
$PY check_outputs.py --repo $REPO --cache-dir $D  # packs SC-MXFP4 weights on first run (cached in $D)
# Sustained regime (primary table): all three lengths and six methods interleaved in one process.
$PY bench_prefill.py --repo $REPO --cache-dir $D --lengths 512,2048,8192 --reps 10
# Burst regime: same protocol without the 8192-token runs, which drop GPU clocks for the rest of the process.
$PY bench_prefill.py --repo $REPO --cache-dir $D --lengths 512,2048 --reps 10 \
  --methods bf16,mlx_mxfp4,mlx_affine4,sc_w4a8,sc_w8a8_res --out ../results/05_end_to_end_burst.csv
