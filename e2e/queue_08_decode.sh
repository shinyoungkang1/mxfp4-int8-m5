#!/bin/zsh
# Decode, one session per model: our W4A8 GEMV decode and MLX's checkpoints (bench_decode.py / bench_qwen35.py), then
# llama.cpp's decode (bench_llamacpp.py --decode-only), back to back under one GPU lock, so that the three are
# measured in the same clock state (MLX's own decode speed differed by up to 9% between separate sessions).
# Usage: queue_08_decode.sh <python-with-mlx-lm> [cache-dir]
set -e
cd "${0:A:h}"
PY=$1; D=${2:-$HOME/.cache/mxfp4-int8-m5}; R=../results
for m in qwen3 olmoe qwen35; do
  [[ -f $R/08_decode_session_$m.csv ]] && continue
  if [[ $m == qwen35 ]]; then ours="$PY bench_qwen35.py decode --out $R/08_decode_session_$m.csv"
  else ours="$PY bench_decode.py --model $m --cache-dir $D --ctx 512,2048 --gen 128 --methods mlx_mxfp4,mlx_affine4,ours --out $R/08_decode_session_$m.csv"; fi
  ./run_timed.sh $R/08_decode_session_$m.log zsh -c "$ours && $PY bench_llamacpp.py --model $m --decode-only --out $R/08_decode_session_llamacpp_$m.csv"
done
