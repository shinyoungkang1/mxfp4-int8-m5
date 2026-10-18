#!/bin/zsh
# A candidate INT8-exact weight format (sim/formats.py) on the fused Qwen3-1.7B kernels, end to end (results/09_format_*):
# pack (pack_format.py), correctness vs eval_ppl's fake-quant "w=fmt:<name>;a=8" (check_fused.py --packed), decode
# bit-identity (check_decode.py --packed), full-test WikiText-2 perplexity on the kernels, then prefill (burst 512 /
# 2048, sustained 8192) and decode against SC-MXFP4 and MLX MXFP4 under one GPU lock, then zero-shot tasks.
# Usage: queue_format.sh <python-with-mlx-lm> [format]     (default e2m1/B16/M1..10)
set -e
cd "${0:A:h}"
PY=$1; FMT=${2:-e2m1/B16/M1..10}; R=../results; EV=$HOME/.cache/mxfp4-int8-m5/venv-eval/bin/python
S=$(echo $FMT | tr '/' '_' | tr -d '.')
./devrun.sh $PY pack_format.py --format $FMT --out $R/09_format_${S}_pack.json
./devrun.sh $PY check_fused.py --packed fmt:$FMT --out $R/09_format_${S}_check.json
./devrun.sh $PY check_decode.py --model qwen3 --packed fmt:$FMT --out $R/09_format_${S}_check_decode.json
./devrun.sh $PY ppl_gptq_qwen3.py --methods "ours_fmt:$FMT" --out $R/09_format_${S}_ppl.csv
./run_timed.sh $R/09_format_${S}_timing.log zsh -c "$PY bench_format.py prefill --format $FMT --lengths 512,2048 \
  --out $R/09_format_${S}_prefill_burst.csv && $PY bench_format.py prefill --format $FMT --lengths 8192 \
  --out $R/09_format_${S}_prefill_sustained.csv && $PY bench_format.py decode --format $FMT \
  --out $R/09_format_${S}_decode.csv"
./devrun.sh $EV eval_tasks.py --model qwen3 --method ours_fmt --format $FMT --out $R/09_format_${S}_tasks.json
