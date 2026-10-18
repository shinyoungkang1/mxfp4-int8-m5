#!/bin/zsh
# Qwen3.8-Flash-Next prefill + decode timing: ours (fused W4A8, SC-MXFP4) vs MLX MXFP4 (W4A16, same matrices).
# The two systems do not fit in memory together (63 + 66 GB), so each session loads one system in its own process,
# under run_timed.sh (GPU lock, idle GPU before/after, AC power), alternating A B A B; one job at a time
# (two concurrent jobs of this step need more than 128 GB of memory).
# Usage: e2e/run_q38_prefill.sh <python of .venv-q38>  -> results/10_prefill_q38.csv, logs 10_prefill_q38_*.log
set -e
py=${1:-$PWD/.venv-q38/bin/python}
root="${0:A:h:h}"
cd "$root/e2e"
r=../results
for s in 1 2; do
  for m in fused mlx_mxfp4; do
    ./run_timed.sh $r/10_prefill_q38_${m}_s$s.log $py bench_q38.py --method $m --session $s --reps 5 --decode 64 \
      --out $r/10_prefill_q38.csv
  done
done
echo "prefill done"
