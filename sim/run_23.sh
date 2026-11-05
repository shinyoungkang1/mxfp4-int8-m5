#!/bin/zsh
# Out-of-sample test of the rotation bound: perplexity with BF16 weights at block sizes not used before, each with INT8 activations and with
# rotation alone (the rotation's own noise floor). Predictions: results/23_prediction_before_sweep.txt. Usage: caffeinate -i -s zsh sim/run_23.sh
set -u
root="${0:A:h:h}"
cd "$root/sim"
r=../results
PY=$root/.venv/bin/python
run() {  # repo method
  grep -qF "$1,$2," $r/23_ppl.csv 2>/dev/null && return
  echo "$(date '+%a %H:%M') start $1 $2" >> $r/23_queue.log
  $PY eval_ppl.py "$1" $r/23_ppl.csv --methods "$2" --windows-dir $r/23_windows >> $r/23_ppl.log 2>&1
  echo "$(date '+%a %H:%M') end   $1 $2 (exit $?)" >> $r/23_queue.log
}
for repo in Qwen/Qwen3-1.7B-Base Qwen/Qwen3-4B-Base HuggingFaceTB/SmolLM3-3B-Base; do
  for b in 8 16 64 128; do run $repo "w=bf16;a=8;r=$b"; done
  for b in 8 16 32 64 128; do run $repo "w=bf16;a=16;r=$b"; done
done
