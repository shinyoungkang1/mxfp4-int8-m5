#!/bin/zsh
# Controls for run 21: NVFP4 with scale search under the same GPTQ, with and without rotations, and rotation with BF16 weights, which shows how much of
# the rotation gain comes from INT8 activations rather than weights. Usage: caffeinate -i -s zsh sim/run_22.sh
set -u
root="${0:A:h:h}"
cd "$root/sim"
r=../results
PY=$root/.venv/bin/python
NV="w=gptq:nvfp4_search;a=8"
BF="w=bf16;a=8;r=32"
run() {  # repo method
  grep -qF "$1,$2," $r/22_ppl.csv 2>/dev/null && return
  echo "$(date '+%a %H:%M') start $1 $2" >> $r/22_queue.log
  $PY eval_ppl.py "$1" $r/22_ppl.csv --methods "$2" --windows-dir $r/22_windows >> $r/22_ppl.log 2>&1
  echo "$(date '+%a %H:%M') end   $1 $2 (exit $?)" >> $r/22_queue.log
}
run Qwen/Qwen3-1.7B-Base "$BF"
run Qwen/Qwen3-1.7B-Base "$NV"
run Qwen/Qwen3-1.7B-Base "$NV;r=32"
for repo in Qwen/Qwen3-4B-Base HuggingFaceTB/SmolLM3-3B-Base; do
  run $repo "$NV"
  run $repo "$NV;r=32"
  run $repo "$BF"
done
for repo in Qwen/Qwen3-4B-Base HuggingFaceTB/SmolLM3-3B-Base; do
  for m in "w=gptq:e2m1/B8/M1..10;a=8;r=32" "w=gptq:e2m1/B16/M1..10;a=8;r=32"; do run $repo "$m"; done
done
