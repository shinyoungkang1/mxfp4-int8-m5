#!/bin/zsh
# Does the GPTQ + rotation result on Qwen3-1.7B (SC-MXFP4 9.75, results/19) hold on other models? Perplexity, fake quant, cached Hessians, one
# method at a time. Also rotations with GPTQ on the wider formats on 1.7B. Usage: caffeinate -i -s zsh sim/run_21.sh
set -u
root="${0:A:h:h}"
cd "$root/sim"
r=../results
PY=$root/.venv/bin/python
SC="w=gptq:sc_mx;a=8"
B8="w=gptq:e2m1/B8/M1..10;a=8"
B16="w=gptq:e2m1/B16/M1..10;a=8"
run() {  # repo method
  grep -qF "$1,$2," $r/21_ppl_gptq.csv 2>/dev/null && return
  echo "$(date '+%a %H:%M') start $1 $2" >> $r/21_queue.log
  $PY eval_ppl.py "$1" $r/21_ppl_gptq.csv --methods "$2" --windows-dir $r/21_windows >> $r/21_ppl_gptq.log 2>&1
  echo "$(date '+%a %H:%M') end   $1 $2 (exit $?)" >> $r/21_queue.log
}
for repo in Qwen/Qwen3-4B-Base HuggingFaceTB/SmolLM3-3B-Base; do
  for m in "$SC;r=32" "$SC" "$B8" "$B16"; do run $repo "$m"; done
done
for m in "$B8;r=32" "$B16;r=32"; do run Qwen/Qwen3-1.7B-Base "$m"; done
