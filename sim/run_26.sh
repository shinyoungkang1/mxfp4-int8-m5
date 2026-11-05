#!/bin/zsh
# Two candidates to beat E2M1 8-wide: top-code-compressed codebooks at the same bits (round-to-nearest, four small models; then GPTQ on the 10-top codebook), and
# rotating only the down projection, with BF16 weights and with GPTQ. Waits for run_25. One job at a time. Usage: caffeinate -i -s zsh sim/run_26.sh
set -u
root="${0:A:h:h}"
cd "$root/sim"
r=../results
py=$root/.venv/bin/python
while pgrep -qf run_25.sh; do sleep 30; done
run() {  # repo method
  grep -qF "$1,$2," $r/26_ppl.csv 2>/dev/null && return
  echo "$(date '+%a %H:%M') start $1 $2" >> $r/26_queue.log
  $py -u eval_ppl.py "$1" $r/26_ppl.csv --methods "$2" --windows-dir $r/26_windows >> $r/26_ppl.log 2>&1
  echo "$(date '+%a %H:%M') end   $1 $2 (exit $?)" >> $r/26_queue.log
}
D="rp=mlp.down_proj"
for repo in Qwen/Qwen3-1.7B-Base Qwen/Qwen3-4B-Base HuggingFaceTB/SmolLM3-3B-Base Qwen/Qwen3-0.6B-Base; do
  run $repo "w=fmt:c10/B8/M1..12;a=8"
  run $repo "w=fmt:c9/B8/M1..14;a=8"
done
for repo in Qwen/Qwen3-1.7B-Base Qwen/Qwen3-4B-Base HuggingFaceTB/SmolLM3-3B-Base; do
  run $repo "w=bf16;a=8;r=128;$D"
  run $repo "w=bf16;a=8;r=32;$D"
done
for repo in Qwen/Qwen3-1.7B-Base Qwen/Qwen3-4B-Base HuggingFaceTB/SmolLM3-3B-Base; do
  run $repo "w=gptq:e2m1/B8/M1..10;a=8;r=128;$D"
  run $repo "w=gptq:s10/B8/M1..12;a=8"
  run $repo "w=gptq:s10/B8/M1..12;a=8;r=128;$D"
done
