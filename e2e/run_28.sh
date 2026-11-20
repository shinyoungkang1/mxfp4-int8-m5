#!/bin/zsh
# Rotation on Qwen3.5-35B (no GPTQ): every W4A8 matrix including the expert matrices, and the expert and shared-expert down projections only. One job at a time,
# each row in its own process. Usage: caffeinate -i -s zsh e2e/run_28.sh     (progress: results/28_queue.log)
set -u
root="${0:A:h:h}"
cd "$root/e2e"
r=../results
py=$root/.venv/bin/python
for m in b8_w4a8_r32 b8_w4a8_base bf16_w4a8 bf16_w4a8_r32 bf16_w4a8_r32_down b8_w4a8_r32_down; do
  grep -qF ",$m," $r/28_ppl_qwen35.csv 2>/dev/null && continue
  echo "$(date '+%a %H:%M') start $m" >> $r/28_queue.log
  $py -u ppl_qwen35.py --methods $m --out $r/28_ppl_qwen35.csv --windows-dir $r/28_windows >> $r/28_ppl_qwen35.log 2>&1
  echo "$(date '+%a %H:%M') end   $m (exit $?)" >> $r/28_queue.log
done
