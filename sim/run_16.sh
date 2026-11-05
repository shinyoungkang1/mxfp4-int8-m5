#!/bin/zsh
# The integer-exact format space (format_space.py): all block sizes, every third layer, three models. One job at a time.
# Usage: caffeinate -i -s zsh sim/run_16.sh     (progress: results/16_queue.log)
set -u
root="${0:A:h:h}"
cd "$root/sim"
r=../results
py=$root/.venv/bin/python
for repo in Qwen/Qwen3-1.7B-Base Qwen/Qwen3-4B-Base HuggingFaceTB/SmolLM3-3B-Base; do
  s=${repo:t}
  echo "$(date '+%a %H:%M') start $s" >> $r/16_queue.log
  $py -u format_space.py $repo $r/16_space_$s.csv --stride 3 > $r/16_space_$s.log 2>&1
  echo "$(date '+%a %H:%M') end   $s (exit $?)" >> $r/16_queue.log
done
