#!/bin/zsh
# Perplexity with per-window losses (paired bootstrap, sim/bootstrap_ppl.py): the main formats and the finalists of format_space.py,
# W4A8, four small models. Usage: caffeinate -i -s zsh sim/run_17.sh     (progress: results/17_queue.log)
set -u
root="${0:A:h:h}"
cd "$root/sim"
r=../results
py=$root/.venv/bin/python
M="bf16,w=sc_mx;a=8,w=nvfp4;a=8,w=nvfp4_search;a=8,w=fmt:e2m1/B16/M1..10;a=8,w=fmt:e2m1/B8/M1..10;a=8,w=fmt:s10/B8/M1..12;a=8,w=fmt:s10/B16/M1..12;a=8,w=fmt:s14/B16/M1..9;a=8,w=fmt:e2m1/B32/M1..8;a=8"
for repo in Qwen/Qwen3-1.7B-Base Qwen/Qwen3-4B-Base Qwen/Qwen3-0.6B-Base HuggingFaceTB/SmolLM3-3B-Base; do
  echo "$(date '+%a %H:%M') start $repo" >> $r/17_queue.log
  $py -u eval_ppl.py $repo $r/17_ppl.csv --methods "$M" --windows-dir $r/17_windows > $r/17_ppl_${repo:t}.log 2>&1
  echo "$(date '+%a %H:%M') end   $repo (exit $?)" >> $r/17_queue.log
done
