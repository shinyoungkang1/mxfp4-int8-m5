#!/bin/zsh
# Code D of paper/ICML_TODO.md "Schedule from 2026-11-05": rotation (r=32, no GPTQ) on the Qwen3.5-35B kernels, then Run C (zero-shot).
#  1. regression: plain 8-wide perplexity and decode check after the change (must match run 33: 5.8892, decode = stepwise prefill)
#  2. rotated pack (first use), perplexity against 5.8496 in simulation (run 28), decode check, prefill and decode speed against plain 8-wide
#  3. Run C: zero-shot, E2M1 8-wide on 35B (as run 18, --limit 1000), then the rest of e2e/run_30.sh (skips finished runs)
# One job at a time; finished steps leave a marker in results/35_done/. Usage: caffeinate -i -s zsh e2e/run_35.sh    (progress: results/35_queue.log)
set -u
root="${0:A:h:h}"
cd "$root/e2e"
r=../results
PY=$root/.venv/bin/python
EV=$HOME/.cache/mxfp4-int8-m5/venv-eval/bin/python
F8=e2m1/B8/M1..10
mkdir -p $r/35_done
step() {
  [[ -f $r/35_done/$1 ]] && return 0
  echo "$(date '+%a %H:%M') start $1" >> $r/35_queue.log
  "${@:2}" > $r/35_$1.log 2>&1
  c=$?
  echo "$(date '+%a %H:%M') end   $1 (exit $c)" >> $r/35_queue.log
  [[ $c == 0 ]] && touch $r/35_done/$1
  return $c
}
timing() {
  local rc=0
  sleep 300
  ./run_timed.sh $r/35_decode_rot.log $PY bench_qwen35.py decode --format $F8 --rot --out $r/35_decode_qwen35_rot.csv || rc=1
  ./run_timed.sh $r/35_decode_b8.log $PY bench_qwen35.py decode --format $F8 --out $r/35_decode_qwen35_b8.csv || rc=1
  sleep 300
  ./run_timed.sh $r/35_prefill_rot.log $PY bench_qwen35.py prefill --format $F8 --rot --out $r/35_prefill_qwen35_rot.csv || rc=1
  ./run_timed.sh $r/35_prefill_b8.log $PY bench_qwen35.py prefill --format $F8 --out $r/35_prefill_qwen35_b8.csv || rc=1
  return $rc
}
step reg_ppl $PY -u ppl_qwen35.py --methods "ours_fmt:$F8" --out $r/35_ppl_qwen35.csv || exit 1
step reg_decode $PY check_decode.py --model qwen35 --packed fmt:$F8 --steps 256 --out $r/35_check_decode_qwen35_b8.json || exit 1
step rot_ppl $PY -u ppl_qwen35.py --methods "ours_rot:$F8" --out $r/35_ppl_qwen35.csv --windows-dir $r/35_windows || exit 1
step rot_decode $PY check_decode.py --model qwen35 --packed rot:$F8 --steps 256 --out $r/35_check_decode_qwen35_rot.json
step timing timing
step tasks35 $EV eval_tasks.py --model qwen35 --method ours_fmt --format $F8 --limit 1000 --out $r/35_tasks_qwen35_b8.json
step run30 zsh run_30.sh
