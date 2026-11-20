#!/bin/zsh
# Run A and Run B of paper/ICML_TODO.md "Schedule from 2026-11-05".
#   A: E2M1 8-wide on the Qwen3.5-35B kernels: pack (first use) and perplexity on the kernels against 5.887 in simulation (runs 18, 28), decode against
#      stepwise prefill, then prefill and decode speed for SC-MXFP4 and 8-wide in separate processes (the kernels' format is global), two sessions,
#      5 min pause before each session as run_18.sh.
#   B: rotation without GPTQ, E2M1 8-wide, on Qwen3-1.7B, Qwen3-4B and SmolLM3 in simulation.
# One job at a time; finished steps leave a marker in results/33_done/. Usage: caffeinate -i -s zsh e2e/run_33.sh    (progress: results/33_queue.log)
set -u
root="${0:A:h:h}"
cd "$root/e2e"
r=../results
PY=$root/.venv/bin/python
F8=e2m1/B8/M1..10
mkdir -p $r/33_done
step() {
  [[ -f $r/33_done/$1 ]] && return 0
  echo "$(date '+%a %H:%M') start $1" >> $r/33_queue.log
  "${@:2}" > $r/33_$1.log 2>&1
  c=$?
  echo "$(date '+%a %H:%M') end   $1 (exit $c)" >> $r/33_queue.log
  [[ $c == 0 ]] && touch $r/33_done/$1
  return $c
}
sim() { cd "$root/sim"; $PY -u eval_ppl.py "$1" $r/33_ppl_sim.csv --methods "w=fmt:$F8;a=8;r=32" --windows-dir $r/33_windows; local c=$?; cd "$root/e2e"; return $c; }
timing() {
  local rc=0
  for s in 1 2; do
    sleep 300
    ./run_timed.sh $r/33_decode_sc_s$s.log $PY bench_qwen35.py decode --out $r/33_decode_qwen35_sc_s$s.csv || rc=1
    ./run_timed.sh $r/33_decode_b8_s$s.log $PY bench_qwen35.py decode --format $F8 --out $r/33_decode_qwen35_b8_s$s.csv || rc=1
    sleep 300
    ./run_timed.sh $r/33_prefill_sc_s$s.log $PY bench_qwen35.py prefill --out $r/33_prefill_qwen35_sc_s$s.csv || rc=1
    ./run_timed.sh $r/33_prefill_b8_s$s.log $PY bench_qwen35.py prefill --format $F8 --out $r/33_prefill_qwen35_b8_s$s.csv || rc=1
  done
  return $rc
}
step ppl35 $PY -u ppl_qwen35.py --methods "ours_fmt:$F8" --out $r/33_ppl_qwen35.csv --windows-dir $r/33_windows || exit 1
step decode35 $PY check_decode.py --model qwen35 --packed fmt:$F8 --steps 256 --out $r/33_check_decode_qwen35.json
step sim17 sim Qwen/Qwen3-1.7B-Base
step sim4b sim Qwen/Qwen3-4B-Base
step simsmol sim HuggingFaceTB/SmolLM3-3B-Base
step timing timing
