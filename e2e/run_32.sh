#!/bin/zsh
# The rewritten rotation kernel (SIMD-lane transform, same outputs): correctness, a longer decode check (with plain 8-wide as control), perplexity, timing, time map. One job at a time;
# finished steps leave a marker in results/32_done/. Usage: caffeinate -i -s zsh e2e/run_32.sh    (progress: results/32_queue.log)
set -u
root="${0:A:h:h}"
cd "$root/e2e"
r=../results
PY=$root/.venv/bin/python
FMT=e2m1/B8/M1..10
mkdir -p $r/32_done
step() {
  [[ -f $r/32_done/$1 ]] && return
  echo "$(date '+%a %H:%M') start $1" >> $r/32_queue.log
  "${@:2}" > $r/32_$1.log 2>&1
  c=$?
  echo "$(date '+%a %H:%M') end   $1 (exit $c)" >> $r/32_queue.log
  [[ $c == 0 ]] && touch $r/32_done/$1
}
step check_rot $PY check_fused.py --packed rot:$FMT --out $r/32_check_rot.json
step check_rotdown $PY check_fused.py --packed rotdown:$FMT --out $r/32_check_rotdown.json
step decode_fmt $PY check_decode.py --model qwen3 --packed fmt:$FMT --steps 512 --out $r/32_check_decode_fmt.json
step decode_rot $PY check_decode.py --model qwen3 --packed rot:$FMT --steps 512 --out $r/32_check_decode_rot.json
step decode_rotdown $PY check_decode.py --model qwen3 --packed rotdown:$FMT --steps 512 --out $r/32_check_decode_rotdown.json
step ppl_rot $PY ppl_gptq_qwen3.py --methods "ours_rot:$FMT,ours_rotdown:$FMT" --out $r/32_ppl.csv
step timing ./run_timed.sh $r/32_timing_run.log zsh -c "$PY bench_format.py prefill --rot --rot-down --format $FMT --lengths 512,2048 --out $r/32_prefill_burst.csv && \
  $PY bench_format.py prefill --rot --rot-down --format $FMT --lengths 8192 --out $r/32_prefill_sustained.csv && \
  $PY bench_format.py decode --rot --rot-down --format $FMT --out $r/32_decode.csv"
step timemap ./run_timed.sh $r/32_timemap_run.log $PY time_map.py --configs fmt:$FMT,rot:$FMT --out $r/32_timemap.csv
step ablate ./run_timed.sh $r/32_ablate_run.log $PY ablate_rot_decode.py --ctx 2048 --out $r/32_ablate_decode_2048.csv
