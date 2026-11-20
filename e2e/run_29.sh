#!/bin/zsh
# Only the down projection rotated (r=32) on the kernels, Qwen3-1.7B, 8-wide format with GPTQ: as run_25.sh, against "w=gptq:e2m1/B8/M1..10;a=8;r=32;rp=mlp.down_proj". Also that
# method in simulation on the three small models. Waits for run_28. One job at a time; finished steps leave a marker in results/29_done/.
# Usage: caffeinate -i -s zsh e2e/run_29.sh
set -u
root="${0:A:h:h}"
cd "$root/e2e"
r=../results
PY=$root/.venv/bin/python
EV=$HOME/.cache/mxfp4-int8-m5/venv-eval/bin/python
FMT=e2m1/B8/M1..10
mkdir -p $r/29_done
while pgrep -qf run_28.sh; do sleep 30; done
step() {  # name command...
  [[ -f $r/29_done/$1 ]] && return
  echo "$(date '+%a %H:%M') start $1" >> $r/29_queue.log
  "${@:2}" > $r/29_$1.log 2>&1
  c=$?
  echo "$(date '+%a %H:%M') end   $1 (exit $c)" >> $r/29_queue.log
  [[ $c == 0 ]] && touch $r/29_done/$1
}
sim() { cd "$root/sim"; $PY -u eval_ppl.py "$1" $r/29_ppl_sim.csv --methods "w=gptq:$FMT;a=8;r=32;rp=mlp.down_proj" --windows-dir $r/29_windows; local c=$?; cd "$root/e2e"; return $c; }
step pack $PY pack_rot.py --down-only --format $FMT --workers 4 --out $r/29_pack.json
step check $PY check_fused.py --packed rotdown:$FMT --out $r/29_check.json
step check_decode $PY check_decode.py --model qwen3 --packed rotdown:$FMT --out $r/29_check_decode.json
step ppl $PY ppl_gptq_qwen3.py --methods "ours_rotdown:$FMT" --out $r/29_ppl.csv
step sim17 sim Qwen/Qwen3-1.7B-Base
step timing ./run_timed.sh $r/29_timing_run.log zsh -c "$PY bench_format.py prefill --rot --rot-down --format $FMT --lengths 512,2048 --out $r/29_prefill_burst.csv && \
  $PY bench_format.py prefill --rot --rot-down --format $FMT --lengths 8192 --out $r/29_prefill_sustained.csv && \
  $PY bench_format.py decode --rot --rot-down --format $FMT --out $r/29_decode.csv"
step timemap ./run_timed.sh $r/29_timemap_run.log $PY time_map.py --configs fmt:$FMT,rotdown:$FMT --out $r/29_timemap.csv
step tasks $EV eval_tasks.py --model qwen3 --method ours_rotdown --format $FMT --out $r/29_tasks.json
step sim4b sim Qwen/Qwen3-4B-Base
step simsmol sim HuggingFaceTB/SmolLM3-3B-Base
