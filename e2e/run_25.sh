#!/bin/zsh
# Rotations on the kernels, Qwen3-1.7B, 8-wide format with GPTQ: correctness against eval_ppl's "w=gptq:e2m1/B8/M1..10;a=8;r=32", decode identity, full-test
# perplexity on the kernels, prefill and decode against the unrotated format and MLX MXFP4 under one GPU lock, then zero-shot tasks. One job at a time;
# finished steps leave a marker in results/25_done/. Usage: caffeinate -i -s zsh e2e/run_25.sh
set -u
root="${0:A:h:h}"
cd "$root/e2e"
r=../results
PY=$root/.venv/bin/python
EV=$HOME/.cache/mxfp4-int8-m5/venv-eval/bin/python
FMT=e2m1/B8/M1..10
mkdir -p $r/25_done
step() {  # name command...
  [[ -f $r/25_done/$1 ]] && return
  echo "$(date '+%a %H:%M') start $1" >> $r/25_queue.log
  "${@:2}" > $r/25_$1.log 2>&1
  c=$?
  echo "$(date '+%a %H:%M') end   $1 (exit $c)" >> $r/25_queue.log
  [[ $c == 0 ]] && touch $r/25_done/$1
}
step pack $PY pack_rot.py --format $FMT --workers 4 --out $r/25_pack.json
step check $PY check_fused.py --packed rot:$FMT --out $r/25_check.json
step check_decode $PY check_decode.py --model qwen3 --packed rot:$FMT --out $r/25_check_decode.json
step ppl $PY ppl_gptq_qwen3.py --methods "ours_rot:$FMT" --out $r/25_ppl.csv
step timing ./run_timed.sh $r/25_timing_run.log zsh -c "$PY bench_format.py prefill --rot --format $FMT --lengths 512,2048 --out $r/25_prefill_burst.csv && \
  $PY bench_format.py prefill --rot --format $FMT --lengths 8192 --out $r/25_prefill_sustained.csv && \
  $PY bench_format.py decode --rot --format $FMT --out $r/25_decode.csv"
step tasks $EV eval_tasks.py --model qwen3 --method ours_rot --format $FMT --out $r/25_tasks.json
