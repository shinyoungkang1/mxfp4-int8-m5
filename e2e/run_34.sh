#!/bin/zsh
# Re-check of the 35B prefill at 512 tokens: E2M1 8-wide ran at 0.81x of MLX in run 33 (median 174 ms, min 109 ms) against 1.16-1.26x for SC-MXFP4.
# More repetitions, both formats, each in its own process, two orders. Usage: caffeinate -i -s zsh e2e/run_34.sh
set -u
root="${0:A:h:h}"
cd "$root/e2e"
r=../results
PY=$root/.venv/bin/python
F8=e2m1/B8/M1..10
echo "$(date '+%a %H:%M') start" >> $r/34_queue.log
./run_timed.sh $r/34_b8_a.log $PY bench_qwen35.py prefill --format $F8 --lengths 512,1024 --reps 15 --out $r/34_prefill_qwen35_b8_a.csv
./run_timed.sh $r/34_sc_a.log $PY bench_qwen35.py prefill --lengths 512,1024 --reps 15 --out $r/34_prefill_qwen35_sc_a.csv
./run_timed.sh $r/34_b8_b.log $PY bench_qwen35.py prefill --format $F8 --lengths 512,1024 --reps 15 --out $r/34_prefill_qwen35_b8_b.csv
echo "$(date '+%a %H:%M') end" >> $r/34_queue.log
