#!/bin/zsh
# Run 39: the 512-token 35B prefill cell of the paper's speed table for SC-MXFP4, in a session of its own (run 38 timed
# 8-wide alone at 512; with two formats and four models loaded, 512-token times were unstable, run 37).
# Usage: caffeinate -i -s zsh e2e/queue_39.sh      (progress: results/39_queue.log)
source "${0:A:h}/lib_queue.sh"
queue_init 39
export MAX_TRIES=2
timed pre35_512_sc $PY bench_qwen35.py prefill --format scmxfp4 --lengths 512 --out $R/39_prefill_qwen35_512_sc.csv
unset MAX_TRIES
_q_log "queue 39 done"
