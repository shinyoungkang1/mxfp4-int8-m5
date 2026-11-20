#!/bin/zsh
# Run 40 = the KL part of PLAN.md step B5 (deferred from run 37): per-token KL divergence and top-1 flips against the
# FP8 model, 32 windows, for 8-wide everywhere and the three mixed assignments that matter for decision D6.
# Usage: caffeinate -i -s zsh e2e/queue_40.sh      (progress: results/40_queue.log)
source "${0:A:h}/lib_queue.sh"
queue_init 40
F8="fmt:e2m1/B8/M1..10"
F16="fmt:e2m1/B16/M1..10"
Q38=$Q_ROOT/.venv-q38/bin/python
step b5_kl $Q38 q38_ppl.py --windows 32 --out $R/40_kl_qwen38.csv \
  --methods "fp8,w=$F8;a=8,w=$F16;a=8;wd=$F8,w=$F16;a=8;wd=int8_row,w=sc_mx;a=8;wd=$F8"
_q_log "queue 40 done"
