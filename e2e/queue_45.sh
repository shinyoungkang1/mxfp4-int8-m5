#!/bin/zsh
# Run 45: the in-kernel INT8 prologue at larger threadgroups beyond the 35B (runs 43-44).
#  - Qwen3-1.7B (o and down) and OLMoE (o): logits identity, then decode vs MLX MXFP4 (tune_prologue_small.py).
#  - Qwen3.8-Flash-Next 125B, SC-MXFP4 and 8-wide packs (one process each, 66 GB): dense projections with MLX's GEMV
#    (SC only), ours at the default tiles, ours tuned (qkvz/qkv prologue at 8 simdgroups, output projections with a
#    separate act_quant); logits of the two "ours" checked equal (tune_decode_q38.py --dense-tuned).
# Usage: caffeinate -i -s zsh e2e/queue_45.sh      (progress: results/45_queue.log)
source "${0:A:h}/lib_queue.sh"
queue_init 45
Q38=$Q_ROOT/.venv-q38/bin/python
timed pro_qwen3 $PY tune_prologue_small.py --model qwen3 --out $R/45_prologue_qwen3.csv
# OLMoE dropped: its base model, pack and MLX conversions are no longer on disk (a ~14 GB download to restore)
export VALIDATE=""
timed q38_sc zsh -c "$Q38 tune_decode_q38.py --dense-tuned --format sc --tokens 48 --reps 6 > $R/45_dense_q38_sc.txt 2>&1"
timed q38_b8 zsh -c "$Q38 tune_decode_q38.py --dense-tuned --format e2m1/B8/M1..10 --tokens 48 --reps 6 > $R/45_dense_q38_b8.txt 2>&1"
unset VALIDATE
_q_log "queue 45 done"
