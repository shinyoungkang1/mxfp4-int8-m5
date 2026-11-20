#!/bin/zsh
# Run 46: the fused o/down prologue (act_tile (4, 16, 16, 1)) as DecodeQwen3's default (run 45: 0.978x -> 0.989x).
#  1. check_decode on Qwen3-1.7B with the new default (decode logits vs the prefill kernels one token at a time).
#  2. Table 3's Qwen3-1.7B decode row re-timed as run 41 (ours, MLX, llama.cpp back to back under one GPU lock).
#  3. Qwen3.8-Flash-Next dense tiles again, with two untimed full passes per arm (run 45's first sessions drifted).
# Usage: caffeinate -i -s zsh e2e/queue_46.sh      (progress: results/46_queue.log)
source "${0:A:h}/lib_queue.sh"
queue_init 46
D=$HOME/.cache/mxfp4-int8-m5
Q38=$Q_ROOT/.venv-q38/bin/python
step check_qwen3 $PY check_decode.py --model qwen3 --out $R/46_check_decode_qwen3.json || exit 1
timed dec_qwen3 zsh -c "$PY bench_decode.py --model qwen3 --cache-dir $D --ctx 512,2048 --gen 128 --methods mlx_mxfp4,mlx_affine4,ours --out $R/46_decode_session_qwen3.csv && \
  $PY bench_llamacpp.py --model qwen3 --decode-only --out $R/46_decode_session_llamacpp_qwen3.csv"
export VALIDATE=""
timed q38_sc zsh -c "$Q38 tune_decode_q38.py --dense-tuned --format sc --tokens 48 --reps 6 > $R/46_dense_q38_sc.txt 2>&1"
timed q38_b8 zsh -c "$Q38 tune_decode_q38.py --dense-tuned --format e2m1/B8/M1..10 --tokens 48 --reps 6 > $R/46_dense_q38_b8.txt 2>&1"
unset VALIDATE
_q_log "queue 46 done"
