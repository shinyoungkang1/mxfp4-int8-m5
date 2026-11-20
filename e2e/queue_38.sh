#!/bin/zsh
# Run 38 = PLAN.md step B3, continued: the crossover of ours against MLX lies above 32 tokens per step (run 37 smoke:
# 0.61x at 8 tokens), so steps of 32 to 512 tokens, both formats. Synchronous steps as in run 37; anchor check off
# (no reference for this protocol), idle-GPU gate on.
# Usage: caffeinate -i -s zsh e2e/queue_38.sh      (progress: results/38_queue.log)
source "${0:A:h}/lib_queue.sh"
queue_init 38
export VALIDATE=""
T=32,48,64,96,128,192,256,384,512
timed b3_b8_long $PY bench_tokens_qwen35.py --format e2m1/B8/M1..10 --tokens $T --reps 15 --out $R/38_tokens_qwen35_b8.csv
timed b3_sc_long $PY bench_tokens_qwen35.py --format scmxfp4 --tokens $T --reps 15 --out $R/38_tokens_qwen35_sc.csv
# 35B prefill re-timing (audit H2): both formats in one process, three methods, two sessions. In run 37 MLX's own
# 512-token time was 16-43% above its reference in both tries with an idle GPU (two packs and two MLX models loaded),
# so 2048 / 8192 here, and 512 in a session of its own with one format and at most two tries.
unset VALIDATE
timed pre35_s1 $PY bench_qwen35.py prefill --format scmxfp4,e2m1/B8/M1..10 --lengths 2048,8192 --out $R/38_prefill_qwen35_s1.csv
timed pre35_s2 $PY bench_qwen35.py prefill --format scmxfp4,e2m1/B8/M1..10 --lengths 2048,8192 --out $R/38_prefill_qwen35_s2.csv
export MAX_TRIES=2
timed pre35_512 $PY bench_qwen35.py prefill --format e2m1/B8/M1..10 --lengths 512 --out $R/38_prefill_qwen35_512.csv
unset MAX_TRIES
# Saved output for the ladder bound numbers of paper section 3.2 (1.18 for m <= 10, 2.15 for powers of two)
step ladder $PY ../sim/ladder_bound.py Qwen/Qwen3-1.7B-Base $R/16_space_Qwen3-1.7B-Base.csv
_q_log "queue 38 done"
