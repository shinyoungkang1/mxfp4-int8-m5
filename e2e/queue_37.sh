#!/bin/zsh
# Run 37 = PLAN.md steps B3 (tokens per step against MLX), B4 (35B GEMV microbenchmark), B6 (activation sites and
# outlier bypass on the dense models), B5 (125B mixed assignments: perplexity, then KL against FP8).
# Short steps first; the 125B passes last (about 3 to 4 hours).
# Usage: caffeinate -i -s zsh e2e/queue_37.sh      (progress: results/37_queue.log)
source "${0:A:h}/lib_queue.sh"
queue_init 37
F8="fmt:e2m1/B8/M1..10"
F16="fmt:e2m1/B16/M1..10"
Q38=$Q_ROOT/.venv-q38/bin/python

# B3. Synchronous steps of T tokens; no MLX reference exists for this protocol, so the anchor check is off
# (the idle-GPU gate before and after still applies).
export VALIDATE=""
step  b3_smoke $PY bench_tokens_qwen35.py --tokens 1,8 --ctx 512 --reps 2 --methods mlx_mxfp4,ours
if [[ -f $R/37_done/b3_smoke ]]; then
  timed b3_b8 $PY bench_tokens_qwen35.py --format e2m1/B8/M1..10 --both-paths --out $R/37_tokens_qwen35_b8.csv
  timed b3_sc $PY bench_tokens_qwen35.py --format scmxfp4 --out $R/37_tokens_qwen35_sc.csv
fi
# Re-timing the 35B speed table (audit H2/M4, PLAN.md F1 for the 35B): SC-MXFP4 and 8-wide in one process, the same
# three methods, the pause and the anchor check; prefill in two sessions.
unset VALIDATE
timed pre35_s1 $PY bench_qwen35.py prefill --format scmxfp4,e2m1/B8/M1..10 --out $R/37_prefill_qwen35_s1.csv
timed pre35_s2 $PY bench_qwen35.py prefill --format scmxfp4,e2m1/B8/M1..10 --out $R/37_prefill_qwen35_s2.csv
timed dec35 $PY bench_qwen35.py decode --format scmxfp4,e2m1/B8/M1..10 --out $R/37_decode_qwen35.csv
export VALIDATE=""
# B4. Synthetic weights, no model load.
step  b4_smoke $PY bench_gemv_qwen35.py --shapes gu,down
[[ -f $R/37_done/b4_smoke ]] && timed b4 $PY bench_gemv_qwen35.py --sweep --out $R/37_gemv_qwen35.csv
unset VALIDATE

# B6. Activations on the dense models (simulation, 145-146 windows, windows saved for pairing).
#   per-site INT8 activations on Qwen3-4B (Qwen3-1.7B has them: results/24_ppl.csv), then outlier bypass.
for p in self_attn.q_proj self_attn.k_proj self_attn.v_proj self_attn.o_proj mlp.gate_proj mlp.up_proj mlp.down_proj; do
  step "b6_4b_${p//./_}" $PY ../sim/eval_ppl.py Qwen/Qwen3-4B-Base $R/37_ppl_dense.csv --methods "w=bf16;a=8;r=0;p=$p"
done
for m in Qwen/Qwen3-1.7B-Base Qwen/Qwen3-4B-Base; do
  t=${${m:t}%-Base}
  step "b6_${t}_bypass" $PY ../sim/eval_ppl.py $m $R/37_ppl_dense.csv --windows-dir $R/37_windows \
    --methods "w=bf16;a=8,w=bf16;a=8;o=1,w=bf16;a=8;o=2,w=bf16;a=8;o=4,w=$F8;a=8,w=$F8;a=8;o=2,w=$F8;a=8;o=4"
done

# B5. 125B (plus NVFP4 with search on every matrix, windows saved, for a paired interval against 8-wide). First the regression on 8 windows: the spec without wd and the same spec with wd = w must both give
# 2.4419 (results/30_check_q38.csv) before the mixed rows are trusted.
step b5_reg $Q38 q38_ppl.py --windows 8 --methods "w=$F8;a=8,w=$F8;a=8;wd=$F8" --out $R/37_check_q38.csv
step b5_reg_check $PY -c "
import csv, sys
v = {r['method']: r['ppl'] for r in csv.DictReader(open('$R/37_check_q38.csv'))}
print(v); sys.exit(0 if all(abs(float(v.get(k, 0)) - 2.4419) < 6e-5 for k in ('w=$F8;a=8', 'w=$F8;a=8;wd=$F8')) else 1)"
if [[ -f $R/37_done/b5_reg_check ]]; then
  step b5_ppl $Q38 q38_ppl.py --out $R/37_ppl_qwen38.csv \
    --methods "w=sc_mx;a=8;wd=$F8,w=$F16;a=8;wd=$F8,w=sc_mx;a=8;wd=int8_row,w=$F16;a=8;wd=int8_row,w=nvfp4_search;a=8"
  step b5_kl $Q38 q38_ppl.py --windows 32 --out $R/37_kl_qwen38.csv \
    --methods "fp8,w=$F8;a=8,w=sc_mx;a=8;wd=$F8,w=$F16;a=8;wd=$F8,w=sc_mx;a=8;wd=int8_row,w=$F16;a=8;wd=int8_row"
fi
_q_log "queue 37 done"
