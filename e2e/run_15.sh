#!/bin/zsh
# Qwen3.8 decode after the gate|up layout fix (fused_qwen38: np.concatenate(...)[:, perm] gave strided bcodes / wscale, which
# every Metal kernel call copied whole, 4x more bytes in the 16-block format). Strictly one job at a time; finished steps
# leave a marker in results/15_done/. Needs the 16-block pack (fused_qwen38.py --pack e2m1/B16/M1..10).
#   1. breakdown, old layout vs fixed, SC-MXFP4 and 16-block                                         ~20 min
#   2. timing: ours SC-MXFP4, ours 16-block, MLX MXFP4; 2 sessions each, alternating                 ~45 min
# Usage: caffeinate -i -s zsh e2e/run_15.sh     (progress: results/15_queue.log)
set -u
root="${0:A:h:h}"
cd "$root/e2e"
r=../results
q38=$root/.venv-q38/bin/python
F16="e2m1/B16/M1..10"
mkdir -p $r/15_done

log() { echo "$(date '+%a %H:%M') $*" >> $r/15_queue.log }
step() {  # step <name> <function>: skipped if done before; logs start, end and exit code
  local name=$1 fn=$2
  if [[ -e $r/15_done/$name ]]; then log "skip  $name (done before)"; return 0; fi
  log "start $name"
  $fn; local rc=$?
  log "end   $name (exit $rc)"
  [[ $rc == 0 ]] && touch $r/15_done/$name
  return $rc
}

breakdown() {
  local rc=0
  for f in sc $F16; do
    local s=${f//\//_}
    ./run_timed.sh $r/15_breakdown_${s}_old.txt $q38 tune_decode_q38.py --format $f --breakdown-only --old-layout || rc=1
    ./run_timed.sh $r/15_breakdown_${s}_fixed.txt $q38 tune_decode_q38.py --format $f --breakdown-only || rc=1
  done
  return $rc
}
timing() {
  local rc=0
  for s in 1 2; do
    for m in fused fused_b16 mlx_mxfp4; do
      ./run_timed.sh $r/15_prefill_q38_${m}_s$s.log $q38 bench_q38.py --method $m --session $s --reps 5 --decode 64 \
        --out $r/15_prefill_q38.csv || { log "      timing $m session $s failed"; rc=1; }
    done
  done
  return $rc
}

timing2() {  # 2026-11-03: session 1 of the 16-block format had an outlier at 8192 tokens, so two more of each
  local rc=0
  for s in 3 4; do
    for m in fused_b16 mlx_mxfp4 fused; do
      ./run_timed.sh $r/15_prefill_q38_${m}_s$s.log $q38 bench_q38.py --method $m --session $s --reps 5 --decode 64 \
        --out $r/15_prefill_q38.csv || { log "      timing $m session $s failed"; rc=1; }
    done
  done
  return $rc
}
check16() {  # kernels vs. reference and decode vs. prefill on the rebuilt pack and fixed layout (as run_11's check16)
  $q38 q38_ppl_kernels.py --format $F16 --windows 8 --decode-check 64 --out $r/15_check_b16.csv > $r/15_check_b16.log 2>&1
}

timing3() {  # sessions 1-4 were bimodal: a session right after another process freed 60+ GB ran up to 1.6x slow in prefill.
  # So a 5 min pause before every session (the clean table; sessions 1-4 stay in the CSV, labelled by session number)
  local rc=0
  for s in 5 6; do
    for m in mlx_mxfp4 fused_b16 fused; do
      sleep 300
      ./run_timed.sh $r/15_prefill_q38_${m}_s$s.log $q38 bench_q38.py --method $m --session $s --reps 5 --decode 64 \
        --out $r/15_prefill_q38.csv || { log "      timing $m session $s failed"; rc=1; }
    done
  done
  return $rc
}

log "queue started (pid $$)"
step breakdown breakdown
step timing timing
step timing2 timing2
step check16 check16
step timing3 timing3
log "queue done"
