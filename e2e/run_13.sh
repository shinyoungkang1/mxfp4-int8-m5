#!/bin/zsh
# Re-timing after the scale-layout fix (fused_qwen38.mxfp4_scales returned column-major E8M0 scales, which MLX's
# quantized kernels copied whole on every call): the Qwen3.8 decode numbers that ran MLX's kernels on our SC-MXFP4
# bytes. Strictly one job at a time; finished steps leave a marker in results/13_done/ (a restarted queue skips them).
# About 50 min.
#   1. timing (run_timed.sh): ours (SC-MXFP4; its dense projections decode with MLX's GEMV on our bytes), MLX's
#      kernels on our SC-MXFP4 bytes (hybrid decode), MLX MXFP4; 2 sessions each, alternating            ~45 min
#   2. timing: dense projections, our W4A8 GEMV vs MLX's MXFP4 GEMV on the same bytes                    ~5 min
# Usage: caffeinate -i -s zsh e2e/run_13.sh     (progress: results/13_queue.log)
set -u
root="${0:A:h:h}"
cd "$root/e2e"
r=../results
q38=$root/.venv-q38/bin/python
mkdir -p $r/13_done

log() { echo "$(date '+%a %H:%M') $*" >> $r/13_queue.log }
step() {  # step <name> <function>: skipped if done before; logs start, end and exit code
  local name=$1 fn=$2
  if [[ -e $r/13_done/$name ]]; then log "skip  $name (done before)"; return 0; fi
  log "start $name"
  $fn; local rc=$?
  log "end   $name (exit $rc)"
  [[ $rc == 0 ]] && touch $r/13_done/$name
  return $rc
}

timing_q38() {
  local rc=0 s m
  for s in 1 2; do
    for m in fused mlx_sc_mxfp4 mlx_mxfp4; do
      ./run_timed.sh $r/13_prefill_q38_${m}_s$s.log $q38 bench_q38.py --method $m --session $s --reps 5 --decode 64 \
        --out $r/13_prefill_q38.csv || { log "      timing $m session $s failed"; rc=1; }
    done
  done
  return $rc
}
dense() { ./run_timed.sh $r/13_tune_decode_q38_dense.txt $q38 tune_decode_q38.py --dense-only }

log "queue started (pid $$)"
step timing_q38 timing_q38
step dense dense
log "queue done"
