#!/bin/zsh
# Further experiments for the paper (results/12_*), strictly one job at a time; a failing step is logged and the queue
# goes on; finished steps leave a marker in results/12_done/ (a restarted queue skips them). About 6-7 h.
#   1. Qwen3.8, 8 windows, DeltaNet recurrence vs Metal kernel: fake quantization (FP8, SC-MXFP4, 16-block, MLX MXFP4)
#      with each, and our kernels with each (SC-MXFP4, 16-block)                                             ~1.5 h
#   2. Qwen3.8 exponent spread with maximum spread and relative weight MSE (Table 12)                     ~1-1.5 h
#   3. NVFP4 W4A8 with scale search on OLMoE-1B-7B; NVFP4 W4A8 with and without search on Qwen3.5-35B-A3B  ~1 h
#   4. Table 1: NVFP4 with scale search and MXFP4 with exponent search (layer-output error, Qwen3-1.7B)     ~20 min
#   5. llama.cpp perplexity (BF16, Q4_0, Q4_K_M, IQ4_XS, MXFP4_MOE) on OLMoE-1B-7B and Qwen3.5-35B-A3B      ~40 min
#   6. timing (run_timed.sh): Qwen3.8 MLX's kernels on our SC-MXFP4 bytes (hybrid decode), MLX affine INT4, and
#      MLX MXFP4, 2 sessions each, alternating                                                            ~50 min
#   7. timing: Qwen3.8 decode profile, SC-MXFP4 and 16-block                                              ~20 min
#   8. timing: linear layers with one timing method (MLX and ours, wall clock)                             ~5 min
#   9. timing: llama.cpp Q4_0, Q4_K_M, IQ4_XS prefill and decode on the three models                      ~30 min
# Usage: caffeinate -i -s zsh e2e/run_12.sh     (progress: results/12_queue.log)
set -u
root="${0:A:h:h}"
cd "$root/e2e"
r=../results
q38=$root/.venv-q38/bin/python
base=$root/.venv/bin/python
B16="e2m1/B16/M1..10"
mkdir -p $r/12_done

log() { echo "$(date '+%a %H:%M') $*" >> $r/12_queue.log }
step() {  # step <name> <function>: skipped if done before; logs start, end and exit code
  local name=$1 fn=$2
  if [[ -e $r/12_done/$name ]]; then log "skip  $name (done before)"; return 0; fi
  log "start $name"
  $fn; local rc=$?
  log "end   $name (exit $rc)"
  [[ $rc == 0 ]] && touch $r/12_done/$name
  return $rc
}

deltanet() {
  local rc=0
  $q38 q38_ppl.py --windows 8 --methods "fp8,sc_mxfp4_w4a8,w=fmt:$B16;a=8,mxfp4_spec_w4a16" \
    --out $r/12_q38_dn_ref.csv > $r/12_q38_dn_ref.log 2>&1 || rc=1
  $q38 q38_ppl.py --windows 8 --deltanet-kernel --methods "fp8,sc_mxfp4_w4a8,w=fmt:$B16;a=8,mxfp4_spec_w4a16" \
    --out $r/12_q38_dn_metal.csv > $r/12_q38_dn_metal.log 2>&1 || rc=1
  for fmt in sc $B16; do
    $q38 q38_ppl_kernels.py --format $fmt --windows 8 --out $r/12_q38_dn_kernels.csv >> $r/12_q38_dn_kernels.log 2>&1 || rc=1
    $q38 q38_ppl_kernels.py --format $fmt --windows 8 --reference-deltanet --out $r/12_q38_dn_kernels.csv \
      >> $r/12_q38_dn_kernels.log 2>&1 || rc=1
  done
  return $rc
}
spread() { $q38 q38_spread.py > $r/12_spread_qwen38.md 2> $r/12_spread_qwen38.log }
nvfp4_search() {
  (cd ../sim && $base eval_ppl.py allenai/OLMoE-1B-7B-0924 ../results/12_ppl_small.csv \
     --model-config '{"rms_norm_eps": 1e-5}' --methods "w=nvfp4_search;a=8") > $r/12_ppl_small.log 2>&1 &&
  $base ppl_qwen35.py --methods nvfp4_w4a8,nvfp4_search_w4a8 --out $r/12_ppl_qwen35.csv > $r/12_ppl_qwen35.log 2>&1
}
table1() {
  (cd ../sim && $base format_sweep.py Qwen/Qwen3-1.7B-Base ../results/12_formats_search.csv 3 \
     "NVFP4, scale search;MXFP4, exponent search;NVFP4;MXFP4 (OCP rule)") > $r/12_formats_search.md 2> $r/12_formats_search.log
}
llama_ppl() { $base ppl_llamacpp.py --models qwen3,olmoe,qwen35 --out $r/12_ppl_llamacpp.csv > $r/12_ppl_llamacpp.log 2>&1 }
timing_q38_mlx() {
  local rc=0 s m
  for s in 1 2; do
    for m in mlx_sc_mxfp4 mlx_affine4 mlx_mxfp4; do
      ./run_timed.sh $r/12_prefill_q38_${m}_s$s.log $q38 bench_q38.py --method $m --session $s --reps 5 --decode 64 \
        --out $r/12_prefill_q38.csv || { log "      timing $m session $s failed"; rc=1; }
    done
  done
  return $rc
}
decode_profile() {
  ./run_timed.sh $r/12_tune_decode_q38_sc.txt $q38 tune_decode_q38.py --format sc --breakdown-only &&
  ./run_timed.sh $r/12_tune_decode_q38_b16.txt $q38 tune_decode_q38.py --format $B16 --breakdown-only
}
layer_wallclock() { ./run_timed.sh $r/12_layer_wallclock.csv $base ../bench/layer_wallclock.py }
llama_speed() {
  local rc=0 m
  for m in qwen3 olmoe qwen35; do
    ./run_timed.sh $r/12_llamacpp_$m.log $base bench_llamacpp.py --model $m --quants Q4_0,Q4_K_M,IQ4_XS \
      --out $r/12_llamacpp_$m.csv || { log "      llama.cpp timing $m failed"; rc=1; }
  done
  return $rc
}

log "queue started (pid $$)"
step deltanet deltanet
step spread spread
step nvfp4_search nvfp4_search
step table1 table1
step llama_ppl llama_ppl
step timing_q38_mlx timing_q38_mlx
step decode_profile decode_profile
step layer_wallclock layer_wallclock
step llama_speed llama_speed
log "queue done"
