#!/bin/zsh
# WikiText-2 perplexity of our formats under llama.cpp's protocol (as llama-perplexity: disjoint 2048-token windows,
# only the second half of each scored; same text as e2e/ppl_llamacpp.py), to compare with llama.cpp's own types
# (results/12_ppl_llamacpp.csv) against each model's BF16 under the same protocol. W4A16 rows compare weight formats
# with llama.cpp's (its Metal kernels keep activations in 16 bits), W4A8 rows are what our kernels compute. Strictly
# one job at a time; finished steps leave a marker in results/14_done/. About 1 h.
#   1. Qwen3-1.7B: BF16, SC-MXFP4 and the 16-block format, W4A16 and W4A8          ~8 min
#   2. OLMoE-1B-7B: the same                                                        ~30 min
#   3. Qwen3.5-35B-A3B: BF16, SC-MXFP4 W4A8 and W4A16                               ~20 min
# Usage: caffeinate -i -s zsh e2e/run_14.sh     (progress: results/14_queue.log)
set -u
root="${0:A:h:h}"
cd "$root/e2e"
r=../results
base=$root/.venv/bin/python
OURS="sc_mxfp4_w4a16,sc_mxfp4_w4a8,w=fmt:e2m1/B16/M1..10;a=16,w=fmt:e2m1/B16/M1..10;a=8"
mkdir -p $r/14_done

log() { echo "$(date '+%a %H:%M') $*" >> $r/14_queue.log }
step() {  # step <name> <function>: skipped if done before; logs start, end and exit code
  local name=$1 fn=$2
  if [[ -e $r/14_done/$name ]]; then log "skip  $name (done before)"; return 0; fi
  log "start $name"
  $fn; local rc=$?
  log "end   $name (exit $rc)"
  [[ $rc == 0 ]] && touch $r/14_done/$name
  return $rc
}

qwen3() {
  (cd ../sim && $base eval_ppl.py Qwen/Qwen3-1.7B-Base ../results/14_ppl_second_half.csv --second-half \
     --methods "bf16,$OURS") > $r/14_ppl_qwen3.log 2>&1
}
olmoe() {
  (cd ../sim && $base eval_ppl.py allenai/OLMoE-1B-7B-0924 ../results/14_ppl_second_half.csv --second-half \
     --model-config '{"rms_norm_eps": 1e-5}' --methods "bf16,$OURS") > $r/14_ppl_olmoe.log 2>&1
}
qwen35() {
  $base ppl_qwen35.py --second-half --methods bf16,sc_mxfp4_w4a8,sc_mxfp4_w4a16 \
    --out $r/14_ppl_second_half.csv > $r/14_ppl_qwen35.log 2>&1
}

log "queue started (pid $$)"
step qwen3 qwen3
step olmoe olmoe
step qwen35 qwen35
log "queue done"
