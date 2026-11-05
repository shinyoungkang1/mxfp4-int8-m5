#!/bin/zsh
# Qwen3.5-35B-A3B after the re-download: 16-block and 8-wide perplexity (fake-quant), 16-block on the kernels, zero-shot on them, and
# prefill and decode re-timed after the gate|up layout fix (5 min pause before each timing session, see run_15.sh). One job at a time;
# finished steps leave a marker in results/18_done/. Usage: caffeinate -i -s zsh e2e/run_18.sh     (progress: results/18_queue.log)
set -u
root="${0:A:h:h}"
cd "$root/e2e"
r=../results
py=$root/.venv/bin/python
D=$HOME/.cache/mxfp4-int8-m5
Q35=Qwen/Qwen3.5-35B-A3B-Base
mkdir -p $r/18_done
log() { echo "$(date '+%a %H:%M') $*" >> $r/18_queue.log }
step() {
  local name=$1 fn=$2
  if [[ -e $r/18_done/$name ]]; then log "skip  $name (done before)"; return 0; fi
  log "start $name"
  $fn; local rc=$?
  log "end   $name (exit $rc)"
  [[ $rc == 0 ]] && touch $r/18_done/$name
  return $rc
}
convert() {
  [[ -d $D/Qwen3.5-35B-A3B-Base-mxfp4 ]] || $py convert_mlx.py $Q35 $D/Qwen3.5-35B-A3B-Base-mxfp4 mxfp4 32 '{}' router8 || return 1
  [[ -d $D/Qwen3.5-35B-A3B-Base-4bit ]] || $py convert_mlx.py $Q35 $D/Qwen3.5-35B-A3B-Base-4bit affine 64 '{}' router8
}
# ppl_qwen35.py keeps fake-quantized matrices between methods, so the fake-quant rows and each kernel row run in their own process
# (one process with all of them was killed for memory while saving the pack, 2026-11-03; its three fake-quant rows are in the CSV)
ppl() {
  grep -q b8_w4a8 $r/18_ppl_qwen35.csv 2>/dev/null || $py ppl_qwen35.py --methods sc_mxfp4_w4a8,b16_w4a8,b8_w4a8 --out $r/18_ppl_qwen35.csv >> $r/18_ppl_qwen35.log 2>&1 || return 1
  grep -q ",ours," $r/18_ppl_qwen35.csv || $py ppl_qwen35.py --methods ours --out $r/18_ppl_qwen35.csv >> $r/18_ppl_qwen35.log 2>&1 || return 1
  $py ppl_qwen35.py --methods ours_b16 --out $r/18_ppl_qwen35.csv >> $r/18_ppl_qwen35.log 2>&1
}
timing() {
  local rc=0
  for s in 1 2; do
    sleep 300
    ./run_timed.sh $r/18_decode_qwen35_s$s.log $py bench_qwen35.py decode --out $r/18_decode_qwen35_s$s.csv || rc=1
    sleep 300
    ./run_timed.sh $r/18_prefill_qwen35_s$s.log $py bench_qwen35.py prefill --out $r/18_prefill_qwen35_s$s.csv || rc=1
  done
  return $rc
}
tasks() {
  $D/venv-eval/bin/python eval_tasks.py --model qwen35 --method ours_fmt --format e2m1/B16/M1..10 --limit 1000 \
    --out $r/18_tasks_qwen35_b16.json > $r/18_tasks_qwen35_b16.log 2>&1
}
log "queue started (pid $$)"
step convert convert && step ppl ppl && step timing timing && step tasks tasks
log "queue done"
