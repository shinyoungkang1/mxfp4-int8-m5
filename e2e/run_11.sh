#!/bin/zsh
# Experiments that complete the Qwen3.8-Flash-Next and system results (results/11_*), strictly one job at a time: each
# Qwen3.8 job needs 60-90 GB, and two concurrent jobs exceed 128 GB. A failing step is logged and the queue goes on;
# the 16-block steps after the kernel check run only if that check passes. Finished steps leave a marker in
# results/11_done/, so a stopped queue can be restarted and skips them. About 15 h in total.
#   1. pack Qwen3.8-Flash-Next in the 16-block integer-exact format (resumable)                           ~75 min
#   2. check it: perplexity on our kernels vs fake quantization (8 windows), decode vs prefill               ~30 min
#   3. GSM8K, all 1319: MLX MXFP4 W4A16 on our SC-MXFP4 weights (INT8 activations vs weight quantizer)       ~1.7 h
#   4. GSM8K, all 1319: the 16-block format on our kernels                                                 ~1.7 h
#   5. timing (run_timed.sh): Qwen3.8 prefill + decode, 16-block vs SC-MXFP4 vs MLX MXFP4, 2 sessions each    ~60 min
#   6. timing: OLMoE-1B-7B prefill in the sustained regime (with 8192 tokens), Qwen3.5-35B-A3B in the burst one ~30 min
#   7. KL to FP8 and top-1 flips (32 windows) for the routed-expert configurations Table 5 lacks             ~2.2 h
#   8. the missing Table 5 perplexities, routed experts only (145 windows)                                  ~4.8 h
#   9. the missing Table 5 perplexities, every matrix: W4A8 for affine INT4 and MXFP4 (OCP rule)            ~1.6 h
# Order the timing steps so they run at night: they wait for an otherwise idle GPU.
# Usage: caffeinate -i -s zsh e2e/run_11.sh     (progress: results/11_queue.log)
set -u
root="${0:A:h:h}"
cd "$root/e2e"
r=../results
q38=$root/.venv-q38/bin/python
base=$root/.venv/bin/python
F16="e2m1/B16/M1..10"
B16="w=fmt:$F16"
mkdir -p $r/11_done

log() { echo "$(date '+%a %H:%M') $*" >> $r/11_queue.log }
step() {  # step <name> <function>: skipped if done before; logs start, end and exit code
  local name=$1 fn=$2
  if [[ -e $r/11_done/$name ]]; then log "skip  $name (done before)"; return 0; fi
  log "start $name"
  $fn; local rc=$?
  log "end   $name (exit $rc)"
  [[ $rc == 0 ]] && touch $r/11_done/$name
  return $rc
}

pack16() { $q38 fused_qwen38.py --pack $F16 >> $r/11_pack_b16.log 2>&1 }
check16() {
  $q38 q38_ppl.py --windows 8 --methods "$B16;a=8" --out $r/11_check_b16.csv > $r/11_check_b16.log 2>&1 &&
  $q38 q38_ppl_kernels.py --format $F16 --windows 8 --decode-check 64 --out $r/11_check_b16.csv >> $r/11_check_b16.log 2>&1 &&
  $q38 - <<'EOF' >> $r/11_check_b16.log 2>&1
import csv, re, sys
rows = {r["method"]: float(r["ppl"]) for r in csv.DictReader(open("../results/11_check_b16.csv"))}
ref, ker = rows.get("w=fmt:e2m1/B16/M1..10;a=8"), rows.get("ours_kernels_e2m1/B16/M1..10")
m = re.search(r"argmax agreement ([0-9.]+)%", open("../results/11_check_b16.log").read())
agree = float(m.group(1)) if m else None
# SC-MXFP4 gave kernels 2.6326 vs reference 2.6381 and 96.9% decode agreement; a broken kernel is far off both
ok = bool(ref and ker and abs(ker / ref - 1) < 0.02 and agree is not None and agree >= 80)
print(f"16-block check: reference {ref}, kernels {ker}, decode argmax agreement {agree}% -> {'ok' if ok else 'FAILED'}")
sys.exit(0 if ok else 1)
EOF
}
timing_q38() {
  local rc=0 s m
  for s in 1 2; do
    for m in $q38_systems; do
      ./run_timed.sh $r/11_prefill_q38_${m}_s$s.log $q38 bench_q38.py --method $m --session $s --reps 5 --decode 64 \
        --out $r/11_prefill_q38.csv || { log "      timing $m session $s failed"; rc=1; }
    done
  done
  return $rc
}
timing_regimes() {
  ./run_timed.sh $r/11_moe_prefill_sustained.log $base bench_moe.py --lengths 512,2048,4096,8192 --reps 7 \
    --regime sustained --out $r/11_moe_prefill_sustained.csv &&
  ./run_timed.sh $r/11_qwen35_prefill_burst.log $base bench_qwen35.py prefill --lengths 512,2048 --reps 7 \
    --out $r/11_qwen35_prefill_burst.csv
}
gsm_mlx_sc() { $q38 gsm8k_q38.py --system mlx_sc_mxfp4 --n 1319 --out $r/11_gsm8k_mlx_sc_mxfp4.jsonl >> $r/11_gsm8k_mlx_sc_mxfp4.log 2>&1 }
gsm_b16() { $q38 gsm8k_q38.py --system fused_b16 --n 1319 --out $r/11_gsm8k_fused_b16.jsonl >> $r/11_gsm8k_fused_b16.log 2>&1 }
kl() {
  $q38 q38_ppl.py --windows 32 --methods "fp8,$B16;a=16;s=experts,w=sc_mx;a=16;s=experts,w=nvfp4_search;a=16;s=experts,w=int4_g64;a=8;s=experts,w=mx_spec;a=8;s=experts" \
    --out $r/11_kl_qwen38.csv > $r/11_kl_qwen38.log 2>&1
}
cells_experts() {
  $q38 q38_ppl.py --methods "$B16;a=16;s=experts,w=sc_mx;a=16;s=experts,w=nvfp4;a=16;s=experts,w=nvfp4_search;a=16;s=experts,w=int4_g64;a=8;s=experts,w=mx_spec;a=8;s=experts" \
    --out $r/11_ppl_qwen38.csv > $r/11_ppl_qwen38_experts.log 2>&1
}
cells_all() { $q38 q38_ppl.py --methods "w=int4_g64;a=8,w=mx_spec;a=8" --out $r/11_ppl_qwen38.csv > $r/11_ppl_qwen38_all.log 2>&1 }

log "queue started (pid $$)"
step pack16 pack16
if step check16 check16; then b16=1; else b16=0; log "      16-block steps skipped: the kernel check did not pass"; fi
q38_systems=(fused mlx_mxfp4); (( b16 )) && q38_systems=(fused_b16 fused mlx_mxfp4)
step gsm_mlx_sc gsm_mlx_sc
(( b16 )) && step gsm_b16 gsm_b16
step timing_q38 timing_q38
step timing_regimes timing_regimes
step kl kl
step cells_experts cells_experts
step cells_all cells_all
log "queue done"
