#!/bin/zsh
# Run 36 = PLAN.md steps A5 (regression of the scope option), B1 (35B: experts or dense projections), B2 (paired
# interval 8-wide against NVFP4 with search) and the 35B checks of B0 (check_decode, one decode timing).
# Simulation rows: 145 windows, INT8 activations at every W4A8 site, windows saved in results/36_windows.
# Usage: caffeinate -i -s zsh e2e/queue_36.sh      (progress: results/36_queue.log)
source "${0:A:h}/lib_queue.sh"
queue_init 36
F8="fmt:e2m1/B8/M1..10"
F16="fmt:e2m1/B16/M1..10"
W=$R/36_windows
ppl() { step "$1" $PY -u ppl_qwen35.py --methods "$2" --out $R/36_ppl_qwen35.csv --windows-dir $W }

# A5 / B0: the default spec must reproduce b8_w4a8 of run 28 (5.8870) before any scoped row is trusted
ppl reg "w=$F8;a=8" || exit 1
step reg_check $PY -c "
import csv, sys
v = [r['ppl'] for r in csv.DictReader(open('$R/36_ppl_qwen35.csv')) if r['method'] == 'w=$F8;a=8']
print('regression', v); sys.exit(0 if v and v[-1] == '5.8870' else 1)" || exit 1

# B1: where the 4-bit loss sits (reference BF16 weights + INT8 activations 5.7794)
ppl exp_b8   "w=$F8;a=8;wd=bf16"
ppl dense_b8 "w=bf16;a=8;wd=$F8"
# B2: NVFP4 with search, windows saved, against the 8-wide windows of the regression row
ppl nvfp4s   "nvfp4_search_w4a8"
# B1 extras and the 35B versions of the B5 mixed assignments (decision D6)
ppl exp_sc     "w=sc_mx;a=8;wd=bf16"
ppl mix_sc_b8  "w=sc_mx;a=8;wd=$F8"
ppl mix_b16_b8 "w=$F16;a=8;wd=$F8"
ppl mix_b8_i8  "w=$F8;a=8;wd=int8_row"
ppl mix_sc_i8  "w=sc_mx;a=8;wd=int8_row"

# B0: kernels after the per-call set_format of A2: decode = stepwise prefill, then one decode timing
step check_decode $PY check_decode.py --model qwen35 --packed $F8 --steps 256 --out $R/36_check_decode_qwen35_b8.json
timed decode_b8 $PY bench_qwen35.py decode --format e2m1/B8/M1..10 --out $R/36_decode_qwen35_b8.csv
_q_log "queue 36 done"
