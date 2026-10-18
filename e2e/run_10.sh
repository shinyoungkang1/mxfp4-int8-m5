#!/bin/zsh
# Step 10 (Qwen3.8-Flash-Next) accuracy jobs, in order, after the FP8 checkpoint is downloaded and packed:
#   1. check: fake-quant reference vs our kernels on the first 8 windows (perplexity must agree)
#   2. layer-streamed WikiText-2 perplexity, pass A (reference, SC-MXFP4, 16-block; W4A8 and W4A16)
#   3. pass B (NVFP4, NVFP4 + scale search, plain MXFP4, INT4 affine g64)
#   4. exponent-spread statistics of every W4A8 matrix
# Usage: e2e/run_10.sh <python of .venv-q38>   (results/10_*)
set -e
py=${1:-$PWD/.venv-q38/bin/python}
root="${0:A:h:h}"
cd "$root/e2e"
r=../results
B16="w=fmt:e2m1/B16/M1..10"
# one heavy job at a time: two concurrent jobs of this step need more than 128 GB of memory
$py q38_pack_when_ready.py sc >> $r/10_pack_sc.log 2>&1
if [ ! -s $r/10_check_q38.csv ]; then
  $py q38_ppl.py --windows 8 --methods "fp8,sc_mxfp4_w4a8" --out $r/10_check_q38.csv > $r/10_check_q38.log 2>&1
  $py q38_ppl_kernels.py --windows 8 --decode-check 64 --out $r/10_check_q38.csv >> $r/10_check_q38.log 2>&1
fi
if ! grep -q "B16/M1..10;a=16" $r/10_ppl_qwen38.csv 2>/dev/null; then
  $py q38_ppl.py --methods "fp8,sc_mxfp4_w4a8,sc_mxfp4_w4a16,$B16;a=8,$B16;a=16" --out $r/10_ppl_qwen38.csv \
    > $r/10_ppl_qwen38_A.log 2>&1
fi
$py q38_ppl.py --methods "w=nvfp4;a=8,nvfp4_w4a16,w=nvfp4_search;a=8,w=nvfp4_search;a=16,mxfp4_spec_w4a16,int4_affine_g64_w4a16" \
  --out $r/10_ppl_qwen38.csv > $r/10_ppl_qwen38_B.log 2>&1
$py q38_spread.py > $r/10_spread_qwen38.md 2> $r/10_spread_qwen38.log
echo "run_10 done"
