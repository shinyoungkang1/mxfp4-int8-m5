#!/bin/zsh
# The timing part of run_07.sh, each step through run_timed.sh (no $AVOID process, idle GPU, lock; disturbed runs
# rejected and retried). Prefill runs are also validated against MLX MXFP4's clean 05 times (anchor_check.py).
# Steps whose CSV exists are skipped (delete a CSV to redo that step).
cd "${0:A:h}"
PY=${1:-../.venv/bin/python}; D=${2:-$HOME/.cache/mxfp4-int8-m5}; R=../results
M="mlx_mxfp4,mlx_affine4,sc_w4a8,fused,fused_row,fused_res"
anchor() { echo "$PY anchor_check.py $R/07_prefill_$1.csv $1 || { mv $R/07_prefill_$1.csv $R/07_prefill_$1.csv.rejected; exit 1; }"; }
[[ -f $R/07_tune_moe.txt ]] || ./run_timed.sh $R/07_tune_moe.txt $PY bench_moe.py --cache-dir $D --sweep 512,2048,4096
[[ -f $R/07_prefill_burst.csv ]] || VALIDATE="$(anchor burst)" ./run_timed.sh $R/07_prefill_burst.log $PY bench_fused.py --cache-dir $D --lengths 512,2048 \
  --reps 10 --methods $M --regime burst --out $R/07_prefill_burst.csv; echo "prefill burst rc=$?"
[[ -f $R/07_prefill_sustained.csv ]] || VALIDATE="$(anchor sustained)" ./run_timed.sh $R/07_prefill_sustained.log $PY bench_fused.py --cache-dir $D \
  --lengths 512,2048,8192 --reps 10 --methods $M --regime sustained --out $R/07_prefill_sustained.csv
echo "prefill sustained rc=$?"
[[ -f $R/07_decode_qwen3.csv ]] || ./run_timed.sh $R/07_decode_qwen3.log $PY bench_decode.py --model qwen3 --cache-dir $D --ctx 512,2048 --gen 128 \
  --reps 5 --out $R/07_decode_qwen3.csv; echo "decode qwen3 rc=$?"
[[ -f $R/07_moe_prefill.csv ]] || ./run_timed.sh $R/07_moe_prefill.log $PY bench_moe.py --cache-dir $D --lengths 512,2048,4096 --reps 7 \
  --methods mlx_mxfp4,mlx_affine4,fused,fused_inkernel,fused_2pass --out $R/07_moe_prefill.csv; echo "moe prefill rc=$?"
[[ -f $R/07_decode_olmoe.csv ]] || ./run_timed.sh $R/07_decode_olmoe.log $PY bench_decode.py --model olmoe --cache-dir $D --ctx 512,2048 --gen 128 \
  --reps 5 --out $R/07_decode_olmoe.csv; echo "decode olmoe rc=$?"
