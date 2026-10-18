#!/bin/zsh
# Reproduces results/layerbench.csv and results/mlx_baseline.csv. Run on AC power, High Power mode.
# Usage: run_layerbench.sh <python-with-mlx-and-numpy> <scratch-dir-for-test-cases>
set -e
cd "${0:A:h}"
PY=$1; TC=$2
swiftc -O bench_layer.swift -o bench_layer
out=../results/layerbench.csv
echo "M,N,K,TM,TN,NSG,BK,GROUP_M,pipeline,ms_median,tflops_eq_median,speedup_vs_fp16,speedup_vs_fused_w4a16" > $out
pmset -g batt | head -1 | sed 's/^/# /' >> $out
for shape in "512 4096 4096" "1024 4096 4096" "2048 4096 4096" "4096 4096 4096" "2048 14336 4096" "2048 4096 14336"; do
  set -- ${=shape}
  d=$TC/tc_$1_$2_$3
  [[ -f $d/meta.json ]] || (cd ../sim && $PY make_testcase.py $1 $2 $3 $d)
  ./bench_layer $d csv=1 | tee -a $out
done
$PY ../bench/mlx_layer_baseline.py | tee ../results/mlx_baseline.csv
