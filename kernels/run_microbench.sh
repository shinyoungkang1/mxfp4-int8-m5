#!/bin/zsh
# Reproduces results/microbench.csv: INT8/INT4 vs FP16 matmul2d throughput on the M5 Neural Accelerators.
# Run on AC power with Low Power Mode off (System Settings > Battery), with other GPU apps closed.
set -e
cd "${0:A:h}"
swiftc -O bench.swift -o bench
out=../results/microbench.csv
echo "M,N,K,TM,TN,NSG,GROUP_M,BK,kernel,ms_median,ms_best,tflops_median,speedup_vs_fp16_median,max_abs_err" > $out
pmset -g batt | head -1 | sed 's/^/# /' >> $out
for shape in "4096 4096 4096" "8192 8192 4096" "4096 14336 4096" "4096 4096 14336" "2048 8192 8192" "512 4096 4096" "1024 4096 4096"; do
  set -- ${=shape}
  ./bench M=$1 N=$2 K=$3 GROUP_M=8 reps=15 csv=1 | tee -a $out
done
