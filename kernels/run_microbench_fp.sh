#!/bin/zsh
# Benchmarks macOS 27.2 FP4 / FP8 / MXFP4 matmul2d operand types against FP16 and INT8 anchors.
# Run on AC power with Low Power Mode off and other GPU apps closed.
set -e
cd "${0:A:h}"
swiftc -O bench_fp.swift -o bench_fp
out=../results/53_microbench_macos27_fp.csv
{
  echo "# date=$(date -Iseconds)"
  echo "# macos=$(sw_vers -productVersion) ($(sw_vers -buildVersion))"
  echo "# git_commit=$(git -C .. rev-parse HEAD)"
  echo "# power_state=$(pmset -g batt | head -1)"
  echo "# command=bench_fp scaled=1 GROUP_M=8 reps=15 warm=1"
  echo "M,N,K,TM,TN,NSG,GROUP_M,BK,kernel,ms_median,ms_best,tflops_median,speedup_vs_fp16_median,speedup_vs_int8_median,max_abs_err"
} > "$out"

KERNELS="mm_f16_f32,mm_i8_i32,mm_i8xi4_i32,mm_fp4_f32,mm_fp4_f16,mm_f16xfp4_f32,mm_f16xfp4_f16,mm_fp8e4m3_f32,mm_fp8e4m3_f16,mm_f16xfp8e4m3_f32,mm_fp8e5m2_f32,mm_f16xfp8e5m2_f32,mm_f16xmxfp4_f32,mm_mxfp4_f32"

for shape in "4096 4096 4096" "8192 8192 4096" "4096 14336 4096" "4096 4096 14336" "2048 8192 8192" "512 4096 4096" "1024 4096 4096"; do
  set -- ${=shape}
  ../tools/gpu_timing.sh ./bench_fp M=$1 N=$2 K=$3 GROUP_M=8 scaled=1 reps=15 warm=1 kernels="$KERNELS" csv=1 | tee -a "$out"
done
