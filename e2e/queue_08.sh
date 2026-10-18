#!/bin/zsh
# Timing runs of results/08 (llama.cpp baselines, Qwen3.5-35B-A3B prefill and decode), each through run_timed.sh
# (no $AVOID process, idle GPU, tools/gpu_timing.sh lock, disturbed runs rejected and retried). Steps whose CSV
# exists are skipped. Usage: queue_08.sh [python]
cd "${0:A:h}"
PY=${1:-../.venv/bin/python}; R=../results
for m in qwen3 olmoe qwen35; do
  [[ -f $R/08_llamacpp_$m.csv ]] || ./run_timed.sh $R/08_llamacpp_$m.log $PY bench_llamacpp.py --model $m \
    --out $R/08_llamacpp_$m.csv; echo "llama.cpp $m rc=$?"
done
[[ -f $R/08_qwen35_prefill.csv ]] || ./run_timed.sh $R/08_qwen35_prefill.log $PY bench_qwen35.py prefill \
  --lengths 512,2048,8192 --reps 7 --out $R/08_qwen35_prefill.csv; echo "qwen35 prefill rc=$?"
[[ -f $R/08_qwen35_decode.csv ]] || ./run_timed.sh $R/08_qwen35_decode.log $PY bench_qwen35.py decode \
  --ctx 512,2048 --out $R/08_qwen35_decode.csv; echo "qwen35 decode rc=$?"
