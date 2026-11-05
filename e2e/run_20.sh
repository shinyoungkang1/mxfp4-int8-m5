#!/bin/zsh
# Zero-shot on Qwen3-1.7B of fake-quantized configurations (eval_tasks.py --method fake): does GPTQ, with and without rotations, hold up outside
# its calibration text? The 8-wide row without GPTQ checks the fake path against the kernels (63.5). Usage: caffeinate -i -s zsh e2e/run_20.sh
set -u
root="${0:A:h:h}"
cd "$root/e2e"
r=../results
EV=$HOME/.cache/mxfp4-int8-m5/venv-eval/bin/python
i=0
for m in "w=fmt:e2m1/B8/M1..10;a=8" "w=gptq:e2m1/B8/M1..10;a=8" "w=gptq:sc_mx;a=8;r=32" "w=gptq:e2m1/B16/M1..10;a=8" "w=gptq:e2m1/B8/M1..10;a=8;r=32"; do
  i=$((i+1))
  [[ -f $r/20_tasks_$i.json ]] && continue
  echo "$(date '+%a %H:%M') start $i $m" >> $r/20_queue.log
  $EV eval_tasks.py --model qwen3 --method fake --fake "$m" --out $r/20_tasks_$i.json > $r/20_tasks_$i.log 2>&1
  echo "$(date '+%a %H:%M') end   $i (exit $?)" >> $r/20_queue.log
done
