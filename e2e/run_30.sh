#!/bin/zsh
# After the down-projection run: E2M1 8-wide on Qwen3.8-Flash-Next from the kept FP8 checkpoint (no download). Layer-streamed perplexity (all W4A8 matrices, and experts only),
# then pack if more than 100 GB would stay free, then kernels against the reference on 8 windows. Then zero-shot in simulation on Qwen3-4B and SmolLM3. One job at a time, finished steps
# leave a marker in results/30_done/. Usage: caffeinate -i -s zsh e2e/run_30.sh    (progress: results/30_queue.log)
set -u
root="${0:A:h:h}"
cd "$root/e2e"
r=../results
q38=$root/.venv-q38/bin/python
EV=$HOME/.cache/mxfp4-int8-m5/venv-eval/bin/python
F8=e2m1/B8/M1..10
S10=s10/B8/M1..12
mkdir -p $r/30_done
while pgrep -qf run_29.sh; do sleep 30; done
step() {
  [[ -f $r/30_done/$1 ]] && return
  echo "$(date '+%a %H:%M') start $1" >> $r/30_queue.log
  "${@:2}" > $r/30_$1.log 2>&1
  c=$?
  echo "$(date '+%a %H:%M') end   $1 (exit $c)" >> $r/30_queue.log
  [[ $c == 0 ]] && touch $r/30_done/$1
}
freegb() { df -g / | tail -1 | awk '{print $4}' }
step ppl125 $q38 q38_ppl.py --methods "w=fmt:$F8;a=8,w=fmt:$F8;a=8;s=experts" --out $r/30_ppl_qwen38.csv
if (( $(freegb) > 170 )); then
  step pack125 $q38 q38_pack_when_ready.py $F8
  step check125ref $q38 q38_ppl.py --windows 8 --methods "w=fmt:$F8;a=8" --out $r/30_check_q38.csv
  step check125 $q38 q38_ppl_kernels.py --format $F8 --windows 8 --decode-check 64 --out $r/30_check_q38.csv
else
  echo "$(date '+%a %H:%M') skipped pack: $(freegb) GB free" >> $r/30_queue.log
fi
task() { step tasks_$1_$2 $EV eval_tasks.py --model $1 --method fake --fake "$3" --out $r/30_tasks_$1_$2.json }
base() { step tasks_$1_bf16 $EV eval_tasks.py --model $1 --method bf16 --out $r/30_tasks_$1_bf16.json }
for m in qwen3_4b smol3; do
  base $m
  task $m e2m1b8 "w=fmt:$F8;a=8"
done
for m in qwen3_4b smol3; do
  task $m gptq_r32 "w=gptq:$F8;a=8;r=32"
  task $m s10b8 "w=fmt:$S10;a=8"
done
