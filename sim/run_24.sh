#!/bin/zsh
# Which layer kinds carry the INT8 perplexity gap, and does each one's gap follow its own rotation gain? INT8 activations (with rotation) on one kind
# of linear layer at a time, BF16 weights, Qwen3-1.7B. Usage: caffeinate -i -s zsh sim/run_24.sh
set -u
root="${0:A:h:h}"
cd "$root/sim"
r=../results
repo=Qwen/Qwen3-1.7B-Base
for b in 0 8 32 128; do
  for p in self_attn.q_proj self_attn.k_proj self_attn.v_proj self_attn.o_proj mlp.gate_proj mlp.up_proj mlp.down_proj; do
    m="w=bf16;a=8;r=$b;p=$p"
    grep -qF "$repo,$m," $r/24_ppl.csv 2>/dev/null && continue
    $root/.venv/bin/python eval_ppl.py $repo $r/24_ppl.csv --methods "$m" >> $r/24_ppl.log 2>&1
    echo "$(date '+%a %H:%M') end $m (exit $?)" >> $r/24_queue.log
  done
done
