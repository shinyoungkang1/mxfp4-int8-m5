#!/bin/zsh
# Run 49 (F3): KL divergence and top-1 agreement against BF16 on LAMBADA test passages (outside WikiText), for the
# systems of the zero-shot table, Qwen3-1.7B and Qwen3.5-35B-A3B (the 125B has its KL from run 40).
# First a sanity check: BF16 against itself must give KL 0 and no top-1 flips, else stop.
# Usage: caffeinate -i -s zsh e2e/queue_49.sh      (progress: results/49_queue.log)
source "${0:A:h}/lib_queue.sh"
queue_init 49
step kl_sanity $EV kl_systems.py --model qwen3 --methods bf16 --windows 2 --out $R/49_kl_sanity.csv || exit 1
step kl_sanity_ok $PY -c "
import csv; r = list(csv.DictReader(open('$R/49_kl_sanity.csv')))[-1]
assert float(r['kl_vs_bf16']) < 1e-6 and float(r['top1_flip_vs_bf16']) == 0 and r['ppl'] == r['ppl_bf16'], r
print('sanity ok', r)" || exit 1
step kl_qwen3 $EV kl_systems.py --model qwen3 --windows 32 --out $R/49_kl_qwen3.csv || _q_log "kl_qwen3 failed"
step kl_qwen35 $EV kl_systems.py --model qwen35 --windows 32 --out $R/49_kl_qwen35.csv || _q_log "kl_qwen35 failed"
_q_log "queue 49 done"
