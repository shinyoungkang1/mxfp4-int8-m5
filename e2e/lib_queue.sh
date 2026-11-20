#!/bin/zsh
# Shared pieces of a run queue: one heavy job at a time, finished steps are skipped, every step leaves its provenance.
# Source it from a queue script and call queue_init first:
#
#   source "${0:A:h}/lib_queue.sh"
#   queue_init 36                                   # tag: everything goes to results/36_*; sets $R, $PY, $EV; cd e2e
#   step  ppl_b8   $PY ppl_qwen35.py --methods "w=fmt:e2m1/B8/M1..10;a=8" --out $R/36_ppl.csv
#   timed prefill  $PY bench_qwen35.py prefill --lengths 512,2048 --out $R/36_prefill.csv
#   step  tasks    $EV eval_tasks.py --model qwen35 --method ours_fmt --out $R/36_tasks.json || exit 1
#
# queue_init <tag>    takes the queue lock (.queue_lock in the project folder). A second queue stops at once (exit 9)
#                     instead of running two heavy jobs side by side; the lock of a queue that is gone is taken over.
# step <name> cmd..   an accuracy or correctness job, through devrun.sh (waits for a timing run and keeps timing runs
#                     out); output in results/<tag>_<name>.log.
# timed <name> cmd..  a timing job, through run_timed.sh with its defaults: PAUSE_S=300 idle seconds first, idle-GPU
#                     gate, anchor check of every "--out <file>.csv", disturbed runs rejected and retried; log in
#                     results/<tag>_<name>.log.
# Both return the command's exit status (the queue decides whether to go on), skip a step whose marker
# results/<tag>_done/<name> exists, wait while a development job started outside the queue (devrun.sh) is running,
# write start and end lines to results/<tag>_queue.log and append two provenance lines per step to
# results/<tag>_steps.jsonl: at the start (commit, files that differ from it, command) and at the end (exit status,
# seconds). The marker is written only after exit status 0, so a failed step runs again in the next pass.
# Stop a queue with Ctrl-C in its terminal (the running step ends with it). After a plain "kill <pid>" the running step
# goes on alone: the next queue then waits for it (devrun.sh marker, timing lock) before its own first step.
# Q_RESULTS=<dir> puts the result files of the queue in another directory (smoke tests).

Q_ROOT="${${(%):-%x}:A:h:h}"

_q_log() { echo "$(date '+%a %m-%d %H:%M') $*" >> "$R/${Q_TAG}_queue.log" }

# pids of development jobs (devrun.sh) that are alive
_q_dev() {
  local f p
  for f in "$Q_ROOT"/e2e/.dev_active/*(N); do
    p=${f:t}; kill -0 "$p" 2>/dev/null && echo "$p"
  done
  return 0
}

queue_init() {
  [[ -n ${1:-} ]] || { echo "usage: queue_init <tag>" >&2; exit 2 }
  Q_TAG=$1
  R=${Q_RESULTS:-$Q_ROOT/results}
  PY=${PY:-$Q_ROOT/.venv/bin/python}
  EV=${EV:-$HOME/.cache/mxfp4-int8-m5/venv-eval/bin/python}
  Q_PY=$PY; [[ -x $Q_PY ]] || Q_PY=python3
  Q_LOCK=$Q_ROOT/.queue_lock
  cd "$Q_ROOT/e2e"
  if ! mkdir "$Q_LOCK" 2>/dev/null; then
    local other=$(cat "$Q_LOCK/pid" 2>/dev/null)
    if [[ -n $other ]] && kill -0 "$other" 2>/dev/null; then
      echo "queue $(cat "$Q_LOCK/tag" 2>/dev/null) is running (pid $other): one queue at a time" >&2
      exit 9
    fi
    echo "taking over the lock of a queue that is gone (pid ${other:-unknown})" >&2
  fi
  echo $$ > "$Q_LOCK/pid"; echo "$Q_TAG" > "$Q_LOCK/tag"
  # zshexit and not "trap ... EXIT": inside a function that trap would fire when the function returns
  zshexit() {
    if [[ $(cat "$Q_LOCK/pid" 2>/dev/null) == $$ ]]; then rm -f "$Q_LOCK/pid" "$Q_LOCK/tag"; rmdir "$Q_LOCK" 2>/dev/null; fi
  }
  trap 'exit 130' INT TERM HUP
  mkdir -p "$R/${Q_TAG}_done"
  _q_log "queue $Q_TAG starts (pid $$)"
}

_q_stamp() {  # <event> <name> <kind> [--exit N --seconds N] -- cmd..
  local event=$1 name=$2 kind=$3; shift 3
  "$Q_PY" "$Q_ROOT/sim/provenance.py" step "$R/${Q_TAG}_steps.jsonl" --queue "$Q_TAG" --step "$name" --kind "$kind" \
    --event "$event" "$@" || echo "no provenance line for $name ($event)" >&2
  return 0
}

_q_run() {  # <kind> <name> cmd..
  local kind=$1 name=$2; shift 2
  [[ -n ${Q_TAG:-} ]] || { echo "call queue_init <tag> first" >&2; return 2 }
  [[ -f $R/${Q_TAG}_done/$name ]] && return 0
  local log=$R/${Q_TAG}_$name.log rc=0 dev waited=0
  while dev=$(_q_dev) && [[ -n $dev ]]; do
    [[ $waited == 1 ]] || { _q_log "wait  $name: a development job is running (pid ${dev//$'\n'/ })"; waited=1 }
    sleep 15
  done
  local t0=$(date +%s)
  _q_log "start $name"
  _q_stamp start "$name" "$kind" -- "$@"
  if [[ $kind == timed ]]; then
    "$Q_ROOT/e2e/run_timed.sh" "$log" "$@" || rc=$?
  else
    "$Q_ROOT/e2e/devrun.sh" "$@" > "$log" 2>&1 || rc=$?
  fi
  local s=$(( $(date +%s) - t0 ))
  _q_log "end   $name (exit $rc, $(( s / 60 )) min)"
  _q_stamp end "$name" "$kind" --exit $rc --seconds $s -- "$@"
  [[ $rc == 0 ]] && touch "$R/${Q_TAG}_done/$name"
  return $rc
}

step() { _q_run eval "$@" }
timed() { _q_run timed "$@" }
