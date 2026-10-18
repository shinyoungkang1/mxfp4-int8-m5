#!/bin/zsh
# Runs a timing command under tools/gpu_timing.sh, but only on an otherwise idle GPU, and rejects disturbed runs.
#
# tools/gpu_timing.sh pauses this project's evaluations, but not other applications (a game and its lobby client on
# the same machine slowed every measured speed by 20-50%). So:
#  1. while any process matching $AVOID runs, wait without touching the lock;
#  2. holding the lock, sample the GPU's "Device Utilization %" (ioreg) while nothing of ours runs; unless it stays
#     <= IDLE_MAX for 3 samples within SETTLE_S seconds, release the lock and retry after RETRY_S seconds;
#  3. after the run (lock still held, so this project's evaluations cannot start) sample again, the lowest of 5 samples
#     over 10 s (the counter lags our own last kernels by a few seconds): above AFTER_MAX means another client is
#     using the GPU, the run is rejected (log kept as <log>.rejected-N) and retried;
#  4. if $VALIDATE is set, it is run afterwards (e.g. anchor_check.py on the CSV) and a failure also rejects the run.
# The utilization samples are appended to the log as '# gpu_util_before=.. after=..'.
#
# Usage: [VALIDATE="cmd"] run_timed.sh <log-file> <command> [args...]
set -e
root="${0:A:h:h}"
log=$1; shift
IDLE_MAX=${IDLE_MAX:-6}; SETTLE_S=${SETTLE_S:-20}; RETRY_S=${RETRY_S:-240}; AFTER_MAX=${AFTER_MAX:-15}
MAX_TRIES=${MAX_TRIES:-8}
AVOID=${AVOID:-}   # pgrep -f pattern of other GPU-heavy applications to wait out (e.g. a game); empty: none
try=0
while true; do
  while [ -n "$AVOID" ] && pgrep -qf "$AVOID"; do sleep 60; done
  # development jobs started through devrun.sh finish first
  while [ -n "$(ls "$root/e2e/.dev_active" 2>/dev/null)" ]; do sleep 5; done
  set +e
  "$root/tools/gpu_timing.sh" zsh -c '
    util() { ioreg -r -d 1 -c IOAccelerator | grep -o "\"Device Utilization %\"=[0-9]*" | head -1 | cut -d= -f2 }
    t=0; ok=0; low=0
    while [ $t -lt '"$SETTLE_S"' ]; do
      u=$(util); if [ "$u" -le '"$IDLE_MAX"' ]; then low=$((low+1)); else low=0; fi
      if [ $low -ge 3 ]; then ok=1; break; fi
      sleep 2; t=$((t+2))
    done
    [ $ok = 1 ] || { echo "GPU busy (utilization $(util)%), not timing" >&2; exit 3; }
    before=$(util)
    pmset -g batt | head -1 | grep -q "AC Power" || { echo "not on AC power" >&2; exit 4; }
    "$@" > "'"$log"'" 2>&1; rc=$?
    # the counter averages over a window: wait for our own last kernels to drain, keep the lowest of 5 samples
    after=100; for i in 1 2 3 4 5; do sleep 2; u=$(util); [ "$u" -lt "$after" ] && after=$u; done
    echo "# gpu_util_before=${before} after=${after}" >> "'"$log"'"
    [ $rc = 0 ] || exit $rc
    [ "$after" -le '"$AFTER_MAX"' ] || { echo "GPU busy right after the run (${after}%): rejected" >&2; exit 5; }
    exit 0' zsh "$@"
  rc=$?
  if [ $rc = 0 ] && [ -n "$VALIDATE" ]; then
    zsh -c "$VALIDATE" >> "$log" 2>&1 || rc=5
  fi
  set -e
  if [ $rc = 5 ]; then
    try=$((try + 1)); mv "$log" "$log.rejected-$try"; echo "$(date +%T) rejected run $try of $log" >&2
    [ $try -lt $MAX_TRIES ] || exit 6
  elif [ $rc != 3 ]; then
    exit $rc
  fi
  sleep $RETRY_S
done
