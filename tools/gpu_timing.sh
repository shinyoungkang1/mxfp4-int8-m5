#!/bin/zsh
# Run a timing-sensitive command with the GPU to itself.
#
# Usage: tools/gpu_timing.sh <command> [args...]
#
# Long accuracy evaluations (sim/) and benchmarks can run on the same machine at the same time.
# Evaluations don't care about contention, timings do. So: this script takes the lock directory
# .gpu_timing (one timing run at a time), then waits until no evaluation step is active
# (.gpu_eval_active, held by sim/gpu_guard.py around each evaluation step), then runs the command.
# Evaluations check .gpu_timing at every step boundary and wait while it exists.
set -e
root="${0:A:h:h}"
lock="$root/.gpu_timing"
until mkdir "$lock" 2>/dev/null; do sleep 5; done
trap 'rmdir "$lock" 2>/dev/null' EXIT INT TERM
while [ -e "$root/.gpu_eval_active" ]; do sleep 5; done
pmset -g batt | head -1 | grep -q "AC Power" || echo "WARNING: not on AC power; timings will be throttled" >&2
"$@"
