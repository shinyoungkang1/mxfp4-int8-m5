#!/bin/zsh
# Runs a development / correctness GPU job without disturbing timing runs: waits while a
# timing run holds the lock, and marks itself in e2e/.dev_active so that run_timed.sh does not start meanwhile.
here="${0:A:h}"
while [ -d "$here/../.gpu_timing" ]; do sleep 5; done
mkdir -p "$here/.dev_active"; touch "$here/.dev_active/$$"
trap 'rm -f "$here/.dev_active/$$"' EXIT INT TERM
"$@"
