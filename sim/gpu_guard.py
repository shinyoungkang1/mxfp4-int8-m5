"""Evaluation side of the GPU lock in tools/gpu_timing.sh.

Wrap each evaluation step (one model x method) in `with eval_step():`. It waits while a timing run holds
.gpu_timing, and marks the step active with .gpu_eval_active so a timing run waits for it to finish.
"""
import contextlib
import os
import time

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
TIMING = os.path.join(ROOT, ".gpu_timing")
ACTIVE = os.path.join(ROOT, ".gpu_eval_active")


@contextlib.contextmanager
def eval_step(poll=5.0):
    while True:
        while os.path.exists(TIMING):
            time.sleep(poll)
        open(ACTIVE, "w").write(str(os.getpid()))
        if not os.path.exists(TIMING):  # a timing run may have taken the lock between the check and the write
            break
        os.remove(ACTIVE)
    try:
        yield
    finally:
        if os.path.exists(ACTIVE):
            os.remove(ACTIVE)
