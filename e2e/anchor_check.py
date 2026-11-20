"""Validity check of a timing run: MLX's own MXFP4 path does not change between our runs, so its time must match
its time in a clean session (idle machine) of the same model and kind of run. A run in which it is more than TOL
slower was disturbed (another GPU client, lowered clocks) and is rejected; run_timed.sh then retries.

References (REF) are keyed by the CSV's `model` column and the kind of run:
  burst      prefill, prompt lengths below 8192 only      median TTFT in ms per prompt length
  sustained  prefill, a session that includes 8192        (8192-token runs lower the clocks for the whole session)
  decode     batch-1 decode                               median tokens/s per context length
Only rows of method mlx_mxfp4 are compared. A model or length without a reference is reported and not checked
(exit 0): add its reference here after its first clean session (two sessions that agree, 5 minutes idle before each).

Usage: python anchor_check.py <csv> [auto|burst|sustained|decode]     exit 0 = valid or not checked, 1 = rejected
auto (default) takes the kind from the CSV: a `context` column means decode; else the `regime` column if it is
filled in; else sustained if a prompt length of 8192 or more is present, burst otherwise.
"""
import csv
import sys

QWEN3 = "Qwen/Qwen3-1.7B-Base"
QWEN35 = "Qwen/Qwen3.5-35B-A3B-Base"
REF = {
    (QWEN3, "burst"): {512: 31.55, 2048: 121.23},                  # results/05_end_to_end_burst.csv
    (QWEN3, "sustained"): {512: 35.55, 2048: 147.25, 8192: 725.83},  # results/05_end_to_end.csv
    (QWEN3, "decode"): {512: 368.3, 2048: 322.6},                  # results/07_decode_qwen3.csv (09: 367.4-368.4 / 321.2-322.8)
    (QWEN35, "burst"): {512: 126.70, 2048: 370.12},                # results/11_qwen35_prefill_burst.csv
    (QWEN35, "sustained"): {512: 130.86, 2048: 370.6, 8192: 1555.0},  # results/18_prefill_qwen35_s1/s2.csv
    (QWEN35, "decode"): {512: 152.0, 2048: 149.2},                 # results/18_decode_qwen35_s1.csv (s2 and run 33: 152.2-154.2 / 149.5-150.6)
}
# 05_end_to_end.md: the sustained regime varies more with run composition. Decode: clean sessions agree within 1%;
# the sessions the 2026-11-05 audit found disturbed are 5-20% slower.
TOL = {"burst": 0.07, "sustained": 0.10, "decode": 0.04}
NO_MODEL = QWEN3  # bench_format.py CSVs written before the model column existed are Qwen3-1.7B


def read(path):
    with open(path) as f:
        if not f.readline().startswith("#"):
            f.seek(0)
        return list(csv.DictReader(f))


def kind_of(rows):
    if rows and "context" in rows[0]:
        return "decode"
    regimes = {r.get("regime") or "" for r in rows} - {""}
    if len(regimes) == 1 and regimes <= {"burst", "sustained"}:
        return regimes.pop()
    return "sustained" if any(int(r["prompt_tokens"]) >= 8192 for r in rows) else "burst"


def main(path, kind="auto"):
    rows = [r for r in read(path) if r.get("method") == "mlx_mxfp4"]
    if not rows:
        print(f"anchor: no mlx_mxfp4 rows in {path}: not checked")
        return True
    if kind == "auto":
        kind = kind_of(rows)
    key, val = ("context", "tok_s_median") if kind == "decode" else ("prompt_tokens", "ttft_ms_median")
    ok = True
    for model in sorted({r.get("model") or NO_MODEL for r in rows}):
        ref = REF.get((model, kind))
        if ref is None:
            print(f"anchor: no {kind} reference for {model}: not checked (add one to anchor_check.REF after a clean session)")
            continue
        for r in rows:
            if (r.get("model") or NO_MODEL) != model:
                continue
            L, t = int(r[key]), float(r[val])
            if L not in ref:
                print(f"anchor {kind} {model} L={L}: no reference, not checked")
                continue
            slow = ref[L] / t if kind == "decode" else t / ref[L]  # > 1: slower than the clean session
            bad = slow > 1 + TOL[kind]
            ok &= not bad
            unit = "tok/s" if kind == "decode" else "ms"
            print(f"anchor {kind} {model} L={L}: MLX MXFP4 {t:.1f} {unit} vs clean {ref[L]:.1f} {unit} "
                  f"(slowdown {slow:.3f}, limit {1 + TOL[kind]:.2f}){' REJECT' if bad else ''}")
    return ok


if __name__ == "__main__":
    sys.exit(0 if main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "auto") else 1)
