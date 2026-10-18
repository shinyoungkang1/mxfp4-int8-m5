"""Validity check of a prefill timing run: MLX's own MXFP4 path is unchanged since results/05, so its time to first
token must match the clean 05 measurement (idle machine) of the same regime. A run where it is more than TOL slower
was disturbed by another GPU client and is rejected (run_timed.sh then retries).
Usage: python anchor_check.py <csv> burst|sustained   (exit 0 = valid)"""
import csv
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REF = {"burst": "05_end_to_end_burst.csv", "sustained": "05_end_to_end.csv"}
TOL = {"burst": 0.07, "sustained": 0.10}  # 05_end_to_end.md: the sustained regime varies more with run composition


def mxfp4(path):
    with open(path) as f:
        if f.readline().startswith("#"):
            pass
        else:
            f.seek(0)
        return {int(r["prompt_tokens"]): float(r["ttft_ms_median"]) for r in csv.DictReader(f)
                if r["method"] == "mlx_mxfp4"}


def main(path, regime):
    ref = mxfp4(os.path.join(HERE, "..", "results", REF[regime]))
    got = mxfp4(path)
    ok = True
    for L, t in sorted(got.items()):
        if L in ref:
            r = t / ref[L]
            bad = r > 1 + TOL[regime]
            ok &= not bad
            print(f"anchor {regime} L={L}: MLX MXFP4 {t:.1f} ms vs clean 05 {ref[L]:.1f} ms ({r:.3f}x){' REJECT' if bad else ''}")
    return ok


if __name__ == "__main__":
    sys.exit(0 if main(sys.argv[1], sys.argv[2]) else 1)
