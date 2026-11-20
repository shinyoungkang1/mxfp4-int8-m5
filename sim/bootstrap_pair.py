"""Paired bootstrap of perplexity differences between two window files of the same test set (the .npz files that
e2e/ppl_qwen35.py and sim/eval_ppl.py write with --windows-dir: arrays nll and count per window).

Same method as bootstrap_ppl.py: windows resampled jointly, 10000 resamples, seed 0, 95% percentile interval of
ppl(a) - ppl(b). Negative means a is better. Windows of one test set are neighbouring text, so this is an interval
over that test set's windows, not over English.
Usage: python bootstrap_pair.py A.npz B.npz [C.npz D.npz ...] [--out file.csv]   (pairs: A-B, C-D, ...)
"""
import argparse
import csv
import os

import numpy as np


def label(fn):
    return os.path.basename(fn)[:-4].split("__", 1)[-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    if len(a.files) % 2:
        raise SystemExit("give the window files in pairs")
    rows = [["a", "b", "windows", "ppl_a", "ppl_b", "diff", "ci_lo", "ci_hi"]]
    for fa, fb in zip(a.files[::2], a.files[1::2]):
        za, zb = np.load(fa), np.load(fb)
        n = len(za["nll"])
        if len(zb["nll"]) != n or not np.array_equal(za["count"], zb["count"]):
            raise SystemExit(f"{fa} and {fb} do not cover the same windows")
        idx = np.random.default_rng(0).integers(0, n, (10000, n))
        ppl = lambda z, j=None: (np.exp(z["nll"][j].sum(-1) / z["count"][j].sum(-1)) if j is not None  # noqa: E731
                                 else float(np.exp(z["nll"].sum() / z["count"].sum())))
        d = ppl(za, idx) - ppl(zb, idx)
        lo, hi = np.percentile(d, [2.5, 97.5])
        p = ppl(za) - ppl(zb)
        print(f"{label(fa)} {ppl(za):.4f} - {label(fb)} {ppl(zb):.4f} = {p:+.4f} [{lo:+.4f}, {hi:+.4f}]"
              + (" *" if lo > 0 or hi < 0 else ""))
        rows.append([label(fa), label(fb), n, f"{ppl(za):.4f}", f"{ppl(zb):.4f}", f"{p:.4f}", f"{lo:.4f}", f"{hi:.4f}"])
    if a.out:
        with open(a.out, "w", newline="") as f:
            csv.writer(f).writerows(rows)


if __name__ == "__main__":
    main()
