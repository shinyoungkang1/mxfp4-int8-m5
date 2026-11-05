"""Paired bootstrap of perplexity differences over windows (results/17_windows, written by eval_ppl.py --windows-dir).

Both methods see the same windows, so windows are resampled jointly: 95% percentile interval of ppl(a) - ppl(b), 10000 resamples.
Windows of one test set are neighbouring text, not independent samples of English, so this is an interval over the test set's windows.
Usage: python bootstrap_ppl.py <windows-dir> [--out results/17_bootstrap.csv]
"""
import argparse
import csv
import glob
import os

import numpy as np

REF = "w-fmt-e2m1_B16_M1..10_a-8"
PAIRS = [("w-fmt-e2m1_B16_M1..10_a-8", "w-sc_mx_a-8"), ("w-fmt-e2m1_B16_M1..10_a-8", "w-nvfp4_a-8"),
         ("w-fmt-e2m1_B16_M1..10_a-8", "w-nvfp4_search_a-8"), ("w-fmt-e2m1_B8_M1..10_a-8", "w-nvfp4_search_a-8"),
         ("w-fmt-e2m1_B8_M1..10_a-8", "w-nvfp4_a-8"), ("w-fmt-s10_B8_M1..12_a-8", "w-fmt-e2m1_B8_M1..10_a-8"),
         ("w-fmt-s10_B8_M1..12_a-8", "w-nvfp4_search_a-8"), ("w-fmt-e2m1_B8_M1..10_a-8", "w-fmt-e2m1_B16_M1..10_a-8"),
         ("w-fmt-s10_B16_M1..12_a-8", REF), ("w-fmt-s14_B16_M1..9_a-8", REF), ("w-fmt-e2m1_B32_M1..8_a-8", "w-nvfp4_a-8")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    data = {}
    for fn in glob.glob(os.path.join(a.dir, "*.npz")):
        repo, method = os.path.basename(fn)[:-4].split("__")
        z = np.load(fn)
        data.setdefault(repo, {})[method] = (z["nll"], z["count"])
    rng = np.random.default_rng(0)
    rows = []
    for repo, ms in sorted(data.items()):
        n = len(next(iter(ms.values()))[0])
        idx = rng.integers(0, n, (10000, n))
        ppl = lambda m, i=None: np.exp(ms[m][0][i].sum(-1) / ms[m][1][i].sum(-1)) if i is not None else np.exp(ms[m][0].sum() / ms[m][1].sum())  # noqa: E731
        print(f"{repo}: {n} windows; " + ", ".join(f"{m}={ppl(m):.3f}" for m in sorted(ms)))
        for x, y in PAIRS:
            if x in ms and y in ms:
                d = ppl(x, idx) - ppl(y, idx)
                lo, hi = np.percentile(d, [2.5, 97.5])
                pt = ppl(x) - ppl(y)
                print(f"  {x:28s} - {y:28s} {pt:+.4f}  [{lo:+.4f}, {hi:+.4f}]" + ("  *" if lo > 0 or hi < 0 else ""))
                rows.append([repo, x, y, f"{ppl(x):.4f}", f"{ppl(y):.4f}", f"{pt:.4f}", f"{lo:.4f}", f"{hi:.4f}"])
    if a.out:
        with open(a.out, "w", newline="") as fh:
            wr = csv.writer(fh)
            wr.writerow(["model", "a", "b", "ppl_a", "ppl_b", "diff", "ci_lo", "ci_hi"])
            wr.writerows(rows)


if __name__ == "__main__":
    main()
