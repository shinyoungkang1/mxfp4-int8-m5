"""Paired bootstrap for run_26 (codebooks and down_proj-only rotation) against the baselines in runs 17, 21, 22 and 23.

Same method as bootstrap_ppl.py: windows resampled jointly, 95% percentile interval of ppl(a) - ppl(b), 10000 resamples. Negative means a is better.
Usage: python bootstrap_26.py --out ../results/26_bootstrap.csv
"""
import argparse
import csv
import glob
import os

import numpy as np

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
E = "w-fmt-e2m1_B8_M1..10_a-8"
GE = "w-gptq-e2m1_B8_M1..10_a-8"
GS = "w-gptq-s10_B8_M1..12_a-8"
D = "r-128_rp-mlp.down_proj"
PAIRS = [(f"w-fmt-c10_B8_M1..12_a-8", E), (f"w-fmt-c9_B8_M1..14_a-8", E), (f"w-fmt-s10_B8_M1..12_a-8", E),
         ("w-fmt-c10_B8_M1..12_a-8", "w-nvfp4_search_a-8"), ("w-fmt-c9_B8_M1..14_a-8", "w-nvfp4_search_a-8"),
         (f"w-bf16_a-8_{D}", "w-bf16_a-8"), (f"w-bf16_a-8_{D}", "w-bf16_a-8_r-32"),
         ("w-bf16_a-8_r-32_rp-mlp.down_proj", "w-bf16_a-8"), (f"w-bf16_a-8_{D}", "w-bf16_a-8_r-32_rp-mlp.down_proj"),
         (GS, GE), (f"{GE}_{D}", GE), (f"{GS}_{D}", GS), (f"{GS}_{D}", f"{GE}_{D}"),
         (f"{GE}_{D}", f"{GE}_r-32"), (f"{GE}_{D}", "w-gptq-nvfp4_search_a-8"), (GE, "w-gptq-nvfp4_search_a-8"),
         (f"{GE}_r-32", "w-gptq-nvfp4_search_a-8_r-32"), (f"{GE}_{D}", E)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    data = {}
    for d in ("17", "19", "21", "22", "23", "26"):
        for fn in glob.glob(os.path.join(R, f"{d}_windows", "*.npz")):
            repo, method = os.path.basename(fn)[:-4].split("__")
            z = np.load(fn)
            data.setdefault(repo, {})[method] = (z["nll"], z["count"])
    # baselines without rotation are stored as a=8 under run 17 for BF16 only as "bf16" with 16-bit activations; flag what is missing rather than guess
    rng = np.random.default_rng(0)
    rows = []
    for repo, ms in sorted(data.items()):
        n = len(next(iter(ms.values()))[0])
        idx = rng.integers(0, n, (10000, n))

        def ppl(m, i=None):
            nll, cnt = ms[m]
            return np.exp(nll[i].sum(-1) / cnt[i].sum(-1)) if i is not None else np.exp(nll.sum() / cnt.sum())
        print(f"{repo}: {n} windows")
        for x, y in PAIRS:
            if x not in ms or y not in ms:
                if x in ms or y in ms:
                    print(f"  missing {y if x in ms else x}")
                continue
            d = ppl(x, idx) - ppl(y, idx)
            lo, hi = np.percentile(d, [2.5, 97.5])
            pt = ppl(x) - ppl(y)
            print(f"  {x:48s} {ppl(x):7.3f} - {y:42s} {ppl(y):7.3f} = {pt:+.4f} [{lo:+.4f}, {hi:+.4f}]" + ("  *" if lo > 0 or hi < 0 else ""))
            rows.append([repo, x, y, f"{ppl(x):.4f}", f"{ppl(y):.4f}", f"{pt:.4f}", f"{lo:.4f}", f"{hi:.4f}"])
    if a.out:
        with open(a.out, "w", newline="") as fh:
            wr = csv.writer(fh)
            wr.writerow(["model", "a", "b", "ppl_a", "ppl_b", "diff", "ci_lo", "ci_hi"])
            wr.writerows(rows)


if __name__ == "__main__":
    main()
