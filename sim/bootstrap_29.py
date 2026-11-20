"""Paired bootstrap for run_29 (GPTQ with down_proj-only rotation at r=32) against runs 21-23. Same method as bootstrap_26.py. Negative means a is better.
Usage: python bootstrap_29.py --out ../results/29_bootstrap.csv
"""
import argparse, csv, glob, os
import numpy as np

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
G = "w-gptq-e2m1_B8_M1..10_a-8"
D32 = f"{G}_r-32_rp-mlp.down_proj"
PAIRS = [(D32, G), (D32, f"{G}_r-32"), (D32, f"{G}_r-128_rp-mlp.down_proj"), (D32, "w-gptq-nvfp4_search_a-8"), (D32, "w-gptq-nvfp4_search_a-8_r-32")]

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="")
a = ap.parse_args()
data = {}
for d in ("17", "19", "21", "22", "23", "26", "29"):
    for fn in glob.glob(os.path.join(R, f"{d}_windows", "*.npz")):
        repo, method = os.path.basename(fn)[:-4].split("__")
        z = np.load(fn)
        data.setdefault(repo, {})[method] = (z["nll"], z["count"])
rng = np.random.default_rng(0)
rows = []
for repo, ms in sorted(data.items()):
    if D32 not in ms:
        continue
    n = len(ms[D32][0])
    idx = rng.integers(0, n, (10000, n))
    def ppl(m, i=None):
        nll, cnt = ms[m]
        return np.exp(nll[i].sum(-1) / cnt[i].sum(-1)) if i is not None else np.exp(nll.sum() / cnt.sum())
    print(f"{repo}: {n} windows")
    for x, y in PAIRS:
        if y not in ms:
            print("  missing", y)
            continue
        d = ppl(x, idx) - ppl(y, idx)
        lo, hi = np.percentile(d, [2.5, 97.5])
        pt = ppl(x) - ppl(y)
        print(f"  {ppl(x):7.3f} - {y:48s} {ppl(y):7.3f} = {pt:+.4f} [{lo:+.4f}, {hi:+.4f}]" + ("  *" if lo > 0 or hi < 0 else ""))
        rows.append([repo, x, y, f"{ppl(x):.4f}", f"{ppl(y):.4f}", f"{pt:.4f}", f"{lo:.4f}", f"{hi:.4f}"])
if a.out:
    with open(a.out, "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["model", "a", "b", "ppl_a", "ppl_b", "diff", "ci_lo", "ci_hi"])
        wr.writerows(rows)
