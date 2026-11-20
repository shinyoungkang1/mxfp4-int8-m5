"""Paired bootstrap for E2M1 8-wide on the 125B model against the runs that saved 145 per-window NLLs (runs 11 and 30). Same method as bootstrap_ppl.py. Negative means a is better.
NVFP4 with search (a=8), the 16-wide format (a=8), SC-MXFP4 and FP8 have no saved full-scope windows, so they have no interval here. Usage: python bootstrap_30.py
"""
import csv, os
import numpy as np

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
a = np.load(f"{R}/11_ppl_qwen38_windows_1792499108.npz")
b = np.load(f"{R}/11_ppl_qwen38_windows_1792502828.npz")
c = np.load(f"{R}/30_ppl_qwen38_windows_1793874503.npz")
F = "w=fmt:e2m1_B8_M1..10;a=8"
E = F + ";s=experts"
PAIRS = [(F, c, "w=int4_g64;a=8", b), (F, c, "w=mx_spec;a=8", b), (F, c, E, c)] + [(E, c, m, a) for m in (
    "w=fmt:e2m1_B16_M1..10;a=16;s=experts", "w=sc_mx;a=16;s=experts", "w=nvfp4_search;a=16;s=experts", "w=int4_g64;a=8;s=experts", "w=mx_spec;a=8;s=experts")]
TOK = 2048  # tokens per window; reproduces the perplexities in 11_ppl_qwen38.csv and 30_ppl_qwen38.csv
idx = np.random.default_rng(0).integers(0, 145, (10000, 145))
ppl = lambda v, j=None: np.exp((v[j].sum(-1) if j is not None else v.sum()) / (145 * TOK))
rows = [["a", "b", "ppl_a", "ppl_b", "diff", "ci_lo", "ci_hi"]]
for x, zx, y, zy in PAIRS:
    d = ppl(zx[x], idx) - ppl(zy[y], idx)
    lo, hi = np.percentile(d, [2.5, 97.5])
    p = ppl(zx[x]) - ppl(zy[y])
    print(f"{x:38s}{ppl(zx[x]):.4f} - {y:38s}{ppl(zy[y]):.4f} = {p:+.4f} [{lo:+.4f},{hi:+.4f}]" + ("*" if lo > 0 or hi < 0 else ""))
    rows.append([x, y, f"{ppl(zx[x]):.4f}", f"{ppl(zy[y]):.4f}", f"{p:.4f}", f"{lo:.4f}", f"{hi:.4f}"])
csv.writer(open(f"{R}/30_bootstrap_qwen38.csv", "w", newline="")).writerows(rows)
