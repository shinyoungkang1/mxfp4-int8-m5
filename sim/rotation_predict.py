"""Summarize results/23_rotbound_*.csv and print the perplexity predicted for each block size from the b=0 gap alone."""
import csv, math
import numpy as np
M = {"q17": ("Qwen3-1.7B", 9.3976, 9.6364, 9.4364), "q4": ("Qwen3-4B", 7.8988, 8.1290, 7.9516), "smol": ("SmolLM3", 7.9059, 7.9967, 7.9162)}
for t, (name, bf, a8, r32) in M.items():
    R = list(csv.DictReader(open(f"../results/23_rotbound_{t}.csv")))
    for r in R:
        for k in "x2 e2 wx2 we2 pred linf2 upper_viol lower_viol peak_ratio rho crest2 tokens n".split():
            r[k] = float(r[k])
    print(f"\n{name}: {len(R)//6} linear layers; sampled tokens per layer {int(R[0]['tokens'])}")
    base = {(r["layer"], r["path"]): r for r in R if r["b"] == "0"}
    gap0 = math.log(a8 / bf)
    print(" b   G_step  G_actual  median-layer pred/actual  upper-viol  lower-viol  peak/lo  sqrt(2ln2n/d)  rho   crest^2   predicted ppl")
    for b in ["0", "8", "16", "32", "64", "128"]:
        rs = [r for r in R if r["b"] == b]
        rel = lambda r: r["we2"] / r["wx2"]
        relp = lambda r: r["pred"] / r["wx2"]
        G_act = sum(rel(base[(r["layer"], r["path"])]) for r in rs) / sum(rel(r) for r in rs)
        G_stp = sum(relp(base[(r["layer"], r["path"])]) for r in rs) / sum(relp(r) for r in rs)
        pa = np.median([relp(r) / rel(r) for r in rs])
        tok = sum(r["tokens"] for r in rs)
        uv, lv = sum(r["upper_viol"] for r in rs) / tok, sum(r["lower_viol"] for r in rs) / tok
        pk = sum(r["peak_ratio"] for r in rs) / tok
        rho = sum(r["rho"] for r in rs) / tok
        cr = sum(r["crest2"] for r in rs) / tok
        hi = np.mean([math.sqrt(2 * math.log(2 * r["n"] / 0.01)) for r in rs])
        print(f"{b:>4} {G_stp:7.2f} {G_act:8.2f} {pa:10.3f} {uv:18.5f} {lv:10.5f} {pk:8.2f} {hi:10.2f} {rho:8.2f} {cr:8.1f}   {bf * math.exp(gap0 / G_act):.4f}")
    print("   gain by layer type at b=32:")
    for p in sorted({r["path"] for r in R}):
        rs = [r for r in R if r["b"] == "32" and r["path"] == p]
        g = sum(base[(r["layer"], p)]["we2"] / r["wx2"] for r in rs) / sum(r["we2"] / r["wx2"] for r in rs)
        rel0 = np.mean([base[(r["layer"], p)]["we2"] / r["wx2"] for r in rs])
        cr0 = np.mean([base[(r["layer"], p)]["crest2"] / base[(r["layer"], p)]["tokens"] for r in rs])
        print(f"     {p:20s} G {g:6.2f}  relative output error unrotated {rel0:.2e}  crest^2 {cr0:7.1f}")
    print(f"   measured at b=32 (already run): {r32:.4f}; gap ratio {gap0 / math.log(r32 / bf):.2f}")
