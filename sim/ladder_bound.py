"""Checks the ladder bound against the measured M ladder (results/16_space_*.csv). A block with peak a in a row with peak A,
ladder {1..M}, beta = A / (Cmax M): m = ceil(a / (beta Cmax)) never clips and s = beta m < a / Cmax + beta, so its scale is within a
factor 1 + A / (M a) of the free scale a / Cmax; the worst-case error scales with s. Powers of two: within a factor 2.
Prints, per M, the energy-weighted squared factor sum s^2 / sum (a / Cmax)^2 from real block peaks (B = 16), and the measured weight MSE.
Usage: python ladder_bound.py <hf-repo> <space.csv>"""
import csv, glob, os, sys
import numpy as np, mlx.core as mx
from huggingface_hub import snapshot_download
repo, space = sys.argv[1:3]
t = {}
for fn in glob.glob(os.path.join(snapshot_download(repo, allow_patterns=["*.safetensors"]), "*.safetensors")):
    t.update(mx.load(fn))
peaks = []
for k, v in t.items():
    if ".layers." in k and k.endswith(".weight") and any(s in k for s in ("_proj.weight",)) and int(k.split(".layers.")[1].split(".")[0]) % 3 == 0:
        w = np.abs(np.array(v.astype(mx.float32)))
        K = w.shape[1] // 16 * 16
        p = w[:, :K].reshape(w.shape[0], -1, 16).max(-1)
        A = p.max(1, keepdims=True)
        A[A == 0] = 1
        peaks.append((p / A).reshape(-1))
r = np.concatenate(peaks)
print("block peak / row peak: median %.3f, 10%% %.3f, 90%% %.3f; n=%d" % (*np.percentile(r, [50, 10, 90]), r.size))
meas = {}
for row in csv.DictReader(open(space)):
    if row["kind"] == "format" and row["codebook"] == "e2m1" and row["block"] == "16" and row["format"].endswith("/M" + ",".join(map(str, range(1, int(row["n_mult"]) + 1)))):
        meas[int(row["n_mult"])] = float(row["weight_rel_mse"])
oracle = [float(x["weight_rel_mse"]) for x in csv.DictReader(open(space)) if x["format"] == "oracle e2m1/B16"][0]
for M in sorted(meas):
    s = np.maximum(np.ceil(r * M), 1) / M
    print(f"M={M:2d}  bound factor {np.sum(s ** 2) / np.sum(r ** 2):.3f}   measured MSE / oracle {meas[M] / oracle:.3f}")
s = 2.0 ** np.ceil(np.log2(np.maximum(r, 1 / 8)))
print("pow2 {1,2,4,8} bound factor %.3f" % (np.sum(s ** 2) / np.sum(r ** 2)))
