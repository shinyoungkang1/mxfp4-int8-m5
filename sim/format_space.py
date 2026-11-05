"""The design space of integer-exact 4-bit formats (formats.py), wider than format_sweep.py's hand-picked list.

w[n, k] = base[n] * m[n, b(k)] * sign * C[idx], max(C) * max(M) <= 127. Varies the codebook C (fixed shapes and Lloyd
shapes rounded to integers), the multiplier set M and the block size B; every format uses all of the INT8 budget, plus an
"M ladder" of e2m1/B16 with growing M. Also, per codebook and block size, an oracle with a free float scale per block (no INT8
constraint, no block-code cost): the part of the error that better multipliers could never remove. Scored as in
format_sweep.py: tr(dW H dW^T) / tr(W H W^T) and weight MSE, every `stride`-th layer. Appends one CSV row per format (resumable).

Usage: python format_space.py <hf-repo> <out.csv> [--stride 3] [--screen]   (--screen: B16 and B32 only)
"""
import argparse
import csv
import glob
import math
import os
import sys

import mlx.core as mx
import numpy as np
from huggingface_hub import snapshot_download

import calib
import format_sweep as FS
import formats as F
import mx_int8 as q
from gpu_guard import eval_step

SHAPES = {
    "e2m1": (0, 1, 2, 3, 4, 6, 8, 12),   # doubled E2M1: the smallest integers with this shape, so M <= 10
    "int4": (0, 1, 2, 3, 4, 5, 6, 7),
    "s8": (0, 1, 2, 3, 4, 5, 6, 8),
    "s10": (0, 1, 2, 3, 4, 6, 8, 10),
    "s14": (0, 1, 2, 3, 5, 7, 10, 14),
    "s18": (0, 1, 2, 4, 6, 9, 13, 18),
    "s25": (0, 1, 2, 4, 7, 11, 17, 25),
}
BLOCKS = (8, 16, 32, 64)


def ladder8(mmax):
    """Eight integer multipliers from 1 to mmax, roughly geometric (3-bit block codes)."""
    vals = sorted({max(1, round(mmax ** (i / 7))) for i in range(8)})
    i = 2
    while len(vals) < 8 and i <= mmax:
        if i not in vals:
            vals = sorted(vals + [i])
        i += 1
    return tuple(vals)


def ladder4(mmax):
    return tuple(sorted({max(1, round(mmax ** (i / 3))) for i in range(4)}))


def lloyd_shape(tensors, cmax, stride=6, max_blocks=2_000_000, block=16, iters=40):
    """Lloyd levels of |w| / (block peak), the ends fixed at 0 and cmax, rounded to distinct integers."""
    xs = []
    for key in sorted(tensors):
        if ".layers." in key and key.endswith(".weight") and any(s in key for s in FS.LINEARS):
            if int(key.split(".layers.")[1].split(".")[0]) % stride:
                continue
            w = np.array(tensors[key].astype(mx.float32))
            b = np.abs(w[:, : w.shape[1] // block * block]).reshape(-1, block)
            p = b.max(1, keepdims=True)
            xs.append((b[p[:, 0] > 0] / p[p[:, 0] > 0]).reshape(-1))
    x = np.concatenate(xs)
    x = np.random.default_rng(0).choice(x, min(x.size, max_blocks), replace=False)
    lv = np.linspace(0, 1, 8)
    for _ in range(iters):
        mids = (lv[1:] + lv[:-1]) / 2
        idx = np.searchsorted(mids, x)
        for i in range(1, 7):
            if (idx == i).any():
                lv[i] = x[idx == i].mean()
    out = [0]
    for v in lv[1:-1] * cmax:   # distinct, increasing integers
        out.append(max(out[-1] + 1, int(round(v))))
    out.append(max(out[-1] + 1, cmax))
    return tuple(out)


def space(tensors, blocks):
    shapes = dict(SHAPES)
    for cmax in (10, 12, 16, 20):
        shapes[f"lloyd{cmax}"] = lloyd_shape(tensors, cmax)
    fmts = []
    for name, c in shapes.items():
        F.CODEBOOKS[name] = np.array(c, np.float32)
        mmax = 127 // max(c)
        for b in blocks:
            sets = {tuple(range(1, mmax + 1)), ladder8(mmax), ladder4(mmax)}
            fmts += [F.Fmt(name, b, s) for s in sorted(sets, key=len, reverse=True)]
    # the M ladder: how much each extra multiplier buys, up to the INT8 limit of the doubled-E2M1 shape
    for mmax in (1, 2, 3, 4, 5, 6, 7, 8, 9, 10):
        for b in (16, 32):
            fmts.append(F.Fmt("e2m1", b, tuple(range(1, mmax + 1))))
    seen, out = set(), []
    for f in fmts:
        if f not in seen:
            seen.add(f)
            out.append(f)
    return shapes, out


def oracle(w, cb, block, per_octave=16, octaves=2, rows_per_chunk=512):
    """Every block with its own free float scale (least squared error of a grid below peak / max(C))."""
    cb_mx, mids = mx.array(cb), mx.array((cb[1:] + cb[:-1]) / 2)
    n, k = w.shape
    nb = k // block
    out = []
    for r0 in range(0, n, rows_per_chunk):
        wb = mx.array(np.ascontiguousarray(w[r0:r0 + rows_per_chunk], dtype=np.float32)).reshape(-1, nb, block)
        peak = mx.abs(wb).max(axis=-1, keepdims=True)
        peak = mx.where(peak == 0, 1.0, peak)
        best_e, best = mx.full(peak.shape[:2], float("inf")), wb
        for j in range(per_octave * octaves):
            s = peak / float(cb[-1]) * 2.0 ** (-j / per_octave)
            d = F._round_to_codebook_mx(wb / s, cb_mx, mids) * s
            e = ((d - wb) ** 2).sum(-1)
            better = e < best_e
            best_e, best = mx.where(better, e, best_e), mx.where(better[..., None], d, best)
            mx.eval(best_e, best)
        out.append(np.array(best).reshape(-1, k))
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("repo")
    ap.add_argument("out")
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--screen", action="store_true")
    a = ap.parse_args()
    calib.compute(a.repo)
    path = snapshot_download(a.repo, allow_patterns=["*.safetensors"])
    tensors = {}
    for fn in sorted(glob.glob(os.path.join(path, "*.safetensors"))):
        tensors.update(mx.load(fn))
    blocks = (16, 32) if a.screen else BLOCKS
    shapes, fmts = space(tensors, blocks)
    print("codebooks:", {k: v for k, v in shapes.items()}, f"; {len(fmts)} formats", flush=True)
    nl = len([k for k in tensors if k.endswith("self_attn.q_proj.weight") and ".layers." in k])
    mats = []
    for i in range(0, nl, a.stride):
        for lin in FS.LINEARS:
            key = f"model.layers.{i}.{lin}.weight"
            if key in tensors:
                w = np.array(tensors[key].astype(mx.float32))
                mats.append((w, mx.array(calib.hessian(a.repo, i, lin).astype(np.float32))))
    base = sum(FS._out_err(w, h) for w, h in mats)
    wden = sum(float((w ** 2).sum()) for w, _ in mats)
    bytes_w = sum(w.size for w, _ in mats)

    done = set()
    if os.path.exists(a.out):
        with open(a.out) as fh:
            done = {r["format"] for r in csv.DictReader(fh)}
    new = not os.path.exists(a.out)
    fh = open(a.out, "a", newline="")
    wr = csv.writer(fh)
    if new:
        wr.writerow(["model", "format", "codebook", "cmax", "block", "n_mult", "mmax", "output_rel_err", "weight_rel_mse",
                     "bits_per_weight", "kind"])

    def score(fn):
        num = wnum = 0.0
        with eval_step():
            for w, h in mats:
                d = fn(w).astype(np.float32) - w
                num += FS._out_err(d, h)
                wnum += float((d ** 2).sum())
        return num / base, wnum / wden

    refs = {
        "SC-MXFP4": (F.sc_mxfp4, 4 + 2 / 32),
        "MXFP4 (OCP rule)": (lambda w: q.dequant_mx(w.astype(np.float64), q.block_exponents(w, "spec"))[0], 4.25),
        "NVFP4": (lambda w: q.nvfp4(w.astype(np.float64)), 4.5),
        "NVFP4, scale search": (lambda w: q.nvfp4_search(w.astype(np.float64)), 4.5),
    }
    for name, (fn, bits) in refs.items():
        if name not in done:
            e, m = score(fn)
            wr.writerow([a.repo, name, "", "", "", "", "", e, m, bits, "reference"])
            fh.flush()
    for name in shapes:
        cb = np.array(shapes[name], np.float32)
        for b in blocks:
            label = f"oracle {name}/B{b}"
            if label not in done:
                e, m = score(lambda w: oracle(w, cb, b))
                wr.writerow([a.repo, label, name, int(cb[-1]), b, "", "", e, m, "", "oracle"])
                fh.flush()
    for i, f in enumerate(fmts):
        label = f"{f.codebook}/B{f.block}/M" + ",".join(map(str, f.mults))
        if label in done:
            continue
        e, m = score(lambda w: F.quantize_mx(w, f)[0])
        bits = sum(f.bits(w.shape[1]) * w.size for w, _ in mats) / bytes_w
        wr.writerow([a.repo, label, f.codebook, int(f.cmax), f.block, len(f.mults), max(f.mults), e, m, bits, "format"])
        fh.flush()
        print(f"{i + 1}/{len(fmts)} {label}: {1000 * e:.2f} (bits {bits:.3f})", flush=True)
    print("done", math.nan if not fmts else len(fmts), file=sys.stderr)


if __name__ == "__main__":
    main()
