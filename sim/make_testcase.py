"""Write a linear-layer test case for kernels/bench_layer: activations, SC-MXFP4 weights in the kernel's
packed layout, and the exact reference output.

Usage: python make_testcase.py M N K outdir

Packed weight layout (W is N x K, K-contiguous):
  codes  uint8 [N, K/2]   two 4-bit codes per byte, even k in the low nibble; code = sign<<3 | e2m1 index
  shifts uint8 [N, K/32]  e_b - E_lo[n] in 0..3 (spread-constrained), one per 32-block
  shifts2 uint8 [N, K/128] the same shifts packed 2 bits each, block j of a row in bits 2*(j%4)
  wscale f32   [N]        2^(E_lo[n] - 1), so W[n, k] = wscale[n] * (±LUT[idx] << shift), LUT = 2v
Also written: x f16 [M, K]; xq int8 [M, K] and xscale f32 [M] (per-token INT8, reference for the
activation kernel); wq int8 [N, K] (unpacked weights, for the W8A8 ceiling); wdq f16 [N, K] (dequantized
weights, for the FP16 baseline); y_ref f32 [M, N] = xscale * wscale * (xq @ wq^T), computed exactly.
"""
import json
import os
import sys

import numpy as np

import mx_int8 as q

LUT2V = np.array([0, 1, 2, 3, 4, 6, 8, 12])


def pack(w):
    e = q.constrain_spread(w, q.block_exponents(w, "search"))
    _, v = q.dequant_mx(w, e)
    qw, sc = q.embed_int8(v, e)  # whole-row chunks: sc is [N, 1]
    n, k = w.shape
    fin = np.where(np.isfinite(e), e, -np.inf)
    hi = fin.max(-1, keepdims=True)
    lo = np.where(np.isfinite(fin), fin, hi).min(-1, keepdims=True)
    shifts = (np.where(np.isfinite(fin), fin, lo) - lo).astype(np.uint8)
    v = v.reshape(n, k)
    idx = np.searchsorted(q.E2M1, np.abs(v)).astype(np.uint8)
    assert np.array_equal(q.E2M1[idx], np.abs(v))
    codes = idx | ((v < 0).astype(np.uint8) << 3)
    packed = (codes[:, 0::2] | (codes[:, 1::2] << 4)).astype(np.uint8)
    # Self-check the layout decodes to qw.
    sh = np.repeat(shifts, q.BLOCK, axis=1)
    dec = np.where(codes & 8, -1, 1) * (LUT2V[codes & 7] << sh)
    assert np.array_equal(dec, qw)
    return packed, shifts, sc[:, 0].astype(np.float32), qw


def main(m, n, k, out):
    rng = np.random.default_rng(0)
    # LLM-like: per-channel scale spread, a few outlier activation channels.
    w = rng.standard_normal((n, k)) * 0.02 * rng.lognormal(0, 0.3, (1, k))
    x = rng.standard_normal((m, k)) * rng.lognormal(0, 0.5, (1, k))
    x[:, rng.choice(k, 8, replace=False)] *= 20
    x = x.astype(np.float16)

    packed, shifts, wscale, wq = pack(w)
    xf = x.astype(np.float64)
    xscale = np.abs(xf).max(1) / 127
    xq = np.clip(np.rint(xf / xscale[:, None]), -127, 127).astype(np.int8)
    acc = xq.astype(np.float64) @ wq.astype(np.float64).T  # exact: |acc| < 127 * 96 * K << 2^53
    y = (acc * xscale[:, None].astype(np.float32) * wscale[None, :]).astype(np.float32)
    wdq = (wq * wscale[:, None]).astype(np.float16)

    os.makedirs(out, exist_ok=True)
    assert shifts.shape[1] % 4 == 0
    s4 = shifts.reshape(n, -1, 4).astype(np.uint8)
    shifts2 = (s4[..., 0] | (s4[..., 1] << 2) | (s4[..., 2] << 4) | (s4[..., 3] << 6)).astype(np.uint8)
    for name, arr in dict(x=x, codes=packed, shifts=shifts, shifts2=shifts2, wscale=wscale, xq=xq,
                          xscale=xscale.astype(np.float32), wq=wq, wdq=wdq, y_ref=y).items():
        np.ascontiguousarray(arr).tofile(os.path.join(out, name + ".bin"))
    json.dump(dict(M=m, N=n, K=k), open(os.path.join(out, "meta.json"), "w"))
    bits = lambda sh: (packed.nbytes + sh.nbytes + wscale.nbytes) * 8 / (n * k)
    print(f"wrote {out}: M={m} N={n} K={k}, weight bits/value = {bits(shifts):.3f} (byte shifts), {bits(shifts2):.3f} (2-bit shifts)")


if __name__ == "__main__":
    main(*map(int, sys.argv[1:4]), sys.argv[4])
