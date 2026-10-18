"""GPTQ rounding (Frantar et al., 2023) for INT8-exact formats with fixed block scales.

Scales are chosen first from the unquantized weights (formats.quantize_mx / SC-MXFP4 exponents) and then
held fixed; GPTQ only changes which codebook level each weight rounds to, compensating each column's
rounding error on the not-yet-quantized columns using the inverse Hessian. Because scales are unchanged, the
result is still exactly representable as INT8 operands times one scale per row.

    scale[n, b]   per-row, per-block scale (base[n] * m[n, b], or 2^e[n, b] for SC-MXFP4)
    codebook      nonnegative magnitudes in units of the scale (sign applied separately)
"""
import numpy as np


def round_cb(x, cb):
    mids = (cb[1:] + cb[:-1]) / 2
    return np.sign(x) * cb[np.searchsorted(mids, np.minimum(np.abs(x), cb[-1]))]


def gptq(w, h, scale, codebook, block_cols=128, damp=0.01):
    """w (N, K), h (K, K) = X^T X, scale (N, K // B) with B = K // scale.shape[1]. Returns quantized w."""
    w = w.astype(np.float64).copy()
    h = h.astype(np.float64).copy()
    n, k = w.shape
    bsz = k // scale.shape[1]
    cb = np.asarray(codebook, np.float64)
    s_full = np.repeat(scale.astype(np.float64), bsz, axis=1)  # (N, K)

    dead = np.diag(h) == 0
    h[dead, dead] = 1.0
    w[:, dead] = 0.0
    h[np.diag_indices(k)] += damp * np.mean(np.diag(h))
    hinv = np.linalg.cholesky(np.linalg.inv(h)).T  # upper Cholesky factor of H^-1

    qw = np.zeros_like(w)
    for i1 in range(0, k, block_cols):
        i2 = min(i1 + block_cols, k)
        w1 = w[:, i1:i2].copy()
        q1 = np.zeros_like(w1)
        err1 = np.zeros_like(w1)
        hinv1 = hinv[i1:i2, i1:i2]
        for i in range(i2 - i1):
            col = w1[:, i]
            s = s_full[:, i1 + i]
            qc = round_cb(col / s, cb) * s
            q1[:, i] = qc
            e = (col - qc) / hinv1[i, i]
            w1[:, i:] -= e[:, None] * hinv1[i, i:][None, :]
            err1[:, i] = e
        qw[:, i1:i2] = q1
        w[:, i2:] -= err1 @ hinv[i1:i2, i2:]
    return qw


def sc_mxfp4_scales(w):
    """Block scales 2^e of spread-constrained MXFP4 (codebook = E2M1 magnitudes in units of the scale)."""
    import mx_int8 as q
    e = q.constrain_spread(w.astype(np.float64), q.block_exponents(w.astype(np.float64), "search"))
    return np.where(np.isfinite(e), 2.0 ** e, 1.0)


def self_test(seed=0):
    import formats as F
    rng = np.random.default_rng(seed)
    n, k = 64, 512
    x = rng.standard_normal((4096, k)) * rng.lognormal(0, 1, (1, k))  # correlated, uneven channels
    x = x @ (np.eye(k) + 0.1 * rng.standard_normal((k, k)))
    h = x.T @ x / len(x)
    w = rng.standard_normal((n, k)) * 0.05
    out_err = lambda wq: np.einsum("ij,jk,ik->", wq - w, h, wq - w) / np.einsum("ij,jk,ik->", w, h, w)
    for name in ["e2m1/B32/M1..10", "int4/B16/M1..16"]:
        f = F.FORMATS[name]
        wq, qi, base, scale = F.quantize(w.astype(np.float32), f)
        wg = gptq(w, h, scale, F.CODEBOOKS[f.codebook])
        # still INT8-exact: wg / (base) must be integers within +-127 that equal code * multiplier
        qg = wg / base.astype(np.float64)[:, None]
        assert np.array_equal(qg, np.round(qg)) and np.abs(qg).max() <= 127  # exact integers
        print(f"{name}: output error RTN {out_err(wq):.4f} -> GPTQ {out_err(wg):.4f}")
        assert out_err(wg) < out_err(wq)
    # SC-MXFP4 with GPTQ: still a valid spread-constrained MXFP4 tensor, so the INT8 embedding stays exact
    import mx_int8 as q
    scale = sc_mxfp4_scales(w)
    wg = gptq(w, h, scale, q.E2M1)
    e = np.log2(scale)
    _, v = q.dequant_mx(wg, e)
    assert np.array_equal(v.reshape(w.shape) * np.repeat(scale, q.BLOCK, 1), wg)  # wg lies on the MXFP4 grid
    q.embed_int8(v, e)
    wr = q.dequant_mx(w, e)[0]
    print(f"SC-MXFP4: output error RTN {out_err(wr):.4f} -> GPTQ {out_err(wg):.4f}")
    print("self-test passed: GPTQ keeps formats INT8-exact and lowers output error")


if __name__ == "__main__":
    self_test()
