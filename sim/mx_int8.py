"""MXFP4 weight quantization and its exact embedding into INT8 tensor-unit operands.

MXFP4 stores w = 2^e_b * v, with v in E2M1 = {0, ±0.5, ±1, ±1.5, ±2, ±3, ±4, ±6} and one E8M0 exponent
e_b per 32-wide block b along K. The INT8 path wants one scale per (row, K-chunk) instead, so within a
chunk every block is rewritten on the chunk's lowest scale E_lo:

    w = 2^(E_lo - 1) * q,   q = 2 v * 2^(e_b - E_lo)   (an integer, since 2v is an integer)

q fits INT8 (|q| <= 127) whenever 12 * 2^(e_b - E_lo) <= 127, i.e. the block exponents inside a chunk
span at most 3. "Spread-constrained" MXFP4 (SC-MXFP4) picks exponents under exactly that constraint, so
it is still a standard MXFP4 tensor, yet runs bit-exactly as INT8 x INT8 with one FP scale per chunk.

Conventions: weight matrices are (N, K) = (out_features, in_features); blocks run along K.
"""
import numpy as np

BLOCK = 32
MAX_SPREAD = 3  # 12 * 2**3 = 96 <= 127
E2M1 = np.array([0, 0.5, 1, 1.5, 2, 3, 4, 6])
_MIDS = (E2M1[1:] + E2M1[:-1]) / 2


def round_e2m1(x):
    """Round to the nearest E2M1 value, saturating at ±6."""
    return np.sign(x) * E2M1[np.searchsorted(_MIDS, np.minimum(np.abs(x), 6.0))]


def blocks(w, bs=BLOCK):
    n, k = w.shape
    assert k % bs == 0, f"K={k} not a multiple of {bs}"
    return w.reshape(n, k // bs, bs)


def _sq_err(b, e):
    v = round_e2m1(b / 2.0 ** e[..., None])
    return ((v * 2.0 ** e[..., None] - b) ** 2).sum(-1)


def block_exponents(w, rule="spec"):
    """Per-block E8M0 exponents, shape (N, K/32).

    spec:   OCP MX v1.0, floor(log2 amax) - 2 (can clip the block max to 6)
    ceil:   ceil(log2(amax / 6)), never clips (BaseRT's converter)
    search: best of spec-1, spec, spec+1 by squared error
    """
    b = blocks(w)
    amax = np.abs(b).max(-1)
    with np.errstate(divide="ignore"):
        lg = np.log2(amax)
    zero = amax == 0
    lg[zero] = 0
    if rule == "ceil":
        e = np.ceil(lg - np.log2(6.0))
    else:
        e = np.floor(lg) - 2
    if rule == "search":
        best, err = e.copy(), _sq_err(b, e)
        for d in (-1, 1):
            cand = e + d
            ce = _sq_err(b, cand)
            better = ce < err
            best[better], err[better] = cand[better], ce[better]
        e = best
    e[zero] = -np.inf  # all-zero block: any exponent works
    return e


def constrain_spread(w, e, chunk_blocks=None, max_spread=MAX_SPREAD):
    """Raise exponents so that within each chunk of `chunk_blocks` blocks (None = whole row) they span at
    most `max_spread`. Raising an exponent only coarsens that (relatively small) block; the block keeps a
    valid MXFP4 encoding. Re-checks spec/spec+1 style choices are not needed: blocks moved up are the
    small-magnitude ones, and their best exponent subject to e >= lo is lo itself."""
    n, nb = e.shape
    c = nb if chunk_blocks is None else chunk_blocks
    assert nb % c == 0
    ec = e.reshape(n, nb // c, c)
    hi = ec.max(-1, keepdims=True)
    hi[~np.isfinite(hi)] = 0  # all-zero chunk
    lo = hi - max_spread
    out = np.maximum(ec, lo)  # -inf (zero blocks) -> lo
    return out.reshape(n, nb)


def dequant_mx(w, e):
    """MXFP4 fake-quantization: returns (dequantized weights, E2M1 values)."""
    b = blocks(w)
    ee = np.where(np.isfinite(e), e, 0.0)[..., None]
    v = round_e2m1(b / 2.0 ** ee)
    return (v * 2.0 ** ee).reshape(w.shape), v


def embed_int8(v, e, chunk_blocks=None):
    """Exact INT8 embedding of MXFP4 values `v` (N, K/32, 32) with exponents `e` (N, K/32).

    Returns q (N, K) int8 and chunk scales (N, n_chunks) such that, within chunk j,
    w = scales[:, j] * q. Raises if some chunk's spread exceeds 3 (not embeddable without loss)."""
    n, nb, _ = v.shape
    c = nb if chunk_blocks is None else chunk_blocks
    ee = np.where(np.isfinite(e), e, -np.inf).reshape(n, nb // c, c)
    hi = ee.max(-1, keepdims=True)
    hi[~np.isfinite(hi)] = 0
    finite = np.where(np.isfinite(ee), ee, hi)
    lo = finite.min(-1, keepdims=True)
    shift = (finite - lo).reshape(n, nb, 1)
    q = 2 * v * 2.0 ** shift
    if np.abs(q).max() > 127:
        raise ValueError(f"chunk exponent spread {int(shift.max())} > {MAX_SPREAD}: not INT8-embeddable")
    assert np.all(q == np.round(q))
    return q.reshape(n, -1).astype(np.int8), 2.0 ** (lo[..., 0] - 1)


def exponent_spread(e, chunk_blocks=None):
    """Max minus min finite block exponent per (row, chunk)."""
    n, nb = e.shape
    c = nb if chunk_blocks is None else chunk_blocks
    ec = e.reshape(n, nb // c, c)
    fin = np.isfinite(ec)
    hi = np.where(fin, ec, -np.inf).max(-1)
    lo = np.where(fin, ec, np.inf).min(-1)
    s = hi - lo
    s[~np.isfinite(s)] = 0
    return s


# ---- reference formats -------------------------------------------------------------------------

def _q_e4m3(x):
    x = np.clip(x, 2.0 ** -9, 448.0)
    e = np.maximum(np.floor(np.log2(x)), -6)
    step = 2.0 ** (e - 3)
    return np.round(x / step) * step


def nvfp4(w):
    """NVFP4: 16-wide blocks, E4M3 block scale, FP32 per-tensor scale."""
    b = blocks(w, 16)
    g = max(np.abs(w).max(), 1e-30) / (6 * 448)
    amax = np.abs(b).max(-1, keepdims=True)
    s = _q_e4m3(amax / 6 / g) * g
    return (round_e2m1(b / s) * s).reshape(w.shape)


def _e4m3_neighbors(x, steps):
    """E4M3 values `steps` ulps above/below x (x already on the E4M3 grid, positive)."""
    out = []
    for d in steps:
        y = x.copy()
        for _ in range(abs(d)):
            e = np.maximum(np.floor(np.log2(np.maximum(y, 2.0 ** -9))), -6)
            ulp = 2.0 ** (e - 3)
            if d > 0:
                y = y + ulp
            else:  # step down: the ulp below a power of two is half as large
                down = np.where(y == 2.0 ** e, ulp / 2, ulp)
                y = np.maximum(y - down, 2.0 ** -9)
        out.append(np.clip(y, 2.0 ** -9, 448.0))
    return out


def nvfp4_search(w, steps=(-2, -1, 0, 1, 2)):
    """NVFP4 with the block scale chosen by squared-error search over neighbouring E4M3 values (the same kind of
    scale search the integer-exact formats use), instead of the amax/6 rule."""
    b = blocks(w, 16)
    g = max(np.abs(w).max(), 1e-30) / (6 * 448)
    amax = np.abs(b).max(-1, keepdims=True)
    s0 = _q_e4m3(amax / 6 / g)
    best, best_err = None, None
    for s in _e4m3_neighbors(s0, steps):
        sc = s * g
        dq = round_e2m1(b / sc) * sc
        err = ((dq - b) ** 2).sum(-1, keepdims=True)
        if best is None:
            best, best_err = dq, err
        else:
            m = err < best_err
            best = np.where(m, dq, best)
            best_err = np.where(m, err, best_err)
    return best.reshape(w.shape)


def int4_affine(w, group=64):
    """MLX-style 4-bit affine (asymmetric) quantization with per-group scale and bias."""
    b = blocks(w, group)
    lo, hi = b.min(-1, keepdims=True), b.max(-1, keepdims=True)
    s = np.maximum(hi - lo, 1e-12) / 15
    return (np.clip(np.round((b - lo) / s), 0, 15) * s + lo).reshape(w.shape)


def rel_err(w, wq):
    return float(((wq - w) ** 2).sum() / (w ** 2).sum())


def self_test(seed=0):
    rng = np.random.default_rng(seed)
    w = rng.standard_normal((64, 1024)) * rng.lognormal(0, 1.0, (1, 1024))
    for chunk in (None, 8):
        e = constrain_spread(w, block_exponents(w, "search"), chunk)
        assert exponent_spread(e, chunk).max() <= MAX_SPREAD
        wq, v = dequant_mx(w, e)
        q, sc = embed_int8(v, e, chunk)
        c = q.shape[1] if chunk is None else chunk * BLOCK
        recon = (q.reshape(64, -1, c) * sc[..., None]).reshape(w.shape)
        assert np.array_equal(recon, wq), "INT8 embedding is not exact"
        # GEMM check: x @ W^T via per-chunk INT8 partial sums equals x @ Wq^T exactly for integer x.
        x = rng.integers(-127, 128, (8, 1024)).astype(np.float64)
        part = np.einsum("mjc,njc->mnj", x.reshape(8, -1, c), q.reshape(64, -1, c).astype(np.float64))
        assert np.array_equal((part * sc[None]).sum(-1), x @ wq.T)
    print("self-test passed: SC-MXFP4 -> INT8 embedding is bit-exact (whole-row and 256-wide chunks)")


if __name__ == "__main__":
    self_test()
