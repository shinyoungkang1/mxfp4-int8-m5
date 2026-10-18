"""INT8-exact 4-bit weight formats: the design space behind SC-MXFP4.

A weight row n is stored as
    w[n, k] = base[n] * m[n, b(k)] * sign * C[idx]
with a 4-bit element (sign + 3-bit index into an integer magnitude codebook C), a small integer block
multiplier m from a set M (one per B-wide block along K), and one float scale base[n] per row.
Because sign * C[idx] * m is an integer, the row runs exactly as INT8 x INT8 -> INT32 with base[n] applied
once per output, provided max(C) * max(M) <= 127 (or 128 for the negative side). SC-MXFP4 is the special
case C = 2 * E2M1 = {0,1,2,3,4,6,8,12}, B = 32, M = {1,2,4,8} with a power-of-two base.

Storage: 4 bits per element + ceil(log2 |M|) bits per block + 32 bits per row.
"""
import math
from dataclasses import dataclass

import numpy as np

import mx_int8 as q

CODEBOOKS = {
    "e2m1": np.array([0, 1, 2, 3, 4, 6, 8, 12], dtype=np.float32),  # doubled E2M1 magnitudes
    "int4": np.arange(8, dtype=np.float32),                         # symmetric INT4, -7..7
}


@dataclass(frozen=True)
class Fmt:
    codebook: str
    block: int
    mults: tuple  # allowed integer block multipliers, ascending
    base_grid: int = 8  # candidate row bases per octave below the max-fitting one

    @property
    def cmax(self):
        return float(CODEBOOKS[self.codebook].max())

    def bits(self, k):
        return 4 + math.ceil(math.log2(len(self.mults))) / self.block + 32 / k

    def name(self):
        m = self.mults
        if all(x & (x - 1) == 0 for x in m) and list(m) == [2 ** i for i in range(len(m))]:
            ms = f"pow2x{len(m)}"
        elif list(m) == list(range(1, len(m) + 1)):
            ms = f"1..{len(m)}"
        else:
            ms = ",".join(map(str, m))
        return f"{self.codebook}/B{self.block}/M[{ms}]"

    def check(self):
        assert self.cmax * max(self.mults) <= 127, f"{self.name()} overflows INT8"


def _round_to_codebook(x, cb):
    """Nearest codebook magnitude with sign, for x already divided by the scale."""
    mids = (cb[1:] + cb[:-1]) / 2
    return np.sign(x) * cb[np.searchsorted(mids, np.minimum(np.abs(x), cb[-1]))]


def quantize(w, fmt: Fmt, rows_per_chunk=256):
    """Returns (dequantized weights, integer operands q (int8), row scales base, block scales base*m (N, K/B))
    with w ~ base[:,None] * q."""
    fmt.check()
    cb = CODEBOOKS[fmt.codebook]
    mults = np.array(fmt.mults, dtype=np.float32)
    n, k = w.shape
    assert k % fmt.block == 0
    out = np.empty((n, k), np.float32)
    qint = np.empty((n, k), np.int8)
    base_out = np.empty(n, np.float32)
    scales_out = np.empty((n, k // fmt.block), np.float64)  # base * m, exact in float64
    for r0 in range(0, n, rows_per_chunk):
        wb = blocks = w[r0:r0 + rows_per_chunk].astype(np.float32).reshape(-1, k // fmt.block, fmt.block)
        rows = wb.shape[0]
        amax_row = np.abs(wb).max((1, 2))
        amax_row = np.where(amax_row == 0, 1.0, amax_row)
        # Largest base that represents the row max exactly-ish: base * max(M) * cmax = amax_row.
        base_hi = amax_row / (mults[-1] * fmt.cmax)
        best_err = np.full(rows, np.inf, np.float32)
        best_base = base_hi.copy()
        best_m = np.zeros((rows, wb.shape[1]), np.int64)
        for j in range(fmt.base_grid):
            base = base_hi * 2.0 ** (-j / fmt.base_grid)
            # error of every block under every multiplier: (rows, nb, |M|)
            scale = base[:, None, None] * mults[None, None, :]
            err = np.empty((rows, wb.shape[1], len(mults)), np.float32)
            for i in range(len(mults)):
                s = scale[:, :, i:i + 1]
                err[:, :, i] = ((_round_to_codebook(wb / s, cb) * s - wb) ** 2).sum(-1)
            mi = err.argmin(-1)
            tot = np.take_along_axis(err, mi[..., None], -1)[..., 0].sum(-1)
            better = tot < best_err
            best_err[better], best_base[better], best_m[better] = tot[better], base[better], mi[better]
        m = mults[best_m]  # (rows, nb)
        s = best_base[:, None, None] * m[..., None]
        codes = _round_to_codebook(wb / s, cb)
        qi = codes * m[..., None]
        assert np.abs(qi).max() <= 127
        out[r0:r0 + rows] = (qi * best_base[:, None, None]).reshape(rows, k)
        qint[r0:r0 + rows] = qi.reshape(rows, k).astype(np.int8)
        base_out[r0:r0 + rows] = best_base
        scales_out[r0:r0 + rows] = best_base.astype(np.float64)[:, None] * m
    return out, qint, base_out, scales_out


def sc_mxfp4(w):
    """Reference: spread-constrained MXFP4 (standard MXFP4 file, power-of-two scales)."""
    e = q.constrain_spread(w, q.block_exponents(w, "search"))
    return q.dequant_mx(w, e)[0].astype(np.float32)


FORMATS = {
    # standard-MXFP4-compatible reference is handled separately (sc_mxfp4)
    "e2m1/B32/pow2 (SC-MXFP4 grid, float row base)": Fmt("e2m1", 32, (1, 2, 4, 8)),
    "e2m1/B16/pow2": Fmt("e2m1", 16, (1, 2, 4, 8)),
    "e2m1/B32/M1..8": Fmt("e2m1", 32, tuple(range(1, 9))),
    "e2m1/B32/M1..10": Fmt("e2m1", 32, tuple(range(1, 11))),
    "e2m1/B16/M1..8": Fmt("e2m1", 16, tuple(range(1, 9))),
    "e2m1/B16/M1..10": Fmt("e2m1", 16, tuple(range(1, 11))),
    "int4/B32/M1..16": Fmt("int4", 32, tuple(range(1, 17))),
    "int4/B16/M1..8": Fmt("int4", 16, tuple(range(1, 9))),
    "int4/B16/M1..16": Fmt("int4", 16, tuple(range(1, 17))),
}


def self_test(seed=0):
    rng = np.random.default_rng(seed)
    w = (rng.standard_normal((64, 512)) * rng.lognormal(0, 1, (1, 512))).astype(np.float32)
    x = rng.integers(-127, 128, (8, 512)).astype(np.float64)
    for name, f in FORMATS.items():
        wq, qi, base, _ = quantize(w, f)
        assert np.array_equal(qi.astype(np.float32) * base[:, None], wq)  # stored form reproduces the weights
        # one INT8 GEMM with one scale per row reproduces x @ wq^T (up to the float rounding of the product)
        # (the stored form is qi with base; the float32 wq is only a rounded view of base * qi)
        acc = x @ qi.astype(np.float64).T
        ref = x @ (qi.astype(np.float64) * base.astype(np.float64)[:, None]).T
        assert np.allclose(acc * base.astype(np.float64)[None, :], ref, rtol=1e-12, atol=0)
    print(f"self-test passed: {len(FORMATS)} formats are INT8-exact with one scale per row")




# ---- GPU implementation (MLX), same decisions as quantize() -------------------------------------
def _round_to_codebook_mx(x, cb_mx, mids_mx):
    import mlx.core as mx
    a = mx.minimum(mx.abs(x), cb_mx[-1])
    idx = (a[..., None] > mids_mx).astype(mx.int32).sum(-1)  # nearest level; ties go down like searchsorted
    return mx.sign(x) * mx.take(cb_mx, idx)


_STEPS = {}


def _candidate_step(fmt: Fmt):
    """Compiled (MLX) error of one row-base candidate: for every block the multiplier with least squared error and
    the row's total error. Rounding to the codebook is written as len(cb) - 1 selects (nearest level, ties down,
    exactly _round_to_codebook_mx) so that mx.compile fuses it; 2x faster, bit-identical decisions."""
    import mlx.core as mx
    if fmt not in _STEPS:
        cb = [float(c) for c in CODEBOOKS[fmt.codebook]]
        mids = [float(m) for m in (CODEBOOKS[fmt.codebook][1:] + CODEBOOKS[fmt.codebook][:-1]) / 2]
        mults = mx.array(np.array(fmt.mults, np.float32))

        def step(wb, base):
            s = base[:, None, None, None] * mults[None, None, :, None]           # (rows, 1, M, 1)
            x = wb[:, :, None, :] / s
            a = mx.minimum(mx.abs(x), cb[-1])
            mag = mx.full(a.shape, cb[0])
            for t in range(len(cb) - 1):
                mag = mx.where(a > mids[t], cb[t + 1], mag)
            err = ((mx.sign(x) * mag * s - wb[:, :, None, :]) ** 2).sum(-1)     # (rows, nb, M)
            mi = mx.argmin(err, axis=-1)
            return mx.take_along_axis(err, mi[..., None], axis=-1)[..., 0].sum(-1), mi

        _STEPS[fmt] = mx.compile(step)
    return _STEPS[fmt]


def quantize_mx(w, fmt: Fmt, rows_per_chunk=512):
    """MLX/GPU version of quantize(): returns numpy (dequantized float32, int8 operands, row bases, block scales)."""
    import mlx.core as mx
    fmt.check()
    cb = CODEBOOKS[fmt.codebook]
    cb_mx, mids_mx = mx.array(cb), mx.array((cb[1:] + cb[:-1]) / 2)
    mults = mx.array(np.array(fmt.mults, np.float32))
    n, k = w.shape
    nb = k // fmt.block
    outs, qis, bases, scales = [], [], [], []
    for r0 in range(0, n, rows_per_chunk):
        wb = mx.array(np.ascontiguousarray(w[r0:r0 + rows_per_chunk], dtype=np.float32)).reshape(-1, nb, fmt.block)
        rows = wb.shape[0]
        amax_row = mx.abs(wb).max(axis=(1, 2))
        amax_row = mx.where(amax_row == 0, 1.0, amax_row)
        base_hi = amax_row / (float(fmt.mults[-1]) * fmt.cmax)
        best_err = mx.full((rows,), float("inf"))
        best_base, best_mi = base_hi, mx.zeros((rows, nb), mx.int32)
        step = _candidate_step(fmt)
        for j in range(fmt.base_grid):
            base = base_hi * 2.0 ** (-j / fmt.base_grid)
            tot, mi = step(wb, base)
            better = tot < best_err
            best_err = mx.where(better, tot, best_err)
            best_base = mx.where(better, base, best_base)
            best_mi = mx.where(better[:, None], mi, best_mi)
            mx.eval(best_err, best_base, best_mi)
        m = mx.take(mults, best_mi)
        s = best_base[:, None, None] * m[..., None]
        qi = _round_to_codebook_mx(wb / s, cb_mx, mids_mx) * m[..., None]
        mx.eval(qi)
        qi_np = np.array(qi).reshape(rows, k)
        assert np.abs(qi_np).max() <= 127 and np.array_equal(qi_np, np.round(qi_np))
        base_np = np.array(best_base)
        qis.append(qi_np.astype(np.int8)); bases.append(base_np)
        scales.append(base_np.astype(np.float64)[:, None] * np.array(m, np.float64))
        outs.append((qi_np * base_np[:, None]).astype(np.float32))
    return np.concatenate(outs), np.concatenate(qis), np.concatenate(bases), np.concatenate(scales)


def self_test_mx(seed=1):
    rng = np.random.default_rng(seed)
    w = (rng.standard_normal((300, 1024)) * rng.lognormal(0, 1, (1, 1024))).astype(np.float32)
    for name, f in FORMATS.items():
        a, qa, ba, sa = quantize(w, f)
        b, qb, bb, sb = quantize_mx(w, f)
        mism = (qa != qb).mean()
        rel = ((a - w) ** 2).sum() / ((b - w) ** 2).sum()
        assert mism < 1e-3 and abs(rel - 1) < 1e-3, (name, mism, rel)
    print("self-test passed: MLX quantizer matches the numpy reference")


if __name__ == "__main__":
    self_test()
    self_test_mx()
