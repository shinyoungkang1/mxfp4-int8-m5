"""Compare MXFP4, NVFP4, and MXFP4-compatible improvements on synthetic weights.

All "MX+" variants keep the exact MXFP4 storage format (32-wide blocks, E2M1
values, E8M0 power-of-two scales); extras are either offline tricks or a
per-row FP16 scale applied once to each output.
"""
import numpy as np

GRID = np.array([0, 0.5, 1, 1.5, 2, 3, 4, 6], dtype=np.float64)
MIDS = (GRID[1:] + GRID[:-1]) / 2


def q_e2m1(x):
    """Round to nearest FP4 E2M1 value (saturating at +-6)."""
    a = np.abs(x)
    return np.sign(x) * GRID[np.searchsorted(MIDS, np.minimum(a, 6.0))]


def q_e4m3(x):
    """Round positive values to FP8 E4M3 (3 mantissa bits, max 448)."""
    x = np.clip(x, 2.0**-9, 448.0)
    e = np.floor(np.log2(x))
    e = np.maximum(e, -6)
    step = 2.0 ** (e - 3)
    return np.round(x / step) * step


def blocks(w, n):
    return w.reshape(w.shape[0], -1, n)


def mxfp4(w, exp_search=False):
    b = blocks(w, 32)
    amax = np.abs(b).max(-1, keepdims=True) + 1e-30
    e = np.floor(np.log2(amax)) - 2  # OCP spec: emax(E2M1) = 2
    if not exp_search:
        s = 2.0**e
        return (q_e2m1(b / s) * s).reshape(w.shape)
    best, best_err = None, None
    for d in (-1, 0, 1):
        s = 2.0 ** (e + d)
        dq = q_e2m1(b / s) * s
        err = ((dq - b) ** 2).sum(-1, keepdims=True)
        if best is None:
            best, best_err = dq, err
        else:
            m = err < best_err
            best = np.where(m, dq, best)
            best_err = np.where(m, err, best_err)
    return best.reshape(w.shape)


def nvfp4(w):
    b = blocks(w, 16)
    g = np.abs(w).max() / (6 * 448)  # per-tensor FP32 scale
    amax = np.abs(b).max(-1, keepdims=True)
    s = q_e4m3(amax / 6 / g) * g
    return (q_e2m1(b / s) * s).reshape(w.shape)


def mx_rowscale(w, fracs=16):
    """MX with per-row fractional scale c in [1,2): store W/c in MXFP4, multiply output by c."""
    best, best_err = None, None
    for c in 2.0 ** (np.arange(fracs) / fracs):
        dq = mxfp4(w / c, exp_search=True) * c
        err = ((dq - w) ** 2).sum(-1, keepdims=True)
        if best is None:
            best, best_err = dq, err
        else:
            m = err < best_err
            best = np.where(m, dq, best)
            best_err = np.where(m, err, best_err)
    return best


def hadamard(n):
    h = np.array([[1.0]])
    while h.shape[0] < n:
        h = np.block([[h, h], [h, -h]])
    return h / np.sqrt(n)


def rotate(w, n=128, seed=0):
    """Block-diagonal randomized Hadamard on the input dim (orthogonal, so MSE is preserved)."""
    rng = np.random.default_rng(seed)
    signs = rng.choice([-1.0, 1.0], size=w.shape[1])
    H = hadamard(n)
    return (blocks(w * signs, n) @ H).reshape(w.shape)


def rel_mse(w, dq):
    return ((dq - w) ** 2).mean() / (w**2).mean()


def make_weights(kind, rows=1024, cols=4096, seed=1):
    rng = np.random.default_rng(seed)
    if kind == "gaussian":
        return rng.standard_normal((rows, cols))
    if kind == "heavy-tail (t, df=4)":
        return rng.standard_t(4, (rows, cols))
    # LLM-like: per-channel scale spread + rare large outliers
    w = rng.standard_normal((rows, cols)) * rng.lognormal(0, 0.5, (1, cols))
    m = rng.random((rows, cols)) < 0.001
    w[m] *= 20
    return w


METHODS = {
    "MXFP4 (spec)": lambda w: mxfp4(w),
    "MX + exponent search": lambda w: mxfp4(w, exp_search=True),
    "MX + exp search + row scale": mx_rowscale,
    "MX + Hadamard + exp + row": lambda w: mx_rowscale(rotate(w)),
    "NVFP4": nvfp4,
    "NVFP4 + Hadamard": lambda w: nvfp4(rotate(w)),
}

if __name__ == "__main__":
    kinds = ["gaussian", "heavy-tail (t, df=4)", "LLM-like (outliers)"]
    print(f"{'method':32s}" + "".join(f"{k:>24s}" for k in kinds))
    print("relative MSE (lower is better; x1000)")
    for name, fn in METHODS.items():
        row = []
        for k in kinds:
            w = make_weights(k)
            ref = rotate(w) if "Hadamard" in name else w
            row.append(rel_mse(ref, fn(w)) * 1000)
        print(f"{name:32s}" + "".join(f"{v:24.2f}" for v in row))
