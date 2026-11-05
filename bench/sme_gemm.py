"""The embedding on a second kind of matrix hardware: the M5 CPU's SME unit, one thread (bench/sme_gemm.c).

Real SC-MXFP4 and 16-block operands of Qwen3-1.7B, INT8 activations: the INT8 SMOPA GEMM's INT32 result must equal numpy's integer
product exactly, and (times the row scales) the FP32 reference. Times INT8 against FP16 and FP32 on the same matrices, with the
same packing, cold panels excluded. Usage (.venv): python bench/sme_gemm.py [--out results/16_sme.csv]
"""
import argparse
import csv
import ctypes as ct
import glob
import os
import platform
import subprocess
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "sim"))
LIB = os.path.join(HERE, "libsme_gemm.dylib")


def load():
    L = ct.CDLL(LIB)
    L.sme_svl_bytes.restype = ct.c_uint64
    L.sme_packed_bytes.restype = ct.c_size_t
    L.sme_packed_bytes.argtypes = [ct.c_int] * 5
    L.sme_pack.argtypes = [ct.c_void_p] + [ct.c_int] * 5 + [ct.c_void_p]
    for f in (L.sme_gemm_i8, L.sme_gemm_f16, L.sme_gemm_f32):
        f.argtypes = [ct.c_void_p, ct.c_void_p] + [ct.c_int] * 3 + [ct.c_void_p]
    return L


def packed(L, a, per, S):
    rows, K = a.shape
    a = np.ascontiguousarray(a)
    out = np.empty(L.sme_packed_bytes(rows, K, a.itemsize, per, S), np.uint8)
    L.sme_pack(a.ctypes.data, rows, K, a.itemsize, per, S, out.ctypes.data)
    return out


def timed(fn, reps=7):
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts)) * 1e3


def matrices():
    import formats as F
    import mlx.core as mx
    from huggingface_hub import snapshot_download
    path = snapshot_download("Qwen/Qwen3-1.7B-Base", allow_patterns=["*.safetensors"])
    t = {}
    for fn in glob.glob(os.path.join(path, "*.safetensors")):
        t.update(mx.load(fn))
    out = []
    for key in ("model.layers.12.mlp.gate_proj.weight", "model.layers.12.mlp.down_proj.weight", "model.layers.12.self_attn.q_proj.weight"):
        w = np.array(t[key].astype(mx.float32))
        for fname, fmt in (("e2m1/B16/M1..10", F.FORMATS["e2m1/B16/M1..10"]), ("e2m1/B32/pow2", F.FORMATS["e2m1/B32/pow2 (SC-MXFP4 grid, float row base)"])):
            _, qi, base, _ = F.quantize_mx(w, fmt)
            out.append((key.split("layers.12.")[1][:-7], fname, qi, base))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="")
    ap.add_argument("--rows", type=int, default=256, help="activation rows M")
    a = ap.parse_args()
    L = load()
    svl = int(L.sme_svl_bytes())
    S = svl // 4
    print(f"SVL {svl * 8} bits, panels of {S} rows; {platform.machine()}, "
          + subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip())
    rng = np.random.default_rng(0)
    rows = []
    for name, fname, qi, base in matrices():
        N, K = qi.shape
        K4 = K // 4 * 4
        qi = qi[:, :K4]
        M = a.rows
        # activation-like: heavy-tailed, scaled to INT8 per row
        xf = rng.standard_normal((M, K4)) * rng.lognormal(0, 1, (1, K4))
        xs = np.abs(xf).max(1, keepdims=True) / 127
        xq = np.rint(xf / xs).astype(np.int8)
        ref = xq.astype(np.int64) @ qi.astype(np.int64).T
        assert np.abs(ref).max() < 2 ** 31

        xp, wp = packed(L, xq, 4, S), packed(L, qi, 4, S)
        y = np.empty((M, N), np.int32)
        run = lambda: L.sme_gemm_i8(xp.ctypes.data, wp.ctypes.data, M, N, K4, y.ctypes.data)  # noqa: E731,B023
        run()
        exact = bool(np.array_equal(y.astype(np.int64), ref))
        t_i8 = timed(run)

        xh, wh = xf.astype(np.float16), (qi.astype(np.float32) * base[:, None]).astype(np.float16)
        xp2, wp2 = packed(L, xh, 2, S), packed(L, wh, 2, S)
        yf = np.empty((M, N), np.float32)
        t_f16 = timed(lambda: L.sme_gemm_f16(xp2.ctypes.data, wp2.ctypes.data, M, N, K4, yf.ctypes.data))  # noqa: B023
        refh = xh.astype(np.float64) @ wh.astype(np.float64).T
        err16 = float(np.abs(yf - refh).max() / np.abs(refh).max())

        xs32, ws32 = xf.astype(np.float32), (qi.astype(np.float32) * base[:, None])
        xp3, wp3 = packed(L, xs32, 1, S), packed(L, ws32, 1, S)
        t_f32 = timed(lambda: L.sme_gemm_f32(xp3.ctypes.data, wp3.ctypes.data, M, N, K4, yf.ctypes.data))  # noqa: B023

        r = dict(matrix=name, format=fname, M=M, N=N, K=K4, int32_exact=exact, int8_ms=round(t_i8, 3), fp16_ms=round(t_f16, 3),
                 fp32_ms=round(t_f32, 3), int8_vs_fp16=round(t_f16 / t_i8, 3), int8_vs_fp32=round(t_f32 / t_i8, 3),
                 fp16_rel_err=err16)
        print(r, flush=True)
        rows.append(r)
    if a.out:
        with open(a.out, "w", newline="") as fh:
            wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
            wr.writeheader()
            wr.writerows(rows)
    assert all(r["int32_exact"] for r in rows), "an INT8 SME GEMM was not exact"


if __name__ == "__main__":
    main()
