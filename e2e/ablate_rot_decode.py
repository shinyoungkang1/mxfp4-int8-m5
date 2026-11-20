"""Why is rotated decode slow? Real decode steps on Qwen3-1.7B, changing one thing at a time, in one process, in alternating rounds. Needs an idle GPU.

 mlx          MLX MXFP4
 sc           SC-MXFP4, plain quantize
 fmt          E2M1 8-wide, plain quantize
 fmt_rotpack  same code, weights from the rotated pack (tests the pack and load path; the outputs are meaningless)
 rot_old      full rotation with the earlier kernel, as benchmarked through run 29
 rot_nt192    full rotation, one block per thread on the 6144-wide input
 rot_nofwht   rot_quant kernel with the butterflies removed: tests the transform against the kernel's structure (outputs meaningless)
 rot_simd     the current S.rot_quant: same outputs as rot_old (--identity), transform across SIMD lanes
 rot_actq     rms_norm kept as a separate dispatch, plain act_quant instead of rot_quant: tests the extra dispatches alone (outputs meaningless)

If rot_actq and rot_nofwht sit near fmt, the time is the transform; if they sit near rot, it is the code around it.
Usage: python ablate_rot_decode.py --out ../results/31_ablate_decode.csv
"""
import argparse
import csv
import os
import time

import mlx.core as mx
import numpy as np

import decode as dec
import fused_qwen3 as fq
import sc_kernels as S
from time_map import CACHE, REPO

F8 = "e2m1/B8/M1..10"
_OLD = """
    const uint row = threadgroup_position_in_grid.x, lid = thread_position_in_threadgroup.x;
    const uint lane = thread_index_in_simdgroup, sg = simdgroup_index_in_threadgroup;
    threadgroup float red[32];
    threadgroup float shared_amax;
    constexpr int J = K / (ROT_B * NT);
    float v[J][ROT_B];
    float amax = 0;
    for (int j = 0; j < J; ++j) {
        const size_t b0 = size_t(ROT_B) * (j * NT + lid);
        for (int i = 0; i < ROT_B; ++i)
            v[j][i] = row < M ? float(x[size_t(row) * K + b0 + i]) * signs[b0 + i] : 0.0f;
        for (int h = 1; h < ROT_B; h *= 2)
            for (int i = 0; i < ROT_B; i += 2 * h)
                for (int k = i; k < i + h; ++k) {
                    const float a = v[j][k], c = v[j][k + h];
                    v[j][k] = a + c;
                    v[j][k + h] = a - c;
                }
        for (int i = 0; i < ROT_B; ++i) {
            v[j][i] = float(static_cast<T>(v[j][i] * ROT_NORM));
            amax = max(amax, abs(v[j][i]));
        }
    }
    amax = simd_max(amax);
    if (lane == 0) red[sg] = amax;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        amax = simd_max(lane < (NT + 31) / 32 ? red[lane] : 0.0f);
        if (lane == 0) shared_amax = amax;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float s = shared_amax / 127.0f;
    const float qi = s > 0 ? 1.0f / s : 0.0f;
    for (int j = 0; j < J; ++j) {
        const size_t b0 = size_t(row) * K + size_t(ROT_B) * (j * NT + lid);
        for (int i = 0; i < ROT_B; ++i) xq[b0 + i] = char(clamp(rint(v[j][i] * qi), -127.0f, 127.0f));
    }
    if (lid == 0) xs[row] = s;
"""
ROT_DEFS = "\n#define ROT_B 32\n#define ROT_NORM 0.17677669529663687f\n"


def ROT0(x, signs, Mp=None, nt=64, deps=()):
    """The earlier kernel: a per-thread array of 32 floats, per-thread transform."""
    M, K = x.shape
    Mp = Mp or S.pad_rows(M)
    nt = min(nt, K // 32)
    while (K // 32) % nt:
        nt -= 1
    k = S._kernel("rot_old", ["x", "signs"], ["xq", "xs"], _OLD, len(deps), header_extra=ROT_DEFS)
    return k(inputs=[x, signs, *deps], template=[("T", x.dtype), ("K", K), ("M", M), ("NT", nt)], grid=(Mp * nt, 1, 1), threadgroup=(nt, 1, 1),
             output_shapes=[(Mp, K), (Mp,)], output_dtypes=[mx.int8, mx.float32])


BUTTERFLY = """        for (int h = 1; h < ROT_B; h *= 2)
            for (int i = 0; i < ROT_B; i += 2 * h)
                for (int k = i; k < i + h; ++k) {
                    const float a = v[j][k], c = v[j][k + h];
                    v[j][k] = a + c;
                    v[j][k + h] = a - c;
                }
"""
assert _OLD.count(BUTTERFLY) == 1
NOFWHT = _OLD.replace(BUTTERFLY, "")


def rot_nt192(x, signs, Mp=None, deps=()):
    return ROT0(x, signs, Mp, nt=192, deps=deps)


def rot_nofwht(x, signs, Mp=None, deps=()):
    M, K = x.shape
    Mp = Mp or S.pad_rows(M)
    nt = min(64, K // 32)
    while (K // 32) % nt:
        nt -= 1
    k = S._kernel("rot_nofwht", ["x", "signs"], ["xq", "xs"], NOFWHT, len(deps), header_extra="\n#define ROT_B 32\n#define ROT_NORM 0.17677669529663687f\n")
    return k(inputs=[x, signs, *deps], template=[("T", x.dtype), ("K", K), ("M", M), ("NT", nt)], grid=(Mp * nt, 1, 1), threadgroup=(nt, 1, 1),
             output_shapes=[(Mp, K), (Mp,)], output_dtypes=[mx.int8, mx.float32])


def rot_actq(x, signs, Mp=None, deps=()):
    return S.act_quant(x, Mp, deps=deps)


def main():
    from mlx_lm import load
    from mlx_lm.models.cache import make_prompt_cache
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctx", type=int, default=512)
    ap.add_argument("--steps", type=int, default=48)
    ap.add_argument("--rounds", type=int, default=7)
    ap.add_argument("--out", default="")
    ap.add_argument("--identity", action="store_true", help="only compare rot_simd with rot_quant on random and on heavy-tailed rows")
    a = ap.parse_args()
    if a.identity:
        rng = np.random.default_rng(3)
        for K in (2048, 6144, 2560, 4096):
            sg = S.rot_signs(K)
            for M, Mp in ((1, 1), (7, None), (64, None)):
                for dt in (mx.bfloat16, mx.float16):
                    raw = rng.standard_normal((M, K)) * np.exp(rng.standard_normal((M, K)))
                    x = mx.array(raw.astype(np.float32)).astype(dt)
                    A = [np.array(v) for v in ROT0(x, sg, Mp)]
                    B = [np.array(v) for v in S.rot_quant(x, sg, Mp)]
                    print(K, M, dt, "codes differing", int((A[0] != B[0]).sum()), "of", A[0].size, "scale equal", bool((A[1] == B[1]).all()))
        return
    mxm = load(os.path.join(CACHE, REPO.split("/")[-1] + "-mxfp4"))[0]
    rng = np.random.default_rng(0)
    prompt = mx.array(rng.integers(0, 151936, (1, a.ctx)).astype(np.int32))
    tok = mx.array(rng.integers(0, 151936, (1, 1)).astype(np.int32))
    fmt = fq.load_fused(REPO, CACHE, packed="fmt:" + F8)[0]
    rot = fq.load_fused(REPO, CACHE, packed="rot:" + F8)[0]
    pack = fq.load_fused(REPO, CACHE, packed="rot:" + F8)[0]
    pack.rot = False
    sc = fq.load_fused(REPO, CACHE)[0]
    SIMD = S.rot_quant
    variants = {"sc": (sc, ROT0), "fmt": (fmt, ROT0), "fmt_rotpack": (pack, ROT0), "rot_old": (rot, ROT0), "rot_nt192": (rot, rot_nt192), "rot_nofwht": (rot, rot_nofwht), "rot_actq": (rot, rot_actq), "rot_simd": (rot, SIMD)}
    ds = {}
    for name, (f, q) in variants.items():  # each decoder traces its own compiled graph here, with its kernel
        S.rot_quant = q
        d = dec.DecodeQwen3(f, head=mxm)
        c = f.make_cache()
        mx.eval(f.hidden(prompt, c))
        mx.eval(d.step(tok, c))
        ds[name] = d
    S.rot_quant = SIMD
    times = {k: [] for k in ["mlx", *variants]}
    for r in range(a.rounds):
        c = make_prompt_cache(mxm)
        mx.eval(mxm(prompt, cache=c))
        for i in range(a.steps):
            t0 = time.perf_counter()
            mx.eval(mxm(tok, cache=c))
            if i >= a.steps // 4:
                times["mlx"].append((time.perf_counter() - t0) * 1e3)
        for name, (f, _) in variants.items():
            S.set_format(f.fmt)
            c = f.make_cache()
            mx.eval(f.hidden(prompt, c))
            for i in range(a.steps):
                t0 = time.perf_counter()
                mx.eval(ds[name].step(tok, c))
                if i >= a.steps // 4:
                    times[name].append((time.perf_counter() - t0) * 1e3)
        print(f"round {r}: " + ", ".join(f"{k} {np.median(v):.2f}" for k, v in times.items()), flush=True)
    rows = [dict(variant=k, ctx=a.ctx, ms_median=f"{np.median(v):.3f}", ms_p10=f"{np.percentile(v, 10):.3f}", ms_p90=f"{np.percentile(v, 90):.3f}", vs_mlx=f"{np.median(times['mlx']) / np.median(v):.3f}") for k, v in times.items()]
    for r in rows:
        print(r)
    if a.out:
        with open(a.out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)


if __name__ == "__main__":
    main()
