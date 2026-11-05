"""GPTQ weights in the rotated basis of sim/eval_ppl.QLinear (r=32), packed for our kernels: exactly what eval_ppl's method
"w=gptq:<format>;a=8;r=32" computes for every block linear, stored as codes, block codes and row scales. Checks per matrix: the
packed layout decodes to the INT8 operands exactly, and they times the row base equal the GPTQ output in float32.

Usage: python pack_rot.py [--repo Qwen/Qwen3-1.7B-Base] [--format e2m1/B8/M1..10] [--workers 8] [--out results/25_pack.json]
python pack_rot.py --self-test   checks the rotation kernel against a NumPy reference, bit for bit.
Writes ~/.cache/mxfp4-int8-m5/<repo>-fmt-<slug>-gptq-rot32.safetensors.
"""
import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "sim"))
BLOCK = 32


def cache_path(repo, name, cache_dir):
    import pack_format as pf
    return os.path.join(cache_dir, repo.replace("/", "--") + f"-fmt-{pf.slug(name)}-gptq-rot32.safetensors")


def rotate(w, h):
    """eval_ppl.QLinear's rotation of (w [N, K], h [K, K]), in float64."""
    from eval_ppl import hadamard
    k = w.shape[1]
    hb = hadamard(BLOCK)
    signs = np.random.default_rng(0).choice([-1.0, 1.0], size=k)
    w = ((w * signs).reshape(w.shape[0], -1, BLOCK) @ hb).reshape(w.shape)
    h1 = h.astype(np.float64) * signs[:, None] * signs[None, :]
    t = (h1.reshape(k, -1, BLOCK) @ hb).reshape(k, k)
    return w, (t.T.reshape(k, -1, BLOCK) @ hb).reshape(k, k).T


def one(args):
    import formats as FF
    import gptq as G
    import pack_format as pf
    key, w32, hpath, name = args
    t0 = time.time()
    fmt = pf.sc_format_of(name)
    f = FF.FORMATS[name]
    w, h = rotate(np.asarray(w32, np.float64), np.load(hpath))
    # as eval_ppl.fake_weight: scales from the format's quantizer on the rotated weights, then GPTQ on its codebook
    _, _, base, scales = FF.quantize_mx(w, f)
    wg = G.gptq(w, h, scales, FF.CODEBOOKS[f.codebook])
    b = base.astype(np.float64)[:, None]
    qi = np.where(b == 0, 0.0, wg / np.where(b == 0, 1.0, b))
    assert np.array_equal(qi, np.rint(qi)), "GPTQ output is not an integer multiple of the row base"
    assert np.abs(qi).max() <= 127
    c, bc, ws, st = pf.pack_parts(wg.astype(np.float32), qi.astype(np.int8), base, scales, fmt)
    st["seconds"] = time.time() - t0
    return key, c, bc, ws, st


def self_test():
    import mlx.core as mx
    import fused_qwen3  # noqa: F401
    import sc_kernels as S
    rng = np.random.default_rng(0)
    ok = True
    for M, K in ((3, 2048), (128, 2048), (130, 6144), (64, 2560)):
        x = rng.standard_normal((M, K)).astype(np.float32) * rng.lognormal(0, 2, (M, 1)).astype(np.float32)
        x[0, 5] = 900.0  # a spike
        xm = mx.array(x).astype(mx.bfloat16)
        signs = S.rot_signs(K)
        xq, xs = S.rot_quant(xm, signs, S.pad_rows(M))
        v = np.array(xm.astype(mx.float32)) * np.array(signs)
        v = v.reshape(M, -1, BLOCK).copy()
        h = 1
        while h < BLOCK:  # the kernel's butterflies, same order, FP32
            for i in range(0, BLOCK, 2 * h):
                for k in range(i, i + h):
                    a, c = v[..., k].copy(), v[..., k + h].copy()
                    v[..., k], v[..., k + h] = a + c, a - c
            h *= 2
        v = np.array(mx.array((v * np.float32(1 / np.sqrt(32))).reshape(M, K)).astype(mx.bfloat16).astype(mx.float32))
        s = (np.abs(v).max(1) / np.float32(127.0)).astype(np.float32)
        qi = (np.float32(1.0) / s).astype(np.float32)
        q = np.clip(np.rint((v * qi[:, None]).astype(np.float32)), -127, 127).astype(np.int8)
        same = np.array_equal(np.array(xq)[:M], q) and np.array_equal(np.array(xs)[:M], s)
        zero = not np.array(xq)[M:].any()
        # the rotated product equals the plain one
        w = rng.standard_normal((16, K)).astype(np.float64)
        from eval_ppl import hadamard
        sg = np.array(signs).astype(np.float64)
        wr = ((w * sg).reshape(16, -1, BLOCK) @ hadamard(BLOCK)).reshape(w.shape)
        xr = ((x.astype(np.float64) * sg).reshape(M, -1, BLOCK) @ hadamard(BLOCK)).reshape(x.shape)
        prod = np.abs(xr @ wr.T - x.astype(np.float64) @ w.T).max() / np.abs(x @ w.T).max()
        print(f"rot_quant M={M} K={K}: bit-identical to reference {same}, padding zero {zero}, rotated product rel. diff {prod:.1e}")
        ok &= same and zero and prod < 1e-12
    print("rot_quant self-test", "passed" if ok else "FAILED")
    sys.exit(0 if ok else 1)


def main():
    import mlx.core as mx
    import calib
    import pack_format as pf
    from mlx_lm import load
    from w4a8_mlx import LINEARS
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="Qwen/Qwen3-1.7B-Base")
    ap.add_argument("--format", default="e2m1/B8/M1..10")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--cache-dir", default=os.path.expanduser("~/.cache/mxfp4-int8-m5"))
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    model, _ = load(a.repo)
    jobs = []
    for i, layer in enumerate(model.model.layers):
        for path in LINEARS:
            lin = layer
            for p in path.split("."):
                lin = getattr(lin, p)
            hp = os.path.join(calib.cache_dir(a.repo), f"{i}_{calib.LINEAR_TO_GROUP[path]}.npy")
            jobs.append((f"layers.{i}.{path}", np.array(lin.weight.astype(mx.float32)), hp, a.format))
    flat, stats = {}, {}
    with ProcessPoolExecutor(a.workers) as ex:
        for n, (key, c, b, s, st) in enumerate(ex.map(one, jobs)):
            flat[key + ".codes"], flat[key + ".bcodes"], flat[key + ".wscale"] = mx.array(c), mx.array(b), mx.array(s)
            stats[key] = st
            print(f"\rpacked {n + 1}/{len(jobs)}", end="", flush=True)
    print()
    path = cache_path(a.repo, a.format, a.cache_dir)
    mx.save_safetensors(path, flat)
    rep = {"format": a.format, "matrices": len(stats), "all_q_exact": all(v["exact_q"] for v in stats.values()),
           "all_w_exact": all(v["exact_w"] for v in stats.values()), "cache": path}
    print(json.dumps(rep, indent=1))
    if a.out:
        with open(a.out, "w") as fh:
            json.dump(dict(rep, per_matrix=stats), fh, indent=1)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--self-test":
        self_test()
    else:
        main()
