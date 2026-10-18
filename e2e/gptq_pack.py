"""GPTQ-rounded SC-MXFP4 (sim/gptq.py; Frantar et al., 2023) packed for our kernels.

sim/gptq.py keeps the SC-MXFP4 block exponents of the unquantized weights fixed and only changes which E2M1 level
each weight rounds to (error compensation with the calibration Hessian of sim/calib.py, WikiText-2 train). So the
result is still a spread-constrained MXFP4 tensor and runs on the same INT8 kernels, with the same bits per weight.
This script computes exactly what sim/eval_ppl.py's fake-quant method "gptq:sc_mx" computes for every block linear
(the same G.gptq call on the same float64 weights and Hessian), packs it from (E2M1 values, exponents) directly
(re-deriving the exponents from the rounded weights, as sim/make_testcase.pack would, could change them), and checks
that the packed weights decode, in float64, to the fake-quantized weights exactly.

Writes the byte-shift cache of e2e/w4a8_mlx.pack_model (loaded by fused_qwen3.load_fused(..., packed="gptq")).
Usage: python gptq_pack.py [--repo Qwen/Qwen3-1.7B-Base] [--workers 14] [--out results/08_gptq_pack_qwen3.json]
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
LUT2V = np.array([0, 1, 2, 3, 4, 6, 8, 12])


def pack_ve(v, e):
    """sim/make_testcase.pack from E2M1 values v (N, K/32, 32) and block exponents e (N, K/32) instead of weights."""
    import mx_int8 as q
    qw, sc = q.embed_int8(v, e)
    n, nb, _ = v.shape
    k = nb * q.BLOCK
    fin = np.where(np.isfinite(e), e, -np.inf)
    hi = fin.max(-1, keepdims=True)
    lo = np.where(np.isfinite(fin), fin, hi).min(-1, keepdims=True)
    shifts = (np.where(np.isfinite(fin), fin, lo) - lo).astype(np.uint8)
    v = v.reshape(n, k)
    idx = np.searchsorted(q.E2M1, np.abs(v)).astype(np.uint8)
    assert np.array_equal(q.E2M1[idx], np.abs(v))
    codes = idx | ((v < 0).astype(np.uint8) << 3)
    packed = (codes[:, 0::2] | (codes[:, 1::2] << 4)).astype(np.uint8)
    sh = np.repeat(shifts, q.BLOCK, axis=1)
    dec = np.where(codes & 8, -1, 1) * (LUT2V[codes & 7] << sh)
    assert np.array_equal(dec, qw)
    return packed, shifts, sc[:, 0].astype(np.float32), qw


def one(args):
    """-> (key, codes uint32, shifts, wscale, stats) for one linear."""
    import gptq as G
    import mx_int8 as q
    key, w32, hpath = args
    w = np.asarray(w32, np.float64)
    h = np.load(hpath)
    t0 = time.time()
    wg = G.gptq(w, h, G.sc_mxfp4_scales(w), q.E2M1)  # = eval_ppl.fake_weight(w, "gptq:sc_mx", h)
    e = q.constrain_spread(w, q.block_exponents(w, "search"))  # the exponents sc_mxfp4_scales used
    ee = np.where(np.isfinite(e), e, 0.0)[..., None]
    v = q.blocks(wg) / 2.0 ** ee
    assert np.all(v[~np.isfinite(e)] == 0), "GPTQ moved weight into an all-zero block"
    packed, shifts, wscale, qw = pack_ve(v, e)
    exact = bool(np.array_equal(qw.astype(np.float64) * wscale.astype(np.float64)[:, None], wg))
    wr = q.dequant_mx(w, e)[0]  # round-to-nearest SC-MXFP4 (the deployed RTN weights)
    rel = lambda x: float(np.einsum("ij,jk,ik->", x - w, h, x - w) / np.einsum("ij,jk,ik->", w, h, w))  # noqa: E731
    stats = dict(exact=exact, changed=float(np.mean(wg != wr)), out_err_rtn=rel(wr), out_err_gptq=rel(wg),
                 seconds=time.time() - t0)
    return key, packed.reshape(-1).view(np.uint32).copy(), shifts, wscale, stats


def main():
    import calib
    import mlx.core as mx
    from mlx_lm import load
    from w4a8_mlx import LINEARS
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="Qwen/Qwen3-1.7B-Base")
    ap.add_argument("--cache-dir", default=os.path.expanduser("~/.cache/mxfp4-int8-m5"))
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--out", default=os.path.join(HERE, "..", "results", "08_gptq_pack_qwen3.json"))
    a = ap.parse_args()
    model, _ = load(a.repo)
    jobs = []
    for i, layer in enumerate(model.model.layers):
        for path in LINEARS:
            lin = layer
            for p in path.split("."):
                lin = getattr(lin, p)
            hp = os.path.join(calib.cache_dir(a.repo), f"{i}_{calib.LINEAR_TO_GROUP[path]}.npy")
            assert os.path.exists(hp), f"missing calibration Hessian {hp} (python sim/calib.py {a.repo})"
            jobs.append((f"layers.{i}.{path}", np.array(lin.weight.astype(mx.float32)), hp))
    flat, stats = {}, {}
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"  # one BLAS thread per worker process
    with ProcessPoolExecutor(a.workers) as ex:
        for n, (key, c, s, ws, st) in enumerate(ex.map(one, jobs)):
            flat[key + ".codes"], flat[key + ".shifts"], flat[key + ".wscale"] = mx.array(c), mx.array(s), mx.array(ws)
            stats[key] = st
            print(f"\r{n + 1}/{len(jobs)} {key}: exact {st['exact']}, output error {st['out_err_rtn']:.4f} -> "
                  f"{st['out_err_gptq']:.4f}", end="", flush=True)
    print()
    path = os.path.join(a.cache_dir, a.repo.replace("/", "--") + "-scmxfp4-gptq.safetensors")
    mx.save_safetensors(path, flat)
    tot = {"matrices": len(stats), "all_exact": all(s["exact"] for s in stats.values()),
           "mean_fraction_of_codes_changed_vs_rtn": float(np.mean([s["changed"] for s in stats.values()])),
           "mean_rel_output_error_rtn": float(np.mean([s["out_err_rtn"] for s in stats.values()])),
           "mean_rel_output_error_gptq": float(np.mean([s["out_err_gptq"] for s in stats.values()])),
           "cache": path, "per_matrix": stats}
    with open(a.out, "w") as f:
        json.dump(tot, f, indent=1)
    print(json.dumps({k: v for k, v in tot.items() if k != "per_matrix"}, indent=1))


if __name__ == "__main__":
    main()
