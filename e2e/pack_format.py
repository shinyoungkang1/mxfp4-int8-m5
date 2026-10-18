"""Packs a model's block linears in one of sim/formats.py's INT8-exact formats (e.g. "e2m1/B16/M1..10") for our
kernels, with sim/formats.quantize_mx as the quantizer (the one sim/eval_ppl.py's "w=fmt:<name>" method uses).

quantize_mx returns, per matrix, INT8 operands qi [N, K], a float32 row base [N] and block scales base x m [N, K/B]
(m an integer from the format's multiplier set). We store m as a block code (index into the multiplier table),
|qi| / m as an index into the element codebook, the sign, and base as the row scale: sc_format.Format(block=B,
code_mag=codebook, mult=multipliers), packed with sc_format.pack_rows. Checks per matrix: the packed layout decodes
to qi exactly, and float32(qi x base) equals the quantizer's float32 weights exactly.

Note: for formats with a non-power-of-two row base, eval_ppl's fake-quant reference stores its weights in BF16, i.e.
rounds base x qi to 8 significant bits, while our kernels apply base in FP32 after the exact INT32 accumulation; the
two then agree in the INT8 operands and the base but not in every BF16 weight (check_format.py reports both).

Usage: python pack_format.py [--repo Qwen/Qwen3-1.7B-Base] [--format e2m1/B16/M1..10]
Writes ~/.cache/mxfp4-int8-m5/<repo>-fmt-<slug>.safetensors (codes / bcodes / wscale per linear).
"""
import argparse
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "sim"))
import sc_format as F  # noqa: E402


def slug(name):
    return name.replace("/", "_").replace(".", "")


def sc_format_of(name):
    """sim/formats.py format name -> sc_format.Format for the kernels."""
    import formats
    f = formats.FORMATS[name]
    bits = max(1, math.ceil(math.log2(len(f.mults))))
    bits = next(b for b in (1, 2, 4, 8) if b >= bits)
    return F.Format(block=f.block, code_mag=tuple(int(c) for c in formats.CODEBOOKS[f.codebook]),
                    mult=tuple(int(m) for m in f.mults), bc_bits=bits)


def pack_matrix(w, name, fmt):
    """float weights [N, K] -> (codes, bcodes, wscale, stats) in `fmt`, quantized by formats.quantize_mx."""
    import formats
    f = formats.FORMATS[name]
    wq, qi, base, scales = formats.quantize_mx(np.asarray(w, np.float32), f)
    n, k = qi.shape
    m = scales / base.astype(np.float64)[:, None]
    mi = np.rint(m).astype(np.int64)
    assert np.array_equal(m, mi), "block scale / row base is not an integer"
    mults = np.array(fmt.mult)
    bcode = np.searchsorted(mults, mi).astype(np.uint8)
    assert np.array_equal(mults[bcode], mi)
    mr = np.repeat(mi, fmt.block, axis=1)
    mag = np.abs(qi.astype(np.int64)) // mr
    assert np.array_equal(mag * mr, np.abs(qi.astype(np.int64))), "INT8 operand is not code x multiplier"
    cm = np.array(fmt.code_mag)
    idx = np.searchsorted(cm, mag)
    assert np.array_equal(cm[idx], mag), "element magnitude not in the codebook"
    code4 = (idx | ((qi < 0).astype(np.int64) << 3)).astype(np.uint8)
    codes, bc, ws = F.pack_rows(code4, bcode, base, fmt)
    dec = F.decode_numpy(codes, bc, n, k, fmt)
    exact_q = bool(np.array_equal(dec, qi))
    exact_w = bool(np.array_equal((dec.astype(np.float32) * ws[:, None]), wq))
    return codes, bc, ws, dict(exact_q=exact_q, exact_w=exact_w)


def cache_path(repo, name, cache_dir):
    return os.path.join(cache_dir, repo.replace("/", "--") + f"-fmt-{slug(name)}.safetensors")


def main():
    import mlx.core as mx
    from mlx_lm import load
    from w4a8_mlx import LINEARS
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="Qwen/Qwen3-1.7B-Base")
    ap.add_argument("--format", default="e2m1/B16/M1..10")
    ap.add_argument("--cache-dir", default=os.path.expanduser("~/.cache/mxfp4-int8-m5"))
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    fmt = sc_format_of(a.format)
    model, _ = load(a.repo)
    flat, stats = {}, {}
    for i, layer in enumerate(model.model.layers):
        for path in LINEARS:
            lin = layer
            for p in path.split("."):
                lin = getattr(lin, p)
            c, b, s, st = pack_matrix(np.array(lin.weight.astype(mx.float32)), a.format, fmt)
            key = f"layers.{i}.{path}"
            flat[key + ".codes"], flat[key + ".bcodes"], flat[key + ".wscale"] = mx.array(c), mx.array(b), mx.array(s)
            stats[key] = st
        print(f"\rpacked layer {i + 1}/{len(model.model.layers)}", end="", flush=True)
    print()
    path = cache_path(a.repo, a.format, a.cache_dir)
    mx.save_safetensors(path, flat)
    ks = (model.args.hidden_size, model.args.intermediate_size)
    rep = {"format": a.format, "sc_format": str(fmt), "bits_per_weight_at_K": {str(k): fmt.bits_per_weight(k) for k in ks},
           "matrices": len(stats), "all_q_exact": all(v["exact_q"] for v in stats.values()),
           "all_w_exact": all(v["exact_w"] for v in stats.values()), "cache": path}
    print(json.dumps(rep, indent=1))
    if a.out:
        with open(a.out, "w") as fh:
            json.dump(dict(rep, per_matrix=stats), fh, indent=1)


def _self_test():
    rng = np.random.default_rng(0)
    w = rng.standard_normal((96, 512)) * 0.02 * rng.lognormal(0, 0.5, (1, 512))
    for name in ("e2m1/B16/M1..10", "e2m1/B32/M1..8", "int4/B16/M1..16"):
        fmt = sc_format_of(name)
        c, b, s, st = pack_matrix(w, name, fmt)
        print(name, fmt, st)
        assert st["exact_q"] and st["exact_w"]
    print("pack_format self-test passed")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--self-test":
        _self_test()
    else:
        main()
