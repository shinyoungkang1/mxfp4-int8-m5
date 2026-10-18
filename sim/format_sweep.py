"""Rank 4-bit weight formats by layer-output error on real activations.

For every transformer-block linear layer, error = tr(dW H dW^T) / tr(W H W^T) with H = X^T X from
calib.py (WikiText-2 train). Reported per format: output error (aggregated over layers), weight MSE, bits per
weight, and whether the format runs exactly on the INT8 path. "+R32" variants first rotate each 32-wide block
of the input dimension with a randomized Hadamard transform (W -> W R, H -> R^T H R), as MR-GPTQ does for MXFP4.

Usage: python format_sweep.py <hf-repo> <out.csv> [layer_stride=3]
(every layer_stride-th transformer block is scored; 1 = all)
"""
import csv
import glob
import os
import sys
from collections import defaultdict

import mlx.core as mx
import numpy as np
from huggingface_hub import snapshot_download

import calib
import formats as F
import mx_int8 as q
from gpu_guard import eval_step

LINEARS = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
           "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]


def hadamard(n):
    h = np.array([[1.0]])
    while h.shape[0] < n:
        h = np.block([[h, h], [h, -h]])
    return (h / np.sqrt(n)).astype(np.float32)


def rotate(w, h, block=32, seed=0):
    k = w.shape[1]
    signs = np.random.default_rng(seed).choice([-1.0, 1.0], size=k).astype(np.float32)
    hb = hadamard(block)
    wr = ((w * signs).reshape(w.shape[0], -1, block) @ hb).reshape(w.shape)
    h1 = h * signs[:, None] * signs[None, :]
    t = (h1.reshape(k, -1, block) @ hb).reshape(k, k)          # H1 B
    hr = (t.T.reshape(k, -1, block) @ hb).reshape(k, k).T       # B H1 B
    return wr, hr


def methods():
    m = {
        "MXFP4 (OCP rule)": (lambda w: q.dequant_mx(w.astype(np.float64), q.block_exponents(w, "spec"))[0], 4.25, False),
        "SC-MXFP4 (standard MXFP4 file)": (F.sc_mxfp4, None, True),
        "NVFP4": (lambda w: q.nvfp4(w.astype(np.float64)), 4.5, False),
        "INT4 affine g64 (MLX)": (lambda w: q.int4_affine(w.astype(np.float64), 64), 4.5, False),
    }
    for name, f in F.FORMATS.items():
        m[name] = ((lambda f: lambda w: F.quantize_mx(w, f)[0])(f), f, True)
    return m


def _out_err(d, h_mx):
    """tr(d H d^T) on the GPU."""
    dm = mx.array(d.astype(np.float32))
    return float(((dm @ h_mx) * dm).sum().item())


def main(repo, out_csv, stride=3):
    calib.compute(repo)
    path = snapshot_download(repo, allow_patterns=["*.safetensors"])
    tensors = {}
    for fn in sorted(glob.glob(os.path.join(path, "*.safetensors"))):
        tensors.update(mx.load(fn))
    ms = methods()
    rot_names = ["SC-MXFP4 (standard MXFP4 file)", "NVFP4", "e2m1/B32/M1..10", "e2m1/B16/M1..8", "int4/B16/M1..16"]
    num = defaultdict(float); den = defaultdict(float); wnum = defaultdict(float); wden = defaultdict(float)
    bits = {}
    nl = len([k for k in tensors if k.endswith("self_attn.q_proj.weight") and ".layers." in k])
    for i in range(0, nl, stride):
        with eval_step():
            for lin in LINEARS:
                key = f"model.layers.{i}.{lin}.weight"
                if key not in tensors:
                    continue
                w = np.array(tensors[key].astype(mx.float32))
                h = calib.hessian(repo, i, lin)
                variants = {"": (w, h)}
                variants["+R32"] = rotate(w, h)
                for suffix, (ww, hh) in variants.items():
                    h_mx = mx.array(hh.astype(np.float32))
                    base = _out_err(ww, h_mx)
                    for name, (fn, b, exact) in ms.items():
                        if suffix and name not in rot_names:
                            continue
                        wq = fn(ww).astype(np.float32)
                        d = wq - ww
                        label = name + suffix
                        num[label] += _out_err(d, h_mx); den[label] += base
                        wnum[label] += float((d ** 2).sum()); wden[label] += float((ww ** 2).sum())
                        kdim = ww.shape[1]
                        bits[label] = (b.bits(kdim) if isinstance(b, F.Fmt) else b if b else 4 + 2 / 32 + 32 / kdim, exact)
        print(f"\rlayer {i + 1}/{nl}", end="", file=sys.stderr)
    print(file=sys.stderr)
    rows = sorted(num, key=lambda l: num[l] / den[l])
    with open(out_csv, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["model", "format", "output_rel_err", "weight_rel_mse", "bits_per_weight_last_layer", "int8_exact"])
        for l in rows:
            wr.writerow([repo, l, num[l] / den[l], wnum[l] / wden[l], bits[l][0], bits[l][1]])
    print(f"## {repo}: layer-output error on WikiText-2 train activations\n")
    print("| format | output error x1000 | weight MSE x1000 | bits/weight | exact on INT8 |\n|---|---|---|---|---|")
    for l in rows:
        print(f"| {l} | {1000 * num[l] / den[l]:.2f} | {1000 * wnum[l] / wden[l]:.2f} | {bits[l][0]:.2f} | {'yes' if bits[l][1] else 'no'} |")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 3)
