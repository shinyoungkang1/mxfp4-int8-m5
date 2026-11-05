"""Test the rotation bound on real activations: does a block rotation shrink the per-token INT8 step as predicted, and does
the output error of each linear layer follow the square of that step?

For every transformer-block linear layer, on a clean BF16 forward pass, and for block sizes b (0 = no rotation), this
sums over sampled tokens:
  e2, x2        ||x - Q(x)||^2 and ||x||^2 (in the rotated basis when b > 0)
  we2, wx2      ||W_b e||^2 and ||W x||^2, the actual output error of the layer (weights BF16)
  pred          ||W||_F^2 D^2 / 12, D the INT8 step: the output error if e were white with variance D^2 / 12
  linf2         ||R x||_inf^2, whose ratio between b = 0 and b gives the predicted gain
  upper_viol    tokens over the high-probability upper bound (delta = 0.01), lower_viol tokens under the sure lower bound
  rel_tok       ||W_b e||^2 / ||W x||^2 per token (the sum, not a ratio of sums, so heavy tokens do not dominate)
  lo_short      largest relative shortfall under the lower bound over tokens (rounding if about 1e-6)
  peak_ratio    ||R x||_inf / (max_B ||x_B||_2 / sqrt(b)), which the bound places in [1, sqrt(2 ln(2n/delta))]
The rotation is the one in eval_ppl.QLinear: fixed random signs, then a Hadamard transform of each b-wide block.

Usage: python rotation_bound.py <hf-repo> <out.csv> [--windows 16] [--every 8] [--blocks 8,16,32,64,128]
"""
import argparse
import csv
import math

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_lm import load

from eval_ppl import LINEARS, hadamard, wikitext_tokens

DELTA = 0.01
FIELDS = ["model", "layer", "path", "b", "n", "tokens", "x2", "e2", "wx2", "we2", "pred", "linf2", "upper_viol", "lower_viol",
          "peak_ratio", "rho", "crest2", "rel_tok", "lo_short"]


def act_q(x):
    s = mx.max(mx.abs(x), axis=-1, keepdims=True) / 127.0
    s = mx.where(s == 0, 1.0, s)
    return mx.clip(mx.round(x / s), -127, 127) * s, s


class Probe(nn.Module):
    def __init__(self, lin, name, blocks, every, stats):
        super().__init__()
        self.lin, self.name, self.every, self.stats = lin, name, every, stats
        w = lin.weight.astype(mx.float32)
        n = w.shape[1]
        signs = mx.array(np.random.default_rng(0).choice([-1.0, 1.0], size=n).astype(np.float32))  # as in QLinear
        self.setup = [(0, None, None, w)]
        for b in blocks:
            hb = mx.array(hadamard(b).astype(np.float32))
            wr = ((w * signs).reshape(w.shape[0], -1, b) @ hb).reshape(w.shape)
            self.setup.append((b, signs, hb, wr))
        self.wf2 = mx.sum(w * w)
        mx.eval(self.wf2, *[s[3] for s in self.setup])

    def __call__(self, x):
        xf = x.astype(mx.float32).reshape(-1, x.shape[-1])[:: self.every]
        n = xf.shape[-1]
        wx = xf @ self.setup[0][3].T
        wx2 = mx.sum(wx * wx)
        out = []
        for b, signs, hb, w in self.setup:
            if b:
                rx = ((xf * signs).reshape(xf.shape[0], -1, b) @ hb).reshape(xf.shape)
            else:
                rx = xf
            q, s = act_q(rx)
            e = rx - q
            we = e @ w.T
            linf = 127.0 * s[:, 0]
            row = [mx.sum(rx * rx), mx.sum(e * e), wx2, mx.sum(we * we), self.wf2 * mx.sum(s * s) / 12.0, mx.sum(linf * linf)]
            if b:
                blocks = xf.reshape(xf.shape[0], -1, b)
                bmax = mx.sqrt(mx.max(mx.sum(blocks * blocks, axis=-1), axis=-1))
                lo, hi = bmax / math.sqrt(b), bmax * math.sqrt(2 * math.log(2 * n / DELTA) / b)
                norm2 = mx.sum(xf * xf, axis=-1)
                row += [mx.sum(linf > hi * (1 + 1e-5)), mx.sum(linf < lo * (1 - 1e-5)), mx.sum(linf / mx.maximum(lo, 1e-30)),
                        mx.sum(bmax * bmax * (n / b) / mx.maximum(norm2, 1e-30))]
            else:
                norm2 = mx.sum(xf * xf, axis=-1)
                row += [mx.array(0.0), mx.array(0.0), mx.array(0.0), mx.array(0.0)]
            row.append(mx.sum(linf * linf * n / mx.maximum(norm2, 1e-30)))
            row.append(mx.sum(mx.sum(we * we, axis=-1) / mx.maximum(mx.sum(wx * wx, axis=-1), 1e-30)))
            row.append(mx.max(1 - linf / mx.maximum(lo, 1e-30)) if b else mx.array(0.0))
            out.append((b, row))
        mx.eval([r for _, r in out])
        for b, row in out:
            k = (self.name, b)
            cur = self.stats.setdefault(k, {"n": n, "tokens": 0, "v": np.zeros(len(row))})
            cur["tokens"] += xf.shape[0]
            cur["v"] += np.array([r.item() for r in row])
        return self.lin(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("repo")
    ap.add_argument("out")
    ap.add_argument("--windows", type=int, default=16)
    ap.add_argument("--every", type=int, default=8)
    ap.add_argument("--seq", type=int, default=2048)
    ap.add_argument("--blocks", default="8,16,32,64,128")
    a = ap.parse_args()
    blocks = [int(b) for b in a.blocks.split(",")]
    model, tok = load(a.repo)
    toks = wikitext_tokens(tok)
    stats = {}
    for i, layer in enumerate(model.model.layers):
        for path in LINEARS:
            pn, name = path.split(".")
            parent = getattr(layer, pn)
            if hasattr(parent, name):
                setattr(parent, name, Probe(getattr(parent, name), (i, path), blocks, a.every, stats))
    for w in range(a.windows):
        # same windows as eval_ppl.perplexity; evaluated on the clean model, so errors do not compound
        mx.eval(model(mx.array(toks[w * a.seq:(w + 1) * a.seq])[None]))
        print(f"window {w + 1}/{a.windows}", flush=True)
    with open(a.out, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(FIELDS)
        for ((i, path), b), s in sorted(stats.items()):
            wr.writerow([a.repo, i, path, b, s["n"], s["tokens"]] + [f"{v:.6g}" for v in s["v"]])
    print("wrote", a.out)


if __name__ == "__main__":
    main()
