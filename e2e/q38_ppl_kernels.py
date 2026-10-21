"""WikiText-2 perplexity of the Qwen3.8-Flash-Next W4A8 model running on our kernels (fused_qwen38.py), with the
windows of q38_ppl.py (2048-token, independent, full test set unless --windows), to compare with the fake-quant
reference rows of q38_ppl.py; optionally also decode-vs-prefill agreement on the first window.
Usage (.venv-q38): python q38_ppl_kernels.py [--format sc|e2m1/B16/M1..10] [--windows N] [--out csv]
"""
import argparse
import csv
import os
import sys

import mlx.core as mx
import mlx.nn as nn
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "sim"))
import eval_ppl as E  # noqa: E402
import fused_qwen38 as FQ  # noqa: E402
import q38_ckpt as C  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--format", default=FQ.SC)
    ap.add_argument("--seq", type=int, default=2048)
    ap.add_argument("--windows", type=int, default=0)
    ap.add_argument("--decode-check", type=int, default=0, help="tokens of window 0 to decode one by one")
    ap.add_argument("--out", default="")
    ap.add_argument("--reference-deltanet", action="store_true",
                    help="the PR's op-by-op DeltaNet recurrence (training mode) instead of its Metal kernel")
    a = ap.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(C.snapshot())
    toks = E.wikitext_tokens(tok)
    n = (len(toks) - 1) // a.seq
    n = min(n, a.windows) if a.windows else n
    f = FQ.FusedQwen38(a.format)
    if a.reference_deltanet:  # FusedQwen38 puts its layers in inference mode, where DeltaNet uses the Metal kernel
        for layer, *_ in f.layers:
            layer.train()
    nll = 0.0
    for b in range(n):
        x = mx.array(toks[b * a.seq:(b + 1) * a.seq + 1])[None]
        logits = f(x[:, :-1]).astype(mx.float32)
        loss = nn.losses.cross_entropy(logits, x[:, 1:], reduction="sum")
        mx.eval(loss)
        nll += loss.item()
    ppl = float(np.exp(nll / (n * a.seq)))
    method = "ours_kernels_" + ("sc_mxfp4_w4a8" if a.format == FQ.SC else a.format)
    row = dict(model=C.REPO, method=method, ppl=round(ppl, 4), windows=n, seq=a.seq,
               deltanet="reference" if a.reference_deltanet else "metal")
    print(row, flush=True)
    if a.decode_check:
        # prefill the first token, then decode one at a time; compare with the full-window prefill logits
        x = mx.array(toks[:a.decode_check])[None]
        full = f(x).astype(mx.float32)
        d = FQ.DecodeQwen38(f)
        cache = f.make_cache()
        outs = [f(x[:, :1], cache)[:, -1]]
        for t in range(1, a.decode_check):
            outs.append(d.step(x[:, t:t + 1], cache)[:, -1])
        dec = mx.stack(outs, axis=1).astype(mx.float32)
        diff = float(mx.max(mx.abs(dec - full)).item())
        agree = float(mx.mean((mx.argmax(dec, -1) == mx.argmax(full, -1)).astype(mx.float32)).item())
        print(f"decode vs prefill over {a.decode_check} tokens: max |logit diff| {diff:.4g}, argmax agreement {agree:.1%}")
    if a.out:
        new = not os.path.exists(a.out)
        with open(a.out, "a", newline="") as fh:
            wr = csv.DictWriter(fh, fieldnames=list(row))
            if new:
                wr.writeheader()
            wr.writerow(row)


if __name__ == "__main__":
    main()
