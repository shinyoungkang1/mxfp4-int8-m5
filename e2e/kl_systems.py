"""F3: KL divergence and top-1 agreement of the deployed systems against the unquantized model, on text outside
WikiText (LAMBADA test passages, BookCorpus fiction, from the local datasets cache that lm-eval already uses).

The systems are exactly those of the zero-shot table: eval_tasks.load_method builds each (mlx_mxfp4 = MLX's default
conversion, mlx_mxfp4_cov = MLX MXFP4 on our W4A8 matrices, ours = SC-MXFP4, ours_fmt:<format> = another INT8-exact
format on our kernels). The reference (bf16) stays loaded while each system is loaded in turn; for every window of
--seq tokens both produce logits, and per token we accumulate
  KL(p_ref || p_sys), top-1 disagreement (argmax differs) and the system's NLL of the next token.
The passages are joined with blank lines into one token stream and cut into disjoint windows (the first --windows).
Per-window sums are saved (<out without .csv>_windows/<model>__<method>.npz) for paired bootstrap intervals.
Usage (lm-eval venv): python kl_systems.py --model qwen3|qwen35 [--methods mlx_mxfp4,...] [--windows 32] --out results/x.csv
"""
import argparse
import csv
import gc
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "sim"))
sys.path.insert(0, HERE)
import provenance as prov  # noqa: E402  (first: its import time is the recorded start of the run)
import eval_tasks as ET  # noqa: E402

import mlx.core as mx  # noqa: E402
import numpy as np  # noqa: E402

LAMBADA = "~/.cache/huggingface/datasets/EleutherAI___lambada_openai/default/0.0.0/900124bf3b8235c6daf21033af9948b3f07346c4/lambada_openai-test.arrow"
METHODS = "mlx_mxfp4,mlx_mxfp4_cov,ours,ours_fmt:e2m1/B16/M1..10,ours_fmt:e2m1/B8/M1..10"
ROWS = 512  # tokens per slice of the KL computation (full-vocabulary float32 tensors stay below ~1 GB)


def lambada_tokens(tok):
    import pyarrow as pa
    with pa.memory_map(os.path.expanduser(LAMBADA)) as src:
        try:
            table = pa.ipc.open_stream(src).read_all()
        except pa.ArrowInvalid:
            table = pa.ipc.open_file(src).read_all()
    text = "\n\n".join(table.column("text").to_pylist())
    return np.array(tok.encode(text), dtype=np.int32), table.num_rows


def load(model_name, spec, cache_dir):
    method, _, fmt = spec.partition(":")
    ET.FORMAT[0] = fmt or ET.FORMAT[0]
    return ET.load_method(model_name, method, cache_dir)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["qwen3", "qwen35"])
    ap.add_argument("--methods", default=METHODS)
    ap.add_argument("--windows", type=int, default=32)
    ap.add_argument("--seq", type=int, default=2048)
    ap.add_argument("--cache-dir", default=os.path.expanduser("~/.cache/mxfp4-int8-m5"))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    specs = a.methods.split(",")
    for s in specs:  # stop before any model is loaded
        m, _, fmt = s.partition(":")
        ET.check(a.model, m, fmt_given=bool(fmt))
    ref, tok, _ = load(a.model, "bf16", a.cache_dir)
    toks, n_passages = lambada_tokens(tok)
    n = min(a.windows, (len(toks) - 1) // a.seq)
    print(f"{a.model}: {n_passages} LAMBADA passages, {len(toks)} tokens, {n} windows of {a.seq}", flush=True)
    chunks = [mx.array(toks[i * a.seq:(i + 1) * a.seq + 1])[None] for i in range(n)]
    wdir = os.path.splitext(a.out)[0] + "_windows"
    os.makedirs(wdir, exist_ok=True)
    rows, ref_nll = [], np.zeros(n)
    new = not os.path.exists(a.out)
    for spec in specs:
        t0 = time.time()
        sysm, _, what = load(a.model, spec, a.cache_dir)
        kl, flip, nll = np.zeros(n), np.zeros(n), np.zeros(n)
        for i, c in enumerate(chunks):
            x, y = c[:, :-1], c[0, 1:]
            lr_all, ls_all = ref(x)[0], sysm(x)[0]
            for r0 in range(0, a.seq, ROWS):
                lr = lr_all[r0:r0 + ROWS].astype(mx.float32)
                ls = ls_all[r0:r0 + ROWS].astype(mx.float32)
                lr = lr - mx.logsumexp(lr, axis=-1, keepdims=True)
                ls = ls - mx.logsumexp(ls, axis=-1, keepdims=True)
                yy = y[r0:r0 + ROWS]
                k = mx.sum(mx.exp(lr) * (lr - ls))
                f = mx.sum((mx.argmax(lr, axis=-1) != mx.argmax(ls, axis=-1)).astype(mx.float32))
                l_sys = -mx.sum(mx.take_along_axis(ls, yy[:, None], axis=-1))
                l_ref = -mx.sum(mx.take_along_axis(lr, yy[:, None], axis=-1))
                mx.eval(k, f, l_sys, l_ref)
                kl[i] += k.item()
                flip[i] += f.item()
                nll[i] += l_sys.item()
                ref_nll[i] += l_ref.item() if spec == specs[0] else 0.0
            del lr_all, ls_all
        tokens = n * a.seq
        row = dict(model=ET.REPOS[a.model], method=spec, text="lambada_openai test passages", windows=n, seq=a.seq,
                   kl_vs_bf16=round(kl.sum() / tokens, 5), top1_flip_vs_bf16=round(flip.sum() / tokens, 5),
                   ppl=round(float(np.exp(nll.sum() / tokens)), 4), ppl_bf16=round(float(np.exp(ref_nll.sum() / tokens)), 4),
                   seconds=round(time.time() - t0))
        rows.append(row)
        np.savez(os.path.join(wdir, f"{a.model}__{spec.replace('/', '_').replace(':', '-')}.npz"),
                 kl=kl, flip=flip, nll=nll, ref_nll=ref_nll, tokens_per_window=a.seq)
        print(row, flush=True)
        with open(a.out, "a", newline="") as fh:
            wr = csv.DictWriter(fh, fieldnames=list(row))
            if new:
                wr.writeheader()
                new = False
            wr.writerow(row)
        prov.sidecar(a.out, model=ET.REPOS[a.model], method=spec, windows=n, seq=a.seq, text="lambada_openai", **what)
        del sysm
        gc.collect()
        mx.clear_cache()
    print("wrote", a.out)


if __name__ == "__main__":
    main()
