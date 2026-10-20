"""GSM8K (5-shot, greedy) on Qwen3.8-Flash-Next systems, batch 1, identical loop for every system:
  fused         ours: W4A8 prefill (fused_qwen38.FusedQwen38) + W4A8 GEMV decode (DecodeQwen38, dense="ours")
  fused_b16     the same with the 16-block integer-exact format (e2m1/B16/M1..10) instead of SC-MXFP4
  mlx_mxfp4     MLX MXFP4 W4A16 on exactly the same matrices (q38_mlx.load_mlx), MLX's own quantization (OCP rule)
  mlx_sc_mxfp4  MLX MXFP4 W4A16 on our SC-MXFP4 weights (separates INT8 activations from the weight quantizer)
Prompt: the first 5 training problems as shots, in chat format with thinking disabled (one user turn "Question: ..."
and one assistant turn "<solution> #### <n>" per shot; with plain completion this instruct model ends its turn at
once). Generation stops as lm-eval's gsm8k task does (at "Question:", "<|im_end|>" or the end-of-sequence token) or
after --max-tokens. Scoring: lm-eval's strict-match ("#### <n>") and flexible-extract (last number) regexes.
Usage (.venv-q38): python gsm8k_q38.py --system fused --n 500 --out ../results/10_gsm8k_fused.jsonl
"""
import argparse
import json
import os
import re
import sys
import time

import mlx.core as mx
import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import q38_ckpt as C  # noqa: E402

STRICT = re.compile(r"#### (\-?[0-9\.\,]+)")
FLEX = re.compile(r"(-?[$0-9.,]{2,})|(-?[0-9]+)")


def norm(s):
    return s.replace(",", "").replace("$", "").rstrip(".").strip()


def load_gsm8k():
    f = lambda split: hf_hub_download("openai/gsm8k", f"main/{split}-00000-of-00001.parquet", repo_type="dataset")  # noqa: E731
    return pq.read_table(f("train")).to_pylist(), pq.read_table(f("test")).to_pylist()


def prompt(shots, q, tok=None):
    if tok is None:  # plain completion (lm-eval's default gsm8k format)
        return "".join(f"Question: {s['question']}\nAnswer: {s['answer']}\n\n" for s in shots) + f"Question: {q}\nAnswer:"
    msgs = []
    for s in shots:
        msgs += [{"role": "user", "content": f"Question: {s['question']}"}, {"role": "assistant", "content": s["answer"]}]
    msgs.append({"role": "user", "content": f"Question: {q}"})
    return tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False, enable_thinking=False)


class Fused:
    def __init__(self, fmt_name, layers=None):
        import fused_qwen38 as FQ
        self.f = FQ.FusedQwen38(fmt_name or FQ.SC, layers=layers)
        self.d = FQ.DecodeQwen38(self.f, dense="ours")

    def start(self, ids):
        cache = self.f.make_cache()
        return self.f.logits(self.f.hidden(ids, cache)[:, -1:]), cache

    def step(self, tok, cache):
        return self.d.step(tok, cache)


class Mlx:
    def __init__(self, method, layers=None):
        import q38_mlx
        self.m = q38_mlx.load_mlx(method, layers=layers)

    def start(self, ids):
        cache = self.m.make_cache()
        return self.m.lm_head(self.m.model(ids, cache=cache)[:, -1:]), cache

    def step(self, tok, cache):
        return self.m(tok, cache=cache)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--system", required=True, choices=["fused", "fused_b16", "mlx_mxfp4", "mlx_sc_mxfp4"])
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--max-tokens", type=int, default=320)
    ap.add_argument("--out", required=True)
    ap.add_argument("--layers", type=int, default=0, help="first N layers only (smoke test; answers are meaningless)")
    a = ap.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(C.snapshot())
    train, test = load_gsm8k()
    shots = train[:5]
    done = {}
    if os.path.exists(a.out):  # resumable
        for line in open(a.out):
            r = json.loads(line)
            done[r["i"]] = r
    fmt = {"fused": None, "fused_b16": "e2m1/B16/M1..10"}
    sysm = Fused(fmt[a.system], a.layers or None) if a.system in fmt else Mlx(a.system, a.layers or None)
    t_all, n_tok = time.time(), 0
    with open(a.out, "a") as fh:
        for i, ex in enumerate(test[:a.n]):
            if i in done:
                continue
            ids = mx.array(tok.encode(prompt(shots, ex["question"], tok), add_special_tokens=False))[None]
            logits, cache = sysm.start(ids)
            out = []
            for _ in range(a.max_tokens):
                t = int(mx.argmax(logits[0, -1]).item())
                out.append(t)
                text = tok.decode(out)
                if "Question:" in text or "<|im_end|>" in text or t in (tok.eos_token_id, 248044):
                    break
                logits = sysm.step(mx.array([[t]]), cache)
            text = tok.decode(out).split("Question:")[0].split("<|im_end|>")[0]
            n_tok += len(out)
            gold = norm(ex["answer"].split("####")[-1])
            m = STRICT.search(text)
            strict = norm(m.group(1)) if m else None
            nums = [x for g in FLEX.findall(text) for x in g if x]
            flex = norm(nums[-1]) if nums else None
            r = dict(i=i, gold=gold, strict=strict, flex=flex, strict_ok=strict == gold, flex_ok=flex == gold,
                     tokens=len(out), text=text)
            fh.write(json.dumps(r) + "\n")
            fh.flush()
            done[i] = r
            if (i + 1) % 25 == 0:
                rs = list(done.values())
                print(f"{i + 1}: strict {np.mean([x['strict_ok'] for x in rs]):.3f} flex "
                      f"{np.mean([x['flex_ok'] for x in rs]):.3f}; {n_tok / (time.time() - t_all):.1f} gen tok/s",
                      flush=True)
    rs = [done[i] for i in sorted(done) if i < a.n]
    s, fl = np.mean([x["strict_ok"] for x in rs]), np.mean([x["flex_ok"] for x in rs])
    se = lambda p: np.sqrt(p * (1 - p) / len(rs))  # noqa: E731
    print(f"{a.system}: n={len(rs)} strict-match {s:.4f} +- {se(s):.4f}, flexible-extract {fl:.4f} +- {se(fl):.4f}, "
          f"mean generated tokens {np.mean([x['tokens'] for x in rs]):.0f}")


if __name__ == "__main__":
    main()
