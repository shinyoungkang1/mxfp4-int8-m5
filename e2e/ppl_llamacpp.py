"""WikiText-2 perplexity of llama.cpp's 4-bit types next to its BF16, measured with llama.cpp's own tool
(llama-perplexity, 2048-token chunks over the whole test set, the text of sim/eval_ppl.py). llama-perplexity scores
the second half of each chunk, so its numbers compare with each other (relative to llama.cpp's BF16), not with
sim/eval_ppl.py's. Types: Q4_0, Q4_K_M, IQ4_XS (llama.cpp's closest type to IQ4_KS, which is only in the
ik_llama.cpp fork; quantized without an importance matrix, like every format in the paper) and, for MoEs,
MXFP4_MOE. Missing GGUFs are quantized from the BF16 ones that e2e/run_08.sh converted.

Usage (.venv): python ppl_llamacpp.py [--models qwen3,olmoe,qwen35] --out ../results/12_ppl_llamacpp.csv
"""
import argparse
import csv
import os
import re
import subprocess

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

D = os.path.expanduser("~/.cache/mxfp4-int8-m5")
BIN = os.path.join(D, "third_party/llama.cpp/build/bin")
G = os.path.join(D, "gguf")
MODELS = {"qwen3": ("Qwen3-1.7B-Base", ["bf16", "Q4_0", "Q4_K_M", "IQ4_XS"]),
          "olmoe": ("OLMoE-1B-7B-0924", ["bf16", "Q4_0", "Q4_K_M", "IQ4_XS", "MXFP4_MOE"]),
          "qwen35": ("Qwen3.5-35B-A3B-Base", ["bf16", "Q4_0", "Q4_K_M", "IQ4_XS", "MXFP4_MOE"])}


def wikitext(path):
    if not os.path.exists(path):
        f = hf_hub_download("Salesforce/wikitext", "wikitext-2-raw-v1/test-00000-of-00001.parquet", repo_type="dataset")
        with open(path, "w") as fh:
            fh.write("\n\n".join(pq.read_table(f).column("text").to_pylist()))
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="qwen3,olmoe,qwen35")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    text = wikitext(os.path.join(D, "wikitext2_test.txt"))
    done = set()
    if os.path.exists(a.out):
        done = {(r["model"], r["quant"]) for r in csv.DictReader(open(a.out))}
    for m in a.models.split(","):
        name, quants = MODELS[m]
        for q in quants:
            if (name, q) in done:
                continue
            gguf = os.path.join(G, f"{name}-{q}.gguf")
            if not os.path.exists(gguf):
                log = os.path.join(G, f"{name}-{q}.quantize.log")
                with open(log, "w") as fh:
                    subprocess.run([os.path.join(BIN, "llama-quantize"), os.path.join(G, f"{name}-bf16.gguf"), gguf, q],
                                   check=True, stdout=fh, stderr=subprocess.STDOUT)
            r = subprocess.run([os.path.join(BIN, "llama-perplexity"), "-m", gguf, "-f", text, "-c", "2048",
                                "-b", "2048", "-ngl", "99"], capture_output=True, text=True)
            hit = re.search(r"Final estimate: PPL = ([0-9.]+) \+/- ([0-9.]+)", r.stdout + r.stderr)
            if not hit:
                print(f"{name} {q}: no perplexity in the output (exit {r.returncode})\n{(r.stdout + r.stderr)[-1500:]}",
                      flush=True)
                continue
            row = dict(model=name, quant=q, ppl=float(hit.group(1)), ppl_stderr=float(hit.group(2)),
                       gguf_gib=round(os.path.getsize(gguf) / 2 ** 30, 2))
            print(row, flush=True)
            new = not os.path.exists(a.out)
            with open(a.out, "a", newline="") as fh:
                wr = csv.DictWriter(fh, fieldnames=list(row))
                if new:
                    wr.writeheader()
                wr.writerow(row)


if __name__ == "__main__":
    main()
