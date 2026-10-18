"""llama.cpp baseline (llama-bench, Metal backend with the tensor API, i.e. the M5 Neural Accelerators, FP16 operands)
on the same models: prefill throughput at our prompt lengths and decode throughput at 512 / 2048 context.

llama.cpp is built from source at a pinned release (third_party/llama.cpp, tag b11706; see run_08.sh); models are
converted from the same Hugging Face checkpoints with its convert_hf_to_gguf.py (BF16) and quantized with
llama-quantize to Q4_0 (4.5 bits/weight; every model) and MXFP4_MOE (MXFP4 experts, Q8_0 elsewhere; MoE models).
Prefill is measured with llama-bench's default batching (-b 2048 -ub 512) and with large micro-batches
(-b 8192 -ub 2048); the faster one is reported. Decode: 128 generated tokens after a prompt of depth 512 / 2048.

Usage: python bench_llamacpp.py --model qwen3|olmoe|qwen35 [--decode-only] [--out results/08_llamacpp_<model>.csv]
--decode-only: decode rows only (queue_08_decode.sh runs it right after our / MLX's decode under the same GPU lock,
so that all three decode speeds come from one session).
"""
import argparse
import csv
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
LC = os.path.expanduser("~/.cache/mxfp4-int8-m5/third_party/llama.cpp")
G = os.path.expanduser("~/.cache/mxfp4-int8-m5/gguf")
MODELS = {
    "qwen3": ("Qwen3-1.7B-Base", ["Q4_0"], [512, 2048, 8192]),
    "olmoe": ("OLMoE-1B-7B-0924", ["Q4_0", "MXFP4_MOE"], [512, 2048, 4096]),
    "qwen35": ("Qwen3.5-35B-A3B-Base", ["Q4_0", "MXFP4_MOE"], [512, 2048, 8192]),
}


def bench(gguf, args):
    cmd = [os.path.join(LC, "build/bin/llama-bench"), "-m", gguf, "-o", "json", "-r", "5", "-ngl", "99"] + args
    out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
    return json.loads(out)


def main():
    import bench_prefill as bp
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(MODELS))
    ap.add_argument("--out", default="")
    ap.add_argument("--decode-only", action="store_true")
    a = ap.parse_args()
    name, quants, lengths = MODELS[a.model]
    rows = []
    for q in quants:
        gguf = os.path.join(G, f"{name}-{q}.gguf")
        pl = ",".join(map(str, lengths))
        best = {}
        for b, ub in () if a.decode_only else ((2048, 512), (8192, 2048)):
            for r in bench(gguf, ["-p", pl, "-n", "0", "-b", str(b), "-ub", str(ub)]):
                L = r["n_prompt"]
                rec = dict(model=name, quant=q, test="prefill", tokens=L, depth=0, batch=b, ubatch=ub,
                           tok_s=r["avg_ts"], tok_s_std=r["stddev_ts"], ttft_ms=1e3 * L / r["avg_ts"],
                           model_bytes=r.get("model_size"), params=r.get("model_n_params"), build=r.get("build_commit"))
                rows.append(rec)
                if L not in best or rec["tok_s"] > best[L]["tok_s"]:
                    best[L] = rec
                print(f"{name} {q} pp{L} b{b}/ub{ub}: {r['avg_ts']:.0f} tok/s (TTFT {1e3 * L / r['avg_ts']:.1f} ms)",
                      flush=True)
        for d in (512, 2048):
            for r in bench(gguf, ["-p", "0", "-n", "128", "-d", str(d)]):
                rows.append(dict(model=name, quant=q, test="decode", tokens=128, depth=d, batch=2048, ubatch=512,
                                 tok_s=r["avg_ts"], tok_s_std=r["stddev_ts"], ttft_ms="",
                                 model_bytes=r.get("model_size"), params=r.get("model_n_params"),
                                 build=r.get("build_commit")))
                print(f"{name} {q} tg128 @ depth {d}: {r['avg_ts']:.1f} tok/s", flush=True)
        for L, rec in best.items():
            rows.append(dict(rec, test="prefill_best"))
    state = bp.power_state()
    print("#", state)
    if a.out:
        with open(a.out, "w", newline="") as f:
            f.write(f"# {state}; llama.cpp {rows[0]['build']}\n")
            wr = csv.DictWriter(f, fieldnames=list(rows[0]))
            wr.writeheader()
            wr.writerows(rows)
        print("wrote", a.out)


if __name__ == "__main__":
    main()
