"""Markdown tables for results/08_system2.md from the CSV / JSON files of e2e/run_08.sh (and results/07 for the MLX /
W4A8 numbers of Qwen3-1.7B and OLMoE). Usage: python make_08_tables.py [results-dir]"""
import csv
import glob
import json
import os
import sys

R = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")


def rows(name):
    p = os.path.join(R, name)
    if not os.path.exists(p):
        return []
    with open(p) as f:
        f.readline()
        return list(csv.DictReader(f))


def pick(rs, **kw):
    for r in rs:
        if all(str(r.get(k)) == str(v) for k, v in kw.items()):
            return r
    return None


def ttft(rs, method, L, key="ttft_ms_median"):
    r = pick(rs, method=method, prompt_tokens=L)
    return float(r[key]) if r else None


def llama_tables():
    out = ["| Model | Prompt | llama.cpp Q4_0 | llama.cpp MXFP4_MOE | MLX MXFP4 | **W4A8 (ours)** | **ours vs llama.cpp** |",
           "|---|---|---|---|---|---|---|"]
    dec = ["| Model | Context | llama.cpp Q4_0 | llama.cpp MXFP4_MOE | MLX MXFP4 | **W4A8 (ours)** | ours vs llama.cpp |",
           "|---|---|---|---|---|---|---|"]
    src = {"qwen3": ("Qwen3-1.7B", "07_prefill_burst.csv", "07_prefill_sustained.csv", "fused", "07_decode_qwen3.csv"),
           "olmoe": ("OLMoE-1B-7B", "07_moe_prefill.csv", None, "best", "07_decode_olmoe.csv"),
           "qwen35": ("Qwen3.5-35B-A3B", "08_qwen35_prefill.csv", None, "best", "08_qwen35_decode.csv")}
    for key, (label, pf, pf2, ours_m, df) in src.items():
        lr = rows(f"08_llamacpp_{key}.csv")
        if not lr:
            continue
        p1, p2 = rows(pf), rows(pf2) if pf2 else []
        for L in sorted({int(r["tokens"]) for r in lr if r["test"] == "prefill_best"}):
            lq = {q: pick(lr, test="prefill_best", quant=q, tokens=L) for q in ("Q4_0", "MXFP4_MOE")}
            lt = {q: float(v["ttft_ms"]) if v else None for q, v in lq.items()}
            src_rows = p1 if ttft(p1, "mlx_mxfp4", L) else p2
            mx_t = ttft(src_rows, "mlx_mxfp4", L)
            if ours_m == "best":
                cands = [ttft(src_rows, m, L) for m in ("fused", "fused_inkernel", "fused_2pass")]
                ours = min(c for c in cands if c)
            else:
                ours = ttft(src_rows, ours_m, L)
            best_l = min(v for v in lt.values() if v)
            f = lambda v: f"{v:.1f}" if v else "–"  # noqa: E731
            out.append(f"| {label} | {L} | {f(lt['Q4_0'])} | {f(lt['MXFP4_MOE'])} | {f(mx_t)} | **{f(ours)}** | "
                       f"**{best_l / ours:.2f}x** |")
        drs, lrd = rows(df), lr
        if rows(f"08_decode_session_{key}.csv") and rows(f"08_decode_session_llamacpp_{key}.csv"):
            drs, lrd = rows(f"08_decode_session_{key}.csv"), rows(f"08_decode_session_llamacpp_{key}.csv")
        for d in (512, 2048):
            lq = {q: pick(lrd, test="decode", quant=q, depth=d) for q in ("Q4_0", "MXFP4_MOE")}
            lt = {q: float(v["tok_s"]) if v else None for q, v in lq.items()}
            m = pick(drs, method="mlx_mxfp4", context=d)
            o = pick(drs, method="ours", context=d)
            if not (m and o):
                continue
            best_l = max(v for v in lt.values() if v)
            f = lambda v: f"{v:.0f}" if v else "–"  # noqa: E731
            dec.append(f"| {label} | {d} | {f(lt['Q4_0'])} | {f(lt['MXFP4_MOE'])} | {float(m['tok_s_median']):.0f} | "
                       f"**{float(o['tok_s_median']):.0f}** | {float(o['tok_s_median']) / best_l:.2f}x |")
    return out, dec


def qwen35_tables():
    rs = rows("08_qwen35_prefill.csv")
    names = ["mlx_mxfp4", "mlx_affine4", "fused", "fused_inkernel", "fused_2pass"]
    out = ["| Prompt | MLX MXFP4 | MLX affine 4-bit | W4A8 (automatic choice) | W4A8, in-kernel unpack | W4A8, two-pass "
           "| **W4A8 vs MXFP4** (automatic / best) |", "|---|---|---|---|---|---|---|"]
    for L in sorted({int(r["prompt_tokens"]) for r in rs}):
        t = {m: ttft(rs, m, L) for m in names}
        best = min(t["fused"], t["fused_inkernel"], t["fused_2pass"])
        out.append(f"| {L} | " + " | ".join(f"{t[m]:.1f}" for m in names)
                   + f" | **{t['mlx_mxfp4'] / t['fused']:.2f}x / {t['mlx_mxfp4'] / best:.2f}x** |")
    dr = rows("08_qwen35_decode.csv")
    dec = ["| Context | MLX MXFP4 | MLX affine 4-bit | **W4A8 GEMV** | W4A8 vs MXFP4 |", "|---|---|---|---|---|"]
    for c in sorted({int(r["context"]) for r in dr}):
        t = {m: float(pick(dr, method=m, context=c)["tok_s_median"]) for m in ("mlx_mxfp4", "mlx_affine4", "ours")}
        dec.append(f"| {c} | {t['mlx_mxfp4']:.0f} | {t['mlx_affine4']:.0f} | **{t['ours']:.0f}** | "
                   f"{t['ours'] / t['mlx_mxfp4']:.2f}x |")
    return out, dec


METRIC = {"arc_easy": "acc,none", "arc_challenge": "acc_norm,none", "hellaswag": "acc_norm,none",
          "piqa": "acc_norm,none", "winogrande": "acc,none", "lambada_openai": "acc,none"}


def task_tables():
    out = []
    for model, label in (("qwen3", "Qwen3-1.7B-Base"), ("qwen35", "Qwen3.5-35B-A3B-Base")):
        files = {os.path.basename(p)[len(f"08_tasks_{model}_"):-5]: p for p in glob.glob(os.path.join(R, f"08_tasks_{model}_*.json"))}
        if not files:
            continue
        lim = None
        out.append(f"\n**{label}**\n")
        out.append("| Method | " + " | ".join(METRIC) + " | mean | lambada ppl |")
        out.append("|---" * (len(METRIC) + 3) + "|")
        for m in ("bf16", "mlx_affine4", "mlx_mxfp4", "mlx_mxfp4_cov", "ours", "ours_gptq"):
            if m not in files:
                continue
            j = json.load(open(files[m]))
            lim = j["config"].get("limit")
            vals = [100 * j["results"][t][k] for t, k in METRIC.items() if t in j["results"]]
            ppl = j["results"].get("lambada_openai", {}).get("perplexity,none")
            name = {"bf16": "BF16", "mlx_affine4": "MLX affine 4-bit (W4A16), default coverage",
                    "mlx_mxfp4": "MLX MXFP4 (W4A16), default coverage",
                    "mlx_mxfp4_cov": "MLX MXFP4 (W4A16), same matrices as ours",
                    "ours": "**SC-MXFP4 W4A8 (our kernels)**",
                    "ours_gptq": "**SC-MXFP4 + GPTQ W4A8 (our kernels)**"}[m]
            out.append(f"| {name} | " + " | ".join(f"{v:.1f}" for v in vals) + f" | {sum(vals) / len(vals):.1f} | "
                       + (f"{ppl:.2f}" if ppl else "–") + " |")
        if lim:
            out.append(f"\n(first {lim} examples of each task)")
    return out


PPL = [("bf16", "BF16", "–", "–"),
       ("mlx_affine4", "MLX affine g64 (checkpoint, default coverage)", "W4A16", "all linears + embeddings + head"),
       ("mlx_mxfp4", "MLX MXFP4 (checkpoint, default coverage)", "W4A16", "all linears + embeddings + head"),
       ("mlx_mxfp4_cov", "MLX MXFP4 (checkpoint)", "W4A16", "W4A8 matrices"),
       ("int4_affine_g64_w4a16", "INT4 affine g64 (fake quant)", "W4A16", "W4A8 matrices"),
       ("nvfp4_w4a16", "NVFP4 (fake quant)", "W4A16", "W4A8 matrices"),
       ("sc_mxfp4_w4a16", "SC-MXFP4 (fake quant)", "W4A16", "W4A8 matrices"),
       ("sc_mxfp4_w4a8", "SC-MXFP4 (fake quant, reference)", "W4A8", "W4A8 matrices"),
       ("ours", "**SC-MXFP4 on our kernels**", "**W4A8**", "W4A8 matrices")]


def ppl_table():
    p = os.path.join(R, "08_ppl_qwen35.csv")
    if not os.path.exists(p):
        return []
    with open(p) as f:
        v = {r["method"]: float(r["ppl"]) for r in csv.DictReader(f)}
    out = ["| Method | Weights x activations | Quantized | WikiText-2 ppl | vs BF16 |", "|---|---|---|---|---|"]
    for m, name, wa, cov in PPL:
        if m in v:
            out.append(f"| {name} | {wa} | {cov} | {v[m]:.3f} | {100 * (v[m] / v['bf16'] - 1):+.1f}% |")
    return out


if __name__ == "__main__":
    pf, dc = llama_tables()
    print("### vs llama.cpp: prefill TTFT (ms)\n\n" + "\n".join(pf) + "\n\n### vs llama.cpp: decode (tokens/s)\n\n" + "\n".join(dc))
    q, qd = qwen35_tables()
    print("\n### Qwen3.5-35B-A3B prefill (TTFT ms)\n\n" + "\n".join(q) + "\n\n### Qwen3.5-35B-A3B decode (tokens/s)\n\n" + "\n".join(qd))
    print("\n### Qwen3.5-35B-A3B WikiText-2 perplexity\n\n" + "\n".join(ppl_table()))
    print("\n### Zero-shot tasks (accuracy %)\n" + "\n".join(task_tables()))
