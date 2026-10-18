"""Markdown + LaTeX tables for results/07_system.md from the CSVs and check JSONs written by e2e/run_07.sh.
Usage: python make_07_tables.py [results-dir]   (prints to stdout)"""
import csv
import json
import os
import sys

R = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")


def rows(name):
    p = os.path.join(R, name)
    if not os.path.exists(p):
        return [], ""
    with open(p) as f:
        head = f.readline().strip()
        return list(csv.DictReader(f)), head.lstrip("# ")


def get(rs, method, key, length_key, length):
    for r in rs:
        if r["method"] == method and int(r[length_key]) == length:
            return float(r[key])
    return None


def prefill_table():
    out_md, out_tex = [], []
    names = {"mlx_mxfp4": "MLX MXFP4", "mlx_affine4": "MLX affine 4-bit", "sc_w4a8": "W4A8 unfused (05)",
             "fused_row": "W4A8 fused, row epilogues", "fused": "**W4A8 fused**", "fused_res": "W4A8 fused, INT8 resident"}
    out_md.append("| Prompt | Regime | " + " | ".join(names.values()) + " | **fused vs MXFP4** | fused vs unfused |")
    out_md.append("|---" * (len(names) + 4) + "|")
    states = []
    for fname, regime in (("07_prefill_burst.csv", "burst"), ("07_prefill_sustained.csv", "sustained")):
        rs, st = rows(fname)
        if not rs:
            continue
        states.append(f"{regime}: {st}")
        for L in sorted({int(r["prompt_tokens"]) for r in rs}):
            t = {m: get(rs, m, "ttft_ms_median", "prompt_tokens", L) for m in names}
            cells = [f"{t[m]:.1f}" if t[m] else "–" for m in names]
            sp = t["mlx_mxfp4"] / t["fused"] if t["fused"] and t["mlx_mxfp4"] else None
            su = t["sc_w4a8"] / t["fused"] if t["fused"] and t["sc_w4a8"] else None
            out_md.append(f"| {L} | {regime} | " + " | ".join(cells) + f" | **{sp:.2f}x** | {su:.2f}x |")
            out_tex.append(f"{L} & {regime} & " + " & ".join(c.replace("**", "") for c in cells)
                           + f" & \\textbf{{{sp:.2f}}}$\\times$ \\\\")
    return out_md, out_tex, states


def moe_table():
    rs, st = rows("07_moe_prefill.csv")
    names = {"mlx_mxfp4": "MLX MXFP4", "mlx_affine4": "MLX affine 4-bit", "fused_inkernel": "W4A8, in-kernel unpack",
             "fused_2pass": "W4A8, two-pass"}
    md = ["| Prompt | " + " | ".join(names.values()) + " | **best W4A8 vs MXFP4** | tok/s, best W4A8 vs MXFP4 |",
          "|---" * (len(names) + 3) + "|"]
    tex = []
    for L in sorted({int(r["prompt_tokens"]) for r in rs}):
        t = {m: get(rs, m, "ttft_ms_median", "prompt_tokens", L) for m in names}
        best = min(t["fused_inkernel"], t["fused_2pass"])
        cells = [(f"**{t[m]:.1f}**" if m.startswith("fused") and t[m] == best else f"{t[m]:.1f}") for m in names]
        sp = t["mlx_mxfp4"] / best
        md.append(f"| {L} | " + " | ".join(cells) + f" | **{sp:.2f}x** | {L / best:.1f}k vs {L / t['mlx_mxfp4']:.1f}k |")
        tex.append(f"{L} & " + " & ".join(c.replace("**", "") for c in cells) + f" & \\textbf{{{sp:.2f}}}$\\times$ \\\\")
    return md, tex, st


def decode_table():
    names = {"mlx_mxfp4": "MLX MXFP4", "mlx_affine4": "MLX affine 4-bit", "ours": "**W4A8 GEMV**",
             "ours_prologue": "W4A8 GEMV, quantization fused into GEMV", "ours_bf16head": "W4A8 GEMV, BF16 head"}
    md = ["| Model | Context | " + " | ".join(names.values()) + " | **W4A8 vs MXFP4** |", "|---" * (len(names) + 3) + "|"]
    tex, states = [], []
    for fname, model in (("07_decode_qwen3.csv", "Qwen3-1.7B"), ("07_decode_olmoe.csv", "OLMoE-1B-7B")):
        rs, st = rows(fname)
        if not rs:
            continue
        states.append(f"{model}: {st}")
        for c in sorted({int(r["context"]) for r in rs}):
            t = {m: get(rs, m, "tok_s_median", "context", c) for m in names}
            cells = [f"{t[m]:.0f}" if t[m] else "–" for m in names]
            sp = t["ours"] / t["mlx_mxfp4"]
            md.append(f"| {model} | {c} | " + " | ".join(cells) + f" | **{sp:.2f}x** |")
            tex.append(f"{model} & {c} & " + " & ".join(cells) + f" & \\textbf{{{sp:.2f}}}$\\times$ \\\\")
    return md, tex, states


def checks():
    out = []
    for name in ("07_check_fused.json", "07_check_moe.json", "07_check_decode_qwen3.json", "07_check_decode_olmoe.json"):
        p = os.path.join(R, name)
        if os.path.exists(p):
            j = json.load(open(p))
            out.append(f"- `{name}`: passed = {j.get('passed')}")
    return out


if __name__ == "__main__":
    for title, (md, tex, st) in (("Dense prefill (Qwen3-1.7B), TTFT ms", prefill_table()),
                                 ("MoE prefill (OLMoE-1B-7B), TTFT ms", moe_table()),
                                 ("Decode, tokens/s", decode_table())):
        print(f"### {title}\n")
        print("\n".join(md))
        print("\nConditions:", st if isinstance(st, str) else "; ".join(st))
        print("\nLaTeX rows:\n```latex\n" + "\n".join(tex) + "\n```\n")
    print("\n".join(checks()))
