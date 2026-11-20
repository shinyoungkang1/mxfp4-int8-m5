"""Zero-shot downstream accuracy (lm-evaluation-harness, through mlx_lm's MLXLM adapter) of the actual systems:
BF16, MLX MXFP4, MLX 4-bit affine, and the SC-MXFP4 W4A8 model running on our kernels (fused prefill path).
mlx_mxfp4 / mlx_affine4 are mlx_lm's default conversions (they also quantize embeddings, lm_head and routers);
mlx_mxfp4_cov is MLX MXFP4 on exactly the matrices that ours runs in W4A8 (convert_mlx.py ... w4a8cov), so that it
differs from ours only in the format and the activations. ours_gptq: ours with GPTQ-rounded SC-MXFP4 weights
(e2e/gptq_pack.py; Qwen3 only). ours_fmt: ours in another INT8-exact format (--format, e2e/pack_format.py; Qwen3).
Every method is scored by the same MLXLM code (prefix once, then each continuation from a copy of its cache);
for ours the model object is the fused model instead of an mlx_lm Model.

Run with the lm-eval environment (~/.cache/mxfp4-int8-m5/venv-eval: same mlx / mlx-lm versions + lm-eval):
  python eval_tasks.py --model qwen3|qwen35 --method bf16|mlx_mxfp4|mlx_affine4|mlx_mxfp4_cov|ours|ours_gptq [--limit N]
Writes results/08_tasks_<model>_<method>.json (lm-eval's results dict; config records format, pack and provenance)
and, next to it, <same name>.samples.json: one record per document and task (doc_id, the task's metrics, the
log-likelihood of every choice), which e2e/paired_tasks.py compares between two systems on the same documents.
Qwen3.8-Flash-Next (--model qwen38, run with .venv-q38): ours / ours_fmt on the fused kernels (fused_qwen38.py),
mlx_mxfp4_cov / mlx_affine4_cov = MLX on the same matrices (q38_mlx.py); results/10_tasks_qwen38_<method>.json.

Which methods exist for which model is the table METHODS; any other pair is an error (nothing is substituted).
--format is for the methods in USES_FORMAT only (default e2m1/B16/M1..10) and an error with any other method.
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "sim"))
sys.path.insert(0, HERE)
import provenance as prov  # noqa: E402  (first: its import time is the recorded start of the run)

TASKS = ["arc_easy", "arc_challenge", "hellaswag", "piqa", "winogrande", "lambada_openai"]
FORMAT = ["e2m1/B16/M1..10"]  # --format, for method ours_fmt
REPOS = {"qwen3": "Qwen/Qwen3-1.7B-Base", "qwen35": "Qwen/Qwen3.5-35B-A3B-Base", "qwen38": "Qwen/Qwen3.8-Flash-Next-FP8",
         "qwen3_4b": "Qwen/Qwen3-4B-Base", "smol3": "HuggingFaceTB/SmolLM3-3B-Base"}
# The methods that are implemented per model. qwen3's ours_rot / ours_rotdown packs are GPTQ-rounded in the rotated
# basis; qwen35's ours_rot pack is round-to-nearest. Simulated ("fake") Qwen3.5 / Qwen3.8 systems: ppl_qwen35.py, q38_ppl.py.
METHODS = {
    "qwen3": ["bf16", "mlx_mxfp4", "mlx_affine4", "mlx_mxfp4_cov", "ours", "ours_gptq", "ours_fmt", "ours_rot", "ours_rotdown", "fake"],
    "qwen35": ["bf16", "mlx_mxfp4", "mlx_affine4", "mlx_mxfp4_cov", "ours", "ours_fmt", "ours_rot"],
    "qwen38": ["mlx_mxfp4_cov", "mlx_affine4_cov", "ours", "ours_fmt"],
    "qwen3_4b": ["bf16", "fake"],
    "smol3": ["bf16", "fake"],
}
USES_FORMAT = ["ours_fmt", "ours_rot", "ours_rotdown"]
MLX_SUFFIX = {"mlx_mxfp4": "-mxfp4", "mlx_affine4": "-4bit", "mlx_mxfp4_cov": "-mxfp4-w4a8cov"}


def check(model_name, method, fmt_given=False):
    """Stops a run whose (model, method) pair has no implementation, instead of evaluating some other system."""
    if method not in METHODS[model_name]:
        raise SystemExit(f"eval_tasks: method {method} is not implemented for {model_name} (implemented: {', '.join(METHODS[model_name])})")
    if fmt_given and method not in USES_FORMAT:
        raise SystemExit(f"eval_tasks: --format has no effect with method {method} (it applies to {', '.join(USES_FORMAT)})")


def make_lm(model, tok):
    from lm_eval.api.model import LM
    from mlx_lm.evaluate import MLXLM
    lm = MLXLM.__new__(MLXLM)
    LM.__init__(lm)
    lm._model, lm.tokenizer = model, tok
    lm._max_tokens, lm._batch_size, lm.use_chat_template, lm._sampler = None, 1, False, None
    return lm


def load_method(model_name, method, cache_dir):
    """-> (model, tokenizer, description of what was loaded: format, pack, head)."""
    check(model_name, method)
    from mlx_lm import load
    fmt = FORMAT[0] if method in USES_FORMAT else None
    if model_name == "qwen38":  # .venv-q38 (mlx-lm PR #1788); systems built from the FP8 checkpoint
        from pathlib import Path
        import q38_ckpt as C
        from mlx_lm.utils import load_tokenizer
        tok = load_tokenizer(Path(C.snapshot()))
        if method.startswith("mlx_"):
            import q38_mlx
            return q38_mlx.load_mlx({"mlx_mxfp4_cov": "mlx_mxfp4", "mlx_affine4_cov": "mlx_affine4"}[method]), tok, dict(format=method, pack="q38_mlx")
        import fused_qwen38 as F8
        f = F8.FusedQwen38(F8.SC if method == "ours" else fmt)
        return f, tok, dict(format=fmt or "scmxfp4", pack=F8.pack_dir(F8.SC if method == "ours" else fmt, os.path.expanduser("~/.cache/mxfp4-int8-m5")))
    repo = REPOS[model_name]
    base = repo.split("/")[-1]
    if method == "bf16":
        return (*load(repo), dict(format="bf16", pack=repo))
    if method.startswith("mlx_"):
        path = os.path.join(cache_dir, base + MLX_SUFFIX[method])
        return (*load(path), dict(format=method, pack=path))
    if model_name == "qwen3":
        import fused_qwen3 as fq
        packed = {"ours": "rtn", "ours_gptq": "gptq", "ours_fmt": f"fmt:{fmt}", "ours_rot": f"rot:{fmt}", "ours_rotdown": f"rotdown:{fmt}"}[method]
        f, tok = fq.load_fused(repo, cache_dir, packed=packed)
        return f, tok, dict(format=fmt or "scmxfp4", pack=packed)
    import fused_qwen35 as fz
    f, tok = fz.load_fused_qwen35(repo, cache_dir, fmt_name=fmt, rot=method == "ours_rot")
    return f, tok, dict(format=f.fmt_name + ("+rot" if f.rot else ""), pack=f.pack_path)


def per_example(samples):
    """lm-eval's per-document records, reduced to what a paired comparison needs: doc_id, hashes of the document and
    the prompt (to verify that two files scored the same thing), target, the task's metric values, and resps = one
    [log-likelihood, is_greedy] per choice (one entry for lambada)."""
    out = {}
    for task, docs in (samples or {}).items():
        rows = []
        for d in docs:
            row = {"doc_id": d["doc_id"], "doc_hash": d.get("doc_hash"), "prompt_hash": d.get("prompt_hash"),
                   "target": d.get("target"), "resps": d.get("filtered_resps")}
            row.update({m: (d[m].item() if hasattr(d[m], "item") else d[m]) for m in d.get("metrics", []) if m in d})
            rows.append(row)
        out[task] = sorted(rows, key=lambda r: r["doc_id"])
    return out


def main():
    import lm_eval
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(REPOS))
    ap.add_argument("--method", required=True, choices=sorted({m for ms in METHODS.values() for m in ms}))
    ap.add_argument("--fake", default="", help="method fake: a sim/eval_ppl.py method string (fake quantization, e.g. w=gptq:sc_mx;a=8;r=32)")
    ap.add_argument("--format", default=None, help=f"for methods {', '.join(USES_FORMAT)}: a sim/formats.py name (default {FORMAT[0]})")
    ap.add_argument("--tasks", default=",".join(TASKS))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--cache-dir", default=os.path.expanduser("~/.cache/mxfp4-int8-m5"))
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    check(a.model, a.method, fmt_given=a.format is not None)
    if (a.method == "fake") != bool(a.fake):
        raise SystemExit("eval_tasks: --fake <method string> goes with --method fake, and only with it")
    FORMAT[0] = a.format or FORMAT[0]
    if a.method == "fake":
        from mlx_lm import load
        import eval_ppl
        model, tok = load(REPOS[a.model])
        eval_ppl.quantize_model(model, a.fake, REPOS[a.model])
        what = dict(format="fake:" + a.fake, pack=None)
    else:
        model, tok, what = load_method(a.model, a.method, a.cache_dir)
    res = lm_eval.simple_evaluate(model=make_lm(model, tok), tasks=a.tasks.split(","), num_fewshot=0,
                                  limit=a.limit or None, random_seed=0, numpy_random_seed=0, torch_random_seed=0,
                                  fewshot_random_seed=0)
    out = a.out or os.path.join(HERE, "..", "results",
                                f"{'10' if a.model == 'qwen38' else '08'}_tasks_{a.model}_{a.method}.json")
    config = {"model": REPOS[a.model], "method": a.method, "fake": a.fake, "format": what["format"], "pack": prov.tilde(what["pack"]),
              "limit": a.limit or None, "num_fewshot": 0, "lm_eval": lm_eval.__version__,
              "provenance": prov.stamp(model=REPOS[a.model], method=a.method, **what)}
    keep = {"results": res["results"], "n-samples": res.get("n-samples"), "config": config}
    with open(out, "w") as f:
        json.dump(keep, f, indent=1, default=str)
    samples = os.path.splitext(out)[0] + ".samples.json"
    with open(samples, "w") as f:
        json.dump({"config": config, "samples": per_example(res.get("samples"))}, f, default=str)
    for t, r in res["results"].items():
        print(t, {k: v for k, v in r.items() if "acc" in k or "ppl" in k})
    print("wrote", out, "and", samples)


if __name__ == "__main__":
    main()
