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
Writes results/08_tasks_<model>_<method>.json (lm-eval's results dict).
Qwen3.8-Flash-Next (--model qwen38, run with .venv-q38): ours / ours_fmt on the fused kernels (fused_qwen38.py),
mlx_mxfp4_cov / mlx_affine4_cov = MLX on the same matrices (q38_mlx.py); results/10_tasks_qwen38_<method>.json.
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
TASKS = ["arc_easy", "arc_challenge", "hellaswag", "piqa", "winogrande", "lambada_openai"]
FORMAT = ["e2m1/B16/M1..10"]  # --format, for method ours_fmt
REPOS = {"qwen3": "Qwen/Qwen3-1.7B-Base", "qwen35": "Qwen/Qwen3.5-35B-A3B-Base", "qwen38": "Qwen/Qwen3.8-Flash-Next-FP8"}


def make_lm(model, tok):
    from lm_eval.api.model import LM
    from mlx_lm.evaluate import MLXLM
    lm = MLXLM.__new__(MLXLM)
    LM.__init__(lm)
    lm._model, lm.tokenizer = model, tok
    lm._max_tokens, lm._batch_size, lm.use_chat_template, lm._sampler = None, 1, False, None
    return lm


def load_method(model_name, method, cache_dir):
    from mlx_lm import load
    if model_name == "qwen38":  # .venv-q38 (mlx-lm PR #1788); systems built from the FP8 checkpoint
        from pathlib import Path
        import q38_ckpt as C
        from mlx_lm.utils import load_tokenizer
        tok = load_tokenizer(Path(C.snapshot()))
        if method.startswith("mlx_"):
            import q38_mlx
            return q38_mlx.load_mlx({"mlx_mxfp4_cov": "mlx_mxfp4", "mlx_affine4_cov": "mlx_affine4"}[method]), tok
        import fused_qwen38 as F8
        return F8.FusedQwen38(F8.SC if method == "ours" else FORMAT[0]), tok
    repo = REPOS[model_name]
    base = repo.split("/")[-1]
    if method == "bf16":
        return load(repo)
    if method.startswith("mlx_"):
        suffix = {"mlx_mxfp4": "-mxfp4", "mlx_affine4": "-4bit", "mlx_mxfp4_cov": "-mxfp4-w4a8cov"}[method]
        return load(os.path.join(cache_dir, base + suffix))
    if model_name == "qwen3":
        import fused_qwen3 as fq
        packed = {"ours_gptq": "gptq", "ours_fmt": "fmt:" + FORMAT[0]}.get(method, "rtn")
        return fq.load_fused(repo, cache_dir, packed=packed)
    import fused_qwen35 as fz
    return fz.load_fused_qwen35(repo, cache_dir)


def main():
    import lm_eval
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(REPOS))
    ap.add_argument("--method", required=True, choices=["bf16", "mlx_mxfp4", "mlx_affine4", "mlx_mxfp4_cov", "mlx_affine4_cov",
                                                         "ours", "ours_gptq", "ours_fmt"])
    ap.add_argument("--format", default=FORMAT[0])
    ap.add_argument("--tasks", default=",".join(TASKS))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--cache-dir", default=os.path.expanduser("~/.cache/mxfp4-int8-m5"))
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    FORMAT[0] = a.format
    model, tok = load_method(a.model, a.method, a.cache_dir)
    res = lm_eval.simple_evaluate(model=make_lm(model, tok), tasks=a.tasks.split(","), num_fewshot=0,
                                  limit=a.limit or None, random_seed=0, numpy_random_seed=0, torch_random_seed=0,
                                  fewshot_random_seed=0)
    out = a.out or os.path.join(HERE, "..", "results",
                                f"{'10' if a.model == 'qwen38' else '08'}_tasks_{a.model}_{a.method}.json")
    keep = {"results": res["results"], "n-samples": res.get("n-samples"), "config": {
        "model": REPOS[a.model], "method": a.method, "limit": a.limit or None, "num_fewshot": 0,
        "lm_eval": lm_eval.__version__}}
    with open(out, "w") as f:
        json.dump(keep, f, indent=1, default=str)
    for t, r in res["results"].items():
        print(t, {k: v for k, v in r.items() if "acc" in k or "ppl" in k})
    print("wrote", out)


if __name__ == "__main__":
    main()
