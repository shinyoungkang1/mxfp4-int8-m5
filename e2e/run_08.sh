#!/bin/zsh
# Reproduces results/08_*: llama.cpp baselines (Qwen3-1.7B, OLMoE-1B-7B, Qwen3.5-35B-A3B), the Qwen3.5-35B-A3B-Base
# system results (prefill, decode, correctness, perplexity, weight statistics, coverage) and zero-shot
# downstream tasks on the actual kernels.
# Run on AC power with High Power mode; timings go through run_timed.sh.
# Downloads: Qwen/Qwen3.5-35B-A3B-Base (~72 GB), llama.cpp sources (tag b11706), the GGUF converter's Python
# requirements (PyTorch CPU etc., into a separate venv), lm-evaluation-harness (separate venv) and its task datasets.
# Usage: run_08.sh <python-with-mlx-lm> [cache-dir]
set -e
cd "${0:A:h}"
PY=$1; D=${2:-$HOME/.cache/mxfp4-int8-m5}; R=../results
Q35=Qwen/Qwen3.5-35B-A3B-Base
LC=$D/third_party/llama.cpp; G=$D/gguf
# ---- llama.cpp (pinned release; Metal backend, tensor API enabled on M5 by default)
if [[ ! -x $LC/build/bin/llama-bench ]]; then
  mkdir -p $D/third_party; git clone --depth 1 --branch b11706 https://github.com/ggml-org/llama.cpp.git $LC
  $PY -m venv $D/venv-tools; $D/venv-tools/bin/pip install cmake ninja
  (cd $LC && PATH=$D/venv-tools/bin:$PATH cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_METAL=ON \
    -DGGML_METAL_EMBED_LIBRARY=ON -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_SERVER=OFF &&
    PATH=$D/venv-tools/bin:$PATH cmake --build build --target llama-bench llama-quantize llama-perplexity -j 12)
fi
if [[ ! -x $D/venv-gguf/bin/python ]]; then
  $PY -m venv $D/venv-gguf
  $D/venv-gguf/bin/pip install -r $LC/requirements/requirements-convert_hf_to_gguf.txt -e $LC/gguf-py
fi
$PY -c "from huggingface_hub import snapshot_download as s; s('$Q35', allow_patterns=['*.safetensors','*.json','*.txt','*.jinja','tokenizer*'])"
mkdir -p $G
for spec in "Qwen3-1.7B-Base:Qwen/Qwen3-1.7B-Base:Q4_0" "OLMoE-1B-7B-0924:allenai/OLMoE-1B-7B-0924:Q4_0 MXFP4_MOE" \
            "Qwen3.5-35B-A3B-Base:$Q35:Q4_0 MXFP4_MOE"; do
  name=${spec%%:*}; rest=${spec#*:}; repo=${rest%%:*}; quants=${rest#*:}
  dir=$($PY -c "from huggingface_hub import snapshot_download as s; print(s('$repo'))")
  [[ -f $G/$name-bf16.gguf ]] || (cd $LC && $D/venv-gguf/bin/python convert_hf_to_gguf.py $dir --outfile $G/$name-bf16.gguf --outtype bf16)
  for q in ${=quants}; do [[ -f $G/$name-$q.gguf ]] || $LC/build/bin/llama-quantize $G/$name-bf16.gguf $G/$name-$q.gguf $q; done
done
# ---- Qwen3.5-35B-A3B: MLX baselines (routers at 8-bit affine, as mlx_lm's quant_predicate), packed SC-MXFP4
[[ -d $D/Qwen3.5-35B-A3B-Base-mxfp4 ]] || $PY convert_mlx.py $Q35 $D/Qwen3.5-35B-A3B-Base-mxfp4 mxfp4 32 '{}' router8
[[ -d $D/Qwen3.5-35B-A3B-Base-4bit ]] || $PY convert_mlx.py $Q35 $D/Qwen3.5-35B-A3B-Base-4bit affine 64 '{}' router8
# coverage-matched MLX MXFP4 (exactly the W4A8 matrices; embeddings, head, routers BF16)
for repo in Qwen/Qwen3-1.7B-Base $Q35; do
  out=$D/${repo#*/}-mxfp4-w4a8cov; [[ -d $out ]] || $PY convert_mlx.py $repo $out mxfp4 32 '{}' w4a8cov
done
# ---- correctness (the reference holds ~70 GB; run it alone)
$PY ref_qwen35.py --out $R/08_ref_qwen35.npz            # packs our weights on first use (cached)
$PY check_fused_qwen35.py --ref $R/08_ref_qwen35.npz --out $R/08_check_qwen35.json
$PY check_decode.py --model qwen35 --out $R/08_check_decode_qwen35.json
# ---- timing
./queue_08.sh $PY
./queue_08_decode.sh $PY $D       # decode of ours, MLX and llama.cpp in one session per model
# ---- accuracy: WikiText-2 perplexity table, weight statistics, W4A8 coverage
$PY ppl_qwen35.py --methods bf16,ours,mlx_mxfp4,mlx_affine4,sc_mxfp4_w4a8,sc_mxfp4_w4a16,nvfp4_w4a16,int4_affine_g64_w4a16,mlx_mxfp4_cov
$PY analyze_qwen35.py $R/08_qwen3.5-35b-a3b
$PY coverage_qwen35.py --out $R/08_coverage_qwen35.json
# ---- GPTQ-rounded SC-MXFP4 on the kernels (Qwen3-1.7B; calibration Hessians of sim/calib.py, CPU-heavy)
[[ -d $D/hessians/Qwen--Qwen3-1.7B-Base ]] || $PY ../sim/calib.py Qwen/Qwen3-1.7B-Base
$PY gptq_pack.py --out $R/08_gptq_pack_qwen3.json
$PY ppl_gptq_qwen3.py --out $R/08_gptq_ppl_qwen3.csv
# ---- zero-shot tasks (lm-evaluation-harness in its own venv with the same mlx / mlx-lm)
if [[ ! -x $D/venv-eval/bin/python ]]; then
  $PY -m venv $D/venv-eval
  $D/venv-eval/bin/pip install mlx==0.32.4 mlx-metal==0.32.4 mlx-lm==0.32.2 transformers==5.20.1 tokenizers==0.23.3 \
    huggingface_hub==2.2.0 numpy==2.5.4 pyarrow==26.0.0 safetensors==0.8.0 lm-eval
fi
for m in ours mlx_mxfp4 mlx_affine4 mlx_mxfp4_cov bf16 ours_gptq; do $D/venv-eval/bin/python eval_tasks.py --model qwen3 --method $m; done
for m in ours mlx_mxfp4 mlx_affine4 mlx_mxfp4_cov bf16; do
  $D/venv-eval/bin/python eval_tasks.py --model qwen35 --method $m --limit 1000
done
$PY make_08_tables.py $R
