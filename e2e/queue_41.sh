#!/bin/zsh
# Run 41: re-time the Qwen3-1.7B decode row of the paper's Table 3 (its run-08 session fails the anchor check:
# MLX 353 vs 368 tok/s). Ours, MLX and llama.cpp back to back under one GPU lock, as queue_08_decode.sh.
# The GGUFs of run 08 are gone, so Qwen3-1.7B is converted again (BF16, then Q4_0; about 4.5 GB in ~/.cache/mxfp4-int8-m5/gguf).
# Usage: caffeinate -i -s zsh e2e/queue_41.sh      (progress: results/41_queue.log)
source "${0:A:h}/lib_queue.sh"
queue_init 41
D=$HOME/.cache/mxfp4-int8-m5; LC=$D/third_party/llama.cpp; G=$D/gguf
mkdir -p $G
step gguf_qwen3 zsh -c "dir=\$($PY -c \"from huggingface_hub import snapshot_download as s; print(s('Qwen/Qwen3-1.7B-Base'))\") && \
  ( [[ -f $G/Qwen3-1.7B-Base-bf16.gguf ]] || (cd $LC && $D/venv-gguf/bin/python convert_hf_to_gguf.py \$dir --outfile $G/Qwen3-1.7B-Base-bf16.gguf --outtype bf16) ) && \
  ( [[ -f $G/Qwen3-1.7B-Base-Q4_0.gguf ]] || $LC/build/bin/llama-quantize $G/Qwen3-1.7B-Base-bf16.gguf $G/Qwen3-1.7B-Base-Q4_0.gguf Q4_0 )" || exit 1
timed dec_qwen3 zsh -c "$PY bench_decode.py --model qwen3 --cache-dir $D --ctx 512,2048 --gen 128 --methods mlx_mxfp4,mlx_affine4,ours --out $R/41_decode_session_qwen3.csv && \
  $PY bench_llamacpp.py --model qwen3 --decode-only --out $R/41_decode_session_llamacpp_qwen3.csv"
_q_log "queue 41 done"
