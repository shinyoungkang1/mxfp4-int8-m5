# Integer-Exact Microscaling: MXFP4 on INT8 matrix units

Code for the paper *Integer-Exact Microscaling: Running MXFP4 Language Models on INT8 Matrix Units* (Shinyoung Kang;
preprint, link to follow).

On hardware whose fastest matrix datapath is INT8×INT8, such as the Neural Accelerators of Apple's M5 GPUs, a row of
MXFP4 weights is exactly one INT8 vector times one power-of-two scale if and only if its block exponents span at most
three. This repository contains the spread-constrained quantizer (SC-MXFP4), the integer-exact FP4 format family, the
W4A8 kernels for the M5 inside MLX (dense and grouped-MoE prefill, batch-1 decode), and the benchmark and evaluation
scripts behind the paper's results.

## Requirements

- A Mac with an Apple M5-family chip (the paper's numbers are from an M5 Max with 128 GB), macOS 26 with Metal 4, and
  the Xcode command-line tools (`swiftc`, `clang`).
- Python 3.12 and MLX 0.32. `./setup.sh` creates `.venv` with MLX, mlx-lm, NumPy, PyArrow and Matplotlib.
- Qwen3.8-Flash-Next runs on mlx-lm pull request 1788 (commit `2097324`), in a separate venv `.venv-q38`:
  ```bash
  git clone https://github.com/ml-explore/mlx-lm ~/src/mlx-lm-pr1788
  git -C ~/src/mlx-lm-pr1788 fetch origin pull/1788/head && git -C ~/src/mlx-lm-pr1788 checkout 2097324
  uv venv --python 3.12 .venv-q38
  uv pip install --python .venv-q38/bin/python mlx==0.32.4 -e ~/src/mlx-lm-pr1788 numpy pyarrow huggingface_hub safetensors
  ```
- Models and datasets are downloaded from Hugging Face on first use; packed weights are cached in
  `~/.cache/mxfp4-int8-m5`. Qwen3.5-35B-A3B takes about 72 GB of disk, the Qwen3.8-Flash-Next FP8 checkpoint about
  173 GiB.

## Layout

| Folder | Contents |
| --- | --- |
| `kernels/` | Metal 4 kernels on the Neural Accelerators (`matmul.metal`: INT8 / INT4 / FP16 `matmul2d`; `w4a8.metal`, `w4a8_packed.metal`: the SC-MXFP4 W4A8 linear layer), their Swift harnesses, and a CPU SME benchmark (`cpu_sme_bench.c`) |
| `sim/` | Formats and quantizers (`mx_int8.py`: SC-MXFP4 and its exact INT8 embedding; `formats.py`: the integer-exact format family), fake-quantized perplexity (`eval_ppl.py`), GPTQ, exponent-spread statistics |
| `e2e/` | The W4A8 system in MLX: fused prefill kernels (`sc_kernels.py`), grouped MoE GEMM (`sc_moe.py`), decode GEMV (`sc_decode.py`), model integrations (`fused_qwen3.py`, `fused_olmoe.py`, `fused_qwen35.py`, `fused_qwen38.py`), correctness checks, benchmarks and evaluations |
| `bench/` | MLX baselines |
| `tools/` | `gpu_timing.sh`: a lock that gives timing runs the GPU to themselves |

## Self-tests

Synthetic data only, a few seconds each:

```bash
cd sim && ../.venv/bin/python mx_int8.py && ../.venv/bin/python formats.py && cd ..
cd e2e && for t in sc_format.py w4a8_mlx.py test_sc_kernels.py sc_moe.py sc_decode.py test_formats.py; do ../.venv/bin/python $t; done
```

## Reproducing the results

Timing runs go through `e2e/run_timed.sh`: on AC power in High Power mode, with an idle GPU before and after each run
(disturbed runs are rejected and retried) and the lock of `tools/gpu_timing.sh`. Set `AVOID` to a `pgrep -f` pattern
to wait until other GPU-heavy applications have quit. Results are written to `results/`. Run heavy jobs one at a
time: the Qwen3.5-35B-A3B reference and every Qwen3.8-Flash-Next job need most of 128 GB.

| Result | Script |
| --- | --- |
| Neural Accelerator throughput by operand type | `kernels/run_microbench.sh` |
| Linear-layer latency against MLX | `kernels/run_layerbench.sh <python> <scratch-dir>` |
| CPU SME: INT8 vs FP16 outer products | `kernels/cpu_sme_bench.c` (build command in its header) |
| Formats ranked by layer-output error | `sim/format_sweep.py` (calibration statistics: `sim/calib.py`) |
| WikiText-2 perplexity on the small models | `sim/run_ppl.sh <python>`, `sim/run_formats_ppl.sh <python>` |
| Exponent-spread statistics | `sim/analyze_weights.py`, `sim/spread_analysis.py` |
| End-to-end prefill, Qwen3-1.7B | `e2e/run_e2e.sh <python>` |
| Fused dense and MoE prefill, decode (Qwen3-1.7B, OLMoE-1B-7B) | `e2e/run_07.sh <python>` (after `run_e2e.sh`) |
| llama.cpp baselines, Qwen3.5-35B-A3B, zero-shot tasks | `e2e/run_08.sh <python>` |
| The integer-exact 16-block format on the kernels | `e2e/queue_format.sh <python>` |
| Qwen3.8-Flash-Next: packing, kernel check, perplexity, spread | `e2e/run_10.sh .venv-q38/bin/python` |
| Qwen3.8-Flash-Next: prefill and decode speed | `e2e/run_q38_prefill.sh .venv-q38/bin/python` |
| Qwen3.8-Flash-Next: decode profile | `e2e/tune_decode_q38.py`, `e2e/bench_gemv_q38.py` |
| Qwen3.8-Flash-Next: the 16-block format on the MoE kernels (pack, check, speed, GSM8K), GSM8K of MLX on our SC-MXFP4 weights, KL and perplexity at equal activation precision, both clock regimes for the smaller MoEs | `e2e/run_11.sh` (one job at a time, about 15 h, restartable) |
| Qwen3.8-Flash-Next: DeltaNet reference recurrence vs. Metal kernel, maximum spread and MSE, MLX on our SC-MXFP4 bytes and MLX affine INT4 end to end, 16-block decode profile; NVFP4 W4A8 with scale search (OLMoE, Qwen3.5); Table 1 searches; linear layers timed alike (`bench/layer_wallclock.py`); llama.cpp Q4_K_M and IQ4_XS perplexity and speed (`e2e/ppl_llamacpp.py`) | `e2e/run_12.sh` (one job at a time, about 6-7 h, restartable) |
| Qwen3.8-Flash-Next decode re-timed after the E8M0 scale-layout fix (ours, MLX on our SC-MXFP4 bytes, MLX MXFP4; dense GEMV comparison) | `e2e/run_13.sh` (about 30-50 min) |
| WikiText-2 perplexity of our formats under llama.cpp's protocol (`--second-half` in `sim/eval_ppl.py` and `e2e/ppl_qwen35.py`: disjoint 2048-token windows, second half scored), next to llama.cpp's types from `e2e/ppl_llamacpp.py` | `e2e/run_14.sh` (one job at a time, about 1 h, restartable) |
| Kernel comparison with oMLX | `e2e/bench_omlx_q38.py` (in a venv with oMLX 0.7.1 and its M5 kernels, under `run_timed.sh`) |

Qwen3.8-Flash-Next accuracy beyond `run_10.sh`, from `e2e/` (`run_11.sh` adds the remaining configurations):

```bash
PY=../.venv-q38/bin/python; B16="w=fmt:e2m1/B16/M1..10"
# routed experts only, full WikiText-2 test set
$PY q38_ppl.py --methods "$B16;a=8;s=experts,w=sc_mx;a=8;s=experts,w=nvfp4;a=8;s=experts,w=nvfp4_search;a=8;s=experts,w=int4_g64;a=16;s=experts,w=mx_spec;a=16;s=experts" --out ../results/10_ppl_qwen38.csv
# KL divergence to the FP8 model and top-1 flips, 32 windows
$PY q38_ppl.py --windows 32 --methods "fp8,$B16;a=8;s=experts,w=sc_mx;a=8;s=experts,w=nvfp4_search;a=8;s=experts,w=int4_g64;a=16;s=experts,w=mx_spec;a=16;s=experts,$B16;a=8,sc_mxfp4_w4a8" --out ../results/10_kl_qwen38.csv
# GSM8K, all 1319 test problems, every matrix quantized: ours (W4A8) with SC-MXFP4 and with the 16-block format,
# MLX MXFP4 (W4A16) on our SC-MXFP4 weights and with MLX's own quantization of the same matrices
$PY gsm8k_q38.py --system fused --n 1319 --out ../results/10_gsm8k_fused.jsonl
$PY gsm8k_q38.py --system fused_b16 --n 1319 --out ../results/11_gsm8k_fused_b16.jsonl
$PY gsm8k_q38.py --system mlx_sc_mxfp4 --n 1319 --out ../results/11_gsm8k_mlx_sc_mxfp4.jsonl
$PY gsm8k_q38.py --system mlx_mxfp4 --n 1319 --out ../results/10_gsm8k_mlx_mxfp4.jsonl
```

## License

MIT, see `LICENSE`.
