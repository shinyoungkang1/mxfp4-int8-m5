"""MLX baselines for the layer shapes in kernels/run_layerbench.sh: FP16 matmul and MLX's quantized
matmuls (4-bit affine g64 and MXFP4), i.e. the dequantize-in-kernel W4A16 path. Sustained timing: 1 s
warm-up, 15 repetitions of 10 evaluated calls, median. Prints CSV."""
import time
import mlx.core as mx

SHAPES = [(512, 4096, 4096), (1024, 4096, 4096), (2048, 4096, 4096), (4096, 4096, 4096),
          (2048, 14336, 4096), (2048, 4096, 14336)]
print("M,N,K,method,ms_median,tflops_median,speedup_vs_fp16")
for M, N, K in SHAPES:
    x = mx.random.normal((M, K)).astype(mx.float16)
    w = (mx.random.normal((N, K)) * 0.02).astype(mx.float16)
    q4 = mx.quantize(w, bits=4, group_size=64)
    qmx = mx.quantize(w, bits=4, group_size=32, mode="mxfp4")
    mx.eval(x, w, q4, qmx)
    fns = {
        "mlx_fp16": lambda: x @ w.T,
        "mlx_int4_affine_g64": lambda: mx.quantized_matmul(x, *q4, transpose=True, bits=4, group_size=64),
        "mlx_mxfp4": lambda: mx.quantized_matmul(x, *qmx, transpose=True, bits=4, group_size=32, mode="mxfp4"),
    }
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < 1.0:
        for f in fns.values():
            mx.eval(f())
    times = {k: [] for k in fns}
    for _ in range(15):  # interleaved, like the Metal harness
        for k, f in fns.items():
            t = time.perf_counter()
            mx.eval([f() for _ in range(10)])
            times[k].append((time.perf_counter() - t) / 10)
    med = {k: sorted(v)[len(v) // 2] for k, v in times.items()}
    for k in fns:
        print(f"{M},{N},{K},{k},{med[k] * 1e3:.4f},{2 * M * N * K / med[k] / 1e12:.2f},{med['mlx_fp16'] / med[k]:.3f}")
