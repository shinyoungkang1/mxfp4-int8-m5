"""Linear-layer latency with one timing method for every kernel: MLX's FP16 matmul and MXFP4 quantized_matmul next to
our W4A8 two-pass path (per-token INT8 activations, unpack to INT8, INT8 GEMM with the scaling in registers;
e2e/sc_kernels.py on SC-MXFP4 weights), all called from MLX and timed alike: wall clock, 1 s warm-up, 15 interleaved
repetitions of 10 evaluated calls, median (the method of bench/mlx_layer_baseline.py). Shapes of
kernels/run_layerbench.sh. Run under e2e/run_timed.sh. Prints CSV.
"""
import os
import sys
import time

import mlx.core as mx
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "e2e"))
sys.path.insert(0, os.path.join(HERE, "..", "sim"))
import sc_format as F  # noqa: E402
import sc_kernels as S  # noqa: E402

SHAPES = [(512, 4096, 4096), (1024, 4096, 4096), (2048, 4096, 4096), (4096, 4096, 4096),
          (2048, 14336, 4096), (2048, 4096, 14336)]


def main():
    print("M,N,K,method,ms_median,speedup_vs_mlx_mxfp4")
    for M, N, K in SHAPES:
        rng = np.random.default_rng(0)
        w = (rng.standard_normal((N, K)) * 0.02).astype(np.float32)
        x = mx.array(rng.standard_normal((M, K)).astype(np.float16))
        codes, bc, ws, _ = F.quantize(w)
        codes, bc, ws = mx.array(codes), mx.array(bc), mx.array(ws)
        wf = mx.array(w.astype(np.float16))
        qmx = mx.quantize(wf, bits=4, group_size=32, mode="mxfp4")
        mx.eval(x, wf, qmx, codes, bc, ws)

        def ours():
            xq, xs = S.act_quant(x)
            return S.mm_epi(xq, S.unpack_i8(codes, bc, N, K), xs, ws, 1, mx.float16)

        fns = {"mlx_fp16": lambda: x @ wf.T,
               "mlx_mxfp4": lambda: mx.quantized_matmul(x, *qmx, transpose=True, bits=4, group_size=32, mode="mxfp4"),
               "w4a8_ours": ours}
        t0 = time.perf_counter()
        while time.perf_counter() - t0 < 1.0:
            for f in fns.values():
                mx.eval(f())
        times = {k: [] for k in fns}
        for _ in range(15):  # interleaved
            for k, f in fns.items():
                t = time.perf_counter()
                mx.eval([f() for _ in range(10)])
                times[k].append((time.perf_counter() - t) / 10)
        med = {k: sorted(v)[len(v) // 2] for k, v in times.items()}
        for k in fns:
            print(f"{M},{N},{K},{k},{med[k] * 1e3:.4f},{med['mlx_mxfp4'] / med[k]:.3f}", flush=True)


if __name__ == "__main__":
    main()
