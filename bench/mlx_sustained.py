"""MLX FP16 and quantized matmul throughput, measured with and without a sustained warm-up."""
import sys, time
import mlx.core as mx

M = N = K = 4096
x = mx.random.normal((M, K)).astype(mx.float16)
w = mx.random.normal((N, K)).astype(mx.float16)
q4 = mx.quantize(w, bits=4, group_size=64)
mxf = mx.quantize(w, bits=4, group_size=32, mode="mxfp4")
fns = {
    "fp16": lambda: x @ w.T,
    "int4 affine": lambda: mx.quantized_matmul(x, *q4, transpose=True, bits=4, group_size=64),
    "mxfp4": lambda: mx.quantized_matmul(x, *mxf, transpose=True, bits=4, group_size=32, mode="mxfp4"),
}
warm = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
t0 = time.perf_counter()
while time.perf_counter() - t0 < warm:
    for f in fns.values():
        mx.eval(f())
for name, f in fns.items():
    ts = []
    for _ in range(9):
        t = time.perf_counter()
        mx.eval([f() for _ in range(10)])  # build all 10 so none are skipped
        ts.append((time.perf_counter() - t) / 10)
    ts.sort()
    print(f"warm={warm:<5} {name:12s} median {2*M*N*K/ts[4]/1e12:6.1f} TFLOPS")
