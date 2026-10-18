"""Time prefill-shaped (many tokens) and decode-shaped (1 token) matmuls in MLX
for FP16 weights vs 8-bit, 4-bit, and MXFP4 quantized weights."""
import time
import mlx.core as mx

K, N = 4096, 4096


def bench(fn, iters=50):
    for _ in range(5):
        mx.eval(fn())
    t = time.perf_counter()
    for _ in range(iters):
        mx.eval(fn())
    return (time.perf_counter() - t) / iters


w = mx.random.normal((N, K)).astype(mx.float16)
variants = {"fp16": None}
for name, kw in [("int8 (affine)", dict(bits=8, group_size=64)),
                 ("int4 (affine)", dict(bits=4, group_size=64)),
                 ("mxfp4", dict(bits=4, group_size=32, mode="mxfp4"))]:
    try:
        variants[name] = (mx.quantize(w, **kw), kw)
    except Exception as e:
        print(f"{name}: unsupported in this MLX ({type(e).__name__}: {e})")

for M, label in [(4096, "prefill 4096 tok"), (1, "decode 1 tok")]:
    x = mx.random.normal((M, K)).astype(mx.float16)
    print(f"\n{label}")
    for name, v in variants.items():
        if v is None:
            f = lambda: x @ w.T
        else:
            q, kw = v
            f = lambda q=q, kw=kw: mx.quantized_matmul(x, *q, transpose=True, **kw)
        s = bench(f, iters=20 if M > 1 else 200)
        tflops = 2 * M * K * N / s / 1e12
        print(f"  {name:14s} {s*1e3:8.3f} ms   {tflops:6.2f} TFLOPS")
