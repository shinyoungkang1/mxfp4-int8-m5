"""Prefill trick: dequantize the 4-bit weights to FP16 once, then use the fast FP16 matmul."""
import mlx.core as mx
from mlx_bench import bench, K, N

w = mx.random.normal((N, K)).astype(mx.float16)
for M in (512, 1024, 4096):
    x = mx.random.normal((M, K)).astype(mx.float16)
    print(f"\nprefill {M} tok")
    for name, kw in [("int4 (affine)", dict(bits=4, group_size=64)),
                     ("mxfp4", dict(bits=4, group_size=32, mode="mxfp4"))]:
        q = mx.quantize(w, **kw)
        fused = bench(lambda: mx.quantized_matmul(x, *q, transpose=True, **kw), 20)
        deq = bench(lambda: x @ mx.dequantize(*q, **kw).T, 20)
        print(f"  {name:14s} quantized_matmul {fused*1e3:6.3f} ms | dequant+fp16 {deq*1e3:6.3f} ms  ({fused/deq:.2f}x)")
