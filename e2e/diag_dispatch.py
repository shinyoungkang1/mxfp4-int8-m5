"""Is Qwen3.5-35B-A3B decode bound by the CPU cost of encoding kernels rather than by the GPU?

1. per decode step (full model, 512-token context): wall time and process CPU time, ours vs MLX MXFP4. A step whose
   CPU time is close to its wall time is limited by graph construction + command encoding on the CPU;
2. per-op cost of a chain of 2000 dependent tiny ops (so that GPU time is negligible): a built-in MLX op, a minimal
   mx.fast.metal_kernel, and our decode kernels (act_quant, GEMV) at tiny sizes; measured as graph-build time and
   eval time per op.
Usage: python diag_dispatch.py [--out results/08_diag_dispatch.txt]   (run through run_timed.sh)
"""
import os
import sys
import time

import mlx.core as mx
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import sc_decode as G  # noqa: E402
import sc_kernels as S  # noqa: E402

CACHE = os.path.expanduser("~/.cache/mxfp4-int8-m5")


def chain_cost(make_op, x0, n=2000, reps=5):
    """-> (build us/op, eval us/op), medians: n dependent applications of make_op on x0."""
    builds, evals = [], []
    for _ in range(reps + 1):
        mx.synchronize()
        t0 = time.perf_counter()
        x = x0
        for _ in range(n):
            x = make_op(x)
        t1 = time.perf_counter()
        mx.eval(x)
        mx.synchronize()
        t2 = time.perf_counter()
        builds.append((t1 - t0) / n * 1e6)
        evals.append((t2 - t1) / n * 1e6)
    return float(np.median(builds[1:])), float(np.median(evals[1:]))


def step_cpu(meth_step, prefill_cache, n=64, reps=5):
    walls, cpus, mains, builds = [], [], [], []
    for _ in range(reps + 1):
        cache, y = prefill_cache()
        mx.synchronize()
        tb = 0.0
        w0, c0, m0 = time.perf_counter(), time.process_time(), time.thread_time()
        nxt = mx.argmax(meth_step(y, cache), -1).astype(mx.int32).reshape(1, 1)
        mx.async_eval(nxt)
        for _ in range(n - 1):
            cur = nxt
            b0 = time.perf_counter()
            nxt = mx.argmax(meth_step(cur, cache), -1).astype(mx.int32).reshape(1, 1)
            tb += time.perf_counter() - b0
            mx.async_eval(nxt)
            cur.item()
        nxt.item()
        walls.append((time.perf_counter() - w0) / n * 1e3)
        cpus.append((time.process_time() - c0) / n * 1e3)
        mains.append((time.thread_time() - m0) / n * 1e3)
        builds.append(tb / (n - 1) * 1e3)
    return tuple(float(np.median(v[1:])) for v in (walls, cpus, mains, builds))


def main():
    lines = []

    def out(s):
        print(s, flush=True)
        lines.append(s)

    # ---- 2. per-op dispatch cost
    tiny = mx.zeros((1, 128), mx.bfloat16)
    mx.eval(tiny)
    k_min = mx.fast.metal_kernel(name="diag_copy", input_names=["a"], output_names=["b"],
                                 source="uint i = thread_position_in_grid.x; b[i] = a[i] + static_cast<T>(1);")

    def custom(x):
        return k_min(inputs=[x], template=[("T", mx.bfloat16)], grid=(128, 1, 1), threadgroup=(128, 1, 1),
                     output_shapes=[x.shape], output_dtypes=[x.dtype])[0]

    K, N = 512, 64
    rng = np.random.default_rng(0)
    import sc_format as F
    codes, bc, ws, _ = F.quantize(rng.standard_normal((N, K)) * 0.02)
    codes, bc, ws = mx.array(codes), mx.array(bc), mx.array(ws)
    xs = mx.array([0.01], mx.float32)
    xin = mx.zeros((1, K), mx.bfloat16)
    mx.eval(codes, bc, ws, xs, xin)

    def ours_actquant(x):  # [1, K] bf16 -> int8 -> back to bf16 so that the chain continues
        q, s = S.act_quant(x, 1)
        return q.astype(mx.bfloat16).reshape(1, K)

    def ours_gemv(x):  # one W4A8 GEMV [1, K] -> [N]; pad back to [1, K]
        q, s = S.act_quant(x, 1)
        y = G.gemv(q, codes, bc, s, ws, N, 1)
        return mx.tile(y, (K // N,)).reshape(1, K)

    for name, op, x0 in () if os.environ.get("SKIP_CHAIN") else (("built-in add", lambda x: x + 1, tiny), ("built-in matmul [1,K]x[K,K]",
                         lambda x, w=mx.zeros((128, 128), mx.bfloat16): x @ w, tiny),
                         ("minimal metal_kernel", custom, tiny), ("act_quant (+astype, reshape)", ours_actquant, xin),
                         ("act_quant + gemv (+tile)", ours_gemv, xin)):
        b, e = chain_cost(op, x0)
        out(f"per-op chain  {name:32s} build {b:6.2f} us/op   eval {e:6.2f} us/op")

    # ---- 1. full decode step: wall vs process CPU time
    import decode as dec
    import fused_qwen35 as fz
    from mlx_lm import load
    from mlx_lm.models.cache import make_prompt_cache
    mxm = load(os.path.join(CACHE, fz.REPO.split("/")[-1] + "-mxfp4"))[0]
    f, _ = fz.load_fused_qwen35()
    f.two_pass = "auto"
    d = dec.DecodeQwen35(f, head=mxm)
    prompt = mx.array(rng.integers(0, 248320, (1, 512)).astype(np.int32))

    def prefill_cache():
        cache = make_prompt_cache(mxm)
        y = mx.argmax(mxm(prompt, cache=cache)[:, -1:], -1).astype(mx.int32)
        mx.eval(y, [c.state for c in cache])
        return cache, y

    for name, st in (("mlx_mxfp4", lambda t, c: mxm(t, cache=c)), ("ours", d.step)):
        step_cpu(st, prefill_cache, 8, 1)
    for name, st in (("mlx_mxfp4", lambda t, c: mxm(t, cache=c)), ("ours", d.step),
                     ("mlx_mxfp4", lambda t, c: mxm(t, cache=c)), ("ours", d.step)):
        w, c, m, b = step_cpu(st, prefill_cache)
        out(f"decode step   {name:12s} wall {w:6.3f} ms/token   process CPU {c:6.3f} ({100 * c / w:.0f}%)   "
            f"main-thread CPU {m:6.3f} ({100 * m / w:.0f}%)   of which graph build {b:6.3f}")
    import bench_prefill as bp
    out("# " + bp.power_state())
    if len(sys.argv) > 2 and sys.argv[1] == "--out":
        open(sys.argv[2], "w").write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
