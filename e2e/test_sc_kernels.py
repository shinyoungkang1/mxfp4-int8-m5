"""Unit checks of e2e/sc_kernels.py against numpy/MLX references (run: python test_sc_kernels.py).

1. unpack_i8 equals sc_format.decode_numpy and the reference quantizer's INT8 operand.
2. INT8 GEMM exact vs a float64 integer GEMM.
3. rmsnorm_quant / res_norm_quant: the normalized row equals MLX's rms_norm bit for bit (checked through the INT8
   codes and scales computed from MLX's output with the kernel's quantization formula).
4. sc_swiglu equals mlx_lm's swiglu for every BF16 gate value (exhaustive) against random up values.
5. qkv_epilogue_headnorm equals MLX's q_norm/k_norm on the BF16 GEMM outputs.
"""
import os
import sys

import mlx.core as mx
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sc_format as F  # noqa: E402
import sc_kernels as S  # noqa: E402


def quant_ref(y):
    """The kernels' per-token INT8 of an f32 [M, K] array, in float32 numpy (s = amax / 127, codes = rint(y / s * ...))."""
    y = np.asarray(y, np.float32)
    s = (np.abs(y).max(1) / np.float32(127)).astype(np.float32)
    qi = np.where(s > 0, np.float32(1) / np.where(s > 0, s, 1), 0).astype(np.float32)
    return np.clip(np.rint(y * qi[:, None]), -127, 127).astype(np.int8), s


def main():
    rng = np.random.default_rng(0)
    ok = True

    # 1. unpack
    N, K = 384, 2048
    w = rng.standard_normal((N, K)) * 0.02 * rng.lognormal(0, 0.3, (1, K))
    codes, bc, ws, qw = F.quantize(w)
    wq = np.array(S.unpack_i8(mx.array(codes), mx.array(bc), N, K))
    d1 = int((wq != qw).sum())
    d2 = int((wq != F.decode_numpy(codes, bc, N, K)).sum())
    print(f"1. unpack: {d1} mismatches vs reference quantizer, {d2} vs decode_numpy; "
          f"{F.DEFAULT.bits_per_weight(K):.3f} bits/weight")
    ok &= d1 == 0 and d2 == 0

    # 2. GEMM
    M = 256
    xq = mx.array(rng.integers(-127, 128, (M, K)).astype(np.int8))
    acc = np.array(S.mm_i8_i32(xq, mx.array(wq))).astype(np.float64)
    ref = np.array(xq).astype(np.float64) @ qw.astype(np.float64).T
    e = np.abs(acc - ref).max()
    print(f"2. INT8 GEMM {M}x{N}x{K}: max abs err {e}")
    ok &= e == 0

    # 3. rmsnorm_quant, res_norm_quant (M not a multiple of the tile: padded rows must be zero)
    M, K = 200, 2048
    x = mx.array((rng.standard_normal((M, K)) * rng.lognormal(0, 1, (1, K))).astype(np.float32)).astype(mx.bfloat16)
    nw = mx.array(rng.lognormal(0, 0.3, K).astype(np.float32)).astype(mx.bfloat16)
    eps = mx.array([1e-6], mx.float32)
    xq, xs = S.rmsnorm_quant(x, nw, eps)
    y = mx.fast.rms_norm(x, nw, 1e-6).astype(mx.float32)
    cq, cs = quant_ref(np.array(y))
    nd = int((np.array(xq)[:M] != cq).sum())
    sd = int((np.array(xs)[:M] != cs).sum())
    pad = int(np.abs(np.array(xq)[M:]).sum() + np.abs(np.array(xs)[M:]).sum())
    print(f"3a. rmsnorm_quant: codes differing from MLX rms_norm + quant: {nd}/{M * K}, scales differing {sd}, "
          f"padding nonzero {pad}")
    ok &= nd == 0 and sd == 0 and pad == 0

    Mp = S.pad_rows(M)
    accr = mx.array(rng.integers(-2 ** 20, 2 ** 20, (Mp, K)).astype(np.int32))
    xsc = mx.array(np.abs(rng.standard_normal(Mp)).astype(np.float32) * 1e-3)
    wsc = mx.array(np.abs(rng.standard_normal(K)).astype(np.float32) * 1e-3)
    res = mx.pad(x, [(0, Mp - M), (0, 0)])
    h, xq2, xs2 = S.res_norm_quant(accr, xsc, wsc, res, nw, eps)
    lin = (accr.astype(mx.float32) * xsc[:, None] * wsc[None, :]).astype(mx.bfloat16)
    h_ref = res + lin
    hd = int(mx.sum(h != h_ref).item())
    cq, cs = quant_ref(np.array(mx.fast.rms_norm(h_ref, nw, 1e-6).astype(mx.float32)))
    nd = int((np.array(xq2) != cq).sum())
    print(f"3b. res_norm_quant: residual differing from MLX bf16 add {hd}/{Mp * K}, codes differing {nd}")
    ok &= hd == 0 and nd == 0

    # 4. swiglu, exhaustive over BF16 gate values
    from mlx_lm.models.activations import swiglu
    g = mx.array(np.arange(65536, dtype=np.uint32).astype(np.uint16)).view(mx.bfloat16)
    g = mx.where(mx.isnan(g) | mx.isinf(g), mx.array(0, mx.bfloat16), g)
    u = mx.array(rng.standard_normal(65536).astype(np.float32) * 3).astype(mx.bfloat16)
    k = mx.fast.metal_kernel(name="t_swiglu", input_names=["g", "u"], output_names=["o"],
                             source="uint i = thread_position_in_grid.x; o[i] = sc_swiglu(g[i], u[i]);",
                             header=S.HEADER + F.DEFAULT.metal_header() + S._SWIGLU_HELPER)
    o = k(inputs=[g, u], grid=(65536, 1, 1), threadgroup=(256, 1, 1), output_shapes=[(65536,)],
          output_dtypes=[mx.bfloat16])[0]
    o_ref = swiglu(g, u)
    nd = int(mx.sum((o != o_ref) & ~(mx.isnan(o) & mx.isnan(o_ref))).item())
    print(f"4. swiglu: {nd}/65536 BF16 outputs differ from mlx_lm swiglu")
    ok &= nd == 0

    # 5. qkv epilogue (Qwen3-1.7B shapes)
    hq, hk, d, L = 16, 8, 128, 200
    Ntot = (hq + 2 * hk) * d
    accq = mx.array(rng.integers(-2 ** 18, 2 ** 18, (Mp, Ntot)).astype(np.int32))
    wsq = mx.array(np.abs(rng.standard_normal(Ntot)).astype(np.float32) * 1e-4)
    qn = mx.array(rng.lognormal(0, 0.3, d).astype(np.float32)).astype(mx.bfloat16)
    kn = mx.array(rng.lognormal(0, 0.3, d).astype(np.float32)).astype(mx.bfloat16)
    q, kk, v = S.qkv_epilogue_headnorm(accq, xsc, wsq, qn, kn, eps, L, hq, hk, d, mx.bfloat16)
    lin = (accq.astype(mx.float32) * xsc[:, None] * wsq[None, :]).astype(mx.bfloat16)[:L]
    qr = mx.fast.rms_norm(lin[:, :hq * d].reshape(L, hq, d), qn, 1e-6).transpose(1, 0, 2)
    kr = mx.fast.rms_norm(lin[:, hq * d:(hq + hk) * d].reshape(L, hk, d), kn, 1e-6).transpose(1, 0, 2)
    vr = lin[:, (hq + hk) * d:].reshape(L, hk, d).transpose(1, 0, 2)
    nd = [int(mx.sum(a != b).item()) for a, b in ((q, qr), (kk, kr), (v, vr))]
    print(f"5. qkv_epilogue_headnorm: q/k/v elements differing from MLX: {nd}")
    ok &= sum(nd) == 0

    ok &= test_epilogues()
    ok &= test_moe_pieces()
    print("ALL PASSED" if ok else "FAILURES")
    return ok




def test_epilogues():
    """mm_epi (EPI 1/2/3) and qkv_headnorm_y against the INT32 GEMM + MLX ops, bit for bit, for the tiles used."""
    from mlx_lm.models.activations import swiglu
    rng = np.random.default_rng(1)
    ok = True
    for M, N, K, tile in ((256, 512, 1024, (128, 64, 4, 8)), (256, 512, 1024, (64, 128, 4, 8)),
                          (128, 256, 2048, (64, 64, 4, 8)), (128, 256, 2048, (64, 64, 2, 8))):
        xq = mx.array(rng.integers(-127, 128, (M, K)).astype(np.int8))
        wq = mx.array(rng.integers(-96, 97, (N, K)).astype(np.int8))
        xs = mx.array(np.abs(rng.standard_normal(M)).astype(np.float32) * 1e-3)
        ws = mx.array(np.abs(rng.standard_normal(N)).astype(np.float32) * 1e-4)
        res = mx.array(rng.standard_normal((M, N)).astype(np.float32)).astype(mx.bfloat16)
        acc = S.mm_i8_i32(xq, wq)
        y_ref = (acc.astype(mx.float32) * xs[:, None] * ws[None, :]).astype(mx.bfloat16)
        y1 = S.mm_epi(xq, wq, xs, ws, 1, mx.bfloat16, tile=tile)
        y2 = S.mm_epi(xq, wq, xs, ws, 2, mx.bfloat16, res=res, tile=tile)
        perm = mx.array(S.interleave_gate_up(N // 2))
        y3 = S.mm_epi(xq, wq[perm], xs, ws[perm], 3, mx.bfloat16, tile=tile)
        a_ref = swiglu(y_ref[:, :N // 2], y_ref[:, N // 2:])
        nd = [int(mx.sum(a != b).item()) for a, b in ((y1, y_ref), (y2, res + y_ref), (y3, a_ref))]
        print(f"6. mm_epi {M}x{N}x{K} tile {tile}: elements differing (EPI 1, 2, 3): {nd}")
        ok &= sum(nd) == 0
    hq, hk, d, L = 16, 8, 128, 200
    Ntot = (hq + 2 * hk) * d
    y = mx.array(rng.standard_normal((256, Ntot)).astype(np.float32)).astype(mx.bfloat16)
    qn = mx.array(rng.lognormal(0, 0.3, d).astype(np.float32)).astype(mx.bfloat16)
    kn = mx.array(rng.lognormal(0, 0.3, d).astype(np.float32)).astype(mx.bfloat16)
    eps = mx.array([1e-6], mx.float32)
    q, kk, v = S.qkv_headnorm_y(y, qn, kn, eps, L, hq, hk, d)
    yl = y[:L]
    qr = mx.fast.rms_norm(yl[:, :hq * d].reshape(L, hq, d), qn, 1e-6).transpose(1, 0, 2)
    kr = mx.fast.rms_norm(yl[:, hq * d:(hq + hk) * d].reshape(L, hk, d), kn, 1e-6).transpose(1, 0, 2)
    vr = yl[:, (hq + hk) * d:].reshape(L, hk, d).transpose(1, 0, 2)
    nd = [int(mx.sum(a != b).item()) for a, b in ((q, qr), (kk, kr), (v, vr))]
    print(f"7. qkv_headnorm_y: q/k/v elements differing from MLX: {nd}")
    ok &= sum(nd) == 0
    return ok




def test_moe_pieces():
    """OLMoE pieces: rmsnorm_quant_n, qkv_fullnorm_y and moe_combine against MLX ops, bit for bit."""
    import fused_olmoe as fo
    rng = np.random.default_rng(2)
    ok = True
    M, K, k = 200, 2048, 8
    Mp = S.pad_rows(M)
    eps = mx.array([1e-5], mx.float32)
    x = mx.array((rng.standard_normal((M, K)) * 2).astype(np.float32)).astype(mx.bfloat16)
    nw = mx.array(rng.lognormal(0, 0.3, K).astype(np.float32)).astype(mx.bfloat16)
    xq, xs, xn = S.rmsnorm_quant_n(x, nw, eps)
    nd = int(mx.sum(xn[:M] != mx.fast.rms_norm(x, nw, 1e-5)).item()) + int(mx.sum(xn[M:] != 0).item())
    xq0, _ = S.rmsnorm_quant(x, nw, eps)
    nd += int(mx.sum(xq != xq0).item())
    print(f"8. rmsnorm_quant_n: normalized rows / codes differing: {nd}")
    ok &= nd == 0
    H, d, L = 16, 128, M
    y = mx.array(rng.standard_normal((Mp, 3 * H * d)).astype(np.float32)).astype(mx.bfloat16)
    qn = mx.array(rng.lognormal(0, 0.3, H * d).astype(np.float32)).astype(mx.bfloat16)
    kn = mx.array(rng.lognormal(0, 0.3, H * d).astype(np.float32)).astype(mx.bfloat16)
    q, kk, v = S.qkv_fullnorm_y(y, qn, kn, eps, L, H, d)
    W = H * d
    qr = mx.fast.rms_norm(y[:L, :W], qn, 1e-5).reshape(L, H, d).transpose(1, 0, 2)
    kr = mx.fast.rms_norm(y[:L, W:2 * W], kn, 1e-5).reshape(L, H, d).transpose(1, 0, 2)
    vr = y[:L, 2 * W:].reshape(L, H, d).transpose(1, 0, 2)
    nd = [int(mx.sum(a != b).item()) for a, b in ((q, qr), (kk, kr), (v, vr))]
    print(f"9. qkv_fullnorm_y: q/k/v elements differing from MLX: {nd}")
    ok &= sum(nd) == 0
    # combine: MLX computes (y * scores[..., None]).sum(-2) on the unsorted [M, k, D] expert outputs, then x + that
    inds = mx.array(np.stack([rng.permutation(64)[:k] for _ in range(M)]).astype(np.int32))
    flat = inds.flatten()
    order = mx.argsort(flat)
    pos = mx.argsort(order).astype(mx.int32)
    ye = mx.array(rng.standard_normal((M, k, K)).astype(np.float32)).astype(mx.bfloat16)
    sc = mx.softmax(mx.array(rng.standard_normal((M, k)).astype(np.float32)), axis=-1).astype(mx.bfloat16)
    res = mx.pad(x, [(0, Mp - M), (0, 0)])
    h_ref = res[:M] + (ye * sc[..., None]).sum(axis=-2)
    y_sorted = ye.reshape(M * k, K)[order]
    h, xq, xs = fo.moe_combine(y_sorted, pos, sc, res, nw, eps, M)
    nd = int(mx.sum(h[:M] != h_ref).item())
    xq_ref, _ = S.rmsnorm_quant(h_ref, nw, eps)
    nq = int(mx.sum(xq[:M] != xq_ref[:M]).item())
    print(f"10. moe_combine: residual-stream elements differing from MLX {nd}/{M * K}, next-norm codes differing {nq}")
    ok &= nd == 0 and nq == 0
    return ok


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
