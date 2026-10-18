"""Batch-1 decode GEMV that reads the packed SC-MXFP4 weights directly (no INT8 scratch), W4A8 like the prefill path.

y[n] = epilogue(xscale * wscale[n] * sum_k xq[k] * q[n, k]) with q = ±CODE_MAG[idx] * MULT[block code] (sc_format).
Each simdgroup computes R = 4 consecutive rows; each lane reads one 16-byte word group (32 codes = one 32-block
when BLOCK = 32) per row per iteration, so a simdgroup streams 32 x 16 bytes of every row per step.

Exactness: E2M1 codes decode to half via the bit trick h = (c & 7) << 9 | (c & 8) << 12 (= 2v * 2^-15 as half),
products with the integer activations are exact in FP32, and a 32-block sum is a multiple of 2^-15 below 2^1, so it
is exact in FP32; times the block multiplier and summed over the <= K/1024 blocks of a lane it stays exact
(< 2^24 ulps); it is converted to the integer sum_k xq q exactly (x 2^15) and reduced over lanes in INT32. So the
GEMV computes the same integer accumulations as the INT8 matmul2d GEMM, and its epilogues are those of mm_epi.
Formats whose code magnitudes are not 2 x E2M1 use a table lookup instead (SC_FAST_E2M1 = 0).
"""
import mlx.core as mx

import sc_kernels as S
import sc_format as F

_GEMV = """
    const uint lane = thread_index_in_simdgroup, sg = simdgroup_index_in_threadgroup;
    const uint n0 = (threadgroup_position_in_grid.x * NSG + sg) * R;
    if (n0 >= N) return;
    constexpr uint WPR = K / 8;                     // code words per row
    constexpr uint BPR = K / (SC_BLOCK * SC_PER_BYTE);
    // MoE: grid.y = top-k slot; slot j uses expert eidx[j] and (XSTRIDE = 1) its own input row
    const uint slot = threadgroup_position_in_grid.y;
    // SHARED >= 0: the slot after the NSLOT routed ones is expert SHARED (the shared expert), not read from eidx
    const uint ex = GATHER ? ((SHARED >= 0 && slot == NSLOT) ? uint(SHARED) : uint(eidx[slot])) : 0;
    auto codes = codes_all + size_t(ex) * N * WPR;
    auto bcodes = bcodes_all + size_t(ex) * N * BPR;
    auto wscale = wscale_all + size_t(ex) * N;
    auto out = out_all + size_t(slot) * (EPI == 3 ? N / 2 : N);
    // ---- prologue: the INT8 input row. PRO 0: given (xq, xscale). PRO 1: per-token INT8 of xin (= act_quant).
    // PRO 2: RMSNorm(xin, nw) then INT8 (= rmsnorm_quant, MLX's reduction order: K/4 virtual threads of 4 values,
    // simd_sum per 32, simd_sum of the K/128 partials). Every threadgroup recomputes it for its own rows.
    threadgroup char sx[PRO ? K : 4];
    threadgroup float red[32];
    threadgroup float shv[2];
    float xsr;
    const uint tid = sg * 32 + lane;
    if (PRO == 0) {
        xsr = xscale[slot * XSTRIDE];
    } else {
        auto xr = xin + size_t(slot) * XSTRIDE * K;
        float inv_rms = 1.0f;
        if (PRO == 2) {
            for (uint g = sg; g < K / 128; g += NSG) {
                const uint v = 32 * g + lane;
                float ss = 0;
                for (int i = 0; i < 4; ++i) { const float xi = float(xr[4 * v + i]); ss += xi * xi; }
                ss = simd_sum(ss);
                if (lane == 0) red[g] = ss;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (sg == 0) {
                const float tot = simd_sum(lane < K / 128 ? red[lane] : 0.0f);
                if (lane == 0) shv[0] = metal::precise::rsqrt(tot / float(K) + eps[0]);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            inv_rms = shv[0];
        }
        float amax = 0;
        for (uint k = tid; k < K; k += 32 * NSG) {
            const float y = PRO == 2 ? float(static_cast<T>(nw[k] * static_cast<T>(float(xr[k]) * inv_rms))) : float(xr[k]);
            amax = max(amax, abs(y));
        }
        amax = simd_max(amax);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (lane == 0) red[sg] = amax;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg == 0) {
            amax = simd_max(lane < NSG ? red[lane] : 0.0f);
            if (lane == 0) shv[1] = amax;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const float sc = shv[1] / 127.0f;
        const float qi = sc > 0 ? 1.0f / sc : 0.0f;
        for (uint k = tid; k < K; k += 32 * NSG) {
            const float y = PRO == 2 ? float(static_cast<T>(nw[k] * static_cast<T>(float(xr[k]) * inv_rms))) : float(xr[k]);
            sx[k] = char(clamp(rint(y * qi), -127.0f, 127.0f));
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        xsr = sc;
    }
    auto xq = xq_all + size_t(slot) * XSTRIDE * K;
    constexpr uint WPL = VPL / 8, WPB = SC_BLOCK / 8;   // code words per lane chunk / per block
    constexpr uint NB = (WPL + WPB - 1) / WPB;           // blocks a lane chunk touches (chunks are block-aligned)
    float acc[R];
#pragma unroll
    for (int r = 0; r < R; ++r) acc[r] = 0;
    for (uint k0 = VPL * lane; k0 < K; k0 += VPL * 32) {
        // VPL activations of this lane as floats (exact integers)
        auto xv = PRO ? (threadgroup const char4 *)(sx + k0) : (threadgroup const char4 *)nullptr;
        auto xd = (device const char4 *)(xq + k0);
        float xf[VPL];
#pragma unroll
        for (uint j = 0; j < VPL / 4; ++j) {
            const char4 c4 = PRO ? xv[j] : xd[j];
            xf[4 * j] = c4.x; xf[4 * j + 1] = c4.y; xf[4 * j + 2] = c4.z; xf[4 * j + 3] = c4.w;
        }
#pragma unroll
        for (int r = 0; r < R; ++r) {
            const uint n = n0 + r;
            auto wp = codes + size_t(n) * WPR + k0 / 8;
            uint wv[WPL];
            if (WPL == 4) { const uint4 t4 = *(device const uint4 *)wp; wv[0] = t4.x; wv[1] = t4.y; wv[2] = t4.z; wv[3] = t4.w; }
            else if (WPL == 2) { const uint2 t2 = *(device const uint2 *)wp; wv[0] = t2.x; wv[1] = t2.y; }
            else { for (uint q = 0; q < WPL; ++q) wv[q] = wp[q]; }
            float bs[NB];
#pragma unroll
            for (uint b = 0; b < NB; ++b) bs[b] = 0;
#pragma unroll
            for (uint q = 0; q < WPL; ++q) {
                const uint w = wv[q];
                const thread float *x8 = xf + 8 * q;
                float sq;
#if SC_FAST_E2M1
                float2 f[4];
#pragma unroll
                for (int p = 0; p < 4; ++p) {   // codes p and p + 4 of this word -> one half2
                    const uint t = w >> (4 * p);
                    f[p] = float2(as_type<half2>(((t & 0x00070007u) << 9) | ((t & 0x00080008u) << 12)));
                }
                sq = ((x8[0] * f[0].x + x8[4] * f[0].y) + (x8[1] * f[1].x + x8[5] * f[1].y))
                   + ((x8[2] * f[2].x + x8[6] * f[2].y) + (x8[3] * f[3].x + x8[7] * f[3].y));
#else
                float v[8];
#pragma unroll
                for (int p = 0; p < 8; ++p) {
                    const uint c = (w >> (4 * p)) & 0xF;
                    const float m = float(sc_mag(c & 7));
                    v[p] = x8[p] * ((c & 8) ? -m : m);
                }
                sq = ((v[0] + v[1]) + (v[2] + v[3])) + ((v[4] + v[5]) + (v[6] + v[7]));
#endif
                bs[q / WPB] += sq;
            }
#pragma unroll
            for (uint b = 0; b < NB; ++b) {
                const uint blk = (k0 + b * SC_BLOCK) / SC_BLOCK;
                const uint bc = (bcodes[size_t(n) * BPR + blk / SC_PER_BYTE] >> (SC_BC_BITS * (blk % SC_PER_BYTE)))
                                & ((1u << SC_BC_BITS) - 1);
                acc[r] = fma(bs[b], float(sc_mult(bc)), acc[r]);
            }
        }
    }
    int tot[R];
#pragma unroll
    for (int r = 0; r < R; ++r) tot[r] = simd_sum(int(rint(acc[r] * SC_ACC_SCALE)));
    if (lane != 0) return;
    float yv[R];
#pragma unroll
    for (int r = 0; r < R; ++r) yv[r] = float(tot[r]) * xsr * wscale[n0 + r];
    if (EPI == 1) {
        for (int r = 0; r < R; ++r) out[n0 + r] = static_cast<T>(yv[r]);
    } else if (EPI == 2) {
        for (int r = 0; r < R; ++r) out[n0 + r] = static_cast<T>(res[n0 + r] + static_cast<T>(yv[r]));
    } else {
        for (int r = 0; r < R; r += 4) {
            out[(n0 + r) / 2] = sc_swiglu(static_cast<T>(yv[r]), static_cast<T>(yv[r + 2]));
            out[(n0 + r) / 2 + 1] = sc_swiglu(static_cast<T>(yv[r + 1]), static_cast<T>(yv[r + 3]));
        }
    }
"""

E2M1X2 = (0, 1, 2, 3, 4, 6, 8, 12)


def _header(fmt):
    fast = tuple(fmt.code_mag) == E2M1X2
    # fast path: products carry 2^-15 (half bit trick gives v * 2^-14 = 2v * 2^-15); LUT path: plain integers
    return f"#define SC_FAST_E2M1 {int(fast)}\n#define SC_ACC_SCALE {32768.0 if fast else 1.0}f\n"


_dummy_xq = []


def _dummy_rows():
    """A device-memory INT8 placeholder (MLX binds very small inputs as `constant`, which device casts reject)."""
    if not _dummy_xq:
        _dummy_xq.append(mx.zeros((4096,), mx.int8))
    return _dummy_xq[0]


GEMV_TILE = (4, 16, 2)  # rows per simdgroup, activations per lane per step, simdgroups per threadgroup
GEMV_TILES = {}         # per-shape overrides {(N, K): tile} (e2e/tune_decode.py)


def gemv(xq, codes, bcodes, xscale, wscale, N, epi, dtype=mx.bfloat16, res=None, nsg=None, eidx=None, xstride=0,
         pro=0, nw=None, eps=None, rows=None, vpl=None, shared=None):
    """W4A8 GEMV. Input: pro=0: xq int8 [K] (or [S, K] with xstride=1) and xscale; pro=1: xq is a T row (or rows),
    quantized per token inside the kernel (act_quant); pro=2: xq is a T row, RMSNorm(nw, eps) + INT8 inside (as
    rmsnorm_quant). Output: EPI 1: T [N]; 2: T [N] = res + y; 3: T [N / 2] (interleaved gate/up rows).
    MoE (eidx int32 [S] given): codes [E, ...], bcodes [E, N, ...], wscale [E, N]; output row j uses expert eidx[j]
    and input row j if xstride = 1 (else the single input row) -> [S, N or N/2]. shared = e: one more slot after the
    eidx.size routed ones uses expert e (int32 or uint32 eidx, e.g. argpartition's output, used as is)."""
    nslot = 0 if eidx is None else eidx.size
    S_ = 1 if eidx is None else nslot + (shared is not None)
    K = xq.shape[-1]
    t = GEMV_TILES.get((N, K), GEMV_TILE)
    R, V, nsg = rows or t[0], vpl or t[1], nsg or t[2]
    # lanes stride K in chunks of V (block-aligned); K need not be a multiple of 32 V (Qwen3.8 down: K = 640)
    assert K % V == 0 and K % 128 == 0 and N % (R * nsg) == 0 and (epi != 3 or R % 4 == 0)
    k = S._kernel("gemv", ["xq_all", "codes_all", "bcodes_all", "xscale", "wscale_all", "res", "eidx", "xin", "nw",
                           "eps"], ["out_all"], _GEMV, header_extra=_header)
    no = N // 2 if epi == 3 else N
    d = S._dummy(dtype)
    # (kernels index flat memory: inputs are passed in their own shapes, no reshape ops)
    out = k(inputs=[xq if pro == 0 else _dummy_rows(), codes, bcodes, xscale if pro == 0 else S._dummy(mx.float32),
                    wscale, res if epi == 2 else d, eidx if eidx is not None else S._dummy(mx.int32),
                    xq if pro else d, nw if pro == 2 else d, eps if pro == 2 else S._dummy(mx.float32)],
            template=[("T", dtype), ("EPI", epi), ("N", N), ("K", K), ("NSG", nsg), ("R", R), ("VPL", V),
                      ("GATHER", eidx is not None), ("XSTRIDE", xstride), ("PRO", pro),
                      ("SHARED", -1 if shared is None else int(shared)), ("NSLOT", nslot)],
            grid=(N // (R * nsg) * 32 * nsg, S_, 1), threadgroup=(32 * nsg, 1, 1),
            output_shapes=[(S_ * no,)], output_dtypes=[dtype])[0]
    return out if eidx is None else out.reshape(S_, no)


def _self_test():
    import numpy as np
    import sc_format as F
    rng = np.random.default_rng(0)
    ok = True
    for N, K in ((256, 2048), (512, 6144), (64, 1024)):
        w = rng.standard_normal((N, K)) * 0.02 * rng.lognormal(0, 0.5, (1, K))
        codes, bc, ws, qw = F.quantize(w)
        codes, bc, ws = mx.array(codes), mx.array(bc), mx.array(ws)
        xq = rng.integers(-127, 128, K).astype(np.int8)
        xq[:16] = 127  # stress the exactness bound
        xs = mx.array([0.01], mx.float32)
        ref_int = qw.astype(np.int64) @ xq.astype(np.int64)
        # EPI 1 against the INT32 GEMM epilogue of the prefill path
        acc = S.mm_i8_i32(mx.array(np.tile(xq, (128, 1))), mx.array(qw))
        y_ref = (acc[0].astype(mx.float32) * xs[0] * ws).astype(mx.bfloat16)
        exact = bool(np.array_equal(np.array(acc[0]).astype(np.int64), ref_int))
        nd0 = 0
        for tile in ((4, 16, 2), (4, 32, 2), (8, 16, 1), (8, 32, 2), (4, 8, 4)):
            if K % (32 * tile[1]) == 0:
                nd0 += int(mx.sum(gemv(mx.array(xq), codes, bc, xs, ws, N, 1, rows=tile[0], vpl=tile[1], nsg=tile[2])
                                  != y_ref).item())
        y1 = gemv(mx.array(xq), codes, bc, xs, ws, N, 1)
        res = mx.array(rng.standard_normal(N).astype(np.float32)).astype(mx.bfloat16)
        y2 = gemv(mx.array(xq), codes, bc, xs, ws, N, 2, res=res)
        nd1 = int(mx.sum(y1 != y_ref).item())
        nd2 = int(mx.sum(y2 != res + y_ref).item())
        # EPI 3 with interleaved rows
        perm = S.interleave_gate_up(N // 2)
        cp = mx.array(np.array(codes).reshape(N, -1)[perm].reshape(-1))
        y3 = gemv(mx.array(xq), cp, mx.array(np.array(bc)[perm]), xs, mx.array(np.array(ws)[perm]), N, 3)
        from mlx_lm.models.activations import swiglu
        nd3 = int(mx.sum(y3 != swiglu(y_ref[:N // 2], y_ref[N // 2:])).item())
        print(f"gemv N={N} K={K}: INT32 GEMM exact {exact}; outputs differing from prefill epilogues "
              f"(EPI 1, 2, 3): {nd1}, {nd2}, {nd3}")
        print(f"   other tiles: outputs differing {nd0}")
        ok &= exact and nd1 == nd2 == nd3 == nd0 == 0
    # fused prologues: act_quant (PRO 1) and rmsnorm_quant (PRO 2) inside the GEMV
    N, K = 512, 2048
    codes, bc, ws, qw = F.quantize(rng.standard_normal((N, K)) * 0.02)
    codes, bc, ws = mx.array(codes), mx.array(bc), mx.array(ws)
    x = mx.array((rng.standard_normal((1, K)) * rng.lognormal(0, 1, K)).astype(np.float32)).astype(mx.bfloat16)
    nw = mx.array(rng.lognormal(0, 0.3, K).astype(np.float32)).astype(mx.bfloat16)
    eps = mx.array([1e-6], mx.float32)
    for pro, (xq_, xs_) in ((1, S.act_quant(x, 1)), (2, S.rmsnorm_quant(x, nw, eps, 1))):
        for nsg in (1, 2, 4):
            y0 = gemv(xq_, codes, bc, xs_, ws, N, 1, nsg=nsg)
            y1 = gemv(x, codes, bc, None, ws, N, 1, nsg=nsg, pro=pro, nw=nw, eps=eps)
            nd = int(mx.sum(y0 != y1).item())
            print(f"gemv prologue PRO {pro} nsg {nsg}: outputs differing from separate kernel {nd}")
            ok &= nd == 0
    # MoE gather: 3 experts, 2 slots, per-slot inputs
    E, N, K = 3, 256, 1024
    packs = [F.quantize(rng.standard_normal((N, K)) * 0.02) for _ in range(E)]
    codes = mx.array(np.stack([p[0] for p in packs]))
    bc = mx.array(np.stack([p[1] for p in packs]))
    ws = mx.array(np.stack([p[2] for p in packs]))
    xq = rng.integers(-127, 128, (2, K)).astype(np.int8)
    xs = mx.array([0.01, 0.02], mx.float32)
    eidx = mx.array([2, 0], mx.int32)
    y = gemv(mx.array(xq), codes, bc, xs, ws, N, 1, eidx=eidx, xstride=1)
    nd = 0
    for j, e in enumerate((2, 0)):
        acc = S.mm_i8_i32(mx.array(np.tile(xq[j], (128, 1))), mx.array(packs[e][3]))
        nd += int(mx.sum(y[j] != (acc[0].astype(mx.float32) * xs[j] * ws[e]).astype(mx.bfloat16)).item())
    print(f"gemv gather (experts 2, 0): outputs differing {nd}")
    ok &= nd == 0
    print("sc_decode self-test", "passed" if ok else "FAILED")
    return ok


if __name__ == "__main__":
    _self_test()
