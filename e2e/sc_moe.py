"""Grouped SC-MXFP4 W4A8 expert GEMM for mixture-of-experts prefill (mx.fast.metal_kernel, Metal 4 matmul2d).

Token-expert assignments are sorted by expert (as mlx_lm's SwitchGLU does for prefill), so expert e owns the
contiguous rows [off[e], off[e+1]) of the gathered INT8 activations. One dispatch covers all experts: threadgroup
(n-tile, s) finds the expert e and M-tile of slot s from a prefix sum of per-expert tile counts, then loops over K
in BK chunks, unpacking expert e's TN x BK weight tile from the packed 4.07-bit layout (e2e/sc_format.py) straight
into threadgroup memory as exact INT8 and running INT8 x INT8 -> INT32 matmul2d on it (no INT8 weight scratch:
with ~M/8 rows per expert, each weight tile is used by only one or a few M-tiles, so a separate unpack pass would
move more bytes than the GEMM reads). The epilogue applies the scales in registers:
  EPI 0: INT32 out;  EPI 1: T(acc * xs * ws_e);  EPI 3: swiglu on gate/up rows interleaved as in mm_epi.

The same kernel with one "expert" is a dense W4A8 GEMM with in-kernel unpack (bench: e2e/bench_moe.py).
"""
import mlx.core as mx
import numpy as np

import sc_kernels as S

_MOE_MM = """
    // ---- which expert / M-tile this threadgroup computes
    const uint s = threadgroup_position_in_grid.y;
    const uint n0 = threadgroup_position_in_grid.x * BN;
    const uint tid = thread_position_in_threadgroup.x;
    if (s >= uint(tstart[E])) return;
    uint lo = 0, hi = E;  // largest e with tstart[e] <= s
    while (hi - lo > 1) { const uint mid = (lo + hi) / 2; if (uint(tstart[mid]) <= s) lo = mid; else hi = mid; }
    const uint e = lo;
    const uint row0 = off[e] + (s - tstart[e]) * BM, row_end = off[e + 1];

    threadgroup int8_t sB[BN * BK];
    tensor<device int8_t, dextents<int32_t, 2>, tensor_inline> tA((device int8_t *)xq, dextents<int32_t, 2>(K, R));
    tensor<threadgroup int8_t, dextents<int32_t, 2>, tensor_inline> tB(sB, dextents<int32_t, 2>(BK, BN));
    constexpr auto desc = matmul2d_descriptor(BM, BN, BK, false, true, false,
                                              matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc, execution_simdgroups<SG>> op;
    auto a0 = tA.template slice<BK, BM>(0, row0);
    auto c = op.template get_destination_cooperative_tensor<decltype(a0), decltype(tB), int32_t>();
#pragma unroll
    for (uint16_t i = 0; i < c.get_capacity(); ++i)
        if (c.is_valid_element(i)) c[i] = 0;

    // ---- K loop: unpack expert e's BN x BK weight tile into threadgroup memory, then accumulate
    device const uint *cw = codes + (size_t(e) * N + n0) * (K / 8);
    device const uchar *bw = bcodes + (size_t(e) * N + n0) * (K / (SC_BLOCK * SC_PER_BYTE));
    constexpr uint WPR = BK / 8;                 // code words per tile row
    constexpr uint WPT = BN * WPR / (32 * SG);   // words per thread
    for (uint k0 = 0; k0 < K; k0 += BK) {
#pragma unroll
        for (uint j = 0; j < WPT; ++j) {
            const uint w = tid + j * 32 * SG, rr = w / WPR, cc = w % WPR;
            const uint word = cw[size_t(rr) * (K / 8) + k0 / 8 + cc];
            const uint blk = (k0 + 8 * cc) / SC_BLOCK;
            const uint bc = (bw[size_t(rr) * (K / (SC_BLOCK * SC_PER_BYTE)) + blk / SC_PER_BYTE]
                             >> (SC_BC_BITS * (blk % SC_PER_BYTE))) & ((1u << SC_BC_BITS) - 1);
            const int mult = sc_mult(bc);
            char8_t v;
#pragma unroll
            for (uint q = 0; q < 8; ++q) {
                const uint cd = (word >> (4 * q)) & 0xF;
                const int mag = sc_mag(cd & 7) * mult;
                v[q] = char((cd & 8) ? -mag : mag);
            }
            *(threadgroup char8_t *)(sB + rr * BK + 8 * cc) = v;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        auto a = tA.template slice<BK, BM>(k0, row0);
        op.run(a, tB, c);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // ---- epilogue (rows past this expert's segment belong to the next expert: not stored)
    device const float *wse = wscale + size_t(e) * N;
#pragma unroll
    for (uint16_t i = 0; i < c.get_capacity(); i += 4) {
        if (!c.is_valid_element(i)) continue;
        auto idx = c.get_multidimensional_index(i);
        const uint n = n0 + idx[0], m = row0 + idx[1];
        if (m >= row_end) continue;
        if (EPI == 0) {
            for (int j = 0; j < 4; ++j) out[size_t(m) * N + n + j] = c[i + j];
        } else {
            const float4 v = float4(c[i], c[i + 1], c[i + 2], c[i + 3]) * xscale[m] * *(device const float4 *)(wse + n);
            if (EPI == 1) {
                for (int j = 0; j < 4; ++j) out[size_t(m) * N + n + j] = static_cast<T>(v[j]);
            } else {
                auto o = out + size_t(m) * (N / 2) + n / 2;
                o[0] = sc_swiglu(static_cast<T>(v[0]), static_cast<T>(v[2]));
                o[1] = sc_swiglu(static_cast<T>(v[1]), static_cast<T>(v[3]));
            }
        }
    }
"""


# Two-pass variant (ablation): B read from an INT8 scratch [E, N, K] unpacked beforehand (sc_kernels.unpack_i8 on all
# experts), one matmul2d over the whole K like the dense two-pass GEMM; same tile mapping and epilogues.
_MOE_MM_I8 = _MOE_MM[:_MOE_MM.index("    threadgroup int8_t sB")] + """
    tensor<device int8_t, dextents<int32_t, 2>, tensor_inline> tA((device int8_t *)xq, dextents<int32_t, 2>(K, R));
    tensor<device int8_t, dextents<int32_t, 2>, tensor_inline> tB((device int8_t *)wq, dextents<int32_t, 2>(K, E * N));
    constexpr auto desc = matmul2d_descriptor(BM, BN, dynamic_length_v<int>, false, true, false,
                                              matmul2d_descriptor::mode::multiply);
    matmul2d<desc, execution_simdgroups<SG>> op;
    auto mA = tA.slice(0, row0);
    auto mB = tB.slice(0, e * N + n0);
    auto c = op.template get_destination_cooperative_tensor<decltype(mA), decltype(mB), int32_t>();
    op.run(mA, mB, c);
""" + _MOE_MM[_MOE_MM.index("    // ---- epilogue"):]


# Counting sort of the token-expert assignments by expert in one threadgroup (E <= 1024 experts): histogram with
# threadgroup atomics, prefix sums for the segment offsets and the per-expert M-tile starts, then each assignment
# takes the next slot of its expert. Rows within an expert segment come out in arbitrary order, which changes no
# result (every row is computed independently), so this replaces mlx_lm's two argsorts + gathers.
_ROUTE_SORT = """
    const uint tid = thread_position_in_threadgroup.x;
    threadgroup atomic_uint cnt[E];
    threadgroup atomic_uint cur[E];
    threadgroup uint offs[E + 1];
    for (uint e = tid; e < E; e += NT) { atomic_store_explicit(&cnt[e], 0u, memory_order_relaxed); }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = tid; i < RR; i += NT) atomic_fetch_add_explicit(&cnt[uint(inds[i])], 1u, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0) {
        uint o = 0, t = 0;
        for (uint e = 0; e < E; ++e) {
            const uint c = atomic_load_explicit(&cnt[e], memory_order_relaxed);
            offs[e] = o; off[e] = int(o); tstart[e] = int(t);
            atomic_store_explicit(&cur[e], o, memory_order_relaxed);
            o += c; t += (c + TMT - 1) / TMT;
        }
        offs[E] = o; off[E] = int(o); tstart[E] = int(t);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = tid; i < RR; i += NT) {
        const uint e = uint(inds[i]);
        const uint p = atomic_fetch_add_explicit(&cur[e], 1u, memory_order_relaxed);
        pos[i] = int(p); tok[p] = int(i / TOPK); esorted[p] = int(e);
    }
    for (uint p = RR + tid; p < RR + TMT; p += NT) tok[p] = 0;   // padding rows gather token 0 (never stored)
"""


def route_sort(inds, n_experts, tm):
    """inds int [M, topk] -> (pos int32 [M*topk]: sorted position of assignment (t, j), tok int32 [M*topk + tm]:
    token of each sorted row (+ tm padding rows), e_sorted int32 [M*topk], off int32 [E+1], tstart int32 [E+1],
    max M-tiles). One kernel instead of argsort / argsort / take / segments."""
    M, k = inds.shape
    R = M * k
    kern = S._kernel("route_sort", ["inds"], ["pos", "tok", "esorted", "off", "tstart"], _ROUTE_SORT)
    pos, tok, es, off, ts = kern(inputs=[inds.astype(mx.int32)],
                                 template=[("E", n_experts), ("RR", R), ("TOPK", k), ("TMT", tm), ("NT", 1024)],
                                 grid=(1024, 1, 1), threadgroup=(1024, 1, 1),
                                 output_shapes=[(R,), (R + tm,), (R,), (n_experts + 1,), (n_experts + 1,)],
                                 output_dtypes=[mx.int32] * 5)
    return pos, tok, es, off, ts, -(-R // tm) + n_experts


def segments(expert_sorted, n_experts, tm):
    """Sorted expert id per row -> (off int32 [E+1], tstart int32 [E+1], max number of M-tiles)."""
    R = expert_sorted.shape[0]
    counts = mx.sum(expert_sorted[:, None] == mx.arange(n_experts, dtype=expert_sorted.dtype)[None], axis=0)
    counts = counts.astype(mx.int32)
    off = mx.concatenate([mx.zeros((1,), mx.int32), mx.cumsum(counts)])
    tiles = (counts + tm - 1) // tm
    tstart = mx.concatenate([mx.zeros((1,), mx.int32), mx.cumsum(tiles)])
    return off, tstart, -(-R // tm) + n_experts


def moe_mm(xq, R, codes, bcodes, wscale, off, tstart, max_tiles, N, epi, xscale=None, dtype=mx.bfloat16,
           tile=(64, 64, 4), bk=128):
    """xq int8 [>= R + TM, K]: R rows sorted by expert, then >= TM rows of padding (read by the last M-tile, never
    stored); packed expert weights codes [E, N*K/8], bcodes [E, N, ...], wscale f32 [E, N]
    -> INT32 [R, N] (EPI 0), BF16 [R, N] (1) or BF16 [R, N/2] (3, interleaved gate/up)."""
    K = xq.shape[1]
    assert xq.shape[0] >= R + tile[0]
    E = wscale.shape[0]
    tm, tn, nsg = tile
    assert N % tn == 0 and K % bk == 0 and (tn * bk // 8) % (32 * nsg) == 0 and tn * bk <= 32768
    k = S._kernel("moe_mm", ["xq", "codes", "bcodes", "wscale", "xscale", "off", "tstart"], ["out"], _MOE_MM)
    out_shape = (R, N // 2) if epi == 3 else (R, N)
    return k(inputs=[xq, codes, bcodes, wscale, xscale if xscale is not None else S._dummy(mx.float32), off, tstart],
             template=[("T", dtype), ("EPI", epi), ("R", R), ("N", N), ("K", K), ("E", E), ("BM", tm), ("BN", tn),
                       ("BK", bk), ("SG", nsg)],
             grid=(N // tn * 32 * nsg, max_tiles, 1), threadgroup=(32 * nsg, 1, 1),
             output_shapes=[out_shape], output_dtypes=[mx.int32 if epi == 0 else dtype])[0]


def moe_mm_i8(xq, R, wq, wscale, off, tstart, max_tiles, N, epi, xscale=None, dtype=mx.bfloat16, tile=(64, 64, 4)):
    """Two-pass ablation of moe_mm: wq int8 [E * N, K] already unpacked (e.g. unpack_i8 of all experts)."""
    K = xq.shape[1]
    E = wscale.shape[0]
    tm, tn, nsg = tile
    assert xq.shape[0] >= R + tm and wq.shape == (E * N, K)
    k = S._kernel("moe_mm_i8", ["xq", "wq", "wscale", "xscale", "off", "tstart"], ["out"], _MOE_MM_I8)
    out_shape = (R, N // 2) if epi == 3 else (R, N)
    return k(inputs=[xq, wq, wscale, xscale if xscale is not None else S._dummy(mx.float32), off, tstart],
             template=[("T", dtype), ("EPI", epi), ("R", R), ("N", N), ("K", K), ("E", E), ("BM", tm), ("BN", tn),
                       ("SG", nsg)],
             grid=(N // tn * 32 * nsg, max_tiles, 1), threadgroup=(32 * nsg, 1, 1),
             output_shapes=[out_shape], output_dtypes=[mx.int32 if epi == 0 else dtype])[0]


def unpack_experts(codes, bcodes, N, K):
    """All experts' packed weights -> INT8 [E * N, K] (one unpack_i8 dispatch)."""
    E = codes.shape[0]
    return S.unpack_i8(codes.reshape(-1), bcodes.reshape(E * N, -1), E * N, K)


def _self_test():
    """Grouped GEMM vs per-expert float64 integer GEMMs (random weights, uneven segments incl. empty experts)."""
    import sc_format as F
    rng = np.random.default_rng(0)
    E, N, K, R = 8, 256, 512, 600
    ex = np.sort(rng.choice(E - 1, R, p=np.r_[0.4, np.full(E - 2, 0.6 / (E - 2))]))  # expert E-1 empty
    ex[ex == 3] = 2  # expert 3 empty
    packs = [F.quantize(rng.standard_normal((N, K)) * 0.02) for _ in range(E)]
    codes = mx.array(np.stack([p[0] for p in packs]))
    bcodes = mx.array(np.stack([p[1] for p in packs]))
    ws = mx.array(np.stack([p[2] for p in packs]))
    qw = [p[3] for p in packs]
    xq = rng.integers(-127, 128, (R, K)).astype(np.int8)
    xs = np.abs(rng.standard_normal(R)).astype(np.float32) * 1e-2
    ref = np.zeros((R, N))
    for e in range(E):
        rows = ex == e
        ref[rows] = xq[rows].astype(np.float64) @ qw[e].astype(np.float64).T
    ok = True
    for tile, bk in (((64, 64, 4), 128), ((128, 64, 4), 128), ((64, 128, 4), 64), ((64, 64, 2), 256)):
        off, tstart, mt = segments(mx.array(ex.astype(np.int32)), E, tile[0])
        xqp = mx.array(np.concatenate([xq, np.zeros((tile[0], K), np.int8)]))
        acc = np.array(moe_mm(xqp, R, codes, bcodes, ws, off, tstart, mt, N, 0, tile=tile, bk=bk))
        err = np.abs(acc - ref).max()
        y1 = moe_mm(xqp, R, codes, bcodes, ws, off, tstart, mt, N, 1, mx.array(xs), tile=tile, bk=bk)
        y1_ref = (mx.array(acc).astype(mx.float32) * mx.array(xs)[:, None]
                  * ws[mx.array(ex.astype(np.int32))]).astype(mx.bfloat16)
        nd = int(mx.sum(y1 != y1_ref).item())
        wq8 = unpack_experts(codes, bcodes, N, K)
        acc2 = np.array(moe_mm_i8(xqp, R, wq8, ws, off, tstart, mt, N, 0, tile=tile))
        err2 = np.abs(acc2 - ref).max()
        print(f"moe_mm tile {tile} bk {bk}: INT32 max err {err}, EPI 1 elements differing {nd}; "
              f"two-pass variant INT32 max err {err2}")
        ok &= err == 0 and nd == 0 and err2 == 0
    print("sc_moe self-test", "passed" if ok else "FAILED")
    return ok


if __name__ == "__main__":
    _self_test()
