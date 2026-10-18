"""Fused SC-MXFP4 W4A8 prefill kernels for MLX (mx.fast.metal_kernel; Metal 4 tensor ops compile there).

Compared with the unfused per-linear pipeline of e2e/w4a8_mlx.py (act_quant -> unpack -> INT8 GEMM -> scale, for
every one of the 7 linears of a block), these kernels
  - quantize a block input once for all consumers: q/k/v and gate/up run as one GEMM each on row-concatenated
    weights (the format is per row, so concatenation is exact);
  - fold RMSNorm into activation quantization (rmsnorm_quant), and the residual add plus the next RMSNorm into the
    GEMM's scaling epilogue (res_norm_quant);
  - fold SiLU(gate) * up into the quantization of the down-projection input (swiglu_quant);
  - fold Qwen3's per-head q/k RMSNorm into the q/k/v epilogue and write q/k/v head-major (qkv_epilogue_headnorm);
  - read the 2-bit packed block codes of e2e/sc_format.py (4.07 bits/weight) in the weight unpack.
Rounding follows MLX's own kernels step by step (BF16 GEMM outputs, MLX's rms_norm thread layout and reduction
order, BF16 residual adds), so the fused model computes the same function as the unfused one.

All kernels take M rows (tokens) and write Mp >= M rows (Mp a multiple of the GEMM tile); rows >= M are zero.
`deps` (optional, every wrapper): extra inputs the kernel ignores. They only order MLX's graph so that an
independent kernel (the next weight unpack) is dispatched next to this one and can overlap with it.
"""
import os

import mlx.core as mx

import sc_format as F

TM, TN, NSG, GROUP_M = 128, 64, 4, 8

HEADER = """
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace mpp::tensor_ops;
typedef vec<char, 8> char8_t;
"""

# ---- weight unpack: packed codes + BC_BITS block codes -> INT8 [N, K]; 8 weights (one uint32) per thread ------------
_UNPACK = """
    const uint i = thread_position_in_grid.x;
    const uint word = codes[i];
    const uint e = i * 8, n = e / K, k = e % K;
    const uint blk = k / SC_BLOCK;
    const uint bc = (bcodes[n * (K / (SC_BLOCK * SC_PER_BYTE)) + blk / SC_PER_BYTE]
                     >> (SC_BC_BITS * (blk % SC_PER_BYTE))) & ((1u << SC_BC_BITS) - 1);
    const int mult = sc_mult(bc);
    char8_t v;
#pragma unroll
    for (uint j = 0; j < 8; ++j) {
        const uint c = (word >> (4 * j)) & 0xF;
        const int mag = sc_mag(c & 7) * mult;
        v[j] = char((c & 8) ? -mag : mag);
    }
    ((device char8_t *)wq)[i] = v;
"""

# ---- INT8 x INT8 -> INT32 matmul2d (kernels/matmul.metal tile_mm, BK = 0, GROUP_M swizzle) -------------------------
_MM = """
    const uint num_m = threadgroups_per_grid.y, num_n = threadgroups_per_grid.x;
    const uint pid = threadgroup_position_in_grid.y * num_n + threadgroup_position_in_grid.x;
    const uint per_group = GM * num_n;
    const uint first_m = (pid / per_group) * GM;
    const uint gsz = min(num_m - first_m, uint(GM));
    const uint r = pid % per_group;
    const int m0 = (first_m + r % gsz) * BM, n0 = (r / gsz) * BN;
    tensor<device int8_t, dextents<int32_t, 2>, tensor_inline> tA((device int8_t *)xq, dextents<int32_t, 2>(K, M));
    tensor<device int8_t, dextents<int32_t, 2>, tensor_inline> tB((device int8_t *)wq, dextents<int32_t, 2>(K, N));
    tensor<device int32_t, dextents<int32_t, 2>, tensor_inline> tC((device int32_t *)acc, dextents<int32_t, 2>(N, M));
    constexpr auto desc = matmul2d_descriptor(BM, BN, dynamic_length_v<int>, false, true, false,
                                              matmul2d_descriptor::mode::multiply);
    matmul2d<desc, execution_simdgroups<SG>> op;
    auto mA = tA.slice(0, m0);
    auto mB = tB.slice(0, n0);
    auto mC = tC.slice(n0, m0);
    op.run(mA, mB, mC);
"""

# ---- INT8 GEMM with the scaling epilogue applied in registers (cooperative destination tensor) -------------------
# The destination layout of matmul2d (probed and asserted in test_sc_kernels.py) gives each thread runs of 4
# consecutive output columns n..n+3 of one row m, so every epilogue works on aligned groups of 4 columns:
#   EPI 1: y = T(acc * xscale[m] * wscale[n])                         -> out [M, N]
#   EPI 2: h = T(res + T(acc * xscale[m] * wscale[n]))                -> out [M, N]   (residual add)
#   EPI 3: rows interleaved g_2p, g_2p+1, u_2p, u_2p+1:  a = swiglu(T(g), T(u)) -> out [M, N / 2]
_MM_EPI = """
    const uint num_m = threadgroups_per_grid.y, num_n = threadgroups_per_grid.x;
    const uint pid = threadgroup_position_in_grid.y * num_n + threadgroup_position_in_grid.x;
    const uint per_group = GM * num_n;
    const uint first_m = (pid / per_group) * GM;
    const uint gsz = min(num_m - first_m, uint(GM));
    const uint r = pid % per_group;
    const int m0 = (first_m + r % gsz) * BM, n0 = (r / gsz) * BN;
    tensor<device int8_t, dextents<int32_t, 2>, tensor_inline> tA((device int8_t *)xq, dextents<int32_t, 2>(K, M));
    tensor<device int8_t, dextents<int32_t, 2>, tensor_inline> tB((device int8_t *)wq, dextents<int32_t, 2>(K, N));
    constexpr auto desc = matmul2d_descriptor(BM, BN, dynamic_length_v<int>, false, true, false,
                                              matmul2d_descriptor::mode::multiply);
    matmul2d<desc, execution_simdgroups<SG>> op;
    auto mA = tA.slice(0, m0);
    auto mB = tB.slice(0, n0);
    auto c = op.template get_destination_cooperative_tensor<decltype(mA), decltype(mB), int32_t>();
    op.run(mA, mB, c);
#pragma unroll
    for (uint16_t i = 0; i < c.get_capacity(); i += 4) {
        if (!c.is_valid_element(i)) continue;
        auto idx = c.get_multidimensional_index(i);
        const uint n = n0 + idx[0], m = m0 + idx[1];
        const float4 a = float4(c[i], c[i + 1], c[i + 2], c[i + 3]) * xscale[m] * *(device const float4 *)(wscale + n);
        if (EPI == 1) {
            device T *o = out + size_t(m) * N + n;
            for (int j = 0; j < 4; ++j) o[j] = static_cast<T>(a[j]);
        } else if (EPI == 2) {
            device T *o = out + size_t(m) * N + n;
            auto rr = res + size_t(m) * N + n;
            for (int j = 0; j < 4; ++j) o[j] = static_cast<T>(rr[j] + static_cast<T>(a[j]));
        } else {
            device T *o = out + size_t(m) * (N / 2) + n / 2;
            o[0] = sc_swiglu(static_cast<T>(a[0]), static_cast<T>(a[2]));
            o[1] = sc_swiglu(static_cast<T>(a[1]), static_cast<T>(a[3]));
        }
    }
"""

# ---- shared pieces of the row kernels ------------------------------------------------------------------------------
_ROW_HELPERS = ""

# MLX rms_norm (rms_single_row, N_READS = 4): each of K/4 threads reads 4 consecutive elements, sums x^2 in FP32,
# simd_sum, then simd_sum over the simdgroup partials; inv = precise::rsqrt(sum / K + eps);
# out = w * T(x * inv) with BF16 rounding of x * inv. Then symmetric per-token INT8 of the normalized row.
_RMS_QUANT_BODY = """
    // xv[4]: this thread's 4 input values (type T) at columns 4 * lid .. 4 * lid + 3 of the row
    float ss = 0;
    for (int i = 0; i < 4; ++i) { const float xi = float(xv[i]); ss += xi * xi; }
    ss = simd_sum(ss);
    if (sg == 0) red[lane] = 0;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0) red[sg] = ss;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        ss = simd_sum(red[lane]);
        if (lane == 0) shared_inv = metal::precise::rsqrt(ss / float(K) + eps[0]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float inv = shared_inv;
    float yv[4];
    float amax = 0;
    for (int i = 0; i < 4; ++i) {
        const T y = static_cast<T>(nw[4 * lid + i] * static_cast<T>(float(xv[i]) * inv));
        yv[i] = float(y);
        amax = max(amax, abs(yv[i]));
    }
    amax = simd_max(amax);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lane == 0) red[sg] = amax;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        amax = simd_max(lane < (K / 4 + 31) / 32 ? red[lane] : 0.0f);
        if (lane == 0) shared_inv = amax;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float s = shared_inv / 127.0f;
    const float qi = s > 0 ? 1.0f / s : 0.0f;
    char4 q;
    for (int i = 0; i < 4; ++i) q[i] = char(clamp(rint(yv[i] * qi), -127.0f, 127.0f));
    *(device char4 *)(xq + size_t(row) * K + 4 * lid) = q;
    if (lid == 0) xs[row] = s;
"""

_RMSNORM_QUANT = """
    const uint row = threadgroup_position_in_grid.x, lid = thread_position_in_threadgroup.x;
    const uint lane = thread_index_in_simdgroup, sg = simdgroup_index_in_threadgroup;
    threadgroup float red[32];
    threadgroup float shared_inv;
    if (row >= M) {
        *(device char4 *)(xq + size_t(row) * K + 4 * lid) = char4(0);
        if (lid == 0) xs[row] = 0;
        return;
    }
    T xv[4];
    for (int i = 0; i < 4; ++i) xv[i] = x[size_t(row) * K + 4 * lid + i];
""" + _RMS_QUANT_BODY

# h = res + T(acc * xscale * wscale) (BF16 GEMM output, BF16 residual add), written out, then rmsnorm_quant(h).
_RES_NORM_QUANT = """
    const uint row = threadgroup_position_in_grid.x, lid = thread_position_in_threadgroup.x;
    const uint lane = thread_index_in_simdgroup, sg = simdgroup_index_in_threadgroup;
    threadgroup float red[32];
    threadgroup float shared_inv;
    const size_t o = size_t(row) * K + 4 * lid;
    const float4 a = float4(*(device const int4 *)(acc + o)) * xscale[row] * *(device const float4 *)(wscale + 4 * lid);
    T xv[4];
    for (int i = 0; i < 4; ++i) {
        xv[i] = static_cast<T>(res[o + i] + static_cast<T>(a[i]));
        h[o + i] = xv[i];
    }
""" + _RMS_QUANT_BODY

_RES_EPILOGUE = """
    const uint i = thread_position_in_grid.x;
    const size_t e = size_t(i) * 4;
    const uint m = e / N, n = e % N;
    const float4 a = float4(*(device const int4 *)(acc + e)) * xscale[m] * *(device const float4 *)(wscale + n);
    for (int j = 0; j < 4; ++j) h[e + j] = static_cast<T>(res[e + j] + static_cast<T>(a[j]));
"""

# Per-token INT8 of x [M, K] (rows >= M -> 0), J chunks of 4 per thread.
_ACT_QUANT = """
    const uint row = threadgroup_position_in_grid.x, lid = thread_position_in_threadgroup.x;
    const uint lane = thread_index_in_simdgroup, sg = simdgroup_index_in_threadgroup;
    threadgroup float red[32];
    threadgroup float shared_amax;
    constexpr int J = K / (4 * NT);
    float v[J][4];
    float amax = 0;
    for (int j = 0; j < J; ++j)
        for (int i = 0; i < 4; ++i) {
            v[j][i] = row < M ? float(x[size_t(row) * K + 4 * (j * NT + lid) + i]) : 0.0f;
            amax = max(amax, abs(v[j][i]));
        }
    amax = simd_max(amax);
    if (lane == 0) red[sg] = amax;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        amax = simd_max(lane < NT / 32 ? red[lane] : 0.0f);
        if (lane == 0) shared_amax = amax;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float s = shared_amax / 127.0f;
    const float qi = s > 0 ? 1.0f / s : 0.0f;
    for (int j = 0; j < J; ++j) {
        char4 q;
        for (int i = 0; i < 4; ++i) q[i] = char(clamp(rint(v[j][i] * qi), -127.0f, 127.0f));
        *(device char4 *)(xq + size_t(row) * K + 4 * (j * NT + lid)) = q;
    }
    if (lid == 0) xs[row] = s;
"""

# SiLU(gate) * up on the BF16 GEMM outputs (gate = columns [0, F), up = [F, 2F) of the concatenated GEMM), rounded
# like MLX's compiled swiglu, then per-token INT8 for the down projection. J chunks of 4 per thread.
_SWIGLU_QUANT = """
    const uint row = threadgroup_position_in_grid.x, lid = thread_position_in_threadgroup.x;
    const uint lane = thread_index_in_simdgroup, sg = simdgroup_index_in_threadgroup;
    threadgroup float red[32];
    threadgroup float shared_amax;
    constexpr int J = F / (4 * NT);
    const float xsr = xscale[row];
    device const int *ar = acc + size_t(row) * (2 * F);
    float v[J][4];
    float amax = 0;
    for (int j = 0; j < J; ++j) {
        const uint c = 4 * (j * NT + lid);
        const float4 g4 = float4(*(device const int4 *)(ar + c)) * xsr * *(device const float4 *)(wscale + c);
        const float4 u4 = float4(*(device const int4 *)(ar + F + c)) * xsr * *(device const float4 *)(wscale + F + c);
        for (int i = 0; i < 4; ++i) {
            const T g = static_cast<T>(g4[i]), u = static_cast<T>(u4[i]);
            v[j][i] = float(sc_swiglu(g, u));
            amax = max(amax, abs(v[j][i]));
        }
    }
    amax = simd_max(amax);
    if (lane == 0) red[sg] = amax;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        amax = simd_max(lane < NT / 32 ? red[lane] : 0.0f);
        if (lane == 0) shared_amax = amax;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const float s = shared_amax / 127.0f;
    const float qi = s > 0 ? 1.0f / s : 0.0f;
    for (int j = 0; j < J; ++j) {
        char4 q;
        for (int i = 0; i < 4; ++i) q[i] = char(clamp(rint(v[j][i] * qi), -127.0f, 127.0f));
        *(device char4 *)(xq + size_t(row) * F + 4 * (j * NT + lid)) = q;
    }
    if (lid == 0) xs[row] = s;
"""

# MLX: sigmoid(x) = x < 0 ? y : 1 - y with y = 1 / (1 + exp(|x|)) in the input type, silu = x * sigmoid(x),
# swiglu = silu(gate) * up, every op rounded to T.
_SWIGLU_HELPER = """
template <typename T>
inline T sc_swiglu(T g, T u) {
    const T e = static_cast<T>(metal::precise::exp(metal::abs(float(g))));  // MLX bf16_math: exp in FP32, rounded
    auto y = 1 / (1 + e);
    const T sig = (g < 0) ? y : 1 - y;
    const T silu = g * sig;
    return silu * u;
}
"""

# Qwen3 q/k/v epilogue: one threadgroup per token row, one simdgroup per 128-wide head (q heads, then k heads, then v
# heads). q/k heads get RMSNorm over the head (MLX rms_single_row for axis 128: 32 threads x 4 values); outputs are
# written head-major [heads, L, D] so RoPE/SDPA need no transpose.
_QKV_HEADNORM = """
    const uint row = threadgroup_position_in_grid.x;
    const uint lane = thread_index_in_simdgroup, g = simdgroup_index_in_threadgroup;
    if (row >= L) return;
    constexpr uint NTOT = (HQ + 2 * HK) * D;
    const uint c = g * D + 4 * lane;
    const float4 a = float4(*(device const int4 *)(acc + size_t(row) * NTOT + c)) * xscale[row]
                     * *(device const float4 *)(wscale + c);
    T yv[4];
    for (int i = 0; i < 4; ++i) yv[i] = static_cast<T>(a[i]);
    if (g < HQ + HK) {
        float ss = 0;
        for (int i = 0; i < 4; ++i) { const float yi = float(yv[i]); ss += yi * yi; }
        ss = simd_sum(ss);
        const float inv = metal::precise::rsqrt(ss / float(D) + eps[0]);
        device const T *w = g < HQ ? qn : kn;
        for (int i = 0; i < 4; ++i) yv[i] = static_cast<T>(w[4 * lane + i] * static_cast<T>(float(yv[i]) * inv));
    }
    device T *out;
    uint hh;
    if (g < HQ) { out = q; hh = g; } else if (g < HQ + HK) { out = k; hh = g - HQ; } else { out = v; hh = g - HQ - HK; }
    device T *o = out + (size_t(hh) * L + row) * D + 4 * lane;
    for (int i = 0; i < 4; ++i) o[i] = yv[i];
"""

# Same, from the BF16 GEMM output y [Mp, (HQ + 2 HK) D] of mm_epi EPI 1.
_QKV_HEADNORM_Y = """
    const uint row = threadgroup_position_in_grid.x;
    const uint lane = thread_index_in_simdgroup, g = simdgroup_index_in_threadgroup;
    if (row >= L) return;
    constexpr uint NTOT = (HQ + 2 * HK) * D;
    const uint c = g * D + 4 * lane;
    T yv[4];
    for (int i = 0; i < 4; ++i) yv[i] = y[size_t(row) * NTOT + c + i];
    if (g < HQ + HK) {
        float ss = 0;
        for (int i = 0; i < 4; ++i) { const float yi = float(yv[i]); ss += yi * yi; }
        ss = simd_sum(ss);
        const float inv = metal::precise::rsqrt(ss / float(D) + eps[0]);
        device const T *w = g < HQ ? qn : kn;
        for (int i = 0; i < 4; ++i) yv[i] = static_cast<T>(w[4 * lane + i] * static_cast<T>(float(yv[i]) * inv));
    }
    device T *out;
    uint hh;
    if (g < HQ) { out = q; hh = g; } else if (g < HQ + HK) { out = k; hh = g - HQ; } else { out = v; hh = g - HQ - HK; }
    device T *o = out + (size_t(hh) * L + row) * D + 4 * lane;
    for (int i = 0; i < 4; ++i) o[i] = yv[i];
"""

# Variants that also write the normalized BF16 row (MoE: the router reads it).
_RMS_QUANT_BODY_N = _RMS_QUANT_BODY.replace("yv[i] = float(y);", "yv[i] = float(y);\n        xn[size_t(row) * K + 4 * lid + i] = y;")
_RMSNORM_QUANT_N = _RMSNORM_QUANT.replace(_RMS_QUANT_BODY, _RMS_QUANT_BODY_N).replace(
    "        if (lid == 0) xs[row] = 0;\n        return;",
    "        if (lid == 0) xs[row] = 0;\n        for (int i = 0; i < 4; ++i) xn[size_t(row) * K + 4 * lid + i] = 0;\n        return;")

# OLMoE q/k/v epilogue from the BF16 GEMM output y [Mp, 3 H D]: q_norm and k_norm are RMSNorm over the whole
# H*D-wide q (resp. k) vector (MLX rms_single_row layout, H*D/4 threads x 4 values), then head-major outputs.
_QKV_FULLNORM_Y = """
    const uint row = threadgroup_position_in_grid.x, lid = thread_position_in_threadgroup.x;
    const uint lane = thread_index_in_simdgroup, sg = simdgroup_index_in_threadgroup;
    threadgroup float red[32];
    threadgroup float shared_inv;
    if (row >= L) return;
    constexpr uint W = H * D;
    const uint c = 4 * lid, hh = c / D, dd = c % D;
    for (uint part = 0; part < 3; ++part) {
        T yv[4];
        for (int i = 0; i < 4; ++i) yv[i] = y[size_t(row) * 3 * W + part * W + c + i];
        if (part < 2) {
            float ss = 0;
            for (int i = 0; i < 4; ++i) { const float yi = float(yv[i]); ss += yi * yi; }
            ss = simd_sum(ss);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (sg == 0) red[lane] = 0;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (lane == 0) red[sg] = ss;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (sg == 0) {
                ss = simd_sum(red[lane]);
                if (lane == 0) shared_inv = metal::precise::rsqrt(ss / float(W) + eps[0]);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            const float inv = shared_inv;
            device const T *w = part == 0 ? qn : kn;
            for (int i = 0; i < 4; ++i) yv[i] = static_cast<T>(w[c + i] * static_cast<T>(float(yv[i]) * inv));
        }
        device T *out = part == 0 ? q : (part == 1 ? k : v);
        device T *o = out + (size_t(hh) * L + row) * D + dd;
        for (int i = 0; i < 4; ++i) o[i] = yv[i];
    }
"""

_kernels = {}
_METAL_TYPES = {mx.bfloat16: "bfloat16_t", mx.float16: "float16_t", mx.float32: "float", mx.int32: "int",
                 mx.int8: "int8_t", mx.uint8: "uint8_t", mx.uint32: "uint32_t"}


class _Baked:
    """mx.fast.metal_kernel with the template arguments baked into the source as constants, one compiled kernel per
    distinct argument tuple. Same call signature; avoids MLX's per-call template processing (~2 us per call, which
    matters for decode, ~400 kernel calls per token, and for the graph-building time of short prefills)."""

    def __init__(self, name, inputs, outputs, source, header):
        self.name, self.inputs, self.outputs, self.source, self.header = name, inputs, outputs, source, header
        self.cache = {}

    def __call__(self, inputs, template=(), grid=None, threadgroup=None, output_shapes=None, output_dtypes=None):
        key = tuple(template)
        k = self.cache.get(key)
        if k is None:
            pre = ""
            for n, v in template:
                if isinstance(v, mx.Dtype):
                    pre += f"    using {n} = {_METAL_TYPES[v]};\n"
                elif isinstance(v, bool):
                    pre += f"    constexpr bool {n} = {'true' if v else 'false'};\n"
                else:
                    pre += f"    constexpr int {n} = {int(v)};\n"
            k = mx.fast.metal_kernel(name=f"{self.name}_{len(self.cache)}", input_names=self.inputs,
                                     output_names=self.outputs, source=pre + self.source, header=self.header)
            self.cache[key] = k
        return k(inputs=inputs, grid=grid, threadgroup=threadgroup, output_shapes=output_shapes,
                 output_dtypes=output_dtypes)


def _kernel(name, inputs, outputs, source, ndeps=0, header_extra=""):
    """Kernel object for the current weight format (sc_format.DEFAULT); callable(header_extra) for format-dependent
    extra headers."""
    fmt = F.DEFAULT
    key = (name, ndeps, fmt)
    if key not in _kernels:
        extra = header_extra(fmt) if callable(header_extra) else header_extra
        tag = abs(hash(fmt)) % 10 ** 8
        _kernels[key] = _Baked(f"sc_{name}_d{ndeps}_f{tag}", inputs + [f"dep{i}" for i in range(ndeps)], outputs,
                               source, HEADER + fmt.metal_header() + _ROW_HELPERS + _SWIGLU_HELPER + extra)
    return _kernels[key]


def set_format(fmt):
    """Switch every kernel (unpack, grouped expert GEMM, GEMV) to another weight format (sc_format.Format)."""
    F.DEFAULT = fmt


def pad_rows(M):
    return -(-M // TM) * TM


def unpack_i8(codes, bcodes, N, K, deps=()):
    k = _kernel("unpack_i8", ["codes", "bcodes"], ["wq"], _UNPACK, len(deps))
    return k(inputs=[codes, bcodes, *deps], template=[("K", K)], grid=(N * K // 8, 1, 1), threadgroup=(256, 1, 1),
             output_shapes=[(N, K)], output_dtypes=[mx.int8])[0]


def mm_i8_i32(xq, wq, deps=(), tile=None):
    """INT8 [M, K] x INT8 [N, K]^T -> INT32 [M, N]; tile = (TM, TN, NSG, GROUP_M), default the config of 03_layer."""
    (M, K), N = xq.shape, wq.shape[0]
    tm, tn, nsg, gm = tile or (TM, TN, NSG, GROUP_M)
    assert M % tm == 0 and N % tn == 0, (M, N, tm, tn)
    k = _kernel("mm_i8_i32", ["xq", "wq"], ["acc"], _MM, len(deps))
    return k(inputs=[xq, wq, *deps], template=[("M", M), ("N", N), ("K", K), ("BM", tm), ("BN", tn), ("SG", nsg),
                                                ("GM", gm)],
             grid=(N // tn * 32 * nsg, M // tm, 1), threadgroup=(32 * nsg, 1, 1),
             output_shapes=[(M, N)], output_dtypes=[mx.int32])[0]


def mm_epi(xq, wq, xscale, wscale, epi, dtype, res=None, deps=(), tile=None):
    """INT8 GEMM + in-register epilogue (EPI 1: scaled BF16, 2: + residual, 3: swiglu of interleaved gate/up)."""
    (M, K), N = xq.shape, wq.shape[0]
    tm, tn, nsg, gm = tile or (TM, TN, NSG, GROUP_M)
    assert M % tm == 0 and N % tn == 0 and (epi != 2 or res.shape == (M, N)), (M, N, tm, tn)
    k = _kernel("mm_epi", ["xq", "wq", "xscale", "wscale", "res"], ["out"], _MM_EPI, len(deps))
    ins = [xq, wq, xscale, wscale, res if epi == 2 else _dummy(dtype)]  # res is only read by EPI 2
    return k(inputs=ins + list(deps), template=[("T", dtype), ("EPI", epi), ("M", M), ("N", N), ("K", K), ("BM", tm),
                                                 ("BN", tn), ("SG", nsg), ("GM", gm)],
             grid=(N // tn * 32 * nsg, M // tm, 1), threadgroup=(32 * nsg, 1, 1),
             output_shapes=[(M, N // 2 if epi == 3 else N)], output_dtypes=[dtype])[0]


_dummies = {}


def _dummy(dtype):
    if dtype not in _dummies:
        _dummies[dtype] = mx.zeros((4,), dtype)
    return _dummies[dtype]


def interleave_gate_up(F):
    """Row permutation of [gate (F rows); up (F rows)] into groups g_2p, g_2p+1, u_2p, u_2p+1 for mm_epi EPI 3."""
    import numpy as np
    p = np.arange(F // 2)
    return np.stack([2 * p, 2 * p + 1, F + 2 * p, F + 2 * p + 1], 1).reshape(-1)


def rmsnorm_quant(x, w, eps, Mp=None, deps=()):
    """x [M, K] -> (xq int8 [Mp, K], xscale f32 [Mp]) of rms_norm(x, w, eps); K/4 threads per row (K <= 4096)."""
    M, K = x.shape
    Mp = Mp or pad_rows(M)
    k = _kernel("rmsnorm_quant", ["x", "nw", "eps"], ["xq", "xs"], _RMSNORM_QUANT, len(deps))
    return k(inputs=[x, w, eps, *deps], template=[("T", x.dtype), ("K", K), ("M", M)], grid=(Mp * K // 4, 1, 1),
             threadgroup=(K // 4, 1, 1), output_shapes=[(Mp, K), (Mp,)], output_dtypes=[mx.int8, mx.float32])


def res_norm_quant(acc, xscale, wscale, res, w, eps, deps=()):
    """h = res + bf16(acc * xscale * wscale); returns (h [Mp, K], xq, xscale') of rms_norm(h)."""
    Mp, K = acc.shape
    assert res.shape == (Mp, K)
    k = _kernel("res_norm_quant", ["acc", "xscale", "wscale", "res", "nw", "eps"], ["h", "xq", "xs"],
                _RES_NORM_QUANT, len(deps))
    return k(inputs=[acc, xscale, wscale, res, w, eps, *deps], template=[("T", res.dtype), ("K", K)],
             grid=(Mp * K // 4, 1, 1), threadgroup=(K // 4, 1, 1), output_shapes=[(Mp, K), (Mp, K), (Mp,)],
             output_dtypes=[res.dtype, mx.int8, mx.float32])


def res_epilogue(acc, xscale, wscale, res, deps=()):
    Mp, N = acc.shape
    k = _kernel("res_epilogue", ["acc", "xscale", "wscale", "res"], ["h"], _RES_EPILOGUE, len(deps))
    return k(inputs=[acc, xscale, wscale, res, *deps], template=[("T", res.dtype), ("N", N)],
             grid=(Mp * N // 4, 1, 1), threadgroup=(256, 1, 1), output_shapes=[(Mp, N)], output_dtypes=[res.dtype])[0]


def act_quant(x, Mp=None, nt=256, deps=()):
    M, K = x.shape
    Mp = Mp or pad_rows(M)
    nt = min(nt, K // 4)
    while (K // 4) % nt or nt % 32:  # threads = whole simdgroups dividing K / 4 (K = 2560: 160)
        nt -= 1
    assert K % (4 * nt) == 0
    k = _kernel("act_quant", ["x"], ["xq", "xs"], _ACT_QUANT, len(deps))
    return k(inputs=[x, *deps], template=[("K", K), ("M", M), ("NT", nt)], grid=(Mp * nt, 1, 1),
             threadgroup=(nt, 1, 1), output_shapes=[(Mp, K), (Mp,)], output_dtypes=[mx.int8, mx.float32])


def swiglu_quant(acc, xscale, wscale, dtype, nt=512, deps=()):
    """acc [Mp, 2F] (gate | up) -> (xq int8 [Mp, F], xscale f32 [Mp]) of swiglu(bf16 gate, bf16 up)."""
    Mp, F2 = acc.shape
    F = F2 // 2
    nt = min(nt, F // 4)
    assert F % (4 * nt) == 0
    k = _kernel("swiglu_quant", ["acc", "xscale", "wscale"], ["xq", "xs"], _SWIGLU_QUANT, len(deps))
    return k(inputs=[acc, xscale, wscale, *deps], template=[("T", dtype), ("F", F), ("NT", nt)],
             grid=(Mp * nt, 1, 1), threadgroup=(nt, 1, 1), output_shapes=[(Mp, F), (Mp,)],
             output_dtypes=[mx.int8, mx.float32])


def qkv_epilogue_headnorm(acc, xscale, wscale, qn, kn, eps, L, hq, hk, d, dtype, deps=()):
    """Qwen3: acc [Mp, (hq + 2 hk) d] -> q [hq, L, d], k [hk, L, d], v [hk, L, d] (q/k per-head RMSNorm)."""
    Mp = acc.shape[0]
    assert d == 128, "one simdgroup x 4 values per lane per head"
    k = _kernel("qkv_headnorm", ["acc", "xscale", "wscale", "qn", "kn", "eps"], ["q", "k", "v"], _QKV_HEADNORM,
                len(deps))
    return k(inputs=[acc, xscale, wscale, qn, kn, eps, *deps],
             template=[("T", dtype), ("L", L), ("HQ", hq), ("HK", hk), ("D", d)],
             grid=(Mp * 32 * (hq + 2 * hk), 1, 1), threadgroup=(32 * (hq + 2 * hk), 1, 1),
             output_shapes=[(hq, L, d), (hk, L, d), (hk, L, d)], output_dtypes=[dtype] * 3)


def qkv_headnorm_y(y, qn, kn, eps, L, hq, hk, d, deps=()):
    """Qwen3: BF16 y [Mp, (hq + 2 hk) d] -> q [hq, L, d], k [hk, L, d], v [hk, L, d] (q/k per-head RMSNorm)."""
    Mp = y.shape[0]
    assert d == 128
    k = _kernel("qkv_headnorm_y", ["y", "qn", "kn", "eps"], ["q", "k", "v"], _QKV_HEADNORM_Y, len(deps))
    return k(inputs=[y, qn, kn, eps, *deps], template=[("T", y.dtype), ("L", L), ("HQ", hq), ("HK", hk), ("D", d)],
             grid=(Mp * 32 * (hq + 2 * hk), 1, 1), threadgroup=(32 * (hq + 2 * hk), 1, 1),
             output_shapes=[(hq, L, d), (hk, L, d), (hk, L, d)], output_dtypes=[y.dtype] * 3)


def rmsnorm_quant_n(x, w, eps, Mp=None, deps=()):
    """rmsnorm_quant that also returns the normalized BF16 rows: (xq, xscale, xn [Mp, K])."""
    M, K = x.shape
    Mp = Mp or pad_rows(M)
    k = _kernel("rmsnorm_quant_n", ["x", "nw", "eps"], ["xq", "xs", "xn"], _RMSNORM_QUANT_N, len(deps))
    return k(inputs=[x, w, eps, *deps], template=[("T", x.dtype), ("K", K), ("M", M)], grid=(Mp * K // 4, 1, 1),
             threadgroup=(K // 4, 1, 1), output_shapes=[(Mp, K), (Mp,), (Mp, K)],
             output_dtypes=[mx.int8, mx.float32, x.dtype])


def qkv_fullnorm_y(y, qn, kn, eps, L, h, d, deps=()):
    """OLMoE: BF16 y [Mp, 3 h d] -> q, k, v [h, L, d] with RMSNorm over the full h*d-wide q and k."""
    Mp = y.shape[0]
    W = h * d
    assert W % 4 == 0 and W // 4 <= 1024 and d % 4 == 0
    k = _kernel("qkv_fullnorm_y", ["y", "qn", "kn", "eps"], ["q", "k", "v"], _QKV_FULLNORM_Y, len(deps))
    return k(inputs=[y, qn, kn, eps, *deps], template=[("T", y.dtype), ("L", L), ("H", h), ("D", d)],
             grid=(Mp * W // 4, 1, 1), threadgroup=(W // 4, 1, 1),
             output_shapes=[(h, L, d)] * 3, output_dtypes=[y.dtype] * 3)
