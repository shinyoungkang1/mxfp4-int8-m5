// SC-MXFP4 linear layer on the M5 Neural Accelerators.
//
// act_quant:  x f16 [M, K] -> xq int8 [M, K], xscale f32 [M]  (symmetric per-token INT8)
// mx_w4a8:    y[m, n] = xscale[m] * wscale[n] * sum_k xq[m, k] * q[n, k]   (INT8 x INT8 -> INT32)
// mx_w4a16:   y[m, n] = wscale[n] * sum_k x[m, k] * q[n, k]               (FP16 x FP16 -> FP32)
//
// Both GEMMs read the same packed 4-bit weights (see sim/make_testcase.py for the layout) and, per BK
// chunk of K, unpack the TN x BK weight tile into threadgroup memory as exact integers
// q = ±LUT[idx] << shift (|q| <= 96), as INT8 for W4A8 or FP16 for W4A16. The only difference between the
// two is the operand type the Neural Accelerators see.
#include <metal_stdlib>
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;
using namespace mpp::tensor_ops;

#ifndef TM
#define TM 128
#endif
#ifndef TN
#define TN 64
#endif
#ifndef NSG
#define NSG 4
#endif
#ifndef BK
#define BK 128
#endif
#ifndef GROUP_M
#define GROUP_M 8
#endif
#define NT (32 * NSG)
#define MXBLOCK 32

constant char LUT2V[8] = {0, 1, 2, 3, 4, 6, 8, 12};

inline uint2 tile_coords(uint2 tgid, uint2 grid)
{
    if (GROUP_M <= 1)
        return uint2(tgid.y, tgid.x);
    const uint num_m = grid.y, num_n = grid.x;
    const uint pid = tgid.y * num_n + tgid.x;
    const uint per_group = GROUP_M * num_n;
    const uint first_m = (pid / per_group) * GROUP_M;
    const uint gsz = min(num_m - first_m, uint(GROUP_M));
    const uint r = pid % per_group;
    return uint2(first_m + r % gsz, r / gsz);
}

// ---- activation quantization: one threadgroup per token row ------------------------------------
kernel void act_quant(device const half *x [[buffer(0)]], device char *xq [[buffer(1)]],
                      device float *xscale [[buffer(2)]], constant uint3 &d [[buffer(3)]],
                      uint row [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],
                      uint ntg [[threads_per_threadgroup]], uint lane [[thread_index_in_simdgroup]],
                      uint sg [[simdgroup_index_in_threadgroup]])
{
    const uint K = d.z;
    device const half *xr = x + size_t(row) * K;
    float amax = 0;
    for (uint k = tid; k < K; k += ntg)
        amax = max(amax, abs(float(xr[k])));
    amax = simd_max(amax);
    threadgroup float part[32];
    if (lane == 0)
        part[sg] = amax;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    amax = lane < (ntg + 31) / 32 ? part[lane] : 0;
    amax = simd_max(amax);
    const float s = amax / 127.0f;
    const float inv = s > 0 ? 1.0f / s : 0.0f;
    device char *qr = xq + size_t(row) * K;
    for (uint k = tid; k < K; k += ntg)
        qr[k] = char(clamp(rint(float(xr[k]) * inv), -127.0f, 127.0f));
    if (tid == 0)
        xscale[row] = s;
}

// ---- unpack one TN x BK weight tile into threadgroup memory ------------------------------------
template <typename T>
inline void unpack_tile(threadgroup T *sB, device const uchar *codes, device const uchar *shifts,
                        uint n0, uint k, uint K, uint tid)
{
    constexpr uint BYTES = TN * BK / 2;
    for (uint j = tid; j < BYTES; j += NT) {
        const uint r = j / (BK / 2), c = j % (BK / 2);
        const uint n = n0 + r, kk = k + 2 * c;
        const uchar byte = codes[size_t(n) * (K / 2) + kk / 2];
        const uchar sh = shifts[size_t(n) * (K / MXBLOCK) + kk / MXBLOCK];  // both nibbles share a block
        const uchar lo = byte & 0xF, hi = byte >> 4;
        const int q0 = (int(LUT2V[lo & 7]) << sh) * ((lo & 8) ? -1 : 1);
        const int q1 = (int(LUT2V[hi & 7]) << sh) * ((hi & 8) ? -1 : 1);
        sB[r * BK + 2 * c] = T(q0);
        sB[r * BK + 2 * c + 1] = T(q1);
    }
}

template <typename TA, typename TAcc>
inline void mx_gemm(device const TA *A, device const uchar *codes, device const uchar *shifts,
                    device const float *wscale, device const float *xscale, device float *Y,
                    uint3 d, uint2 tgid, uint2 grid, uint tid, threadgroup TA *sB)
{
    const int M = d.x, N = d.y, K = d.z;
    const uint2 t = tile_coords(tgid, grid);
    const int m0 = t.x * TM, n0 = t.y * TN;

    tensor<device TA, dextents<int32_t, 2>, tensor_inline> tA((device TA *)A, dextents<int32_t, 2>(K, M));
    tensor<threadgroup TA, dextents<int32_t, 2>, tensor_inline> tB(sB, dextents<int32_t, 2>(BK, TN));

    constexpr auto desc = matmul2d_descriptor(TM, TN, BK, false, true, false,
                                              matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc, execution_simdgroups<NSG>> op;
    auto a0 = tA.template slice<BK, TM>(0, m0);
    auto acc = op.template get_destination_cooperative_tensor<decltype(a0), decltype(tB), TAcc>();
#pragma unroll
    for (uint16_t i = 0; i < acc.get_capacity(); ++i)
        if (acc.is_valid_element(i))
            acc[i] = 0;

    for (int k = 0; k < K; k += BK) {
        unpack_tile(sB, codes, shifts, n0, k, K, tid);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        auto a = tA.template slice<BK, TM>(k, m0);
        op.run(a, tB, acc);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Epilogue: per-token x per-row scales, written as FP32.
    for (uint16_t i = 0; i < acc.get_capacity(); ++i) {
        if (!acc.is_valid_element(i))
            continue;
        auto idx = acc.get_multidimensional_index(i);  // (n, m) within the tile
        const int n = n0 + idx[0], m = m0 + idx[1];
        const float xs = xscale ? xscale[m] : 1.0f;
        Y[size_t(m) * N + n] = float(acc[i]) * xs * wscale[n];
    }
}

kernel void mx_w4a8(device const char *xq [[buffer(0)]], device const uchar *codes [[buffer(1)]],
                    device const uchar *shifts [[buffer(2)]], device const float *wscale [[buffer(3)]],
                    device const float *xscale [[buffer(4)]], device float *Y [[buffer(5)]],
                    constant uint3 &d [[buffer(6)]], uint2 tgid [[threadgroup_position_in_grid]],
                    uint2 grid [[threadgroups_per_grid]], uint tid [[thread_index_in_threadgroup]])
{
    threadgroup int8_t sB[TN * BK];
    mx_gemm<int8_t, int32_t>((device const int8_t *)xq, codes, shifts, wscale, xscale, Y, d, tgid, grid, tid, sB);
}

kernel void mx_w4a16(device const half *x [[buffer(0)]], device const uchar *codes [[buffer(1)]],
                     device const uchar *shifts [[buffer(2)]], device const float *wscale [[buffer(3)]],
                     device const float *unused [[buffer(4)]], device float *Y [[buffer(5)]],
                     constant uint3 &d [[buffer(6)]], uint2 tgid [[threadgroup_position_in_grid]],
                     uint2 grid [[threadgroups_per_grid]], uint tid [[thread_index_in_threadgroup]])
{
    threadgroup half sB[TN * BK];
    mx_gemm<half, float>(x, codes, shifts, wscale, nullptr, Y, d, tgid, grid, tid, sB);
}

// ---- two-pass path: unpack the whole weight matrix once, then GEMM from device memory -------------

// Each thread expands 8 weights (4 code bytes; all inside one 32-block) into INT8 or FP16 scratch.
// LUT2V packed as nibbles: value(idx) = (0xC8643210 >> 4*idx) & 0xF.
template <typename T, typename V8>
inline void unpack8(device const uint *codes, device const uchar *shifts, device V8 *out, uint i, uint K)
{
    const uint word = codes[i];             // 8 codes, element 0 in the low nibble
    const uint e = i * 8;                    // flat element index n*K + k
    const uint n = e / K, k = e % K;
    const uint sh = shifts[n * (K / MXBLOCK) + k / MXBLOCK];
    V8 v;
#pragma unroll
    for (uint j = 0; j < 8; ++j) {
        const uint c = (word >> (4 * j)) & 0xF;
        const int mag = int((0xC8643210u >> (4 * (c & 7))) & 0xF) << sh;
        v[j] = T((c & 8) ? -mag : mag);
    }
    out[i] = v;
}

typedef vec<char, 8> char8_t;
typedef vec<half, 8> half8_t;
kernel void unpack_w_i8(device const uint *codes [[buffer(0)]], device const uchar *shifts [[buffer(1)]],
                        device char8_t *out [[buffer(2)]], constant uint3 &d [[buffer(3)]], uint i [[thread_position_in_grid]])
{ unpack8<char, char8_t>(codes, shifts, out, i, d.z); }
kernel void unpack_w_f16(device const uint *codes [[buffer(0)]], device const uchar *shifts [[buffer(1)]],
                         device half8_t *out [[buffer(2)]], constant uint3 &d [[buffer(3)]], uint i [[thread_position_in_grid]])
{ unpack8<half, half8_t>(codes, shifts, out, i, d.z); }

// Single matmul2d over all of K from device memory, scales applied in the epilogue.
template <typename TA, typename TAcc>
inline void gemm_scaled(device const TA *A, device const TA *B, device const float *wscale, device const float *xscale,
                        device float *Y, uint3 d, uint2 tgid, uint2 grid)
{
    const int M = d.x, N = d.y, K = d.z;
    const uint2 t = tile_coords(tgid, grid);
    const int m0 = t.x * TM, n0 = t.y * TN;
    tensor<device TA, dextents<int32_t, 2>, tensor_inline> tA((device TA *)A, dextents<int32_t, 2>(K, M));
    tensor<device TA, dextents<int32_t, 2>, tensor_inline> tB((device TA *)B, dextents<int32_t, 2>(K, N));
    constexpr auto desc = matmul2d_descriptor(TM, TN, dynamic_length_v<int>, false, true, false,
                                              matmul2d_descriptor::mode::multiply);
    matmul2d<desc, execution_simdgroups<NSG>> op;
    auto mA = tA.slice(0, m0);
    auto mB = tB.slice(0, n0);
    auto acc = op.template get_destination_cooperative_tensor<decltype(mA), decltype(mB), TAcc>();
    op.run(mA, mB, acc);
    for (uint16_t i = 0; i < acc.get_capacity(); ++i) {
        if (!acc.is_valid_element(i))
            continue;
        auto idx = acc.get_multidimensional_index(i);
        const int n = n0 + idx[0], m = m0 + idx[1];
        Y[size_t(m) * N + n] = float(acc[i]) * (xscale ? xscale[m] : 1.0f) * wscale[n];
    }
}

kernel void gemm_i8_scaled(device const char *xq [[buffer(0)]], device const char *wq [[buffer(1)]],
                           device const float *wscale [[buffer(2)]], device const float *xscale [[buffer(3)]],
                           device float *Y [[buffer(4)]], constant uint3 &d [[buffer(5)]],
                           uint2 tgid [[threadgroup_position_in_grid]], uint2 grid [[threadgroups_per_grid]])
{ gemm_scaled<int8_t, int32_t>((device const int8_t *)xq, (device const int8_t *)wq, wscale, xscale, Y, d, tgid, grid); }

kernel void gemm_f16_scaled(device const half *x [[buffer(0)]], device const half *wq [[buffer(1)]],
                            device const float *wscale [[buffer(2)]], device const float *unused [[buffer(3)]],
                            device float *Y [[buffer(4)]], constant uint3 &d [[buffer(5)]],
                            uint2 tgid [[threadgroup_position_in_grid]], uint2 grid [[threadgroups_per_grid]])
{ gemm_scaled<half, float>(x, wq, wscale, nullptr, Y, d, tgid, grid); }

// Separate epilogue for the plain INT32-output GEMM: y = acc * xscale[m] * wscale[n], 4 outputs per thread.
kernel void scale_epilogue(device const int4 *acc [[buffer(0)]], device const float *xscale [[buffer(1)]],
                           device const float *wscale [[buffer(2)]], device float4 *Y [[buffer(3)]],
                           constant uint3 &d [[buffer(4)]], uint i [[thread_position_in_grid]])
{
    const uint N = d.y, e = i * 4, m = e / N, n = e % N;
    const float4 ws = *(device const float4 *)(wscale + n);
    Y[i] = float4(acc[i]) * xscale[m] * ws;
}
