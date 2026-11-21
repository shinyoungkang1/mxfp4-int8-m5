// Metal 4.1 tensor-API GEMMs testing native FP4/FP8 matmul2d operand types on macOS 27.2.
// Same tiling and memory layout as matmul.metal: C[M,N] = A[M,K] x B^T with A row-major M x K
// and B row-major N x K (K-contiguous). Each threadgroup computes one TM x TN output tile.
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
#define BK 0
#endif
#ifndef GROUP_M
#define GROUP_M 1
#endif
#ifndef ENABLE_SCALED
#define ENABLE_SCALED 0
#endif

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

template <typename TA, typename TB, typename TOut, typename PA, typename PB>
inline void tile_mm(PA A, PB B, device TOut *C, uint3 dims, uint2 tgid, uint2 grid)
{
    const int M = dims.x, N = dims.y, K = dims.z;
    tensor<device TA, dextents<int32_t, 2>, tensor_inline> tA(A, dextents<int32_t, 2>(K, M));
    tensor<device TB, dextents<int32_t, 2>, tensor_inline> tB(B, dextents<int32_t, 2>(K, N));
    tensor<device TOut, dextents<int32_t, 2>, tensor_inline> tC(C, dextents<int32_t, 2>(N, M));

    const uint2 t = tile_coords(tgid, grid);
    const int m0 = t.x * TM, n0 = t.y * TN;
#if BK == 0
    constexpr auto desc = matmul2d_descriptor(TM, TN, dynamic_length_v<int>, false, true, false,
                                              matmul2d_descriptor::mode::multiply);
    matmul2d<desc, execution_simdgroups<NSG>> op;
    auto mA = tA.slice(0, m0);
    auto mB = tB.slice(0, n0);
    auto mC = tC.slice(n0, m0);
    op.run(mA, mB, mC);
#else
    constexpr auto desc = matmul2d_descriptor(TM, TN, BK, false, true, false,
                                              matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc, execution_simdgroups<NSG>> op;
    auto a0 = tA.template slice<BK, TM>(0, m0);
    auto b0 = tB.template slice<BK, TN>(0, n0);
    auto acc = op.get_destination_cooperative_tensor<decltype(a0), decltype(b0), TOut>();
#pragma unroll
    for (uint16_t i = 0; i < acc.get_capacity(); ++i)
        if (acc.is_valid_element(i))
            acc[i] = 0;
    for (int k = 0; k < K; k += BK) {
        auto a = tA.template slice<BK, TM>(k, m0);
        auto b = tB.template slice<BK, TN>(k, n0);
        op.run(a, b, acc);
    }
    auto mC = tC.template slice<TN, TM>(n0, m0);
    acc.store(mC);
#endif
}

#define KERNEL_GEN(name, TA, TB, TO, PA, PB)                                             \
    kernel void name(PA A [[buffer(0)]], PB B [[buffer(1)]],                             \
                     device TO *C [[buffer(2)]], constant uint3 &d [[buffer(3)]],         \
                     uint2 tgid [[threadgroup_position_in_grid]],                         \
                     uint2 grid [[threadgroups_per_grid]])                                \
    { tile_mm<TA, TB, TO>(A, B, C, d, tgid, grid); }

// Anchors matching kernels/matmul.metal
KERNEL_GEN(mm_f16_f32, half, half, float, device half *, device half *)
KERNEL_GEN(mm_i8_i32, int8_t, int8_t, int32_t, device int8_t *, device int8_t *)
KERNEL_GEN(mm_i8xi4_i32, int8_t, int4b_format, int32_t, device int8_t *, device uchar *)

// macOS 27.2 FP4 / FP8 matmul2d combinations
#if defined(__HAVE_METAL_FP4_E2M1_FORMAT_TYPE__)
KERNEL_GEN(mm_fp4_f32, metal_fp4_e2m1_format, metal_fp4_e2m1_format, float, device uchar *, device uchar *)
KERNEL_GEN(mm_fp4_f16, metal_fp4_e2m1_format, metal_fp4_e2m1_format, half, device uchar *, device uchar *)
KERNEL_GEN(mm_f16xfp4_f32, half, metal_fp4_e2m1_format, float, device half *, device uchar *)
KERNEL_GEN(mm_f16xfp4_f16, half, metal_fp4_e2m1_format, half, device half *, device uchar *)
#endif

#if defined(__HAVE_METAL_FP8_E4M3_FORMAT_TYPE__)
KERNEL_GEN(mm_fp8e4m3_f32, metal_fp8_e4m3_format, metal_fp8_e4m3_format, float, device uchar *, device uchar *)
KERNEL_GEN(mm_fp8e4m3_f16, metal_fp8_e4m3_format, metal_fp8_e4m3_format, half, device uchar *, device uchar *)
KERNEL_GEN(mm_f16xfp8e4m3_f32, half, metal_fp8_e4m3_format, float, device half *, device uchar *)
#endif

#if defined(__HAVE_METAL_FP8_E5M2_FORMAT_TYPE__)
KERNEL_GEN(mm_fp8e5m2_f32, metal_fp8_e5m2_format, metal_fp8_e5m2_format, float, device uchar *, device uchar *)
KERNEL_GEN(mm_fp8e5m2_f16, metal_fp8_e5m2_format, metal_fp8_e5m2_format, half, device uchar *, device uchar *)
KERNEL_GEN(mm_f16xfp8e5m2_f32, half, metal_fp8_e5m2_format, float, device half *, device uchar *)
#endif

#if ENABLE_SCALED && defined(__HAVE_TENSOR_MULTIPLANE__) && defined(__HAVE_METAL_FP4_E2M1_FORMAT_TYPE__)
using scale_plane_t = tensor_blockwise<tensor_plane_scales, device metal_fp8_ue8m0_format, 32, 1>;

kernel void mm_f16xmxfp4_f32(device half *A [[buffer(0)]], device uchar *B [[buffer(1)]],
                             device float *C [[buffer(2)]], constant uint3 &d [[buffer(3)]],
                             device uchar *SB [[buffer(4)]],
                             uint2 tgid [[threadgroup_position_in_grid]],
                             uint2 grid [[threadgroups_per_grid]])
{
    const int M = d.x, N = d.y, K = d.z;
    scale_plane_t sb(SB);
    tensor<device half, dextents<int32_t, 2>, tensor_inline> tA(A, dextents<int32_t, 2>(K, M));
    tensor<device metal_fp4_e2m1_format, dextents<int32_t, 2>, tensor_inline, scale_plane_t>
        tB(B, dextents<int32_t, 2>(K, N), sb);
    tensor<device float, dextents<int32_t, 2>, tensor_inline> tC(C, dextents<int32_t, 2>(N, M));
    const uint2 t = tile_coords(tgid, grid);
    constexpr auto desc = matmul2d_descriptor(TM, TN, dynamic_length_v<int>, false, true, false,
                                              matmul2d_descriptor::mode::multiply);
    matmul2d<desc, execution_simdgroups<NSG>> op;
    auto mA = tA.slice(0, int(t.x) * TM);
    auto mB = tB.slice(0, int(t.y) * TN);
    auto mC = tC.slice(int(t.y) * TN, int(t.x) * TM);
    op.run(mA, mB, mC);
}

kernel void mm_mxfp4_f32(device uchar *A [[buffer(0)]], device uchar *B [[buffer(1)]],
                         device float *C [[buffer(2)]], constant uint3 &d [[buffer(3)]],
                         device uchar *SB [[buffer(4)]], device uchar *SA [[buffer(5)]],
                         uint2 tgid [[threadgroup_position_in_grid]],
                         uint2 grid [[threadgroups_per_grid]])
{
    const int M = d.x, N = d.y, K = d.z;
    scale_plane_t sa(SA), sb(SB);
    tensor<device metal_fp4_e2m1_format, dextents<int32_t, 2>, tensor_inline, scale_plane_t>
        tA(A, dextents<int32_t, 2>(K, M), sa);
    tensor<device metal_fp4_e2m1_format, dextents<int32_t, 2>, tensor_inline, scale_plane_t>
        tB(B, dextents<int32_t, 2>(K, N), sb);
    tensor<device float, dextents<int32_t, 2>, tensor_inline> tC(C, dextents<int32_t, 2>(N, M));
    const uint2 t = tile_coords(tgid, grid);
    constexpr auto desc = matmul2d_descriptor(TM, TN, dynamic_length_v<int>, false, true, false,
                                              matmul2d_descriptor::mode::multiply);
    matmul2d<desc, execution_simdgroups<NSG>> op;
    auto mA = tA.slice(0, int(t.x) * TM);
    auto mB = tB.slice(0, int(t.y) * TN);
    auto mC = tC.slice(int(t.y) * TN, int(t.x) * TM);
    op.run(mA, mB, mC);
}
#endif
