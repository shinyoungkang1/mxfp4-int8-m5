// Metal 4 tensor-API GEMMs routed through the M5 GPU Neural Accelerators.
// C[M,N] = A[M,K] x B^T with A row-major M x K and B row-major N x K (weights K-contiguous, the
// usual nn.Linear layout). M, N, K must be multiples of TM, TN, BK. Each threadgroup computes one
// TM x TN output tile; operands stream straight from device memory.
//
// BK = 0: one matmul2d call over the whole K (the op picks its own K tiling).
// BK > 0: explicit K loop in BK chunks, accumulating in a register-resident cooperative tensor.
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
#define GROUP_M 1  // >1: grouped tile order so neighbouring threadgroups share B tiles in cache
#endif

// Map the linear threadgroup id to an output tile (pid_m, pid_n). With GROUP_M > 1, consecutive ids
// sweep GROUP_M tile-rows before moving to the next tile-column (Triton-style swizzle).
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

// PB is the B pointer type: TB itself, or `device uchar *` for packed 4-bit (int4b_format).
template <typename TA, typename TB, typename TOut, typename PB>
inline void tile_mm(device TA *A, PB B, device TOut *C, uint3 dims, uint2 tgid, uint2 grid)
{
    const int M = dims.x, N = dims.y, K = dims.z;
    // Extents are (innermost, outer): a row-major M x K matrix is (K, M).
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

#define KERNEL(name, TA, TB, TO)                                                         \
    kernel void name(device TA *A [[buffer(0)]], device TB *B [[buffer(1)]],              \
                     device TO *C [[buffer(2)]], constant uint3 &d [[buffer(3)]],         \
                     uint2 tgid [[threadgroup_position_in_grid]],                         \
                     uint2 grid [[threadgroups_per_grid]])                                \
    { tile_mm<TA, TB, TO>(A, B, C, d, tgid, grid); }

// B packed two 4-bit values per byte, even element in the low nibble.
#define KERNEL4(name, TA, TO)                                                            \
    kernel void name(device TA *A [[buffer(0)]], device uchar *B [[buffer(1)]],           \
                     device TO *C [[buffer(2)]], constant uint3 &d [[buffer(3)]],         \
                     uint2 tgid [[threadgroup_position_in_grid]],                         \
                     uint2 grid [[threadgroups_per_grid]])                                \
    { tile_mm<TA, int4b_format, TO>(A, B, C, d, tgid, grid); }

KERNEL(mm_f16_f16, half, half, half)          // FP16, FP16 out
KERNEL(mm_f16_f32, half, half, float)         // FP16, FP32 out (baseline)
KERNEL(mm_i8_i32, int8_t, int8_t, int32_t)    // W8A8
KERNEL(mm_f16xi8_f32, half, int8_t, float)    // W8A16
KERNEL4(mm_i8xi4_i32, int8_t, int32_t)        // W4A8 (INT4 weights)
KERNEL4(mm_f16xi4_f32, half, float)           // W4A16 (INT4 weights)
