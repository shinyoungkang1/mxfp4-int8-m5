// SC-MXFP4 weight unpack with 2-bit block shifts (4.07 bits per weight instead of 4.26).
//
// Same as unpack_w_i8 in w4a8.metal, except the per-32-block shift e_b - E_lo (always 0..3 for
// spread-constrained MXFP4) is packed four to a byte: block j of a row lives in bits 2*(j%4) of byte j/4.
// Layout of codes is unchanged (two 4-bit codes per byte, even k in the low nibble).
#include <metal_stdlib>
using namespace metal;

#define MXBLOCK 32
typedef vec<char, 8> char8_t;

// Each thread expands 8 weights (one uint of codes; all inside one 32-block) into INT8 scratch.
// value(idx) = (0xC8643210 >> 4*idx) & 0xF  encodes LUT 2v = {0,1,2,3,4,6,8,12}.
kernel void unpack_w_i8_p2(device const uint *codes [[buffer(0)]], device const uchar *shifts2 [[buffer(1)]],
                           device char8_t *out [[buffer(2)]], constant uint3 &d [[buffer(3)]],
                           uint i [[thread_position_in_grid]])
{
    const uint K = d.z;
    const uint word = codes[i];
    const uint e = i * 8, n = e / K, k = e % K;
    const uint blk = k / MXBLOCK;
    const uint sh = (shifts2[n * (K / (4 * MXBLOCK)) + blk / 4] >> (2 * (blk % 4))) & 3;
    char8_t v;
#pragma unroll
    for (uint j = 0; j < 8; ++j) {
        const uint c = (word >> (4 * j)) & 0xF;
        const int mag = int((0xC8643210u >> (4 * (c & 7))) & 0xF) << sh;
        v[j] = char((c & 8) ? -mag : mag);
    }
    out[i] = v;
}
