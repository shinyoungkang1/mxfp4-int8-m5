"""SC-MXFP4 W4A8 linear layers inside MLX, via mx.fast.metal_kernel (Metal 4 tensor ops compile there).

The two-pass pipeline of kernels/w4a8.metal ("W4A8 2-pass+sep" in results/03_layer.md), per linear call:
  act_quant       x [M, K] (bf16/f16) -> xq int8 [M, K], xscale f32 [M]   (symmetric per-token INT8)
  unpack_w_i8     packed 4.26-bit SC-MXFP4 weights -> INT8 scratch [N, K]   (exact: q = ±LUT[idx] << shift)
  mm_i8_i32       INT8 x INT8 -> INT32 matmul2d on the Neural Accelerators (kernels/matmul.metal, BK = 0)
  scale_epilogue  y = acc * xscale[m] * wscale[n], written in the model dtype

Weight layout is the one written by sim/make_testcase.py (codes [N, K/2] two nibbles per byte, shifts [N, K/32],
wscale [N]); packing reuses sim/make_testcase.pack, so it is the same SC-MXFP4 quantizer that
sim/eval_ppl.py's sc_mxfp4_w4a8 fake-quantizes with.
"""
import os
import sys

import mlx.core as mx
import mlx.nn as nn
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "sim"))

LINEARS = ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
           "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]
TM, TN, NSG, GROUP_M = 128, 64, 4, 8  # the tile config of results/03_layer.md

_HEADER = """
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace mpp::tensor_ops;
typedef vec<char, 8> char8_t;
"""

_ACT_QUANT = """
    const uint row = threadgroup_position_in_grid.x;
    const uint tid = thread_index_in_threadgroup, ntg = threads_per_threadgroup.x;
    const uint lane = thread_index_in_simdgroup, sg = simdgroup_index_in_threadgroup;
    device const T *xr = x + size_t(row) * K;
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
    device int8_t *qr = xq + size_t(row) * K;
    for (uint k = tid; k < K; k += ntg)
        qr[k] = int8_t(clamp(rint(float(xr[k]) * inv), -127.0f, 127.0f));
    if (tid == 0)
        xscale[row] = s;
"""

# 8 weights (one uint32 of codes, all inside one 32-block) per thread. LUT2V as nibbles: (0xC8643210 >> 4*idx) & 0xF.
_UNPACK = """
    const uint i = thread_position_in_grid.x;
    const uint word = codes[i];
    const uint e = i * 8, n = e / K, k = e % K;
    const uint sh = shifts[n * (K / 32) + k / 32];
    char8_t v;
#pragma unroll
    for (uint j = 0; j < 8; ++j) {
        const uint c = (word >> (4 * j)) & 0xF;
        const int mag = int((0xC8643210u >> (4 * (c & 7))) & 0xF) << sh;
        v[j] = char((c & 8) ? -mag : mag);
    }
    ((device char8_t *)wq)[i] = v;
"""

# kernels/matmul.metal tile_mm<int8_t, int8_t, int32_t> with BK = 0 and the GROUP_M tile swizzle.
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

_EPILOGUE = """
    const uint i = thread_position_in_grid.x;
    const uint e = i * 4, m = e / N, n = e % N;
    const float4 ws = *(device const float4 *)(wscale + n);
    const float4 y4 = float4(((device const int4 *)acc)[i]) * xscale[m] * ws;
    device T *yo = y + e;
    yo[0] = T(y4.x); yo[1] = T(y4.y); yo[2] = T(y4.z); yo[3] = T(y4.w);
"""

_k_act = mx.fast.metal_kernel(name="sc_act_quant", input_names=["x"], output_names=["xq", "xscale"],
                              source=_ACT_QUANT, header=_HEADER)
_k_unpack = mx.fast.metal_kernel(name="sc_unpack_w_i8", input_names=["codes", "shifts"], output_names=["wq"],
                                 source=_UNPACK, header=_HEADER)
_k_mm = mx.fast.metal_kernel(name="sc_mm_i8_i32", input_names=["xq", "wq"], output_names=["acc"],
                             source=_MM, header=_HEADER)
_k_epi = mx.fast.metal_kernel(name="sc_scale_epilogue", input_names=["acc", "xscale", "wscale"], output_names=["y"],
                              source=_EPILOGUE, header=_HEADER)


def act_quant(x):
    M, K = x.shape
    return _k_act(inputs=[x], template=[("T", x.dtype), ("K", K)], grid=(M * 256, 1, 1), threadgroup=(256, 1, 1),
                  output_shapes=[(M, K), (M,)], output_dtypes=[mx.int8, mx.float32])


def unpack_w_i8(codes, shifts, N, K):
    return _k_unpack(inputs=[codes, shifts], template=[("K", K)], grid=(N * K // 8, 1, 1), threadgroup=(256, 1, 1),
                     output_shapes=[(N, K)], output_dtypes=[mx.int8])[0]


def mm_i8_i32(xq, wq):
    (M, K), N = xq.shape, wq.shape[0]
    assert M % TM == 0 and N % TN == 0, (M, N)
    return _k_mm(inputs=[xq, wq], template=[("M", M), ("N", N), ("K", K), ("BM", TM), ("BN", TN), ("SG", NSG),
                                            ("GM", GROUP_M)],
                 grid=(N // TN * 32 * NSG, M // TM, 1), threadgroup=(32 * NSG, 1, 1),
                 output_shapes=[(M, N)], output_dtypes=[mx.int32])[0]


def scale_epilogue(acc, xscale, wscale, dtype):
    M, N = acc.shape
    return _k_epi(inputs=[acc, xscale, wscale], template=[("T", dtype), ("N", N)], grid=(M * N // 4, 1, 1),
                  threadgroup=(256, 1, 1), output_shapes=[(M, N)], output_dtypes=[dtype])[0]


class W4A8Linear(nn.Module):
    """Bias-free linear y = x W^T with SC-MXFP4 weights, computed as INT8 x INT8 on the Neural Accelerators.

    resident_int8=True keeps the unpacked INT8 weights in memory instead of unpacking per call (8 bits/weight;
    the W8A8 speed ceiling of results/03_layer.md), for measuring what the unpack pass costs end to end."""

    def __init__(self, codes, shifts, wscale, resident_int8=False):
        super().__init__()
        self.N, self.K = shifts.shape[0], shifts.shape[1] * 32
        self.codes = codes          # uint32 [N * K / 8]
        self.shifts = shifts        # uint8  [N, K / 32]
        self.wscale = wscale        # f32    [N]
        self.wq = unpack_w_i8(codes, shifts, self.N, self.K) if resident_int8 else None

    def int8_weight(self):
        return self.wq if self.wq is not None else unpack_w_i8(self.codes, self.shifts, self.N, self.K)

    def __call__(self, x):
        lead, K = x.shape[:-1], x.shape[-1]
        x2 = x.reshape(-1, K)
        M = x2.shape[0]
        Mp = -(-M // TM) * TM
        if Mp != M:  # zero rows quantize to xscale = 0 and produce zero outputs
            x2 = mx.pad(x2, [(0, Mp - M), (0, 0)])
        xq, xs = act_quant(x2)
        y = scale_epilogue(mm_i8_i32(xq, self.int8_weight()), xs, self.wscale, x.dtype)
        if Mp != M:
            y = y[:M]
        return y.reshape(*lead, self.N)


def pack_linear(w):
    """(N, K) float weights -> (codes uint32 [N*K/8], shifts uint8 [N, K/32], wscale f32 [N]) as numpy."""
    from make_testcase import pack
    packed, shifts, wscale, _ = pack(np.asarray(w, dtype=np.float64))
    return packed.reshape(-1).view(np.uint32).copy(), shifts, wscale


def pack_model(model, cache_path=None):
    """Packs every transformer-block linear of an mlx_lm model; returns {name: (codes, shifts, wscale)} as mx arrays.
    Cached as one safetensors file (packing in float64 numpy takes a few minutes for a 1.7B model)."""
    if cache_path and os.path.exists(cache_path):
        flat = mx.load(cache_path)
    else:
        flat = {}
        for i, layer in enumerate(model.model.layers):
            for path in LINEARS:
                lin = layer
                for p in path.split("."):
                    lin = getattr(lin, p)
                c, s, ws = pack_linear(np.array(lin.weight.astype(mx.float32)))
                key = f"layers.{i}.{path}"
                flat[key + ".codes"], flat[key + ".shifts"], flat[key + ".wscale"] = mx.array(c), mx.array(s), mx.array(ws)
            print(f"packed layer {i + 1}/{len(model.model.layers)}", flush=True)
        if cache_path:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            mx.save_safetensors(cache_path, flat)
    return {k[:-len(".codes")]: (flat[k], flat[k[:-6] + ".shifts"], flat[k[:-6] + ".wscale"])
            for k in flat if k.endswith(".codes")}


def apply_w4a8(model, packed, resident_int8=False):
    """Swaps every transformer-block linear for W4A8Linear (embeddings, norms, attention and lm_head unchanged)."""
    for i, layer in enumerate(model.model.layers):
        for path in LINEARS:
            parent_name, name = path.split(".")
            parent = getattr(layer, parent_name)
            assert "bias" not in getattr(parent, name)
            setattr(parent, name, W4A8Linear(*packed[f"layers.{i}.{path}"], resident_int8=resident_int8))
    mx.eval(model.parameters())
    return model


def load_w4a8(repo, cache_dir, resident_int8=False):
    from mlx_lm import load
    model, tok = load(repo)
    cache = os.path.join(cache_dir, repo.replace("/", "--") + "-scmxfp4.safetensors")
    return apply_w4a8(model, pack_model(model, cache), resident_int8), tok


def _self_test(seed=0):
    """Exactness on a random layer: kernel output vs float64 integer reference (make_testcase's y_ref)."""
    from make_testcase import pack
    rng = np.random.default_rng(seed)
    M, N, K = 200, 1024, 2048  # M not a multiple of 128: exercises padding
    w = rng.standard_normal((N, K)) * 0.02 * rng.lognormal(0, 0.3, (1, K))
    x = (rng.standard_normal((M, K)) * rng.lognormal(0, 0.5, (1, K))).astype(np.float16)
    packed, shifts, wscale, wq = pack(w)
    lin = W4A8Linear(mx.array(packed.reshape(-1).view(np.uint32)), mx.array(shifts), mx.array(wscale))
    assert np.array_equal(np.array(lin.int8_weight()), wq), "unpack mismatch"
    xq, xs = act_quant(mx.pad(mx.array(x), [(0, 56), (0, 0)]))
    xf = x.astype(np.float64)
    s_ref = np.abs(xf).max(1) / 127
    xq_ref = np.clip(np.rint(xf / s_ref[:, None]), -127, 127)
    nd = int((np.array(xq)[:M] != xq_ref).sum())
    acc = np.array(mm_i8_i32(xq, lin.int8_weight()))[:M].astype(np.float64)
    acc_ref = np.array(xq)[:M].astype(np.float64) @ wq.astype(np.float64).T
    assert np.array_equal(acc, acc_ref), "INT8 GEMM not exact"
    y = np.array(lin(mx.array(x).astype(mx.float32)))
    y_ref = acc_ref * np.array(xs)[:M, None] * wscale[None, :]
    print(f"self-test: unpack exact, INT32 GEMM exact on {M}x{N}x{K}, act-quant codes differing from float64: "
          f"{nd}/{M * K}, f32 output max rel err {np.abs(y - y_ref).max() / np.abs(y_ref).max():.1e}")


if __name__ == "__main__":
    _self_test()
