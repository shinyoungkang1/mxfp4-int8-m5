"""The kernels on other weight formats (sc_format.Format): unpack, grouped expert GEMM and decode GEMV against
numpy references, on random codes and block codes. Shows that a format change (block size, block multiplier table,
code magnitudes) needs only a new Format and its quantizer/packer. Run: python test_formats.py"""
import os
import sys

import mlx.core as mx
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sc_format as F  # noqa: E402
import sc_kernels as S  # noqa: E402
import sc_moe  # noqa: E402
import sc_decode as G  # noqa: E402

FORMATS = [F.Format(),                                                        # published: 32-blocks, 2^shift
           F.Format(block=16),                                                # 16-wide blocks, 2^shift
           F.Format(block=32, mult=(1, 2, 3, 4)),                            # small integer multipliers
           F.Format(block=16, mult=(1, 2, 3, 4, 5, 6, 8, 10), bc_bits=4),     # 8 integer multipliers, 4-bit codes
           F.Format(block=32, code_mag=(0, 1, 2, 3, 4, 5, 6, 7), mult=(1, 3, 9, 15))]  # non-E2M1 magnitudes (LUT path)


def main():
    rng = np.random.default_rng(0)
    ok = True
    for fmt in FORMATS:
        S.set_format(fmt)
        N, K, M = 256, 2048, 128
        c4 = rng.integers(0, 16, (N, K)).astype(np.uint8)
        bcode = rng.integers(0, len(fmt.mult), (N, K // fmt.block)).astype(np.uint8)
        codes, bc, ws = F.pack_rows(c4, bcode, np.abs(rng.standard_normal(N)) * 1e-3, fmt)
        q = F.decode_numpy(codes, bc, N, K, fmt)
        wq = np.array(S.unpack_i8(mx.array(codes), mx.array(bc), N, K))
        e1 = int((wq != q).sum())
        xq = rng.integers(-127, 128, (M, K)).astype(np.int8)
        ref = xq.astype(np.int64) @ q.astype(np.int64).T
        xs = mx.array(np.full(M, 1e-2, np.float32))
        y_ref = (mx.array(ref.astype(np.float32)) * xs[:, None] * mx.array(ws)[None]).astype(mx.bfloat16)
        # dense GEMV, all tile variants
        e2 = 0
        for tile in ((4, 16, 2), (4, 32, 2), (8, 16, 1)):
            y = G.gemv(mx.array(xq[0]), mx.array(codes), mx.array(bc), xs[:1], mx.array(ws), N, 1,
                       rows=tile[0], vpl=tile[1], nsg=tile[2])
            e2 += int(mx.sum(y != y_ref[0]).item())
        # grouped expert GEMM: 2 experts (same weights), uneven segments
        ex = np.sort(rng.integers(0, 2, M)).astype(np.int32)
        off, tstart, mt = sc_moe.segments(mx.array(ex), 2, 64)
        cE, bE, sE = (mx.array(np.stack([a, a])) for a in (codes, bc, ws))
        xp = mx.array(np.concatenate([xq, np.zeros((64, K), np.int8)]))
        acc = np.array(sc_moe.moe_mm(xp, M, cE, bE, sE, off, tstart, mt, N, 0))
        e3 = int((acc != ref).sum())
        print(f"block {fmt.block:2d} mult {fmt.mult} code_mag {fmt.code_mag} ({fmt.bits_per_weight(K):.3f} bits): "
              f"unpack mismatches {e1}, GEMV outputs differing {e2}, grouped GEMM INT32 mismatches {e3}")
        ok &= e1 == e2 == e3 == 0
    S.set_format(F.Format())
    print("ALL FORMATS PASSED" if ok else "FAILURES")
    return ok


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
