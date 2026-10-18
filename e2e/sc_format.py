"""Storage format of SC-MXFP4 weights for the fused kernels (e2e/sc_kernels.py), kept in one place so that a format
change (block size, block multipliers) is a small edit here.

A weight row n of length K is stored as
  codes   4 bits per weight: sign << 3 | idx, idx in 0..7 indexes CODE_MAG (|2 x E2M1| = 0,1,2,3,4,6,8,12)
  bcodes  BC_BITS bits per block of BLOCK weights along K: index into the integer block multiplier table MULT
  wscale  one FP32 scale per row
and decodes exactly to the INT8 operand  q[n, k] = ±CODE_MAG[idx] * MULT[bcode[n, k / BLOCK]]  (|q| <= 127),
with W[n, k] = wscale[n] * q[n, k]. The published SC-MXFP4 format is BLOCK = 32 and MULT = 2^shift, shift 0..3
(results/02_mx_to_int8.md), with 2-bit block codes: 4 + 2/32 + 32/K = 4.07 bits/weight at K = 4096.

Packed layouts (all row-major over N, K-contiguous):
  codes   uint32 [N, K / 8]           8 codes per word, element k = 8 j + i in bits 4 i .. 4 i + 3
  bcodes  uint8  [N, K * BC_BITS / (8 BLOCK)]   block b of a row in bits BC_BITS (b % per_byte) of byte b / per_byte
  wscale  f32    [N]
"""
import os
import sys
from dataclasses import dataclass

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "sim"))


@dataclass(frozen=True)
class Format:
    block: int = 32
    code_mag: tuple = (0, 1, 2, 3, 4, 6, 8, 12)
    mult: tuple = (1, 2, 4, 8)
    bc_bits: int = 2

    def __post_init__(self):
        assert self.block % 8 == 0, "a thread decodes 8 consecutive codes, which must share a block"
        assert self.bc_bits in (1, 2, 4, 8) and len(self.mult) <= 2 ** self.bc_bits
        assert len(self.code_mag) == 8 and max(self.code_mag) * max(self.mult) <= 127, "must fit INT8"

    @property
    def per_byte(self):
        return 8 // self.bc_bits

    def bits_per_weight(self, K):
        return 4 + self.bc_bits / self.block + 32 / K

    def metal_header(self):
        """Inline Metal decode helpers for this format (nibble-packed constants when values fit in 4 bits)."""
        def table(name, vals):
            if max(vals) < 16 and len(vals) <= 8:
                word = sum(v << (4 * i) for i, v in enumerate(vals))
                return f"inline int {name}(uint i) {{ return int((0x{word:08X}u >> (4 * i)) & 0xFu); }}\n"
            arr = ", ".join(str(v) for v in vals)
            return f"constant int {name}_T[{len(vals)}] = {{{arr}}};\ninline int {name}(uint i) {{ return {name}_T[i]; }}\n"
        return (f"#define SC_BLOCK {self.block}\n#define SC_BC_BITS {self.bc_bits}\n#define SC_PER_BYTE {self.per_byte}\n"
                + table("sc_mag", self.code_mag) + table("sc_mult", self.mult))


DEFAULT = Format()


def pack_rows(code4, bcode, wscale, fmt=DEFAULT):
    """code4 uint8 [N, K] (sign << 3 | idx), bcode uint8 [N, K / BLOCK], wscale [N] -> packed numpy arrays."""
    n, k = code4.shape
    assert k % fmt.block == 0
    c = code4.astype(np.uint32).reshape(n, k // 8, 8)
    codes = np.zeros((n, k // 8), np.uint32)
    for i in range(8):
        codes |= c[..., i] << np.uint32(4 * i)
    nb = k // fmt.block
    assert nb % fmt.per_byte == 0 and bcode.max() < len(fmt.mult)
    b = bcode.astype(np.uint8).reshape(n, nb // fmt.per_byte, fmt.per_byte)
    bc = np.zeros((n, nb // fmt.per_byte), np.uint8)
    for i in range(fmt.per_byte):
        bc |= (b[..., i] << (fmt.bc_bits * i)).astype(np.uint8)
    return codes.reshape(-1), bc, np.asarray(wscale, np.float32)


def decode_numpy(codes, bc, n, k, fmt=DEFAULT):
    """Reference decode of the packed layout to the INT8 operand q [N, K]."""
    w = codes.reshape(n, k // 8)
    c = np.stack([(w >> np.uint32(4 * i)) & 0xF for i in range(8)], -1).reshape(n, k).astype(np.int32)
    b = np.stack([(bc >> (fmt.bc_bits * i)) & (2 ** fmt.bc_bits - 1) for i in range(fmt.per_byte)], -1)
    b = b.reshape(n, k // fmt.block).astype(np.int32)
    mag = np.array(fmt.code_mag)[c & 7] * np.repeat(np.array(fmt.mult)[b], fmt.block, axis=1)
    return np.where(c & 8, -mag, mag).astype(np.int8)


def from_shift_layout(codes_u32, shifts, wscale, fmt=DEFAULT):
    """Converts the byte-shift layout of sim/make_testcase.pack / e2e/w4a8_mlx.pack_model (codes uint32 [N*K/8],
    shifts uint8 [N, K/32] in 0..3, wscale [N]) to this format's 2-bit packed block codes (no requantization)."""
    assert fmt.block == 32 and tuple(fmt.mult) == (1, 2, 4, 8), "shift layout is the power-of-two BLOCK=32 format"
    n, nb = shifts.shape
    b = shifts.astype(np.uint8).reshape(n, nb // fmt.per_byte, fmt.per_byte)
    bc = np.zeros((n, nb // fmt.per_byte), np.uint8)
    for i in range(fmt.per_byte):
        bc |= (b[..., i] << (fmt.bc_bits * i)).astype(np.uint8)
    return np.asarray(codes_u32, np.uint32).reshape(-1), bc, np.asarray(wscale, np.float32)


def quantize(w, fmt=DEFAULT):
    """(N, K) float weights -> packed (codes, bcodes, wscale) with the reference SC-MXFP4 quantizer of sim/
    (the one sim/eval_ppl.py's sc_mxfp4_w4a8 fake-quantizes with), plus the INT8 operand for checking."""
    from make_testcase import pack
    packed, shifts, wscale, qw = pack(np.asarray(w, dtype=np.float64))
    codes, bc, ws = from_shift_layout(packed.reshape(-1).view(np.uint32), shifts, wscale, fmt)
    return codes, bc, ws, qw


def _self_test():
    rng = np.random.default_rng(0)
    w = rng.standard_normal((64, 512)) * 0.02 * rng.lognormal(0, 0.3, (1, 512))
    codes, bc, ws, qw = quantize(w)
    assert np.array_equal(decode_numpy(codes, bc, 64, 512), qw)
    # A generic format round trip: random codes and block multipliers.
    f = Format(block=16, mult=(1, 2, 3, 4, 5, 6, 8, 10), bc_bits=4)
    c4 = rng.integers(0, 16, (8, 256)).astype(np.uint8)
    b = rng.integers(0, 8, (8, 256 // 16)).astype(np.uint8)
    cc, bb, _ = pack_rows(c4, b, np.ones(8), f)
    q = decode_numpy(cc, bb, 8, 256, f)
    mag = np.array(f.code_mag)[c4 & 7] * np.repeat(np.array(f.mult)[b], 16, axis=1)
    assert np.array_equal(q, np.where(c4 & 8, -mag, mag))
    print(f"sc_format self-test passed; default format {DEFAULT.bits_per_weight(4096):.3f} bits/weight at K=4096")


if __name__ == "__main__":
    _self_test()
