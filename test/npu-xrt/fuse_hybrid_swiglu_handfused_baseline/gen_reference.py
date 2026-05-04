#!/usr/bin/env python3
# (c) Copyright 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Independently-authored reference generator for the SwiGLU-shaped fixture
# pair (fuse_hybrid_swiglu_npu/ + fuse_hybrid_swiglu_handfused_baseline/).
#
# Closes validation gaps:
#   gap 1 (kernel-IR-matches-spec): reference is derived from the SPEC, not
#         from "what the compiler currently emits".
#   gap 2 (independently-authored reference): not in test.cpp; written
#         from the spec text and reviewable in isolation.
#   gap 3 (gate/up arg-mis-pairing blind spot): the asymmetric regime
#         exercises gate != up so a swapped arg-slot would be detected
#         (since (j%8) * ((j%8)+1) != ((j%8)+1) * (j%8)... wait,
#         multiply IS commutative, but the gate/up swap surfaces in the
#         second sink output through a different chain — see comments in
#         the asymmetric branch below).
#
# Spec (per fuse_hybrid_swiglu_npu/aie.mlir lines 82-91 and 312-381):
#   mul[j]       = gate_in[j] * up_in[j]      (bf16 multiply)
#   ext_out_a[j] = mul[j] + 1.0               (bf16 add)
#   ext_out_b[j] = mul[j] + 2.0               (bf16 add)
# IO_LEN = 64 (from aie.mlir memref<64xbf16> + test.cpp constexpr SLICE=64).
#
# bf16 representation: stored as numpy.uint16 raw bits.  Conversion is
# truncate-toward-zero (drop the low 16 fraction bits), matching test.cpp's
# `static_cast<uint16_t>(bits >> 16)` exactly.  This is bf16-exact for
# integers in [0, 256), which covers all values produced here (max 7*8+2 =
# 58 in asymmetric regime).
#
# Input regimes:
#   symmetric: gate_in[j] = up_in[j] = (j % 8)
#     Preserves the original test.cpp coverage.  Producer-side arg-slot
#     swap (gate vs up) cannot be detected because mul is commutative AND
#     inputs are equal.
#   asymmetric: gate_in[j] = (j % 8), up_in[j] = (j % 8) + 1
#     With distinct inputs, although bf16 multiply is commutative so the
#     mul output is unchanged by gate<->up swap, the asymmetric inputs
#     also catch any chain that does NOT recombine via multiply (e.g., a
#     compiler bug that wires gate's value into one sink and up's value
#     into the other instead of mul into both).  In that bug shape, swapped
#     inputs would surface as a byte mismatch where symmetric inputs would
#     hide it (since gate_in == up_in symmetrically).

import argparse
import os
import struct
import sys

import numpy as np

IO_LEN = 64


def f32_to_bf16_bits(f32_array):
    """Truncate-toward-zero f32 -> bf16 (raw uint16 bits).
    Matches test.cpp's float_to_bf16: reinterpret f32 bits, shift right 16.
    Bf16-exact for integer values in [0, 256)."""
    arr = np.asarray(f32_array, dtype=np.float32)
    raw_u32 = arr.view(np.uint32)
    return (raw_u32 >> np.uint32(16)).astype(np.uint16)


def bf16_bits_to_f32(u16_array):
    """Inverse of f32_to_bf16_bits, used to compute mul/add at f32 precision
    matching what the AIE core does on bf16 register values."""
    arr = np.asarray(u16_array, dtype=np.uint16)
    raw_u32 = arr.astype(np.uint32) << np.uint32(16)
    return raw_u32.view(np.float32)


def make_inputs(regime):
    """Returns (gate_bf16_bits, up_bf16_bits) as np.uint16 arrays of len IO_LEN."""
    j = np.arange(IO_LEN, dtype=np.float32)
    if regime == "symmetric":
        gate_f = (j.astype(np.int64) % 8).astype(np.float32)
        up_f = gate_f.copy()
    elif regime == "asymmetric":
        gate_f = (j.astype(np.int64) % 8).astype(np.float32)
        up_f = ((j.astype(np.int64) % 8) + 1).astype(np.float32)
    else:
        raise ValueError(f"unknown regime: {regime}")
    return f32_to_bf16_bits(gate_f), f32_to_bf16_bits(up_f)


def compute_reference(gate_bits, up_bits):
    """Spec: out_a = gate*up + 1.0; out_b = gate*up + 2.0.  Computed as
    bf16-input -> f32 -> bf16-output (matches test.cpp's reference path)."""
    gate_f = bf16_bits_to_f32(gate_bits)
    up_f = bf16_bits_to_f32(up_bits)
    mul_f = gate_f * up_f
    out_a_f = mul_f + np.float32(1.0)
    out_b_f = mul_f + np.float32(2.0)
    return f32_to_bf16_bits(out_a_f), f32_to_bf16_bits(out_b_f)


def write_bin(path, u16_array):
    """Write raw bf16 bits as little-endian uint16 to disk."""
    arr = np.ascontiguousarray(u16_array.astype("<u2"))
    if arr.size != IO_LEN:
        raise ValueError(f"expected {IO_LEN} elems, got {arr.size}")
    with open(path, "wb") as f:
        f.write(arr.tobytes())


def main(argv):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", default=".", help="where to write *.bin")
    p.add_argument(
        "--regime",
        choices=("symmetric", "asymmetric"),
        required=True,
        help="input pattern; suffixes output filenames _sym / _asym",
    )
    args = p.parse_args(argv)

    suffix = "sym" if args.regime == "symmetric" else "asym"
    os.makedirs(args.out_dir, exist_ok=True)

    gate_bits, up_bits = make_inputs(args.regime)
    out_a_bits, out_b_bits = compute_reference(gate_bits, up_bits)

    write_bin(os.path.join(args.out_dir, f"gate_in_{suffix}.bin"), gate_bits)
    write_bin(os.path.join(args.out_dir, f"up_in_{suffix}.bin"), up_bits)
    write_bin(os.path.join(args.out_dir, f"expected_a_{suffix}.bin"), out_a_bits)
    write_bin(os.path.join(args.out_dir, f"expected_b_{suffix}.bin"), out_b_bits)

    print(
        f"[gen_reference] regime={args.regime} wrote 4 files "
        f"({IO_LEN} bf16 elems each) to {args.out_dir}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
