# test_ternary.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# RUN: %pytest %s
"""Ternary code order, zero codes and group checks (no NPU required)."""

import numpy as np
import pytest
from ml_dtypes import bfloat16

from aie.iron.kernels import ternary_gemv, ternary_dequant_ref


def _block(weights, scales):
    """The bytes of a (32, 256) block of trits and its (32, 2) bf16 scales,
    packed as ``ternary_dequant_ref`` documents, written out by hand.
    """
    codes = np.where(weights == 1, 1, np.where(weights == -1, 2, 0)).astype(np.uint8)
    codes = codes.reshape(32, 2, 4, 32)
    packed = np.zeros((32, 2, 32), np.uint8)
    for j in range(4):
        packed |= codes[:, :, j, :] << (2 * j)
    s = np.asarray(scales, bfloat16).view(np.uint16).astype("<u2").view(np.uint8)
    return np.concatenate([packed.reshape(-1), s.reshape(-1)])


def test_the_reference_reads_the_documented_layout():
    rng = np.random.default_rng(0)
    trits = rng.integers(-1, 2, (32, 256))
    scales = rng.choice([0.5, -2.0, 0.375], (32, 2)).astype(np.float32)
    got = ternary_dequant_ref(_block(trits, scales))
    np.testing.assert_array_equal(got, trits * np.repeat(scales, 128, axis=1))


def test_code_three_is_zero():
    block = _block(np.zeros((32, 256)), np.ones((32, 2)))
    block[0] = 0b11
    assert ternary_dequant_ref(block)[0, 0] == 0


def test_the_gemv_refuses_a_partial_group():
    with pytest.raises(ValueError, match="k_tile"):
        ternary_gemv(k_tile=96)


def test_the_gemv_refuses_rows_not_in_sixteens():
    with pytest.raises(ValueError, match="m_tile"):
        ternary_gemv(m_tile=24)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-q"]))
