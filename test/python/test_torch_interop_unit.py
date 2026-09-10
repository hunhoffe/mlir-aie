# test_torch_interop_unit.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

# RUN: %pytest %s
"""Unit tests for the torch <-> numpy bridge.

No device involved: these are pure host-memory conversions.
"""

import numpy as np
import pytest

torch = pytest.importorskip("torch")
ml_dtypes = pytest.importorskip("ml_dtypes")

from aie.utils.hostruntime.torch_interop import (  # noqa: E402
    _array_to_torch,
    torch_to_numpy,
)

# ---------------------------------------------------------------------------
# Dtypes numpy and torch share
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "torch_dtype, np_dtype",
    [
        (torch.float32, np.float32),
        (torch.float16, np.float16),
        (torch.int32, np.int32),
        (torch.int8, np.int8),
        (torch.uint8, np.uint8),
    ],
)
def test_native_dtypes_round_trip(torch_dtype, np_dtype):
    t = torch.arange(12, dtype=torch_dtype).reshape(3, 4)
    arr = torch_to_numpy(t)
    assert arr.dtype == np_dtype
    assert arr.shape == (3, 4)
    assert np.array_equal(arr, np.arange(12, dtype=np_dtype).reshape(3, 4))


def test_native_dtype_shares_memory():
    """The common case must not pay for a copy."""
    t = torch.zeros(8, dtype=torch.float32)
    arr = torch_to_numpy(t)
    arr[3] = 7.0
    assert t[3].item() == 7.0


# ---------------------------------------------------------------------------
# Dtypes only ml_dtypes has
# ---------------------------------------------------------------------------


def test_bfloat16_reinterprets_rather_than_converting():
    """bfloat16 must come back bit-for-bit, not through a float32 detour."""
    t = torch.tensor([1.5, -2.25, 0.0, 3.75], dtype=torch.bfloat16)
    arr = torch_to_numpy(t)
    assert arr.dtype == np.dtype(ml_dtypes.bfloat16)
    assert np.array_equal(
        arr.astype(np.float32), np.array([1.5, -2.25, 0.0, 3.75], dtype=np.float32)
    )


def test_bfloat16_shares_memory():
    t = torch.zeros(4, dtype=torch.bfloat16)
    arr = torch_to_numpy(t)
    arr[1] = ml_dtypes.bfloat16(2.5)
    assert t[1].item() == 2.5


def test_bfloat16_survives_a_round_trip_through_torch():
    """torch_to_numpy is the inverse of _array_to_torch, so both ways must agree."""
    original = torch.tensor([[1.5, -2.25], [0.5, 8.0]], dtype=torch.bfloat16)
    back = _array_to_torch(torch_to_numpy(original))
    assert back.dtype == torch.bfloat16
    assert torch.equal(back, original)


def test_multidimensional_bfloat16_keeps_its_shape():
    """The reinterpret view is per-element, so rank must survive it."""
    t = torch.ones((2, 3, 4), dtype=torch.bfloat16)
    arr = torch_to_numpy(t)
    assert arr.shape == (2, 3, 4)
    assert np.all(arr.astype(np.float32) == 1.0)


# ---------------------------------------------------------------------------
# The normalization every caller would otherwise redo
# ---------------------------------------------------------------------------


def test_a_tensor_carrying_grad_is_detached():
    """.numpy() refuses a tensor that requires grad; the helper must not."""
    t = torch.ones(4, dtype=torch.float32, requires_grad=True)
    arr = torch_to_numpy(t)
    assert np.array_equal(arr, np.ones(4, dtype=np.float32))


def test_a_non_contiguous_tensor_is_made_contiguous():
    """A transposed view has no numpy equivalent layout until it is compacted."""
    t = torch.arange(6, dtype=torch.float32).reshape(2, 3).t()
    assert not t.is_contiguous()
    arr = torch_to_numpy(t)
    assert arr.shape == (3, 2)
    assert np.array_equal(arr, np.arange(6, dtype=np.float32).reshape(2, 3).T)


def test_a_non_contiguous_bfloat16_tensor_is_made_contiguous():
    """The reinterpret view needs contiguity too, not just .numpy()."""
    t = torch.arange(6, dtype=torch.bfloat16).reshape(2, 3).t()
    assert not t.is_contiguous()
    arr = torch_to_numpy(t)
    assert arr.shape == (3, 2)
    assert np.array_equal(
        arr.astype(np.float32), np.arange(6, dtype=np.float32).reshape(2, 3).T
    )


# ---------------------------------------------------------------------------
# Input that is not a torch tensor
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value", [[1.0, 2.0, 3.0], (1, 2, 3), np.arange(4, dtype=np.float32), 5.0]
)
def test_array_likes_pass_through(value):
    """Callers holding "a tensor or an array" should not have to branch."""
    assert isinstance(torch_to_numpy(value), np.ndarray)


def test_a_numpy_array_is_returned_as_is():
    arr = np.arange(4, dtype=np.float32)
    assert torch_to_numpy(arr) is arr


# ---------------------------------------------------------------------------
# The two direction maps must describe the same set of types
# ---------------------------------------------------------------------------


def test_the_dtype_maps_are_inverses():
    from aie.utils.hostruntime.torch_interop import (
        _ml_dtype_to_torch_map,
        _torch_to_ml_dtype_map,
    )

    forward = _ml_dtype_to_torch_map()
    reverse = _torch_to_ml_dtype_map()
    assert len(forward) == len(reverse)
    for np_dtype, torch_dtype in forward.items():
        assert reverse[torch_dtype] == np_dtype
