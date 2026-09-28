# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# RUN: %python %s

import numpy as np
from aie.helpers.taplib import Layout, TensorAccessPattern
from aie.iron.runtime.data import BoundGrid, RuntimeData, View


def test_slice_returns_metadata_without_runtime_storage():
    for dtype in (np.int8, np.int32, np.float32):
        data = RuntimeData(np.ndarray[(4, 8), np.dtype[dtype]])
        view = data[1::2, 2::3]
        assert isinstance(view, View) and view.data is data
        assert view.offset == 10
        assert view.sizes == [2, 2]
        assert view.strides == [16, 3]
        assert isinstance(view.tap(None), TensorAccessPattern)
        assert view.tap(None) == Layout.full((4, 8))[1::2, 2::3].tap(None)

        scalar = data[-1, -1]
        assert isinstance(scalar, View)
        assert scalar.offset == 31
        assert scalar.sizes == [1]


def test_numpy_spellings_stay_bound():
    data = RuntimeData(np.ndarray[(64, 32), np.dtype[np.int16]])
    tile = data.reshape(8, 8, 4, 8).transpose(0, 2, 1, 3)[1, 2]
    assert isinstance(tile, View) and tile.data is data
    assert tile.layout == Layout.full((64, 32)).tile((8, 8)).at(1, 2)
    assert data.T.strides == [1, 32]
    assert data.broadcast_to((3, 64, 32)).strides == [0, 32, 1]
    assert data.repeat(3).layout == data.broadcast_to((3, 64, 32)).layout
    # Anything else the algebra offers comes back bound too.
    assert isinstance(data.view.coalesce(), View)
    assert isinstance(data.view.permute((1, 0)), View)
    grid = data.tile((8, 8))
    assert isinstance(grid, BoundGrid) and len(grid) == 32
    assert isinstance(grid.group((1, 4))[0], View)
    assert isinstance(grid.order("col"), BoundGrid)
    assert isinstance(grid.inverse(), View)
    assert all(isinstance(t, View) for t in grid)
    assert data.view.numel == 64 * 32


test_slice_returns_metadata_without_runtime_storage()
test_numpy_spellings_stay_bound()
print("RuntimeData slicing returns bound views")
