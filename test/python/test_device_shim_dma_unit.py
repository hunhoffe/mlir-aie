# test_device_shim_dma_unit.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#

# RUN: %pytest %s
"""Unit tests for the shim DMA capability queries on Device.

These read the target model only — no device is opened, so they run anywhere
the Python bindings are built.
"""

import pytest
from aie.dialects._aie_enum_gen import (  # pyright: ignore[reportMissingImports]
    WireBundle,
)
from aie.iron.device import NPU1, NPU2, NPU1Col1, NPU1Col2, NPU2Col1, NPU2Col2

ALL_DEVICES = [NPU1Col1, NPU1Col2, NPU1, NPU2Col1, NPU2Col2, NPU2]


@pytest.fixture(params=ALL_DEVICES, ids=lambda c: c.__name__)
def device(request):
    return request.param()


# ---------------------------------------------------------------------------
# shim_cols
# ---------------------------------------------------------------------------


def test_shim_cols_are_columns_of_this_device(device):
    assert device.shim_cols, "every NPU has at least one shim column"
    assert all(0 <= col < device.cols for col in device.shim_cols)


def test_shim_cols_are_sorted_and_unique(device):
    assert device.shim_cols == sorted(set(device.shim_cols))


def test_shim_cols_agree_with_the_target_model(device):
    """The property must report what the target model says, not a guess."""
    expected = [
        c for c in range(device.cols) if device._tm.is_shim_noc_or_pl_tile(c, 0)
    ]
    assert device.shim_cols == expected


# ---------------------------------------------------------------------------
# num_shim_dma_channels
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("output", [True, False])
def test_channel_count_is_positive(device, output):
    """A device with no shim DMA in either direction could not be driven."""
    assert device.num_shim_dma_channels(output=output) > 0


def test_defaults_to_the_output_direction(device):
    assert device.num_shim_dma_channels() == device.num_shim_dma_channels(output=True)


@pytest.mark.parametrize("output", [True, False])
def test_count_is_the_sum_over_shim_tiles(device, output):
    """The total must be the per-tile budgets added up, not a per-device constant."""
    query = (
        device._tm.get_num_source_shim_mux_connections
        if output
        else device._tm.get_num_dest_shim_mux_connections
    )
    expected = sum(query(col, 0, WireBundle.DMA) for col in device.shim_cols)
    assert device.num_shim_dma_channels(output=output) == expected


@pytest.mark.parametrize("output", [True, False])
@pytest.mark.parametrize(
    "smaller, larger",
    [(NPU1Col1, NPU1Col2), (NPU1Col2, NPU1), (NPU2Col1, NPU2Col2), (NPU2Col2, NPU2)],
    ids=lambda c: c.__name__,
)
def test_more_columns_never_means_fewer_channels(smaller, larger, output):
    """Column variants of one part differ only in how many columns they expose."""
    assert larger().num_shim_dma_channels(
        output=output
    ) >= smaller().num_shim_dma_channels(output=output)


def test_direction_must_be_passed_by_keyword(device):
    """Reject a positionally-passed direction.

    Positional truthiness at a call site would read as its opposite half the
    time, so the argument is keyword-only.
    """
    with pytest.raises(TypeError):
        device.num_shim_dma_channels(False)  # pyright: ignore[reportCallIssue]
