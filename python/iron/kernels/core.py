# core.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""Core state kernels: ``set_rounding``, ``set_saturation``, ``read_core_state``.

The AIE core narrows accumulators (an ``srs`` shift, a bf16 store) in the
rounding mode its mode register holds, and clamps or wraps as its saturation
register says; a fresh core boots in ``floor`` with saturation off. Both
registers are sticky per core: whatever one kernel leaves, the next kernel on
that core inherits. A kernel whose contract names a ``rounding_mode`` or
``saturation_mode`` expects the design to have set it before its first call;
``set_rounding`` and ``set_saturation`` are the kernels that do so, and
``aie.utils.kernel_harness`` calls them for such contracts. ``read_core_state``
copies both registers into a tile, which is how the harness's dirty-state
sweep checks what state a kernel ran in and what it left behind.
"""

import numpy as np
from aie.iron.kernel import ExternalFunction

from ._common import (
    ROUNDING_MODES,
    SATURATION_MODES,
    _default_source_path,
    _make_extern,
)

# The probe's output tile: [rounding code, saturation code, marker, reserved].
# Four int32 keep the tile DMA-aligned; the codes index ROUNDING_MODES[2:] and
# SATURATION_MODES[2:], as read_core_state.cc documents.
CORE_STATE_WORDS = 4
CORE_STATE_MARKER = 0x50524F42  # "PROB"


def set_rounding(mode: str = "conv_even") -> ExternalFunction:
    """Kernel that sets the core's rounding mode to ``mode`` and returns.

    Call it once in a Worker before the first kernel whose contract's
    ``rounding_mode`` is ``mode``; the mode persists on that core until
    another kernel changes it. ``mode`` is an ``aie::rounding_mode`` name:
    ``floor``, ``ceil``, ``positive_inf``, ``negative_inf``,
    ``symmetric_inf``, ``symmetric_zero``, ``conv_even`` or ``conv_odd``.

    Args:
        mode: The ``aie::rounding_mode`` to set.

    Returns:
        ExternalFunction ``set_rounding_<mode>``, which takes no arguments.

    Raises:
        ValueError: When ``mode`` is not an ``aie::rounding_mode`` name.
    """
    if mode not in ROUNDING_MODES or mode in ("unspecified", "sets_own"):
        raise ValueError(
            f"set_rounding() mode must be an aie::rounding_mode name, got {mode!r}."
        )
    return _make_extern(
        f"set_rounding_{mode}",
        _default_source_path("set_rounding.cc", subdir="generic"),
        [],
        compile_flags=[f"-DROUNDING_MODE={mode}"],
    )


def set_saturation(mode: str = "saturate") -> ExternalFunction:
    """Kernel that sets the core's saturation mode to ``mode`` and returns.

    The saturation counterpart of :func:`set_rounding`: call it once before
    the first kernel whose contract's ``saturation_mode`` is ``mode``.
    ``mode`` is an ``aie::saturation_mode`` name: ``none`` (results wrap),
    ``saturate`` (clamp to the output range) or ``symmetric`` (clamp to a
    range symmetric about zero).

    Args:
        mode: The ``aie::saturation_mode`` to set.

    Returns:
        ExternalFunction ``set_saturation_<mode>``, which takes no arguments.

    Raises:
        ValueError: When ``mode`` is not an ``aie::saturation_mode`` name.
    """
    if mode not in SATURATION_MODES or mode in ("unspecified", "sets_own"):
        raise ValueError(
            f"set_saturation() mode must be an aie::saturation_mode name, got {mode!r}."
        )
    return _make_extern(
        f"set_saturation_{mode}",
        _default_source_path("set_saturation.cc", subdir="generic"),
        [],
        compile_flags=[f"-DSATURATION_MODE={mode}"],
    )


def read_core_state() -> ExternalFunction:
    """Kernel that writes the core's rounding and saturation modes into a tile.

    The tile holds :data:`CORE_STATE_WORDS` int32: the rounding mode as an
    index into ``ROUNDING_MODES[2:]``, the saturation mode as an index into
    ``SATURATION_MODES[2:]``, :data:`CORE_STATE_MARKER`, and a reserved word.
    :func:`aie.utils.kernel_harness.decode_core_state` turns it back into
    names. It carries no contract: it has no inputs and its "reference" is
    whatever the core's registers hold.

    Returns:
        ExternalFunction ``read_core_state``, taking the output tile.
    """
    return _make_extern(
        "read_core_state",
        _default_source_path("read_core_state.cc", subdir="generic"),
        [np.ndarray[(CORE_STATE_WORDS,), np.dtype[np.int32]]],
    )
