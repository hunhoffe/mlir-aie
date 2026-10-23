# _explore.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
"""An optional view of each compiled design in IRON Design Explorer.

IRON Design Explorer (the ``iron_explorer`` package) draws a design layer
by layer, from its ObjectFIFOs down to the switchboxes they are routed
through, as one static HTML page. Where it is installed,
``IRON_EXPLORE=1`` or ``@iron.jit(explore=True)`` writes that page beside each
build after every compile and prints where, and ``design.explore()`` returns
it (Jupyter shows it inline). Where it is not, nothing here runs and nothing
else changes: the explorer is never imported unless asked for.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

ENV = "IRON_EXPLORE"
OFF = ("", "0", "false", "no", "off")
MISSING = "IRON Design Explorer (the iron_explorer package) is not installed"

# What this process has already said, so a design called in a loop prints
# its page (or the missing explorer) once rather than on every cache hit.
_announced: set[Path | str] = set()


def requested(explore: bool | None) -> bool:
    """Whether to write a page after compiling: the design's own ``explore``
    setting, or else ``IRON_EXPLORE``."""
    if explore is not None:
        return explore
    return os.environ.get(ENV, "").strip().lower() not in OFF


def write_page(
    build_dir: Path,
    name: str,
    flags: list[str],
    *,
    required: bool = False,
    failed: bool = False,
):
    """Write the explorer page for the build in ``build_dir`` and return it.

    ``failed`` says the compile it follows failed; the page then opens on
    aiecc's first error, and says so when announced.

    With ``required`` (``design.explore()``), a missing explorer or a page
    that cannot be made raises. Otherwise (after a compile) it is logged and
    ``None`` returned: the compile's own outcome stands either way.
    """
    try:
        import iron_explorer
    except ImportError:
        if required:
            raise ImportError(f"design.explore(): {MISSING}") from None
        if MISSING not in _announced:
            _announced.add(MISSING)
            logger.warning("%s is set, but %s", ENV, MISSING)
        return None
    try:
        page = iron_explorer.explore(build_dir, name=name, flags=list(flags))
    except Exception as e:
        if required:
            raise
        logger.warning("[iron] no explorer page for %s: %s", name, e)
        return None
    if page.path not in _announced:
        _announced.add(page.path)
        what = "failed to compile; see why at" if failed else "explore"
        print(f"[iron] {name}: {what} {page.path}", flush=True)
    return page
