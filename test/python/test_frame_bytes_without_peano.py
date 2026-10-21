# test_frame_bytes_without_peano.py -*- Python -*-
#
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# RUN: %pytest %s

"""Without Peano a kernel's frame reads as 0, and the host says so once."""

import logging

import numpy as np
from aie.iron import kernel
from aie.iron.kernel import ExternalFunction
from aie.utils import config


def test_frame_bytes_warns_once_without_peano(monkeypatch, caplog):
    def no_peano():
        raise RuntimeError("no Peano")

    monkeypatch.setattr(config, "peano_cxx_path", no_peano)
    kernel._warn_no_peano.cache_clear()
    kernels = [
        ExternalFunction(
            name,
            source_string=f'extern "C" void {name}(int *a) {{}}',
            arg_types=[np.ndarray[(64,), np.dtype[np.int32]]],
        )
        for name in ("a", "b")
    ]
    with caplog.at_level(logging.WARNING, logger=kernel.__name__):
        assert [k.frame_bytes("aie2p") for k in kernels] == [0, 0]
    assert ["Peano is not installed" in r.message for r in caplog.records] == [True]
