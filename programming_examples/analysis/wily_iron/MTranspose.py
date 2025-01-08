# dma_transpose/dma_transpose_alt.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2024 Advanced Micro Devices, Inc. or its affiliates
import argparse
import numpy as np
import sys

from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.extras.context import mlir_mod_ctx
from aie.helpers.dialects.ext.scf import _for as range_


def my_passthrough(M, K):
    tensor_ty = np.ndarray[(M, K), np.dtype[np.int32]]

    with mlir_mod_ctx() as ctx:

        @device(AIEDevice.npu1_1col)
        def device_body():
            # Tile declarations
            ShimTile = tile(0, 0)
            ComputeTile2 = tile(0, 2)

            # AIE-array data movement with object fifos
            of_in = object_fifo("in", ShimTile, ComputeTile2, 2, tensor_ty)
            of_out = object_fifo("out", ComputeTile2, ShimTile, 2, tensor_ty)
            object_fifo_link(of_in, of_out)

            # Set up compute tiles

            # To/from AIE-array data movement
            @runtime_sequence(tensor_ty, tensor_ty, tensor_ty)
            def sequence(A, B, C):
                # The strides below are configured to read across all rows in the same column
                # Stride of K in dim/wrap 2 skips an entire row to read a full column
                in_task = shim_dma_single_bd_task(
                    of_in, A, sizes=[1, 1, K, M], strides=[0, 0, 1, K]
                )
                out_task = shim_dma_single_bd_task(
                    of_out, C, sizes=[1, 1, 1, M * K], issue_token=True
                )

                dma_start_task(in_task, out_task)
                dma_await_task(out_task)
                dma_free_task(in_task)

    print(ctx.module)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("dims", help="M K", type=int, nargs="*", default=[64, 64])
    args = p.parse_args()

    if len(args.dims) != 2:
        print(
            "ERROR: Must provide either no dimensions or both M and K", file=sys.stderr
        )
        exit(-1)
    my_passthrough(M=args.dims[0], K=args.dims[1])
