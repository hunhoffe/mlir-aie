#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Copyright (C) 2024, Advanced Micro Devices, Inc.
import numpy as np
import sys

from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.extras.context import mlir_mod_ctx
from aie.helpers.dialects.ext.scf import _for as range_

width = 32
height = 32
in_channels = 64
out_channels = 64

if len(sys.argv) == 3:
    width = int(sys.argv[1])
    height = int(sys.argv[2])


actIn = width * in_channels  # 32*64 = 2048
bufIn = actIn * 2  # double buffer

weights = in_channels * out_channels

actOut = width * out_channels  # 32*64 = 2048
bufOut = actOut * 2  # double buffer

tensorSize = width * height * in_channels


def conv2dk1():
    with mlir_mod_ctx() as ctx:

        @device(AIEDevice.npu1_1col)
        def device_body():

            actIn_ty = np.ndarray[(actIn,), np.dtype[np.int8]]
            bufIn_ty = np.ndarray[(bufIn,), np.dtype[np.int8]]

            weights_ty = np.ndarray[(weights,), np.dtype[np.int8]]

            out_ty = np.ndarray[(actOut,), np.dtype[np.int8]]
            bufOut_ty = np.ndarray[(bufOut,), np.dtype[np.int8]]

            tensor_ty = np.ndarray[(tensorSize,), np.dtype[np.int8]]

            # AIE Core Function declarations
            conv2dk1_i8 = external_func(
                "conv2dk1_i8",
                inputs=[
                    actIn_ty,
                    weights_ty,
                    out_ty,
                    np.int32,
                    np.int32,
                    np.int32,
                    np.int32,
                ],
            )

            # Tile declarations
            ShimTile = tile(0, 0)
            MemTile = tile(0, 1)
            ComputeTile2 = tile(0, 2)
            compute_tile2_col, compute_tile2_row = 0, 2

            # AIE-array data movement with object fifos
            # Input
            of_inOF_act_L3L2 = object_fifo(
                "inOF_act_L3L2", ShimTile, MemTile, 2, bufIn_ty
            )
            of_act_L2_02 = object_fifo("act_L2_02", MemTile, ComputeTile2, 2, actIn_ty)
            object_fifo_link(of_inOF_act_L3L2, of_act_L2_02)

            # wts
            of_inOF_wts_0_L3L2 = object_fifo(
                "inOF_wts_0_L3L2", ShimTile, [ComputeTile2], 1, weights_ty
            )

            # Output
            of_out_02_L2 = object_fifo("out_02_L2", ComputeTile2, [MemTile], 2, out_ty)
            of_outOFL2L3 = object_fifo("outOFL2L3", MemTile, [ShimTile], 2, bufOut_ty)
            object_fifo_link(of_out_02_L2, of_outOFL2L3)

            # Set up compute tiles

            rtp2 = buffer(
                ComputeTile2, T.memref(16, T.i32()), "rtp2", use_write_rtp=True
            )

            # Compute tile 2
            @core(ComputeTile2, "conv2dk1.o")
            def core_body():
                y_dim = 32
                x_dim = 32
                ci = 64
                co = 64

                for _ in range_(sys.maxsize):
                    elemWts = of_inOF_wts_0_L3L2.acquire(ObjectFifoPort.Consume, 1)

                    scale = rtp2[0]
                    # scale = memref.load(rtpComputeTile2, [0])

                    for _ in range_(y_dim):
                        elemIn = of_act_L2_02.acquire(ObjectFifoPort.Consume, 1)
                        elemOut0 = of_out_02_L2.acquire(ObjectFifoPort.Produce, 1)

                        conv2dk1_i8(elemIn, elemWts, elemOut0, x_dim, ci, co, scale)

                        of_act_L2_02.release(ObjectFifoPort.Consume, 1)
                        of_out_02_L2.release(ObjectFifoPort.Produce, 1)
                    of_inOF_wts_0_L3L2.release(ObjectFifoPort.Consume, 1)

            # To/from AIE-array data movement

            @runtime_sequence(tensor_ty, weights_ty, tensor_ty)
            def sequence(I, W, O):
                rtp2[0] = 1

                in_act_task = shim_dma_single_bd_task(
                    of_inOF_act_L3L2, I, sizes=[1, 1, 1, tensorSize]
                )
                in_wts_task = shim_dma_single_bd_task(
                    of_inOF_wts_0_L3L2, W, sizes=[1, 1, 1, weights]
                )
                out_task = shim_dma_single_bd_task(
                    of_outOFL2L3,
                    O,
                    sizes=[1, 1, 1, tensorSize],
                    issue_token=True,
                )

                dma_start_task(in_act_task, in_wts_task, out_task)
                # out_task will only complete after in_act_task and in_wts_task complete, so we just wait on out_task instead of all
                dma_await_task(out_task)
                dma_free_task(in_act_task, in_wts_task)

    #    print(ctx.module.operation.verify())
    print(ctx.module)


conv2dk1()
