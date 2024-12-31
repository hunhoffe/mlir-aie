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
from aie.helpers.util import np_ndarray_type_get_shape

# Define bottleneck layer sizes

tensorInW = 32
tensorInH = 32
tensorInC = 256

tensorL1InC = tensorInC
tensorL1OutC = tensorL1InC // 4

tensorL2InC = tensorL1OutC
tensorL2OutC = tensorL2InC

tensorL3InC = tensorL2OutC
tensorL3OutC = tensorL3InC * 4

activationsIn = tensorInW * tensorInH * tensorInC
acitivationsOut = activationsIn
totalWeights = (
    tensorL1InC * tensorL1OutC
    + 3 * 3 * tensorL2InC * tensorL2OutC
    + tensorL3InC * tensorL3OutC
)


def bottleneck4AIEs():
    with mlir_mod_ctx() as ctx:

        @device(AIEDevice.npu1_1col)
        def deviceBody():

            # define types
            activationsInL3_ty = np.ndarray[(activationsIn,), np.dtype[np.int8]]
            weightsInL3_ty = np.ndarray[(totalWeights,), np.dtype[np.uint8]]
            weightsAll_ty = np.ndarray[(totalWeights,), np.dtype[np.int8]]

            tensorLayer1In_ty = np.ndarray[
                (tensorInW, 1, tensorL1InC), np.dtype[np.int8]
            ]
            weightsLayer1_ty = np.ndarray[
                (tensorL1InC * tensorL1OutC,), np.dtype[np.int8]
            ]
            tensorLayer1Out_ty = np.ndarray[
                (tensorInW, 1, tensorL1OutC), np.dtype[np.uint8]
            ]

            tensorLayer2In_ty = np.ndarray[
                (tensorInW, 1, tensorL2InC), np.dtype[np.uint8]
            ]
            weightsLayer2_ty = np.ndarray[
                (3 * 3 * tensorL2InC * tensorL2OutC,), np.dtype[np.int8]
            ]
            tensorLayer2Out_ty = np.ndarray[
                (tensorInW, 1, tensorL2OutC // 2), np.dtype[np.uint8]
            ]

            tensorLayer3In_ty = np.ndarray[
                (tensorInW, 1, tensorL3InC // 2), np.dtype[np.uint8]
            ]
            weightsLayer3_ty = np.ndarray[
                (tensorL3InC * tensorL3OutC,), np.dtype[np.int8]
            ]
            tensorLayer3Out_ty = np.ndarray[
                (tensorInW, 1, tensorL3OutC), np.dtype[np.uint8]
            ]

            # kernel definitions
            conv2dk1 = external_func(
                "conv2dk1_i8",
                inputs=[
                    tensorLayer1In_ty,
                    weightsLayer1_ty,
                    tensorLayer1Out_ty,
                    np.int32,
                    np.int32,
                    np.int32,
                    np.int32,
                ],
            )
            conv2dk3 = external_func(
                "conv2dk3_ui8",
                inputs=[
                    tensorLayer2In_ty,
                    tensorLayer2In_ty,
                    tensorLayer2In_ty,
                    weightsLayer2_ty,
                    tensorLayer2Out_ty,
                    np.int32,
                    np.int32,
                    np.int32,
                    np.int32,
                    np.int32,
                    np.int32,
                    np.int32,
                    np.int32,
                ],
            )
            conv2dk1_skip = external_func(
                "conv2dk1_skip_i8",
                inputs=[
                    tensorLayer3In_ty,
                    tensorLayer3In_ty,
                    weightsLayer3_ty,
                    tensorLayer3Out_ty,
                    tensorLayer1In_ty,
                    np.int32,
                    np.int32,
                    np.int32,
                    np.int32,
                    np.int32,
                ],
            )

            ShimTile = tile(0, 0)
            MemTile = tile(0, 1)
            ComputeTile2 = tile(0, 2)
            ComputeTile3 = tile(0, 3)
            ComputeTile4 = tile(0, 4)
            ComputeTile5 = tile(0, 5)

            # runtime parameters

            rtpComputeTile2 = buffer(
                ComputeTile2,
                np.ndarray[(16,), np.dtype[np.int32]],
                "rtpComputeTile2",
                use_write_rtp=True,
            )
            rtpComputeTile3 = buffer(
                ComputeTile3,
                np.ndarray[(16,), np.dtype[np.int32]],
                "rtpComputeTile3",
                use_write_rtp=True,
            )
            rtpComputeTile4 = buffer(
                ComputeTile4,
                np.ndarray[(16,), np.dtype[np.int32]],
                "rtpComputeTile4",
                use_write_rtp=True,
            )
            rtpComputeTile5 = buffer(
                ComputeTile5,
                np.ndarray[(16,), np.dtype[np.int32]],
                "rtpComputeTile5",
                use_write_rtp=True,
            )

            # set up data movement with OFs
            # input tensor (with broadcast for skip connection)
            of_inOF_act_L3L2 = object_fifo(
                "inOF_act_L3L2",
                ShimTile,
                [ComputeTile2, MemTile],
                [2, 2, 4],
                tensorLayer1In_ty,
            )
            of_skip_buf = object_fifo(
                "skip_buf", MemTile, ComputeTile4, 2, tensorLayer1In_ty
            )
            object_fifo_link(of_inOF_act_L3L2, of_skip_buf)

            # weights
            inOF_wts_0_L3L2 = object_fifo(
                "inOF_wts_0_L3L2", ShimTile, MemTile, 1, weightsAll_ty
            )
            of_wts_buf_00 = object_fifo(
                "wts_buf_00", MemTile, ComputeTile2, 1, weightsLayer1_ty
            )
            wts_buf_01 = object_fifo(
                "wts_buf_01",
                MemTile,
                [ComputeTile3, ComputeTile5],
                1,
                weightsLayer2_ty,
            )
            wts_buf_02 = object_fifo(
                "wts_buf_02", MemTile, ComputeTile4, 1, weightsLayer3_ty
            )
            of_offsets = [
                0,
                np.prod(np_ndarray_type_get_shape(weightsLayer1_ty)),
                np.prod(np_ndarray_type_get_shape(weightsLayer1_ty))
                + np.prod(np_ndarray_type_get_shape(weightsLayer2_ty)),
            ]
            object_fifo_link(
                inOF_wts_0_L3L2, [of_wts_buf_00, wts_buf_01, wts_buf_02], [], of_offsets
            )

            # activation tensor
            of_act_2_3_5 = object_fifo(
                "act_2_3_5",
                ComputeTile2,
                [ComputeTile3, ComputeTile5],
                [2, 4, 4],
                tensorLayer1Out_ty,
            )  # 1x1 -> 3x3
            act_3_4 = object_fifo(
                "act_3_4", ComputeTile3, ComputeTile4, 2, tensorLayer2Out_ty
            )  # 3x3 -> 1x1
            act_5_4 = object_fifo(
                "act_5_4", ComputeTile5, ComputeTile4, 2, tensorLayer2Out_ty
            )  # 3x3 -> 1x1

            # output tensor
            outOFL2L3 = object_fifo(
                "outOFL2L3", ComputeTile4, ShimTile, 2, tensorLayer3Out_ty
            )

            # 1x1 conv2d
            @core(ComputeTile2, "conv2dk1.o")
            def core_body():
                for _ in range_(sys.maxsize):
                    # acquire weights once
                    element0Weights = of_wts_buf_00.acquire(ObjectFifoPort.Consume, 1)
                    scale = rtpComputeTile2[0]
                    for _ in range_(tensorInH):
                        element0ActivactionsIn = of_inOF_act_L3L2.acquire(
                            ObjectFifoPort.Consume, 1
                        )
                        element0ActivactionsOut = of_act_2_3_5.acquire(
                            ObjectFifoPort.Produce, 1
                        )
                        conv2dk1(
                            element0ActivactionsIn,
                            element0Weights,
                            element0ActivactionsOut,
                            tensorInW,
                            tensorL1InC,
                            tensorL1OutC,
                            scale,
                        )
                        of_inOF_act_L3L2.release(ObjectFifoPort.Consume, 1)
                        of_act_2_3_5.release(ObjectFifoPort.Produce, 1)
                    of_wts_buf_00.release(ObjectFifoPort.Consume, 1)

            # 3x3 conv2d OFM 0-31
            @core(ComputeTile3, "conv2dk3.o")
            def core_body():
                scale = 11
                for _ in range_(sys.maxsize):

                    # acquire weights and rtps once
                    element0Weights = wts_buf_01.acquire(ObjectFifoPort.Consume, 1)
                    # scale = memref.load(rtpComputeTile3, 0)

                    # pre-amble: top row
                    elementActivactionsIn = of_act_2_3_5.acquire(
                        ObjectFifoPort.Consume, 2
                    )
                    element0ActivactionsOut = act_3_4.acquire(ObjectFifoPort.Produce, 1)
                    conv2dk3(
                        elementActivactionsIn[0],
                        elementActivactionsIn[0],
                        elementActivactionsIn[1],
                        element0Weights,
                        element0ActivactionsOut,
                        tensorInW,
                        tensorL2InC,
                        tensorL2OutC,
                        3,
                        3,
                        0,
                        scale,
                        0,
                    )
                    act_3_4.release(ObjectFifoPort.Produce, 1)

                    # middle
                    for _ in range_(tensorInH - 2):
                        elementActivactionsIn = of_act_2_3_5.acquire(
                            ObjectFifoPort.Consume, 3
                        )
                        element0ActivactionsOut = act_3_4.acquire(
                            ObjectFifoPort.Produce, 1
                        )
                        conv2dk3(
                            elementActivactionsIn[0],
                            elementActivactionsIn[1],
                            elementActivactionsIn[2],
                            element0Weights,
                            element0ActivactionsOut,
                            tensorInW,
                            tensorL2InC,
                            tensorL2OutC,
                            3,
                            3,
                            1,
                            scale,
                            0,
                        )
                        of_act_2_3_5.release(ObjectFifoPort.Consume, 1)
                        act_3_4.release(ObjectFifoPort.Produce, 1)

                    # last part
                    elementActivactionsIn = of_act_2_3_5.acquire(
                        ObjectFifoPort.Consume, 2
                    )
                    element0ActivactionsOut = act_3_4.acquire(ObjectFifoPort.Produce, 1)
                    conv2dk3(
                        elementActivactionsIn[0],
                        elementActivactionsIn[1],
                        elementActivactionsIn[1],
                        element0Weights,
                        element0ActivactionsOut,
                        tensorInW,
                        tensorL2InC,
                        tensorL2OutC,
                        3,
                        3,
                        2,
                        scale,
                        0,
                    )

                    of_act_2_3_5.release(ObjectFifoPort.Consume, 2)
                    act_3_4.release(ObjectFifoPort.Produce, 1)
                    wts_buf_01.release(ObjectFifoPort.Consume, 1)

            # 3x3 conv2d OFM 32-63
            @core(ComputeTile5, "conv2dk3.o")
            def core_body():
                scale = 11
                for _ in range_(sys.maxsize):

                    # acquire weights and rtps once
                    element0Weights = wts_buf_01.acquire(ObjectFifoPort.Consume, 1)
                    # scale = memref.load(rtpComputeTile5, 0)

                    # pre-amble: top row
                    elementActivactionsIn = of_act_2_3_5.acquire(
                        ObjectFifoPort.Consume, 2
                    )
                    element0ActivactionsOut = act_5_4.acquire(ObjectFifoPort.Produce, 1)
                    conv2dk3(
                        elementActivactionsIn[0],
                        elementActivactionsIn[0],
                        elementActivactionsIn[1],
                        element0Weights,
                        element0ActivactionsOut,
                        tensorInW,
                        tensorL2InC,
                        tensorL2OutC,
                        3,
                        3,
                        0,
                        scale,
                        tensorL2OutC // 2,
                    )
                    act_5_4.release(ObjectFifoPort.Produce, 1)

                    # middle
                    for _ in range_(tensorInH - 2):
                        elementActivactionsIn = of_act_2_3_5.acquire(
                            ObjectFifoPort.Consume, 3
                        )
                        element0ActivactionsOut = act_5_4.acquire(
                            ObjectFifoPort.Produce, 1
                        )
                        conv2dk3(
                            elementActivactionsIn[0],
                            elementActivactionsIn[1],
                            elementActivactionsIn[2],
                            element0Weights,
                            element0ActivactionsOut,
                            tensorInW,
                            tensorL2InC,
                            tensorL2OutC,
                            3,
                            3,
                            1,
                            scale,
                            tensorL2OutC // 2,
                        )
                        of_act_2_3_5.release(ObjectFifoPort.Consume, 1)
                        act_5_4.release(ObjectFifoPort.Produce, 1)

                    # last part
                    elementActivactionsIn = of_act_2_3_5.acquire(
                        ObjectFifoPort.Consume, 2
                    )
                    element0ActivactionsOut = act_5_4.acquire(ObjectFifoPort.Produce, 1)
                    conv2dk3(
                        elementActivactionsIn[0],
                        elementActivactionsIn[1],
                        elementActivactionsIn[1],
                        element0Weights,
                        element0ActivactionsOut,
                        tensorInW,
                        tensorL2InC,
                        tensorL2OutC,
                        3,
                        3,
                        2,
                        scale,
                        tensorL2OutC // 2,
                    )
                    of_act_2_3_5.release(ObjectFifoPort.Consume, 2)
                    act_5_4.release(ObjectFifoPort.Produce, 1)
                    wts_buf_01.release(ObjectFifoPort.Consume, 1)

            # # 1x1 conv2d and add skip
            @core(ComputeTile4, "conv2dk1_skip.o")
            def core_body():
                for _ in range_(sys.maxsize):

                    # acquire weights and rtps once
                    element0Weights = wts_buf_02.acquire(ObjectFifoPort.Consume, 1)
                    scale = rtpComputeTile4[0]
                    skipScale = rtpComputeTile4[1]

                    for _ in range_(tensorInH):
                        element0ActivactionsIn = act_3_4.acquire(
                            ObjectFifoPort.Consume, 1
                        )
                        element1ActivactionsIn = act_5_4.acquire(
                            ObjectFifoPort.Consume, 1
                        )
                        elementSkipsIn = of_skip_buf.acquire(ObjectFifoPort.Consume, 1)
                        elementActivactionsOut = outOFL2L3.acquire(
                            ObjectFifoPort.Produce, 1
                        )

                        conv2dk1_skip(
                            element0ActivactionsIn,
                            element1ActivactionsIn,
                            element0Weights,
                            elementActivactionsOut,
                            elementSkipsIn,
                            tensorInW,
                            tensorL3InC,
                            tensorL3OutC,
                            scale,
                            skipScale,
                        )
                        outOFL2L3.release(ObjectFifoPort.Produce, 1)
                        act_3_4.release(ObjectFifoPort.Consume, 1)
                        act_5_4.release(ObjectFifoPort.Consume, 1)
                        of_skip_buf.release(ObjectFifoPort.Consume, 1)
                    wts_buf_02.release(ObjectFifoPort.Consume, 1)

            # instruction stream generation
            @runtime_sequence(activationsInL3_ty, weightsInL3_ty, activationsInL3_ty)
            def sequence(inputFromL3, weightsFromL3, outputToL3):
                # write RTP parameters
                rtpComputeTile2[0] = 1  # scale
                rtpComputeTile3[0] = 1  # scale
                rtpComputeTile5[0] = 1  # scale
                # scale: conv1x1 with the same scale as the input so we match the scaling factor of output after conv1x1 and the initial input
                rtpComputeTile4[0] = 1
                rtpComputeTile4[1] = 0  # skip_scale

                in_act_task = shim_dma_single_bd_task(
                    of_inOF_act_L3L2, inputFromL3, sizes=[1, 1, 1, activationsIn]
                )
                in_wts_task = shim_dma_single_bd_task(
                    inOF_wts_0_L3L2, weightsFromL3, sizes=[1, 1, 1, totalWeights]
                )
                out_task = shim_dma_single_bd_task(
                    outOFL2L3,
                    outputToL3,
                    sizes=[1, 1, 1, acitivationsOut],
                    issue_token=True,
                )

                dma_start_task(in_act_task, in_wts_task, out_task)

                dma_await_task(out_task)
                dma_free_task(in_act_task, in_wts_task)

    print(ctx.module)


bottleneck4AIEs()
