#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2023 AMD Inc.
import numpy as np
import sys

from aie.extras.context import mlir_mod_ctx
from aie.dialects.aie import *
from aie.dialects.aiex import *
from aie.helpers.dialects.ext.scf import _for as range_


def my_matmul():
    M = 288
    K = 288
    m = 32
    k = 32

    n_cores = 1
    C_sz_div_n_cores = M // n_cores
    M_div_m_div_n_cores = M // (m * n_cores)
    K_div_k = K // k
    m_x_K = m * K

    dtype_in = np.dtype[np.int16]
    dtype_in_str = "i16"
    dtype_out = np.dtype[np.int32]
    dtype_out_str = "i32"

    with mlir_mod_ctx() as ctx:

        @device(AIEDevice.npu1_4col)
        def device_body():
            inA_ty = np.ndarray[(m, k), dtype_in]
            inB_ty = np.ndarray[(k,), dtype_in]
            outC_ty = np.ndarray[(m,), dtype_out]

            # AIE Core Function declarations
            zero = external_func(f"zero_scalar_{dtype_out_str}", inputs=[outC_ty])
            matvec = external_func(
                f"matvec_scalar_{dtype_in_str}_{dtype_out_str}",
                inputs=[inA_ty, inB_ty, outC_ty],
            )

            # Tile declarations
            ShimTiles = []
            MemTiles = []
            cores = []
            for i in range(n_cores):
                ShimTiles.append(tile(i, 0))
                MemTiles.append(tile(i, 1))
                cores.append(tile(i, 2))

            memA_fifos = []
            inA_fifos = []
            outC_fifos = []

            # AIE-array data movement with object fifos
            # Input A
            for i in range(n_cores):
                memA_fifos.append(
                    object_fifo(f"memA{i}", ShimTiles[i], MemTiles[i], 2, inA_ty)
                )
                inA_fifos.append(
                    object_fifo(
                        f"inA{i}",
                        MemTiles[i],
                        cores[i],
                        2,
                        inA_ty,
                    )
                )
                object_fifo_link(memA_fifos[i], inA_fifos[i])

                # Output C
                outC_fifos.append(
                    object_fifo(f"outC{i}", cores[i], ShimTiles[i], 2, outC_ty)
                )

            # Input B
            inB_fifo = object_fifo(
                "inB", ShimTiles[1 % n_cores], cores[0:n_cores], 2, inB_ty
            )

            # Set up compute tiles
            for i in range(n_cores):
                # Compute tile i
                @core(cores[i], f"mv_{m}x{k}.o")
                def core_body():
                    for _ in range_(sys.maxsize):
                        elem_out = outC_fifos[i].acquire(
                            ObjectFifoPort.Produce,
                            1,
                        )
                        zero(elem_out)

                        for _ in range_(K_div_k):
                            elem_in_a = inA_fifos[i].acquire(ObjectFifoPort.Consume, 1)
                            elem_in_b = inB_fifo.acquire(ObjectFifoPort.Consume, 1)
                            matvec(elem_in_a, elem_in_b, elem_out)
                            inA_fifos[i].release(ObjectFifoPort.Consume, 1)
                            inB_fifo.release(ObjectFifoPort.Consume, 1)

                        outC_fifos[i].release(ObjectFifoPort.Produce, 1)

            # To/from AIE-array data movement

            @runtime_sequence(
                np.ndarray[(M, K), dtype_in],
                np.ndarray[(K,), dtype_in],
                np.ndarray[(M,), dtype_out],
            )
            def sequence(A, B, C):
                b_task = shim_dma_single_bd_task(
                    inB_fifo,
                    B,
                    sizes=[M_div_m_div_n_cores, 1, 1, K],
                    strides=[0, 0, 0, 1],
                )

                ab_tasks = [b_task]
                c_tasks = []
                for i in range(n_cores):
                    A_offset = i * M_div_m_div_n_cores * m * K
                    C_offset = i * M_div_m_div_n_cores * m

                    a_task = shim_dma_single_bd_task(
                        memA_fifos[i],
                        A,
                        offset=A_offset,
                        sizes=[M_div_m_div_n_cores, K_div_k, m, k],
                        strides=[m_x_K, k, K, 1],
                    )
                    ab_tasks.append(a_task)

                    c_task = shim_dma_single_bd_task(
                        outC_fifos[i],
                        C,
                        offset=C_offset,
                        sizes=[1, 1, 1, C_sz_div_n_cores],
                        issue_token=True,
                    )
                    c_tasks.append(c_task)

                dma_start_task(*ab_tasks)
                dma_start_task(*c_tasks)

                dma_await_task(*c_tasks)
                dma_free_task(*ab_tasks)

    print(ctx.module)


my_matmul()
