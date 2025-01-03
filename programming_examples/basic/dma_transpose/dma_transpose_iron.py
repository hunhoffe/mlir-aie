# dma_transpose/dma_transpose_iron.py -*- Python -*-
#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2024 Advanced Micro Devices, Inc. or its affiliates
import argparse
import numpy as np
import sys

from aie.iron import ObjectFifo, Program, Runtime
from aie.iron.device import NPU1Col1, AnyComputeTile
from aie.iron.placers import SequentialPlacer
from aie.helpers.taplib import TensorTiler2D


def my_passthrough(M, K):

    # Define types
    tensor_ty = np.ndarray[(M, K), np.dtype[np.int32]]

    # Define tensor access pattern
    tap_in = TensorTiler2D.simple_tiler((M, K), tile_col_major=True)[0]

    # Dataflow with ObjectFifos
    of_in = ObjectFifo(tensor_ty, name="in")
    of_out = of_in.cons().forward(AnyComputeTile, name="out")

    # Runtime operations to move data to/from the AIE-array
    rt = Runtime()
    with rt.sequence(tensor_ty, tensor_ty, tensor_ty) as (a_in, _, c_out):
        rt.fill(of_in.prod(), a_in, tap_in)
        rt.drain(of_out.cons(), c_out, wait=True)

    # Create the program from the device type and runtime
    my_program = Program(NPU1Col1(), rt)

    # Place program components (assign them resources on the device) and generate an MLIR module
    module = my_program.resolve_program(SequentialPlacer())

    # Print the generated MLIR
    print(module)


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
