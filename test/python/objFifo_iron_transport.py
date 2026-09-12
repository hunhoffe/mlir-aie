# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# RUN: %python %s | FileCheck %s

import numpy as np

from aie.iron import ObjectFifo, Program, Runtime, Transport, Worker
from aie.iron.controlflow import range_
from aie.iron.device import NPU1Col1


# The IRON `transport` argument lands on the op as the transport attribute,
# whether it is given as a Transport or as a bare mode name.
# CHECK: aie.objectfifo @of_in({{.*}}) {transport = #aie.transport<dma>} : !aie.objectfifo<memref<16xi32>>
# CHECK: aie.objectfifo @of_out({{.*}}) {transport = #aie.transport<dma>} : !aie.objectfifo<memref<16xi32>>
def test_objectfifo_transport():
    dev = NPU1Col1()
    tile_ty = np.ndarray[(16,), np.dtype[np.int32]]

    of_in = ObjectFifo(tile_ty, depth=2, name="of_in", transport=Transport.dma())
    of_out = ObjectFifo(tile_ty, depth=2, name="of_out", transport="dma")

    def body(of_in_c, of_out_p):
        for _ in range_(2):
            elem_in = of_in_c.acquire(1)
            elem_out = of_out_p.acquire(1)
            for i in range_(16):
                elem_out[i] = elem_in[i]
            of_in_c.release(1)
            of_out_p.release(1)

    worker = Worker(body, fn_args=[of_in.cons(), of_out.prod()])

    tensor_ty = np.ndarray[(32,), np.dtype[np.int32]]

    def sequence(a, b, in_h, out_h):
        in_h.fill(a)
        out_h.drain(b, wait=True)

    rt = Runtime(sequence, [tensor_ty, tensor_ty, of_in.prod(), of_out.cons()])
    module = Program(dev, rt, workers=[worker]).resolve_program()
    print(module)


# A stream transport is spelled with both of its parameters, and a bare
# "stream" is refused because it would leave the ports unplaced.
# CHECK: #aie.transport<stream, ends = both, port = 1>
# CHECK: a stream transport needs ends and port
def test_transport_spellings():
    print(Transport.stream(ends="both", port=1))
    try:
        Transport.coerce("stream")
    except ValueError as e:
        print(e)


if __name__ == "__main__":
    test_objectfifo_transport()
    test_transport_spellings()
