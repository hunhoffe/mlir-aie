# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# RUN: %python %s | FileCheck %s

"""fill()/drain() take a View: a Layout bound to the runtime buffer it walks.

A GEMM sequence written with NumPy spellings on the runtime tensors
(``A.reshape(...).transpose(...)[i]``) generates the same runtime sequence as
the same design written with ``tile()``/``group()`` and ``tap=``, statically
and with a dispatch-time row count. No NPU: the test compares MLIR.
"""

import aie.iron as iron
import numpy as np
from aie.helpers.taplib import Layout
from aie.iron import (
    DispatchTime,
    In,
    ObjectFifo,
    Out,
    Program,
    Runtime,
    TaskGroup,
    View,
    Worker,
)
from aie.iron.controlflow import range_
from aie.iron.device import NPU1Col1

iron.set_current_device(NPU1Col1())
M, K, N = 128, 64, 64
m, k, n = 32, 16, 32


def build(sequence, n_rows=None):
    a_ty = np.ndarray[(M, K), np.dtype[np.int16]]
    b_ty = np.ndarray[(K, N), np.dtype[np.int16]]
    c_ty = np.ndarray[(M, N), np.dtype[np.int32]]
    tile_a = np.ndarray[(m, k), np.dtype[np.int16]]
    tile_b = np.ndarray[(k, n), np.dtype[np.int16]]
    tile_c = np.ndarray[(m, n), np.dtype[np.int32]]
    of_a = ObjectFifo(tile_a, name="of_a")
    of_b = ObjectFifo(tile_b, name="of_b")
    of_c = ObjectFifo(tile_c, name="of_c")

    def core_fn(a, b, c):
        for _ in range_(K // k):
            a.acquire(1)
            b.acquire(1)
            a.release(1)
            b.release(1)
        c.acquire(1)
        c.release(1)

    worker = Worker(core_fn, [of_a.cons(), of_b.cons(), of_c.prod()])
    args = [a_ty, b_ty, c_ty] + ([] if n_rows is None else [n_rows])
    rt = Runtime(sequence, args + [of_a.prod(), of_b.prod(), of_c.cons()])
    return Program(iron.get_current_device(), rt, workers=[worker]).resolve_program()


def seq_text(design):
    mlir = design.specialize().as_mlir()
    return mlir[mlir.index("aie.runtime_sequence") :]


# ------------------------------------------------- static: two spellings agree


@iron.jit
def with_grid(A: In, B: In, C: Out):
    a_rows = Layout.full((M, K)).tile((m, k)).group((1, K // k))
    b_cols = Layout.full((K, N)).tile((k, n)).group((K // k, N // n), col_major=True)
    c_rows = Layout.full((M, N)).tile((m, n)).group((1, N // n))

    def sequence(A, B, C, a_prod, b_prod, c_cons):
        for i in range(M // m):
            tg = TaskGroup()
            a_prod.fill(A, tap=a_rows[i].repeat(N // n), group=tg)
            b_prod.fill(B, tap=b_cols[0], group=tg)
            c_cons.drain(C, tap=c_rows[i], group=tg, wait=True)
            tg.finish()

    return build(sequence)


@iron.jit
def with_views(A: In, B: In, C: Out):
    def sequence(A, B, C, a_prod, b_prod, c_cons):
        # (M/m, K/k, m, k): tile row i, walked tile by tile, once per column of C.
        A4 = A.reshape(M // m, m, K // k, k).transpose(0, 2, 1, 3)
        # (N/n, K/k, k, n): every tile column of B, down each column first.
        B4 = B.reshape(K // k, k, N // n, n).transpose(2, 0, 1, 3)
        C4 = C.reshape(M // m, m, N // n, n).transpose(0, 2, 1, 3)
        for i in range(M // m):
            tg = TaskGroup()
            a_prod.fill(A4[i].broadcast_to((N // n, *A4[i].shape)), group=tg)
            b_prod.fill(B4, group=tg)
            c_cons.drain(C4[i], group=tg, wait=True)
            tg.finish()

    return build(sequence)


grid_text, view_text = seq_text(with_grid), seq_text(with_views)
print("static: views generate the grid design's sequence:", grid_text == view_text)
print(view_text)
# CHECK: static: views generate the grid design's sequence: True
# CHECK-LABEL: aie.runtime_sequence
# Tile row 0 of A, tile by tile, twice (once per tile column of C); all of B,
# one tile column at a time; tile row 0 of C; then tile row 1 of A.
# CHECK: @of_a
# CHECK-NEXT: aie.dma_bd(%arg0 : {{.*}} offset = 0 {{.*}} sizes = [2, 4, 32, 16] strides = [0, 16, 64, 1])
# CHECK: @of_b
# CHECK-NEXT: aie.dma_bd(%arg1 : {{.*}} offset = 0 {{.*}} sizes = [2, 4, 16, 32] strides = [32, 1024, 64, 1])
# CHECK: @of_c
# CHECK-NEXT: aie.dma_bd(%arg2 : {{.*}} offset = 0 {{.*}} sizes = [1, 2, 32, 32] strides = [0, 32, 64, 1])
# CHECK: @of_a
# CHECK-NEXT: aie.dma_bd(%arg0 : {{.*}} offset = 2048 {{.*}} sizes = [2, 4, 32, 16] strides = [0, 16, 64, 1])


# -------------------------------------- dispatch-time row count, staged index


@iron.jit
def rows_at_dispatch(A: In, B: In, C: Out, *, n_rows: DispatchTime[np.int32] = 2):
    def sequence(A, B, C, n_rows, a_prod, b_prod, c_cons):
        A4 = A.reshape(M // m, m, K // k, k).transpose(0, 2, 1, 3)
        B4 = B.reshape(K // k, k, N // n, n).transpose(2, 0, 1, 3)
        C4 = C.reshape(M // m, m, N // n, n).transpose(0, 2, 1, 3)
        for i in range_(n_rows):
            tg = TaskGroup()
            a_prod.fill(A4[i].broadcast_to((N // n, *A4[i].shape)), group=tg)
            b_prod.fill(B4, group=tg)
            c_cons.drain(C4[i], group=tg, wait=True)
            tg.finish()

    return build(sequence, n_rows)


dyn = rows_at_dispatch.specialize().as_mlir()
print("dispatch-time:", "scf.for" in dyn, "aiex.npu.require" in dyn)
# A staged row index becomes a runtime offset on A and C; B is unchanged.
# CHECK: dispatch-time: True True


# ------------------------------------------------------------- argument rules


@iron.jit
def view_plus_tap(A: In, B: In, C: Out):
    def sequence(A, B, C, a_prod, b_prod, c_cons):
        a_prod.fill(A[0:m], tap=Layout.full((M, K))[0:m])

    return build(sequence)


@iron.jit
def foreign_view(A: In, B: In, C: Out):
    def sequence(A, B, C, a_prod, b_prod, c_cons):
        a_prod.fill(A, tap=B[0:k])

    return build(sequence)


@iron.jit
def old_spelling(A: In, B: In, C: Out):
    def sequence(A, B, C, a_prod, b_prod, c_cons):
        a_prod.fill(A, tap=A[0:m])  # a View as tap= of its own buffer
        a_prod.fill(A, A[m : 2 * m])

    return build(sequence)


for name, design in [
    ("view plus tap", view_plus_tap),
    ("foreign view", foreign_view),
    ("old spelling", old_spelling),
]:
    try:
        design.specialize().as_mlir()
        print(f"{name}: accepted")
    except ValueError as e:
        print(f"{name}: refused: {str(e)[:52]}")
# CHECK: view plus tap: refused: A View already carries its access pattern
# CHECK: foreign view: refused: tap= is a View of a different buffer
# CHECK: old spelling: accepted

assert isinstance(View, type)
