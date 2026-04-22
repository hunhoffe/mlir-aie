//===- aiecc_add_dma_task_conversion_flag.mlir ---------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//
//
// Verifies the new aiecc cl::opt --conduit-add-dma-task-conversion injects
// `dma-task-to-conduit` into the conduit resource-allocation pipeline
// INDEPENDENTLY of fusion flags. This lets us exercise Pass C lowering of
// aiex.dma_task ops on full Llama IR without enabling any fusion.
//
// Baseline: --use-conduit alone does NOT include dma-task-to-conduit.
// New flag: --use-conduit --conduit-add-dma-task-conversion DOES include it,
// without adding any fuse-operators / fuse-core-bodies / aie-combine-device
// passes.
//
//===----------------------------------------------------------------------===//

// Baseline: --use-conduit only — no dma-task-to-conduit, no fusion passes.
// RUN: aiecc --no-xchesscc --no-xbridge -n --verbose --use-conduit %s 2>&1 | FileCheck %s --check-prefix=BASE

// New flag enables dma-task-to-conduit independently of fusion.
// RUN: aiecc --no-xchesscc --no-xbridge -n --verbose --use-conduit --conduit-add-dma-task-conversion %s 2>&1 | FileCheck %s --check-prefix=DTC

// Compose with existing fusion flags: both gates active, dma-task-to-conduit
// appears once.
// RUN: aiecc --no-xchesscc --no-xbridge -n --verbose --use-conduit --conduit-add-dma-task-conversion --conduit-fuse-spatial %s 2>&1 | FileCheck %s --check-prefix=BOTH

// BASE: Conduit pipeline: objectfifo-to-conduit,conduit-depth-promote,conduit-to-dma

// DTC: Conduit pipeline: objectfifo-to-conduit,dma-task-to-conduit,conduit-depth-promote,conduit-to-dma

// BOTH: Conduit pipeline: objectfifo-to-conduit,dma-task-to-conduit,aie-combine-device,conduit-fuse-operators,conduit-depth-promote,conduit-to-dma

module {
  aie.device(npu1_1col) {
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)

    aie.objectfifo @of_in(%tile_0_0, {%tile_0_2}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @of_out(%tile_0_2, {%tile_0_0}, 2 : i32) : !aie.objectfifo<memref<16xi32>>

    %core_0_2 = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c16 = arith.constant 16 : index
      %c1_i32 = arith.constant 1 : i32

      %subview_in = aie.objectfifo.acquire @of_in(Consume, 1) : !aie.objectfifosubview<memref<16xi32>>
      %elem_in = aie.objectfifo.subview.access %subview_in[0] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>

      %subview_out = aie.objectfifo.acquire @of_out(Produce, 1) : !aie.objectfifosubview<memref<16xi32>>
      %elem_out = aie.objectfifo.subview.access %subview_out[0] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>

      scf.for %i = %c0 to %c16 step %c1 {
        %val = memref.load %elem_in[%i] : memref<16xi32>
        %result = arith.addi %val, %c1_i32 : i32
        memref.store %result, %elem_out[%i] : memref<16xi32>
      }

      aie.objectfifo.release @of_in(Consume, 1)
      aie.objectfifo.release @of_out(Produce, 1)
      aie.end
    }

    aie.runtime_sequence(%in : memref<16xi32>, %out : memref<16xi32>) {
      %c0 = arith.constant 0 : i64
      %c1 = arith.constant 1 : i64
      %c16 = arith.constant 16 : i64
      aiex.npu.dma_memcpy_nd(%out[%c0,%c0,%c0,%c0][%c1,%c1,%c1,%c16][%c0,%c0,%c0,%c1]) {metadata = @of_out, id = 1 : i64} : memref<16xi32>
      aiex.npu.dma_memcpy_nd(%in[%c0,%c0,%c0,%c0][%c1,%c1,%c1,%c16][%c0,%c0,%c0,%c1]) {metadata = @of_in, id = 0 : i64, issue_token = true} : memref<16xi32>
      aiex.npu.dma_wait {symbol = @of_out}
    }
  }
}
