//===- pin_silent_no_op_errors.mlir -----------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: not aie-opt --split-input-file --aie-objectFifo-stateful-transform %s 2>&1 | FileCheck %s

// On NPU2, adjacent CoreTiles (0,2)/(0,3) share memory, so the OF
// produces a single buffer pool on the producer's tile. A consumer-side
// bank pin can't be honored because no consumer-side buffer exists.
// CHECK: 'aie.objectfifo' op `consumer_mem_banks[0]` is set on a shared-memory ObjectFifo
module @cons_pin_on_shared_mem {
  aie.device(npu2) {
    %prod = aie.tile(0, 2)
    %cons = aie.tile(0, 3)
    aie.objectfifo @of(%prod, {%cons}, 2 : i32)
      {consumer_mem_banks = [[0 : i32, 1 : i32]]}
      : !aie.objectfifo<memref<16xi32>>
  }
}

// -----

// aie.objectfifo.allocate (delegate_tile) redirects the buffer pool to a
// third tile, so neither the producer nor consumer endpoint pin applies.
// CHECK: 'aie.objectfifo' op has a `delegate_tile`
module @pin_with_delegate {
  aie.device(npu2) {
    %prod = aie.tile(0, 2)
    %cons = aie.tile(0, 3)
    %delegate = aie.tile(0, 1)
    aie.objectfifo @of(%prod, {%cons}, 2 : i32)
      {producer_mem_bank = [0 : i32, 1 : i32]}
      : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo.allocate @of(%delegate)
  }
}
