//===- cascade_test.mlir ------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-objectfifo-split %s | FileCheck %s
// RUN: aie-opt --aie-objectFifo-stateful-transform %s | FileCheck %s

// A cascade fifo names the connection between two neighbouring cores and
// nothing else: there is no buffer to fill, no lock to take and no route to
// program, so the producer's store becomes the value on the wire and the
// consumer's load becomes the value off it.

// CHECK-LABEL: @cascade
// CHECK:         aie.cascade_flow(%{{.*}}tile_0_3, %{{.*}}tile_1_3)

// CHECK:         aie.core(%{{.*}}tile_0_3)
// CHECK:           %[[V:.*]] = arith.constant dense<42> : vector<16xi32>
// CHECK:           aie.put_cascade(%[[V]] : vector<16xi32>)
// CHECK:           aie.end

// CHECK:         aie.core(%{{.*}}tile_1_3)
// CHECK:           %[[R:.*]] = aie.get_cascade() : vector<16xi32>
// CHECK:           vector.print %[[R]] : vector<16xi32>
// CHECK:           aie.end

// Nothing of the fifo survives, and it leaves no buffers or locks behind.
// CHECK-NOT:     aie.objectfifo
// CHECK-NOT:     aie.buffer
// CHECK-NOT:     aie.lock

module @cascade {
 aie.device(npu1) {
    %tile03 = aie.tile(0, 3)
    %tile13 = aie.tile(1, 3)

    aie.objectfifo @cas (%tile03, {%tile13}, 1 : i32) {transport = #aie.transport<cascade>}
        : !aie.objectfifo<memref<1xvector<16xi32>>>

    %core03 = aie.core(%tile03) {
      %c0 = arith.constant 0 : index
      %v = arith.constant dense<42> : vector<16xi32>
      %e = aie.objectfifo.acquire @cas (Produce, 1) : memref<1xvector<16xi32>>
      memref.store %v, %e[%c0] : memref<1xvector<16xi32>>
      aie.objectfifo.release @cas (Produce, 1)
      aie.end
    }

    %core13 = aie.core(%tile13) {
      %c0 = arith.constant 0 : index
      %e = aie.objectfifo.acquire @cas (Consume, 1) : memref<1xvector<16xi32>>
      %r = memref.load %e[%c0] : memref<1xvector<16xi32>>
      vector.print %r : vector<16xi32>
      aie.objectfifo.release @cas (Consume, 1)
      aie.end
    }
 }
}
