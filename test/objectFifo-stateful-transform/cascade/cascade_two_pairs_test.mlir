//===- cascade_two_pairs_test.mlir --------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-objectfifo-split %s | FileCheck %s

// Two acquire and release pairs for one fifo in one block. Each pair is
// rewritten against the object its own acquire returned, so the values keep
// the order the core wrote them in.

// CHECK-LABEL: @two_pairs
// CHECK:         aie.core(%{{.*}}tile_0_3)
// CHECK-DAG:       %[[A:.*]] = arith.constant dense<1> : vector<16xi32>
// CHECK-DAG:       %[[B:.*]] = arith.constant dense<2> : vector<16xi32>
// CHECK:           aie.put_cascade(%[[A]] : vector<16xi32>)
// CHECK:           aie.put_cascade(%[[B]] : vector<16xi32>)

// CHECK:         aie.core(%{{.*}}tile_1_3)
// CHECK:           %[[R0:.*]] = aie.get_cascade() : vector<16xi32>
// CHECK:           vector.print %[[R0]] : vector<16xi32>
// CHECK:           %[[R1:.*]] = aie.get_cascade() : vector<16xi32>
// CHECK:           vector.print %[[R1]] : vector<16xi32>

module @two_pairs {
 aie.device(npu1) {
    %tile03 = aie.tile(0, 3)
    %tile13 = aie.tile(1, 3)

    aie.objectfifo @cas (%tile03, {%tile13}, 1 : i32) {transport = #aie.transport<cascade>}
        : !aie.objectfifo<memref<1xvector<16xi32>>>

    %core03 = aie.core(%tile03) {
      %c0 = arith.constant 0 : index
      %a = arith.constant dense<1> : vector<16xi32>
      %b = arith.constant dense<2> : vector<16xi32>
      %e1 = aie.objectfifo.acquire @cas (Produce, 1) : memref<1xvector<16xi32>>
      memref.store %a, %e1[%c0] : memref<1xvector<16xi32>>
      aie.objectfifo.release @cas (Produce, 1)
      %e2 = aie.objectfifo.acquire @cas (Produce, 1) : memref<1xvector<16xi32>>
      memref.store %b, %e2[%c0] : memref<1xvector<16xi32>>
      aie.objectfifo.release @cas (Produce, 1)
      aie.end
    }

    %core13 = aie.core(%tile13) {
      %c0 = arith.constant 0 : index
      %e1 = aie.objectfifo.acquire @cas (Consume, 1) : memref<1xvector<16xi32>>
      %r1 = memref.load %e1[%c0] : memref<1xvector<16xi32>>
      vector.print %r1 : vector<16xi32>
      aie.objectfifo.release @cas (Consume, 1)
      %e2 = aie.objectfifo.acquire @cas (Consume, 1) : memref<1xvector<16xi32>>
      %r2 = memref.load %e2[%c0] : memref<1xvector<16xi32>>
      vector.print %r2 : vector<16xi32>
      aie.objectfifo.release @cas (Consume, 1)
      aie.end
    }
 }
}
