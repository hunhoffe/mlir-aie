// RUN: aie-opt --objectfifo-to-conduit --conduit-to-dma %s | FileCheck %s
//
// Regression test (A-2): Phase 4.5 must process ALL shim consumers.
//
// When an objectfifo has two shim consumer tiles (both row==0), Pass A must
// emit one aie.shim_dma_allocation for EACH shim consumer, not just the first.
//
// Before the fix, a `break` statement inside the consumer loop stopped after
// finding the first row-0 consumer, silently dropping the second.
//
// Topology:
//   tile(0,2) [compute producer]
//     -> tile(0,0) [shim consumer 0]
//     -> tile(1,0) [shim consumer 1]
//
// Expected: two distinct aie.shim_dma_allocation ops.
//
// CHECK-LABEL: module @two_shim_consumers
// CHECK: aie.device(xcve2302)
// CHECK:     aie.shim_dma_allocation @{{.*}}(
// CHECK:     aie.shim_dma_allocation @{{.*}}(
// CHECK-NOT: conduit.create

module @two_shim_consumers {
  aie.device(xcve2302) {
    %shim0 = aie.tile(0, 0)
    %shim1 = aie.tile(1, 0)
    %tile02 = aie.tile(0, 2)

    // Objectfifo with two shim consumers.
    aie.objectfifo @multi_shim_fifo(%tile02, {%shim0, %shim1}, 1 : i32) :
        !aie.objectfifo<memref<8xi32>>
  }
}
