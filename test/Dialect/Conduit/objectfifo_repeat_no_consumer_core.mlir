// RUN: aie-opt --objectfifo-to-conduit --conduit-to-dma %s | FileCheck %s
//
// Tests repeat_count=3 with a producer core but no consumer core.
// Adapted from objectFifo-stateful-transform/repeat_count/repeat_count_test.mlir
// (tile_1_3 is declared as a consumer tile but has no aie.core block).
//
// The DMA still needs to fire even when no consumer core is present — the
// consumer-tile aie.mem S2MM BD chain must be emitted.
//
// Lock correctness:
// - Producer-side lock (tile_1_2): init = depth * repeat_count = 1 * 3 = 3
//   because the producer core acquires all 3 slots before releasing to DMA.
// - Consumer-side lock (tile_1_3): init = depth = 1
//   The consumer FIFO has only depth=1 slot; repeat_count does NOT multiply here.
//   The DMA fires 3 times per buffer slot but the FIFO lock tracks only
//   how many slots are ready to receive, which is always depth.

// CHECK-LABEL: module @repeatCount
// Producer-side locks: init = depth * repeat_count = 1 * 3 = 3
// CHECK-DAG:   aie.lock({{.*tile_1_2.*}}) {init = 3 : i32, sym_name = "of1_prod_lock_0"}
// CHECK-DAG:   aie.lock({{.*tile_1_2.*}}) {init = 0 : i32, sym_name = "of1_cons_lock_0"}
// Consumer-tile lock: init = depth = 1 (NOT repeat_count * depth)
// CHECK-DAG:   aie.lock({{.*tile_1_3.*}}) {init = 1 : i32, sym_name = "of1_cons_prod_lock_0"}
// CHECK-DAG:   aie.lock({{.*tile_1_3.*}}) {init = 0 : i32, sym_name = "of1_cons_cons_lock_0"}
// Producer core acquires/releases repeat_count=3 units at once
// CHECK:       aie.use_lock({{.*}}, AcquireGreaterEqual, 3)
// CHECK:       aie.use_lock({{.*}}, Release, 3)
// Flow is emitted even with no consumer core
// CHECK:       aie.flow(%{{.*}}, DMA : 0, %{{.*}}, DMA : 0)
// Producer DMA: 3 BD blocks (one per repeat_count), circular
// CHECK:       aie.mem(%{{.*tile_1_2.*}})
// CHECK:         aie.dma_start(MM2S
// CHECK:         aie.dma_bd
// CHECK:         aie.next_bd
// CHECK:         aie.dma_bd
// CHECK:         aie.next_bd
// CHECK:         aie.dma_bd
// Circular: loops back to first BD
// CHECK:         aie.next_bd ^bb1
// Consumer DMA is emitted even with no consumer core
// CHECK:       aie.mem(%{{.*tile_1_3.*}})
// CHECK:         aie.dma_start(S2MM
// CHECK:         aie.dma_bd
// No residual Conduit ops
// CHECK-NOT: conduit.create
// CHECK-NOT: conduit.acquire
// CHECK-NOT: conduit.release

module @repeatCount {
  aie.device(npu1) {
    %tile12 = aie.tile(1, 2)
    %tile13 = aie.tile(1, 3)

    aie.objectfifo @of1 (%tile12, {%tile13}, 1 : i32) {repeat_count = 3 : i32}
        : !aie.objectfifo<memref<16xi32>>

    func.func @some_work(%lineOut : memref<16xi32>) -> () {
       return
    }

    %core12 = aie.core(%tile12) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %height = arith.constant 12 : index

      scf.for %indexInHeight = %c0 to %height step %c1 {
         %subview = aie.objectfifo.acquire @of1 (Produce, 1) : !aie.objectfifosubview<memref<16xi32>>
         %elem0 = aie.objectfifo.subview.access %subview[0] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>
         func.call @some_work(%elem0) : (memref<16xi32>) -> ()
         aie.objectfifo.release @of1 (Produce, 1)
      }

      aie.end
    }
  }
}
