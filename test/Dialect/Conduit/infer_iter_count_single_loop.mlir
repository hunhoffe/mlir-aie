// RUN: aie-opt --objectfifo-to-conduit %s | FileCheck %s
//
// Task #11 — Pass A dma_repeat inference, baseline.
//
// When the source aie.objectfifo carries no explicit iter_count and the
// consumer core has a single statically-bounded scf.for around its
// acquire/release, Pass A infers dma_repeat = trip_count.
//
// Geometry:
//   outer trip = 16
//   no aiex.dma_configure_task_for → acquires_per_BD defaults to 1
//   → dma_repeat = 16 / 1 = 16

// CHECK-LABEL: module @infer_single_loop
// CHECK: conduit.create @chan
// CHECK-SAME: dma_repeat = 16

module @infer_single_loop {
  aie.device(npu1) {
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)

    aie.objectfifo @chan(%tile_0_0, {%tile_0_2}, 2 : i32)
        : !aie.objectfifo<memref<128xbf16>>

    aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c16 = arith.constant 16 : index
      %c1 = arith.constant 1 : index
      scf.for %i = %c0 to %c16 step %c1 {
        %sub = aie.objectfifo.acquire @chan (Consume, 1)
            : !aie.objectfifosubview<memref<128xbf16>>
        %elem = aie.objectfifo.subview.access %sub[0]
            : !aie.objectfifosubview<memref<128xbf16>> -> memref<128xbf16>
        aie.objectfifo.release @chan (Consume, 1)
      }
      aie.end
    }
  }
}
