// RUN: aie-opt --objectfifo-to-conduit %s | FileCheck %s
//
// Task #11 — Pass A dma_repeat inference recognizes scf.parallel as a
// loop-bearing op (in addition to scf.for).
//
// Geometry:
//   single-dim scf.parallel from 0 to 8 step 1 → trip = 8
//   no aiex.dma_configure_task_for → acquires_per_BD = 1
//   → dma_repeat = 8

// CHECK-LABEL: module @infer_scf_parallel
// CHECK: conduit.create @chan
// CHECK-SAME: dma_repeat = 8

module @infer_scf_parallel {
  aie.device(npu1) {
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)

    aie.objectfifo @chan(%tile_0_0, {%tile_0_2}, 2 : i32)
        : !aie.objectfifo<memref<64xbf16>>

    aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.parallel (%i) = (%c0) to (%c8) step (%c1) {
        %sub = aie.objectfifo.acquire @chan (Consume, 1)
            : !aie.objectfifosubview<memref<64xbf16>>
        %elem = aie.objectfifo.subview.access %sub[0]
            : !aie.objectfifosubview<memref<64xbf16>> -> memref<64xbf16>
        aie.objectfifo.release @chan (Consume, 1)
        scf.reduce
      }
      aie.end
    }
  }
}
