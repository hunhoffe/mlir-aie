// RUN: aie-opt --objectfifo-to-conduit %s | FileCheck %s
//
// Task #32 — Pass A dma_repeat inference handles MULTI-DIM scf.parallel.
//
// The single-dim scf.parallel case is pinned by infer_iter_count_scf_parallel.mlir.
// This test pins the multi-dim variant: scf.parallel(0,4,1)(0,2,1) wraps the
// acquire site with two induction variables.  tripCountOfLoop's scf.parallel
// handler multiplies trips across all dims:
//   trip = 4 * 2 = 8
//   no aiex.dma_configure_task_for → emissions=1, acquires_per_BD=1
//   → dma_repeat = 8
//
// Pinning this guards against a future regression where someone walks only
// the first induction variable of an scf.parallel.

// CHECK-LABEL: module @infer_multidim_scf_parallel
// CHECK: conduit.create @chan
// CHECK-SAME: dma_repeat = 8

module @infer_multidim_scf_parallel {
  aie.device(npu1) {
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)

    aie.objectfifo @chan(%tile_0_0, {%tile_0_2}, 2 : i32)
        : !aie.objectfifo<memref<64xbf16>>

    aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c2 = arith.constant 2 : index
      %c4 = arith.constant 4 : index
      %c1 = arith.constant 1 : index
      // 2-dim scf.parallel: outer dim trips 4, inner dim trips 2 → total 8.
      scf.parallel (%i, %j) = (%c0, %c0) to (%c4, %c2) step (%c1, %c1) {
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
