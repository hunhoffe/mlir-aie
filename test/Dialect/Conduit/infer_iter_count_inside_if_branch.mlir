// RUN: aie-opt --objectfifo-to-conduit %s | FileCheck %s
//
// Task #32 — Pass A dma_repeat inference behaviour when the acquire site lives
// INSIDE an scf.if then-branch under an always-taken (`arith.constant true`)
// guard.
//
// productOfEnclosingLoops in ObjectFifoToConduit.cpp now special-cases
// scf.if: it constant-folds the condition via `foldIfCondition` and, when
// the descending child sits in the always-taken region, continues walking
// past the if as if it weren't there.  This preserves the prior pinned
// behaviour for the constant-true case below: outer scf.for trip = 6 is
// folded in and `dma_repeat = 6` is stamped.
//
// Geometry:
//   outer scf.for trip = 6
//   acquire nested in scf.if then-branch (else exists, empty), guard = true
//   no aiex.dma_configure_task_for → acquires_per_BD = 1
//   → dma_repeat = 6
//
// Sibling: `infer_iter_count_scf_if_conditional_acquire_overcount.mlir`
// pins the unfoldable-condition counterpart (Task #38) where the same
// helper now returns Dynamic and Pass A skips with a remark.
//
// Hint cross-ref: gemm uses `if rtp_n_tiles_per_core > 1: loop = range_(...)`
// Python-level branching that may emit conditional IR shaped like this.

// CHECK-LABEL: module @infer_inside_if_branch
// CHECK: conduit.create @chan
// CHECK-SAME: dma_repeat = 6

module @infer_inside_if_branch {
  aie.device(npu1) {
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)

    aie.objectfifo @chan(%tile_0_0, {%tile_0_2}, 2 : i32)
        : !aie.objectfifo<memref<128xbf16>>

    aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c6 = arith.constant 6 : index
      %true = arith.constant true
      scf.for %i = %c0 to %c6 step %c1 {
        scf.if %true {
          %sub = aie.objectfifo.acquire @chan (Consume, 1)
              : !aie.objectfifosubview<memref<128xbf16>>
          %elem = aie.objectfifo.subview.access %sub[0]
              : !aie.objectfifosubview<memref<128xbf16>> -> memref<128xbf16>
          aie.objectfifo.release @chan (Consume, 1)
        } else {
        }
      }
      aie.end
    }
  }
}
