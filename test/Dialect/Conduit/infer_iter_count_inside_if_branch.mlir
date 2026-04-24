// RUN: aie-opt --objectfifo-to-conduit %s | FileCheck %s
//
// Task #32 — Pass A dma_repeat inference behaviour when the acquire site lives
// INSIDE an scf.if then-branch.
//
// productOfEnclosingLoops walks parent chain via getParentOp(), and
// tripCountOfLoop returns NotALoop for any op that isn't scf.for / scf.parallel.
// scf.if therefore falls into the NotALoop bucket — silently skipped while
// walking — and the helper continues upward to multiply in the enclosing
// scf.for trip.
//
// Current pinned behaviour (a): scf.if is transparent to the walk; the trip
// count of the enclosing scf.for is honoured as if no conditional existed.
//
// Geometry:
//   outer scf.for trip = 6
//   acquire nested in scf.if then-branch (else exists, empty)
//   no aiex.dma_configure_task_for → acquires_per_BD = 1
//   → dma_repeat = 6
//
// Note: this pin records present semantics.  A future correctness pass might
// argue the acquire is conditional and therefore inference should back off
// (behaviour (b) — skip with remark).  Either change requires updating this
// test deliberately, which is the point of pinning it.
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
