// RUN: aie-opt --objectfifo-to-conduit %s | FileCheck %s
//
// Task #32 — Pass A precedence pin: explicit iter_count wins over inference
// even when the two values DISAGREE.
//
// Closely related to infer_iter_count_explicit_iron_set_iter_count.mlir, but
// specifically constructed so the inferred and explicit values are different.
// Without the `if (!iterCountAttr && ...)` guard in ObjectFifoToConduit.cpp
// the inference would overwrite the IRON-set value and silently change the
// shim BD-chain replay count.
//
// Note: aie.objectfifo's iter_count attr is verifier-restricted to MemTile
// producers, so the topology here is MemTile(0,1) → compute(0,2).
//
// Geometry:
//   explicit iter_count = 4 on the source aie.objectfifo
//   consumer scf.for trip = 12  (would otherwise infer dma_repeat = 12)
//   → inference would compute 12; explicit 4 must win
//   → conduit.create @chan dma_repeat = 4

// CHECK-LABEL: module @infer_explicit_overrides_inference
// CHECK: conduit.create @chan
// CHECK-SAME: dma_repeat = 4

module @infer_explicit_overrides_inference {
  aie.device(npu1) {
    %tile_0_1 = aie.tile(0, 1)
    %tile_0_2 = aie.tile(0, 2)

    aie.objectfifo @chan(%tile_0_1, {%tile_0_2}, 2 : i32)
        {iter_count = 4 : i32}
        : !aie.objectfifo<memref<64xbf16>>

    aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c12 = arith.constant 12 : index
      // trip=12 — inference would derive dma_repeat=12, but explicit
      // iter_count=4 must take precedence.
      scf.for %i = %c0 to %c12 step %c1 {
        %sub = aie.objectfifo.acquire @chan (Consume, 1)
            : !aie.objectfifosubview<memref<64xbf16>>
        %elem = aie.objectfifo.subview.access %sub[0]
            : !aie.objectfifosubview<memref<64xbf16>> -> memref<64xbf16>
        aie.objectfifo.release @chan (Consume, 1)
      }
      aie.end
    }
  }
}
