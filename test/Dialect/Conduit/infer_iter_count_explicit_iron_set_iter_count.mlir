// RUN: aie-opt --objectfifo-to-conduit %s | FileCheck %s
//
// Task #32 — Pass A defers to explicit iter_count from IRON's
// set_iter_count(N) machinery and SKIPS inference entirely.
//
// In ObjectFifoToConduit.cpp the wiring is:
//   if (op.getIterCount().has_value()) iterCountAttr = ...explicit...;
//   if (!iterCountAttr && !routingSkipsDma) {
//     auto inferred = inferDmaRepeatForChannel(...);
//     if (inferred.dmaRepeat) iterCountAttr = ...inferred...;
//   }
// i.e., explicit always wins, inference is bypassed when the source
// aie.objectfifo carries iter_count.
//
// Note: aie.objectfifo's iter_count attr is verifier-restricted to MemTile
// producers, so the topology here is MemTile(0,1) → compute(0,2).
//
// Geometry:
//   explicit iter_count = 5 on the source aie.objectfifo
//   consumer scf.for trip = 10  (would otherwise infer dma_repeat = 10)
//   → conduit.create @chan dma_repeat = 5
//
// Pinning this guards the IRON-set explicit value from being silently
// overwritten by a future implementation that computes inference first then
// overwrites or that drops the explicit-defer guard.

// CHECK-LABEL: module @infer_explicit_iron_set_iter_count
// CHECK: conduit.create @chan
// CHECK-SAME: dma_repeat = 5

module @infer_explicit_iron_set_iter_count {
  aie.device(npu1) {
    %tile_0_1 = aie.tile(0, 1)
    %tile_0_2 = aie.tile(0, 2)

    aie.objectfifo @chan(%tile_0_1, {%tile_0_2}, 2 : i32)
        {iter_count = 5 : i32}
        : !aie.objectfifo<memref<128xbf16>>

    aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c10 = arith.constant 10 : index
      // trip=10 here; if inference fired it would stamp dma_repeat=10.
      // Explicit iter_count=5 must win.
      scf.for %i = %c0 to %c10 step %c1 {
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
