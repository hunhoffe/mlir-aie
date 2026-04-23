// RUN: aie-opt --objectfifo-to-conduit %s | FileCheck %s --check-prefix=PASSA
// RUN: aie-opt --objectfifo-to-conduit --conduit-to-dma %s | FileCheck %s --check-prefix=PASSC
//
// Task #128 / Routing Step 4a — round-trip from aie.objectfifo with the
// `routing_mode = "dma"` discardable StringAttr through Pass A and Pass C.
//
//   1. Pass A (--objectfifo-to-conduit) propagates the StringAttr into a
//      typed `routing_mode = #conduit.routing_mode<dma>` enum attr on the
//      resulting conduit.create (already covered by
//      objectfifo_to_conduit_routing_mode_passthrough.mlir test (6); pinned
//      again here as a regression guard).
//
//   2. Pass C (--conduit-to-dma) reads that enum attr and sets
//      forceDMA = true in ConduitToDMACollect, so the lowered IR takes
//      the DMA path (aie.flow emitted) even though tile(0,2) and tile(0,3)
//      are adjacent and would normally take the shared-memory path.
//
// Without Step 4a's predicate change, only `routing_mode = circuit` would
// have set forceDMA — the `dma` enum value would be a cosmetic no-op and
// the conduit would silently land on shared memory.

// PASSA-LABEL: module @rm_dma_passthrough
// PASSA:       conduit.create @chan
// PASSA-SAME:  routing_mode = #conduit.routing_mode<dma>

// PASSC-LABEL: module @rm_dma_passthrough
// PASSC:       aie.device(npu1)
// PASSC:       aie.flow
module @rm_dma_passthrough {
  aie.device(npu1) {
    %t02 = aie.tile(0, 2)
    %t03 = aie.tile(0, 3)
    aie.objectfifo @chan (%t02, {%t03}, 2 : i32)
        {routing_mode = "dma"} : !aie.objectfifo<memref<16xi32>>
    %core02 = aie.core(%t02) {
      %w = aie.objectfifo.acquire @chan(Produce, 1)
          : !aie.objectfifosubview<memref<16xi32>>
      aie.objectfifo.release @chan(Produce, 1)
      aie.end
    }
    %core03 = aie.core(%t03) {
      %w = aie.objectfifo.acquire @chan(Consume, 1)
          : !aie.objectfifosubview<memref<16xi32>>
      aie.objectfifo.release @chan(Consume, 1)
      aie.end
    }
  }
}
