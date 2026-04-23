// RUN: aie-opt --conduit-to-dma -split-input-file %s | FileCheck %s
//
// Task #128 / Routing Step 4a — enum_dma functional in Pass C.
//
// `routing_mode = #conduit.routing_mode<dma>` must set forceDMA = true in
// ConduitToDMACollect, so adjacent-tile compute→compute conduits (which
// would normally take the shared-memory path) are forced onto a
// DMA-mediated path (aie.flow emitted, no shared-memory buffers placed).
//
// Discriminator: presence/absence of `aie.flow` in the lowered IR.
//   - shared-memory path (no DMA): NO aie.flow emitted; buffers + locks
//     placed on the producer tile.
//   - DMA path (forceDMA=true):   aie.flow emitted; conduit takes the
//     normal Phase 4.5a non-adjacent flow path even though tiles are
//     adjacent.
//
// Two split-input cases: identical adjacent compute→compute topology
// (tile(0,2) → tile(0,3) on npu1), differing only in routing_mode.

// -----

// Case A: no routing_mode → forceDMA = false → adjacent → shared memory.
// Shared-memory path emits no aie.flow and no aie.shim_dma_allocation.
//
// CHECK-LABEL: module @rm_absent_shared_memory
// CHECK:       aie.device(npu1)
// CHECK-NOT:   aie.flow
module @rm_absent_shared_memory {
  aie.device(npu1) {
    %t02 = aie.tile(0, 2)
    %t03 = aie.tile(0, 3)
    conduit.create @adj {element_type = memref<4xi32>, depth = 1 : i64}
    %core02 = aie.core(%t02) {
      %w = conduit.acquire {name = @adj, count = 1 : i64,
                            port = #conduit.port<Produce>}
                           : !conduit.window<memref<4xi32>>
      conduit.release %w {count = 1 : i64,
                          port = #conduit.port<Produce>}
                         : !conduit.window<memref<4xi32>>
      aie.end
    }
    %core03 = aie.core(%t03) {
      %w = conduit.acquire {name = @adj, count = 1 : i64,
                            port = #conduit.port<Consume>}
                           : !conduit.window<memref<4xi32>>
      conduit.release %w {count = 1 : i64,
                          port = #conduit.port<Consume>}
                         : !conduit.window<memref<4xi32>>
      aie.end
    }
  }
}

// -----

// Case B: routing_mode = dma → forceDMA = true → adjacent path skipped →
// aie.flow emitted (DMA path).
//
// CHECK-LABEL: module @rm_dma_forces_dma
// CHECK:       aie.device(npu1)
// CHECK:       aie.flow
module @rm_dma_forces_dma {
  aie.device(npu1) {
    %t02 = aie.tile(0, 2)
    %t03 = aie.tile(0, 3)
    conduit.create @adj {element_type = memref<4xi32>, depth = 1 : i64,
                         routing_mode = #conduit.routing_mode<dma>}
    %core02 = aie.core(%t02) {
      %w = conduit.acquire {name = @adj, count = 1 : i64,
                            port = #conduit.port<Produce>}
                           : !conduit.window<memref<4xi32>>
      conduit.release %w {count = 1 : i64,
                          port = #conduit.port<Produce>}
                         : !conduit.window<memref<4xi32>>
      aie.end
    }
    %core03 = aie.core(%t03) {
      %w = conduit.acquire {name = @adj, count = 1 : i64,
                            port = #conduit.port<Consume>}
                           : !conduit.window<memref<4xi32>>
      conduit.release %w {count = 1 : i64,
                          port = #conduit.port<Consume>}
                         : !conduit.window<memref<4xi32>>
      aie.end
    }
  }
}
