// RUN: aie-opt --conduit-to-dma -split-input-file %s | FileCheck %s
//
// Task #128 / Routing Step 4a — both `dma` and `circuit` modes hit the
// forceDMA = true branch in ConduitToDMACollect.
//
// Case A (routing_mode = dma) and Case B (routing_mode = circuit) on the
// same adjacent compute→compute topology must both bypass the
// shared-memory short-circuit and emit aie.flow (DMA path).
//
// The DIFFERENCE between dma and circuit lives downstream:
//   - circuit: pinned to circuit-switched DMA (Step 3.5 packet fallback
//     skipped).
//   - dma:     downstream is free to pick circuit OR packet under budget
//     constraints.
//
// Step 4a (this task) only validates the forceDMA → aie.flow cascade for
// both modes; downstream packet-vs-circuit budget logic is out of scope.

// -----

// Case A: routing_mode = dma → forceDMA → aie.flow.
//
// CHECK-LABEL: module @rm_dma_emits_flow
// CHECK:       aie.device(npu1)
// CHECK:       aie.flow
module @rm_dma_emits_flow {
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

// -----

// Case B: routing_mode = circuit → forceDMA → aie.flow.
//
// CHECK-LABEL: module @rm_circuit_emits_flow
// CHECK:       aie.device(npu1)
// CHECK:       aie.flow
module @rm_circuit_emits_flow {
  aie.device(npu1) {
    %t02 = aie.tile(0, 2)
    %t03 = aie.tile(0, 3)
    conduit.create @adj {element_type = memref<4xi32>, depth = 1 : i64,
                         routing_mode = #conduit.routing_mode<circuit>}
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
