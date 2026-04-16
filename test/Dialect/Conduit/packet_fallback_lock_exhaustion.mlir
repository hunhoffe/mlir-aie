// RUN: aie-opt --conduit-to-dma --verify-diagnostics %s
//
// P2-A Step 3.5b: Packet DMA fallback — lock budget exhausted on producer.
//
// Producer tile (0,3) gets its 16-lock budget consumed by allocations in
// Phase 3:
//   - 2 packet conduits from (0,3) to other tiles: 2×2 = 4 prod-side locks
//   - 6 circuit conduits from (0,3) to adjacent tile (0,2): Phase 3c
//     (shared-memory path) allocates 2 locks per conduit on the producer
//     tile → 6×2 = 12 locks on (0,3)
//   Total: 4 + 12 = 16 locks used on (0,3)
//
// A mode=any conduit (0,3)→(2,5) triggers Step 3.5 since both MM2S channels
// are in packet-mode.  Step 3.5b finds prodLockTotal(16) - prodLockUsed(16)
// = 0 < 2 → returns false → Step 4 error.
//
// Note: the 6 shared-memory conduits go to (0,2) (adjacent, same column).
// npu1_1col: rows 2-5 are compute tiles; (0,2) is adjacent to (0,3).
// The packet conduits go to non-adjacent tiles in other columns.

module @pkt_fallback_lock_exhaustion {
  // expected-error @+1 {{conduit-to-dma: S2MM DMA channel exhausted on tile (0,2): all 2 channels in use}}
  aie.device(npu1) {
    %t02 = aie.tile(0, 2)
    %t03 = aie.tile(0, 3)
    %t23 = aie.tile(2, 3)
    %t33 = aie.tile(3, 3)
    %t25 = aie.tile(2, 5)

    // 2 explicit packet conduits from (0,3): MM2S ch 0 and ch 1 (packet-mode).
    // Phase 3: 2×2 = 4 locks on (0,3) prod side; 2×2 = 4 on consumer tiles.
    conduit.create @pkt_a {                    element_type = memref<4xi32>, depth = 1 : i64,
                    routing_mode = #conduit.routing_mode<packet>}
    conduit.create @pkt_b {                    element_type = memref<4xi32>, depth = 1 : i64,
                    routing_mode = #conduit.routing_mode<packet>}

    // 6 shared-memory conduits from (0,3)→(0,2) [adjacent, same column].
    // Phase 3c: 2 locks each allocated on producer tile (0,3).
    // 6×2 = 12 locks on (0,3). Total with pkt_a/b: 4+12 = 16 = limit.
    conduit.create @sm0 {                    element_type = memref<4xi32>, depth = 1 : i64,
                                        routing_mode = #conduit.routing_mode<circuit>}
    conduit.create @sm1 {                    element_type = memref<4xi32>, depth = 1 : i64,
                                        routing_mode = #conduit.routing_mode<circuit>}
    conduit.create @sm2 {                    element_type = memref<4xi32>, depth = 1 : i64,
                                        routing_mode = #conduit.routing_mode<circuit>}
    conduit.create @sm3 {                    element_type = memref<4xi32>, depth = 1 : i64,
                                        routing_mode = #conduit.routing_mode<circuit>}
    conduit.create @sm4 {                    element_type = memref<4xi32>, depth = 1 : i64,
                                        routing_mode = #conduit.routing_mode<circuit>}
    conduit.create @sm5 {                    element_type = memref<4xi32>, depth = 1 : i64,
                                        routing_mode = #conduit.routing_mode<circuit>}

    // mode=any: (0,3) → (2,5); circuit MM2S exhausted; Step 3.5 fires.
    // Step 3.5b: prodLockTotal(16) - prodLockUsed(16) = 0 < 2 → fail → Step 4.
    conduit.create @fallback {                    element_type = memref<4xi32>, depth = 1 : i64
                    }

    %core03 = aie.core(%t03) {
      conduit.acquire {name = @pkt_a, port = #conduit.port<Produce>, count = 1 : i64} : !conduit.window<memref<4xi32>>
      conduit.acquire {name = @pkt_b, port = #conduit.port<Produce>, count = 1 : i64} : !conduit.window<memref<4xi32>>
      conduit.acquire {name = @sm0, port = #conduit.port<Produce>, count = 1 : i64} : !conduit.window<memref<4xi32>>
      conduit.acquire {name = @sm1, port = #conduit.port<Produce>, count = 1 : i64} : !conduit.window<memref<4xi32>>
      conduit.acquire {name = @sm2, port = #conduit.port<Produce>, count = 1 : i64} : !conduit.window<memref<4xi32>>
      conduit.acquire {name = @sm3, port = #conduit.port<Produce>, count = 1 : i64} : !conduit.window<memref<4xi32>>
      conduit.acquire {name = @sm4, port = #conduit.port<Produce>, count = 1 : i64} : !conduit.window<memref<4xi32>>
      conduit.acquire {name = @sm5, port = #conduit.port<Produce>, count = 1 : i64} : !conduit.window<memref<4xi32>>
      conduit.acquire {name = @fallback, port = #conduit.port<Produce>, count = 1 : i64} : !conduit.window<memref<4xi32>>
      aie.end
    }
    %core02 = aie.core(%t02) {
      conduit.acquire {name = @sm0, port = #conduit.port<Consume>, count = 1 : i64} : !conduit.window<memref<4xi32>>
      conduit.acquire {name = @sm1, port = #conduit.port<Consume>, count = 1 : i64} : !conduit.window<memref<4xi32>>
      conduit.acquire {name = @sm2, port = #conduit.port<Consume>, count = 1 : i64} : !conduit.window<memref<4xi32>>
      conduit.acquire {name = @sm3, port = #conduit.port<Consume>, count = 1 : i64} : !conduit.window<memref<4xi32>>
      conduit.acquire {name = @sm4, port = #conduit.port<Consume>, count = 1 : i64} : !conduit.window<memref<4xi32>>
      conduit.acquire {name = @sm5, port = #conduit.port<Consume>, count = 1 : i64} : !conduit.window<memref<4xi32>>
      aie.end
    }
    %core23 = aie.core(%t23) {
      conduit.acquire {name = @pkt_a, port = #conduit.port<Consume>, count = 1 : i64} : !conduit.window<memref<4xi32>>
      aie.end
    }
    %core33 = aie.core(%t33) {
      conduit.acquire {name = @pkt_b, port = #conduit.port<Consume>, count = 1 : i64} : !conduit.window<memref<4xi32>>
      aie.end
    }
    %core25 = aie.core(%t25) {
      conduit.acquire {name = @fallback, port = #conduit.port<Consume>, count = 1 : i64} : !conduit.window<memref<4xi32>>
      aie.end
    }
  }
}
