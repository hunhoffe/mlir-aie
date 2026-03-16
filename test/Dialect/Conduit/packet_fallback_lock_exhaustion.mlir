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
  // expected-error @+1 {{conduit-to-dma: no DMA resources available for conduit 'fallback'}}
  aie.device(npu1) {
    %t02 = aie.tile(0, 2)
    %t03 = aie.tile(0, 3)
    %t23 = aie.tile(2, 3)
    %t33 = aie.tile(3, 3)
    %t25 = aie.tile(2, 5)

    // 2 explicit packet conduits from (0,3): MM2S ch 0 and ch 1 (packet-mode).
    // Phase 3: 2×2 = 4 locks on (0,3) prod side; 2×2 = 4 on consumer tiles.
    conduit.create {name = "pkt_a", capacity = 4 : i64,
                    producer_tile = array<i64: 0, 3>,
                    consumer_tiles = array<i64: 2, 3>,
                    element_type = memref<4xi32>, depth = 1 : i64,
                    routing_mode = "packet"}
    conduit.create {name = "pkt_b", capacity = 4 : i64,
                    producer_tile = array<i64: 0, 3>,
                    consumer_tiles = array<i64: 3, 3>,
                    element_type = memref<4xi32>, depth = 1 : i64,
                    routing_mode = "packet"}

    // 6 shared-memory conduits from (0,3)→(0,2) [adjacent, same column].
    // Phase 3c: 2 locks each allocated on producer tile (0,3).
    // 6×2 = 12 locks on (0,3). Total with pkt_a/b: 4+12 = 16 = limit.
    conduit.create {name = "sm0", capacity = 4 : i64,
                    producer_tile = array<i64: 0, 3>,
                    consumer_tiles = array<i64: 0, 2>,
                    element_type = memref<4xi32>, depth = 1 : i64,
                    routing_mode = "circuit"}
    conduit.create {name = "sm1", capacity = 4 : i64,
                    producer_tile = array<i64: 0, 3>,
                    consumer_tiles = array<i64: 0, 2>,
                    element_type = memref<4xi32>, depth = 1 : i64,
                    routing_mode = "circuit"}
    conduit.create {name = "sm2", capacity = 4 : i64,
                    producer_tile = array<i64: 0, 3>,
                    consumer_tiles = array<i64: 0, 2>,
                    element_type = memref<4xi32>, depth = 1 : i64,
                    routing_mode = "circuit"}
    conduit.create {name = "sm3", capacity = 4 : i64,
                    producer_tile = array<i64: 0, 3>,
                    consumer_tiles = array<i64: 0, 2>,
                    element_type = memref<4xi32>, depth = 1 : i64,
                    routing_mode = "circuit"}
    conduit.create {name = "sm4", capacity = 4 : i64,
                    producer_tile = array<i64: 0, 3>,
                    consumer_tiles = array<i64: 0, 2>,
                    element_type = memref<4xi32>, depth = 1 : i64,
                    routing_mode = "circuit"}
    conduit.create {name = "sm5", capacity = 4 : i64,
                    producer_tile = array<i64: 0, 3>,
                    consumer_tiles = array<i64: 0, 2>,
                    element_type = memref<4xi32>, depth = 1 : i64,
                    routing_mode = "circuit"}

    // mode=any: (0,3) → (2,5); circuit MM2S exhausted; Step 3.5 fires.
    // Step 3.5b: prodLockTotal(16) - prodLockUsed(16) = 0 < 2 → fail → Step 4.
    conduit.create {name = "fallback", capacity = 4 : i64,
                    producer_tile = array<i64: 0, 3>,
                    consumer_tiles = array<i64: 2, 5>,
                    element_type = memref<4xi32>, depth = 1 : i64,
                    routing_mode = "any"}

    %core02 = aie.core(%t02) { aie.end }
    %core03 = aie.core(%t03) { aie.end }
    %core23 = aie.core(%t23) { aie.end }
    %core33 = aie.core(%t33) { aie.end }
    %core25 = aie.core(%t25) { aie.end }
  }
}
