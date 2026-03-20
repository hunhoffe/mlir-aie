// RUN: aie-opt --allow-unregistered-dialect --air-channel-to-conduit --conduit-to-dma %s | FileCheck %s
//
// Regression test: Tier 3 ops (put/get_memref_async) inside aie.core bodies
// must be lowered to use_lock ops (Pass C Phase 6, Steps 8e-8f).
//
// When hierarchy-produced IR (via --air-hierarchy-to-aie -> Pass B) places
// conduit.put_memref_async / conduit.get_memref_async inside aie.core
// regions, Pass C must emit use_lock to synchronize the core with the DMA
// engine.  Without these, the core never waits for DMA completion and all
// hardware data would be corrupted or stale.
//
// Producer (put): acquire prodLock (empty slot) + release consLock (data ready)
// Consumer (get): acquire consLock (data arrived) + release prodLock (slot free)
//
// Input: hierarchy-produced IR with non-adjacent tiles (2,3) and (4,5).
// Pipeline: --air-channel-to-conduit --conduit-to-dma
//
// Verifies:
//   1. use_lock ops appear in both core bodies
//   2. Correct lock polarity (acquire/release on correct locks)
//   3. No residual conduit ops
//   4. DMA BD chains exist on both tiles

// All four lock definitions appear before the core bodies.
// CHECK:       %[[PROD_LOCK:.*]] = aie.lock(%{{.*}}, 0) {init = 1 : i32, sym_name = "channel_0_prod_lock_0"}
// CHECK:       %[[CONS_LOCK:.*]] = aie.lock(%{{.*}}, 1) {init = 0 : i32, sym_name = "channel_0_cons_lock_0"}
// CHECK:       %[[C_PROD_LOCK:.*]] = aie.lock(%{{.*}}, 0) {init = 1 : i32, sym_name = "channel_0_cons_prod_lock_0"}
// CHECK:       %[[C_CONS_LOCK:.*]] = aie.lock(%{{.*}}, 1) {init = 0 : i32, sym_name = "channel_0_cons_cons_lock_0"}

// --- Producer core: acquire prodLock, release consLock ---
// CHECK:       aie.core
// CHECK:         aie.use_lock(%[[PROD_LOCK]], AcquireGreaterEqual, 1)
// CHECK-NEXT:    aie.use_lock(%[[CONS_LOCK]], Release, 1)

// --- Consumer core: acquire consLock, release prodLock ---
// CHECK:       aie.core
// CHECK:         aie.use_lock(%[[C_CONS_LOCK]], AcquireGreaterEqual, 1)
// CHECK-NEXT:    aie.use_lock(%[[C_PROD_LOCK]], Release, 1)

// --- DMA BD chains exist ---
// CHECK:       aie.mem
// CHECK:         aie.dma_start(MM2S
// CHECK:       aie.mem
// CHECK:         aie.dma_start(S2MM

// --- No residual conduit ops ---
// CHECK-NOT: conduit.put_memref_async
// CHECK-NOT: conduit.get_memref_async
// CHECK-NOT: conduit.create

module @test_tier3_locks {
  aie.device(xcve2802) @segment_0 {
    %tile_2_3 = aie.tile(2, 3)
    %tile_4_5 = aie.tile(4, 5)
    %core_2_3 = aie.core(%tile_2_3) {
      %alloc = memref.alloc() : memref<32xi32, 2>
      "air.channel.put"(%alloc) {chan_name = @channel_0, id = 1 : i32, operand_segment_sizes = array<i32: 0, 0, 1, 0, 0, 0>} : (memref<32xi32, 2>) -> ()
      memref.dealloc %alloc : memref<32xi32, 2>
      aie.end
    }
    %core_4_5 = aie.core(%tile_4_5) {
      %alloc = memref.alloc() : memref<32xi32, 2>
      "air.channel.get"(%alloc) {chan_name = @channel_0, id = 2 : i32, operand_segment_sizes = array<i32: 0, 0, 1, 0, 0, 0>} : (memref<32xi32, 2>) -> ()
      memref.dealloc %alloc : memref<32xi32, 2>
      aie.end
    }
    "air.channel"() {sym_name = "channel_0"} : () -> ()
  }
}
