// RUN: aie-opt --conduit-to-dma %s 2>&1 | FileCheck %s
//
// Distribroad pattern end-to-end test: Pass C only (input is already Conduit IR).
//
// The distribroad pattern combines distribute + broadcast:
//   - "input_slice"    : shim → tile(2,2); carries a unique data slice
//   - "shared_weights" : shim → tile(2,2); carries an identical broadcast copy
//
// Both channels are acquired asynchronously via conduit.acquire_async.
// The consumer core uses conduit.wait_all to fan-in across both pending grants,
// then conduit.wait_window to materialize each window.
//
// Two separate conduit channels → two buffer allocations, two lock pairs.
// The async path defers use_lock to the wait_window site (not acquire_async or wait_all).
//
// Note on buffer/lock ordering: Phase 3 inserts resources right after the last
// tile op using a fixed insertion point.  Each new conduit's resources are
// inserted at the same insertion point, pushing the previous conduit's resources
// down.  The result is that the LAST conduit declared ("shared_weights") appears
// first in the device body, followed by "input_slice".
//
// Buffer allocation (Phase 3): one buffer per conduit on the consumer tile.
// CHECK-LABEL: module @distribroad
// CHECK:   aie.device(xcve2302) {
// CHECK:     %[[TILE_2_0:.*]] = aie.tile(2, 0)
// CHECK:     %[[TILE_2_2:.*]] = aie.tile(2, 2)
//
// --- shared_weights: consumer-tile buffer and lock pair (appears first due to insertion order) ---
// CHECK:     aie.buffer(%[[TILE_2_2]])
// CHECK-SAME:   sym_name = "shared_weights_cons_buff_0"
//
// CHECK:     %[[SW_PROD_LOCK:.*]] = aie.lock(%[[TILE_2_2]]
// CHECK-SAME:   init = 1
// CHECK-SAME:   sym_name = "shared_weights_cons_prod_lock_0"
// CHECK:     %[[SW_CONS_LOCK:.*]] = aie.lock(%[[TILE_2_2]]
// CHECK-SAME:   init = 0
// CHECK-SAME:   sym_name = "shared_weights_cons_cons_lock_0"
//
// --- input_slice: consumer-tile buffer and lock pair ---
// CHECK:     aie.buffer(%[[TILE_2_2]])
// CHECK-SAME:   sym_name = "input_slice_cons_buff_0"
//
// CHECK:     %[[IS_PROD_LOCK:.*]] = aie.lock(%[[TILE_2_2]]
// CHECK-SAME:   init = 1
// CHECK-SAME:   sym_name = "input_slice_cons_prod_lock_0"
// CHECK:     %[[IS_CONS_LOCK:.*]] = aie.lock(%[[TILE_2_2]]
// CHECK-SAME:   init = 0
// CHECK-SAME:   sym_name = "input_slice_cons_cons_lock_0"
//
// --- Consumer core: async acquire path ---
// use_lock emitted at wait_window site for input_slice (deferred from acquire_async):
// CHECK:     aie.core(%[[TILE_2_2]]) {
// CHECK:       scf.for
// CHECK:         aie.use_lock(%[[IS_CONS_LOCK]], AcquireGreaterEqual, 1)
//
// use_lock emitted at wait_window site for shared_weights (deferred):
// CHECK:         aie.use_lock(%[[SW_CONS_LOCK]], AcquireGreaterEqual, 1)
//
// Release use_locks after compute (Phase 6 Step 2, window from wait_window):
// CHECK:         aie.use_lock(%[[IS_PROD_LOCK]], Release, 1)
// CHECK:         aie.use_lock(%[[SW_PROD_LOCK]], Release, 1)
//
// --- Shim-side locks (Phase 4a, one pair per shim conduit) ---
// CHECK:     aie.lock(%[[TILE_2_0]]
// CHECK-SAME:   sym_name = "input_slice_prod_lock_0"
// CHECK:     aie.lock(%[[TILE_2_0]]
// CHECK-SAME:   sym_name = "input_slice_cons_lock_0"
// CHECK:     aie.shim_dma_allocation @input_slice_shim_alloc
// CHECK:     aie.flow(%[[TILE_2_0]], DMA : 0, %[[TILE_2_2]], DMA : 0)
// CHECK:     aie.lock(%[[TILE_2_0]]
// CHECK-SAME:   sym_name = "shared_weights_prod_lock_0"
// CHECK:     aie.lock(%[[TILE_2_0]]
// CHECK-SAME:   sym_name = "shared_weights_cons_lock_0"
// CHECK:     aie.shim_dma_allocation @shared_weights_shim_alloc
// CHECK:     aie.flow(%[[TILE_2_0]], DMA : 0, %[[TILE_2_2]], DMA : 0)
//
// --- Tile DMA region (Phase 5.5): single aie.mem with S2MM per conduit ---
// CHECK:     aie.mem(%[[TILE_2_2]]) {
// CHECK:       aie.dma_start(S2MM, 0,
// CHECK:       aie.use_lock(%[[IS_PROD_LOCK]], AcquireGreaterEqual, 1)
// CHECK:       aie.dma_bd({{.*}}input_slice{{.*}}buff_0
// CHECK:       aie.use_lock(%[[IS_CONS_LOCK]], Release, 1)
// CHECK:       aie.next_bd
// CHECK:       aie.dma_start(S2MM, 1,
// CHECK:       aie.use_lock(%[[SW_PROD_LOCK]], AcquireGreaterEqual, 1)
// CHECK:       aie.dma_bd({{.*}}shared_weights{{.*}}buff_0
// CHECK:       aie.use_lock(%[[SW_CONS_LOCK]], Release, 1)
// CHECK:       aie.next_bd
// CHECK:       aie.end
//
// No residual Conduit ops:
// CHECK-NOT: conduit.create
// CHECK-NOT: conduit.acquire_async
// CHECK-NOT: conduit.wait_all
// CHECK-NOT: conduit.wait_window
// CHECK-NOT: conduit.subview_access
// CHECK-NOT: conduit.release
//
// ============================================================
// TEST INPUT: Conduit IR (not ObjectFIFO — Pass C only)
// ============================================================
//
// Topology:
//   shim tile(2,0) --[input_slice]--> tile(2,2)    unique data slice
//   shim tile(2,0) --[shared_weights]--> tile(2,2) broadcast weights (same data)
//
// The consumer core uses conduit.acquire_async + conduit.wait_all to launch
// both grants concurrently, then conduit.wait_window to materialize each window
// at the point where the lock is actually needed.  This expresses the
// distribroad pattern: distribute (unique slice) + broadcast (same weights)
// via two concurrent acquire_async tokens combined in a single wait_all.

module @distribroad {
  aie.device(xcve2302) {
    %tile_2_0 = aie.tile(2, 0)
    %tile_2_2 = aie.tile(2, 2)

    // Conduit 1: shim DMA (MM2S) → tile(2,2).
    // Carries a unique input slice (distribute: each consumer gets different data).
    conduit.create {name = "input_slice",
                    capacity = 16 : i64,
                    producer_tile = array<i64: 2, 0>,
                    consumer_tiles = array<i64: 2, 2>,
                    element_type = memref<16xi32>,
                    depth = 1 : i64}

    // Conduit 2: shim DMA (MM2S) → tile(2,2).
    // Carries broadcast shared weights (all consumers receive the same data).
    conduit.create {name = "shared_weights",
                    capacity = 8 : i64,
                    producer_tile = array<i64: 2, 0>,
                    consumer_tiles = array<i64: 2, 2>,
                    element_type = memref<8xi32>,
                    depth = 1 : i64}

    %core_2_2 = aie.core(%tile_2_2) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c4 = arith.constant 4 : index

      scf.for %i = %c0 to %c4 step %c1 {

        // Non-blocking async acquires: emit NOTHING in hardware here.
        // Each token represents a pending lock grant; the hardware DMA fills
        // the buffers while the core continues to the wait_all below.
        %tok_slice   = conduit.acquire_async {name = "input_slice",
                                               count = 1 : i64}
                           : !conduit.window.token
        %tok_weights = conduit.acquire_async {name = "shared_weights",
                                               count = 1 : i64}
                           : !conduit.window.token

        // Cross-tier fan-in: the key distribroad op.
        // Waits for BOTH the unique-slice grant AND the broadcast-weights grant
        // to be satisfied simultaneously.  In the fixed lowering, this op
        // emits NO hardware instructions (use_lock deferred to wait_window).
        conduit.wait_all %tok_slice, %tok_weights : !conduit.window.token, !conduit.window.token

        // Materialize windows — in the fixed lowering these are the sites
        // where aie.use_lock(consLock, AcquireGreaterEqual, 1) is emitted,
        // deferred from the acquire_async sites above.
        %win_slice   = conduit.wait_window %tok_slice   for "input_slice"
                           : !conduit.window.token -> !conduit.window<memref<16xi32>>
        %win_weights = conduit.wait_window %tok_weights for "shared_weights"
                           : !conduit.window.token -> !conduit.window<memref<8xi32>>

        // Access element 0 of each granted window.
        %elem_slice   = conduit.subview_access %win_slice   {index = 0 : i64}
                           : !conduit.window<memref<16xi32>>   -> memref<16xi32>
        %elem_weights = conduit.subview_access %win_weights {index = 0 : i64}
                           : !conduit.window<memref<8xi32>>    -> memref<8xi32>

        // Compute (placeholder: in a real program this would consume elem_slice
        // and elem_weights, e.g. a matmul or stencil over the two buffers).

        // Release both windows (Consume port → free the producer lock slot).
        // In the fixed lowering these emit aie.use_lock(prodLock, Release, 1).
        conduit.release %win_slice   {count = 1 : i64, port = #conduit.port<Consume>}
                           : !conduit.window<memref<16xi32>>
        conduit.release %win_weights {count = 1 : i64, port = #conduit.port<Consume>}
                           : !conduit.window<memref<8xi32>>
      }

      aie.end
    }
  }
}
