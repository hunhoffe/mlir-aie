// RUN: aie-opt --conduit-to-dma %s | FileCheck %s
//
// Pass C test: sliding window pattern with conduit.acquire_async + conduit.wait_window.
//
// Pattern: consumer acquires 3 rows (count=3), processes them, releases 1 (count=1).
// This is the Conduit-native sliding window expression that cannot be represented
// in air.channel without architectural restructuring (Claim d).
//
// Topology: shim(0,0) → tile(0,2).  Depth=4 (≥ acquire count 3 + 1 for pipelining).
// For loop trips = 8.
//
// The sliding window means the consumer holds 3 of 4 buffer slots during computation
// and releases 1 slot each iteration, sliding forward by 1 row.
//
// Expected lowering:
//   aie.use_lock(cons_lock, AcquireGreaterEqual, 3) at wait_window site
//   aie.use_lock(prod_lock, Release, 1)             at release site
//   4 aie.buffer ops (depth=4)
//   4 aie.dma_bd ops in the BD ring

// CHECK-LABEL: module @async_sliding_window
// CHECK:   aie.device(npu1_1col) {
// CHECK:     aie.tile(0, 0)
// CHECK:     aie.tile(0, 2)

// --- 4 buffers (depth=4) on consumer tile ---
// CHECK:     aie.buffer(%{{.*}}tile_0_2)
// CHECK-SAME:   sym_name = "sw_fifo_cons_buff_0"
// CHECK:     aie.buffer(%{{.*}}tile_0_2)
// CHECK-SAME:   sym_name = "sw_fifo_cons_buff_1"
// CHECK:     aie.buffer(%{{.*}}tile_0_2)
// CHECK-SAME:   sym_name = "sw_fifo_cons_buff_2"
// CHECK:     aie.buffer(%{{.*}}tile_0_2)
// CHECK-SAME:   sym_name = "sw_fifo_cons_buff_3"

// --- Locks: prod_lock init=4 (depth), cons_lock init=0 ---
// CHECK:     %[[PROD_LOCK:.*]] = aie.lock(%{{.*}}tile_0_2
// CHECK-SAME:   init = 4
// CHECK-SAME:   sym_name = "sw_fifo_cons_prod_lock_0"
// CHECK:     %[[CONS_LOCK:.*]] = aie.lock(%{{.*}}tile_0_2
// CHECK-SAME:   init = 0
// CHECK-SAME:   sym_name = "sw_fifo_cons_cons_lock_0"

// --- Core: acquire_async → wait_window emits AcquireGreaterEqual(3) ---
// CHECK:     aie.core(%{{.*}}tile_0_2) {
// CHECK:       scf.for
// CHECK:         aie.use_lock(%[[CONS_LOCK]], AcquireGreaterEqual, 3)
// CHECK:         func.call @process_window
// --- Release with count=1 (slide forward by 1) ---
// CHECK:         aie.use_lock(%[[PROD_LOCK]], Release, 1)
// CHECK:     }

// --- DMA BD ring: 4 BDs for depth=4 ---
// CHECK:     aie.mem(%{{.*}}tile_0_2) {
// CHECK:       aie.dma_start(S2MM
// CHECK:       aie.dma_bd
// CHECK:       aie.next_bd
// CHECK:       aie.dma_bd
// CHECK:       aie.next_bd
// CHECK:       aie.dma_bd
// CHECK:       aie.next_bd
// CHECK:       aie.dma_bd
// CHECK:       aie.next_bd
// CHECK:       aie.end
// CHECK:     }

// --- No residual Conduit ops ---
// CHECK-NOT: conduit.create
// CHECK-NOT: conduit.acquire_async
// CHECK-NOT: conduit.wait_window
// CHECK-NOT: conduit.subview_access
// CHECK-NOT: conduit.release

module @async_sliding_window {
  aie.device(npu1_1col) {
    func.func @process_window(%row0: memref<32xi8>, %row1: memref<32xi8>, %row2: memref<32xi8>) -> () {
      return
    }

    %tile_0_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)

    // Conduit IR: depth=4 channel, shim → compute tile.
    conduit.create {name = "sw_fifo", capacity = 32 : i64,
                    producer_tile = array<i64: 0, 0>,
                    consumer_tiles = array<i64: 0, 2>,
                    element_type = memref<32xi8>,
                    depth = 4 : i64}

    %core_0_2 = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c8 = arith.constant 8 : index

      scf.for %arg0 = %c0 to %c8 step %c1 {
        // Sliding window: acquire 3 rows asynchronously, wait, process, release 1.
        %tok = conduit.acquire_async {name = "sw_fifo", count = 3 : i64}
                   : !conduit.window.token

        %win = conduit.wait_window %tok for "sw_fifo"
                   : !conduit.window.token -> !conduit.window<memref<32xi8>>

        %row0 = conduit.subview_access %win {index = 0 : i64}
                    : !conduit.window<memref<32xi8>> -> memref<32xi8>
        %row1 = conduit.subview_access %win {index = 1 : i64}
                    : !conduit.window<memref<32xi8>> -> memref<32xi8>
        %row2 = conduit.subview_access %win {index = 2 : i64}
                    : !conduit.window<memref<32xi8>> -> memref<32xi8>

        func.call @process_window(%row0, %row1, %row2) : (memref<32xi8>, memref<32xi8>, memref<32xi8>) -> ()

        // Release 1 row: slide the window forward by 1
        conduit.release %win {count = 1 : i64, port = #conduit.port<Consume>}
            : !conduit.window<memref<32xi8>>
      }
      aie.end
    } {dynamic_objfifo_lowering = true}
  }
}
