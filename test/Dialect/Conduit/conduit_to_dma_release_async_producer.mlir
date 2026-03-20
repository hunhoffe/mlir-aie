// RUN: aie-opt --conduit-to-dma %s | FileCheck %s
//
// Pass C test: BLOCK-2 fix — conduit.release_async on Port::Produce with
// depth=2 must emit the producer rotation counter increment.
//
// Background:
//   The synchronous conduit.release handler (Step 2 in lowerPhase) emits the
//   producer rotation counter increment (load/addi/remui/store) after the
//   aie.use_lock Release op.  The async counterpart (conduit.release_async,
//   Step 8d in lowerPhase) was not updated in the initial Bug 2 fix, so it
//   emitted the lock op but left the producer counter permanently at 0.
//
//   Before the BLOCK-2 fix: the counter stayed at slot 0 permanently for
//   async producers, causing every buffer selection to resolve to buff_0 and
//   silently bypassing buff_1 in double-buffered pipelines.
//
//   After the fix: release_async on Port::Produce emits the same
//   load/addi/remui/store sequence as the synchronous Release handler.
//
// This test uses raw Pass C IR (conduit dialect, no objectfifo) to directly
// exercise conduit.release_async with Port::Produce.  Using Pass C IR avoids
// Pass A converting objectfifo.release to conduit.release (synchronous);
// conduit.release_async must be spelled out explicitly.
//
// Topology: producer tile(0,2) → consumer tile(0,4), depth=2.
// effectiveDepth = min(2, 1+1) = 2.  Rotation counter modulus = 2.

// CHECK-LABEL: module @release_async_producer_block2
// CHECK:   aie.device(npu1_1col) {

// --- Producer tile: 2 ping-pong buffers (effectiveDepth=2) ---
// CHECK:     %[[PBUF0:.*]] = aie.buffer(%{{.*}}tile_0_2)
// CHECK-SAME:   sym_name = "fifo_async_buff_0"
// CHECK:     %[[PBUF1:.*]] = aie.buffer(%{{.*}}tile_0_2)
// CHECK-SAME:   sym_name = "fifo_async_buff_1"
// CHECK:     %[[PROD_LOCK:.*]] = aie.lock(%{{.*}}tile_0_2
// CHECK-SAME:   init = 2
// CHECK-SAME:   sym_name = "fifo_async_prod_lock_0"
// CHECK:     %[[CONS_LOCK:.*]] = aie.lock(%{{.*}}tile_0_2
// CHECK-SAME:   init = 0
// CHECK-SAME:   sym_name = "fifo_async_cons_lock_0"

// --- Producer core ---
// CHECK:     aie.core(%{{.*}}tile_0_2) {
// --- Rotation counter allocated as memref.alloc inside core body ---
// CHECK:       %[[ROT:.*]] = memref.alloca() : memref<1xi32>
// --- Counter init to 0 at core entry ---
// CHECK:       memref.store {{.*}}, %[[ROT]][{{.*}}] : memref<1xi32>
// CHECK:       scf.for
// --- Blocking acquire: waits for free slot on Produce port ---
// CHECK:         aie.use_lock(%[[PROD_LOCK]], AcquireGreaterEqual, 1)
// --- release_async: signals buffer filled ---
// CHECK:         aie.use_lock(%[[CONS_LOCK]], Release, 1)
// --- BLOCK-2 fix: producer rotation counter increment in release_async path ---
// CHECK:         memref.load %[[ROT]]
// CHECK:         arith.addi
// CHECK:         %[[C2:.*]] = arith.constant 2 : i32
// CHECK:         arith.remui {{.*}}, %[[C2]] : i32
// CHECK:         memref.store {{.*}}, %[[ROT]]
// CHECK:     }
// CHECK-NOT: conduit.create
// CHECK-NOT: conduit.acquire
// CHECK-NOT: conduit.release_async

module @release_async_producer_block2 {
  aie.device(npu1_1col) {
    %tile_0_2 = aie.tile(0, 2)
    %tile_0_4 = aie.tile(0, 4)

    // Producer tile(0,2) → consumer tile(0,4), depth=2.
    // Pass C allocates 2 producer buffers + prod/cons locks on tile_0_2,
    // plus the producer rotation counter memref<1xi32>.
    conduit.create {name = "fifo_async", capacity = 16 : i64,
                    producer_tile = array<i64: 0, 2>,
                    consumer_tiles = array<i64: 0, 4>,
                    element_type = memref<8xi32>,
                    depth = 2 : i64}

    %core_0_2 = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c4 = arith.constant 4 : index

      scf.for %arg0 = %c0 to %c4 step %c1 {
        // Blocking acquire on Produce port: waits for a free buffer slot.
        // Emits: aie.use_lock(prod_lock, AcquireGreaterEqual, 1)
        %win = conduit.acquire {name = "fifo_async", count = 1 : i64,
                                port = #conduit.port<Produce>}
                   : !conduit.window<memref<8xi32>>

        // release_async on Produce port: signals that the buffer is filled.
        // BLOCK-2 fix: Pass C must emit the producer rotation counter
        // increment (load/addi/remui/store) after the use_lock Release op.
        %rel_tok = conduit.release_async {name = "fifo_async", count = 1 : i64,
                                          port = #conduit.port<Produce>}
                       : !conduit.window.token

        // wait_all_async + wait: no hardware op (erased in Phase 7).
        %merged = conduit.wait_all_async %rel_tok :
            (!conduit.window.token) -> !conduit.dma.token
        conduit.wait %merged : !conduit.dma.token
      }
      aie.end
    } {dynamic_objfifo_lowering = true}
  }
}
