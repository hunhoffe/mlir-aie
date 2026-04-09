// RUN: aie-opt --conduit-to-dma %s | FileCheck %s
//
// Pass C test: conduit.release_async lowering (Step 8d fix).
//
// This test verifies that conduit.release_async is lowered to
// aie.use_lock(prodLock, Release, count) instead of hard-erroring with
// "conduit-to-dma: unimplemented — conduit.release_async lowering not yet
// supported".
//
// The test uses a consumer core (tile_0_2) that calls release_async after
// acquiring data. The consumer endpoint uses conduit.acquire (blocking)
// and conduit.release_async (non-blocking release, token dropped via wait_all).
//
// Resources expected (depth=1, shim tile_0_0 → compute tile_0_2):
//   aie.buffer:   1  (fifo_rel_cons_buff_0 on tile_0_2, consumer side)
//   aie.lock:     4  (prod_lock init=1, cons_lock init=0 on tile_0_2;
//                     prod_lock, cons_lock on shim tile_0_0)
//   aie.flow:     1  (tile_0_2 DMA:0 → shim tile_0_0 DMA:0)
//
// In the producer core body:
//   aie.use_lock(consLock, AcquireGreaterEqual, 1)  from conduit.acquire
//   (write data to buffer)
//   aie.use_lock(prodLock, Release, 1)               from conduit.release_async

// CHECK-LABEL: module @release_async_lowering
// CHECK:   aie.device(npu1_1col) {
// CHECK:     aie.tile(0, 0)
// CHECK:     aie.tile(0, 2)
// CHECK:     %[[BUFF0:.*]] = aie.buffer(%{{.*}}tile_0_2)
// CHECK-SAME:   sym_name = "fifo_rel_cons_buff_0"
// CHECK:     %[[CONS_PROD:.*]] = aie.lock(%{{.*}}tile_0_2
// CHECK-SAME:   init = 1
// CHECK-SAME:   sym_name = "fifo_rel_cons_prod_lock_0"
// CHECK:     %[[CONS_CONS:.*]] = aie.lock(%{{.*}}tile_0_2
// CHECK-SAME:   init = 0
// CHECK-SAME:   sym_name = "fifo_rel_cons_cons_lock_0"
// CHECK:     aie.core(%{{.*}}tile_0_2) {
// CHECK:       scf.for
// --- acquire emits use_lock on cons_lock (consumer acquires data) ---
// CHECK:         aie.use_lock(%[[CONS_CONS]], AcquireGreaterEqual, 1)
// --- release_async emits use_lock on prod_lock (release slot to producer) ---
// CHECK:         aie.use_lock(%[[CONS_PROD]], Release, 1)
// --- no surviving conduit ops ---
// CHECK-NOT: conduit.release_async
// CHECK-NOT: conduit.acquire
// CHECK-NOT: conduit.create

module @release_async_lowering {
  aie.device(npu1_1col) {
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)

    // conduit.create: shim (row=0) is the producer, tile_0_2 is the consumer.
    // Pass C allocates aie.buffer + aie.lock on the consumer tile (tile_0_2).
    conduit.create @fifo_rel {slot_elems = 8 : i64,
                    producer_tile = array<i64: 0, 0>,
                    consumer_tiles = array<i64: 0, 2>,
                    element_type = memref<8xi32>,
                    depth = 1 : i64}

    %core_0_2 = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c4 = arith.constant 4 : index

      scf.for %arg0 = %c0 to %c4 step %c1 {
        // Blocking acquire: emits use_lock(consLock, AcquireGreaterEqual, 1).
        %win = conduit.acquire {name = @fifo_rel, count = 1 : i64,
                                port = #conduit.port<Consume>}
                   : !conduit.window<memref<8xi32>>

        // release_async: non-blocking release.  Pass C must emit
        // use_lock(prodLock, Release, 1) at this point (Step 8d fix).
        %rel_tok = conduit.release_async {name = @fifo_rel, count = 1 : i64, port = #conduit.port<Consume>}
                       : !conduit.window.token

        // wait_all consumes the window token directly.
        conduit.wait_all %rel_tok : !conduit.window.token
      }
      aie.end
    } {dynamic_objfifo_lowering = true}
  }
}
