// RUN: aie-opt --conduit-to-dma %s | FileCheck %s
//
// Pass C test: conduit.acquire_async → conduit.wait_window → conduit.subview_access
// → conduit.release path (the async window acquisition path, Fix NF3).
//
// This test verifies that Phase 6 Step 1 (SubviewAccess lowering) correctly
// handles the case where the window operand is produced by conduit.wait_window
// (the async path) rather than conduit.acquire (the blocking path).
//
// Before Fix NF3 the WaitWindow defining-op cast failed, conduitName was empty,
// replaced stayed false, and the pass crashed with a hard error.
//
// This test does NOT go through Pass A (--objectfifo-to-conduit) because the
// async ops are not produced by Pass A yet.  It exercises Pass C directly on
// hand-written Conduit IR inside an aie.device.
//
// Expected: no crash; use_lock emitted at the wait_window point; buffer refs
// from the allocated aie.buffer ops replace the subview_access and release.
//
// Resources expected (depth=1, shim→tile_0_2):
//   aie.buffer:   1  (fifo_async_cons_buff_0 on tile_0_2)
//   aie.lock:     4  (cons prod_lock init=1, cons cons_lock init=0 on tile_0_2;
//                     prod_lock, cons_lock on shim tile_0_0)
//   aie.flow:     1  (shim DMA:0 → tile_0_2 DMA:0)
//   aie.mem:      1  (S2MM for tile_0_2)
//
// In the core body, the async acquire path emits:
//   (nothing at acquire_async point — deferred to wait_window)
//   aie.use_lock(cons_lock, AcquireGreaterEqual, 1)  at wait_window
//   func.call @process(%buffer)                       at subview_access
//   aie.use_lock(prod_lock, Release, 1)               at release

// CHECK-LABEL: module @async_window_path
// CHECK:   aie.device(npu1_1col) {
// CHECK:     aie.tile(0, 0)
// CHECK:     aie.tile(0, 2)
// CHECK:     %[[BUFF0:.*]] = aie.buffer(%{{.*}}tile_0_2)
// CHECK-SAME:   sym_name = "fifo_async_cons_buff_0"
// CHECK:     %[[CONS_PROD:.*]] = aie.lock(%{{.*}}tile_0_2
// CHECK-SAME:   init = 1
// CHECK-SAME:   sym_name = "fifo_async_cons_prod_lock_0"
// CHECK:     %[[CONS_CONS:.*]] = aie.lock(%{{.*}}tile_0_2
// CHECK-SAME:   init = 0
// CHECK-SAME:   sym_name = "fifo_async_cons_cons_lock_0"
// CHECK:     aie.core(%{{.*}}tile_0_2) {
// CHECK:       scf.for
// --- acquire_async emits nothing (deferred to wait_window) ---
// --- wait_window emits the deferred use_lock ---
// CHECK:         aie.use_lock(%[[CONS_CONS]], AcquireGreaterEqual, 1)
// --- subview_access replaced with the allocated buffer ---
// CHECK:         func.call @process(%[[BUFF0]])
// --- release emits use_lock on prod_lock ---
// CHECK:         aie.use_lock(%[[CONS_PROD]], Release, 1)
// CHECK:     aie.shim_dma_allocation @fifo_async_shim_alloc
// CHECK:     aie.flow(%{{.*}}tile_0_0, DMA : 0, %{{.*}}tile_0_2, DMA : 0)
// CHECK:     aie.mem(%{{.*}}tile_0_2) {
// CHECK:       aie.dma_start(S2MM
// CHECK:       aie.use_lock(%[[CONS_PROD]], AcquireGreaterEqual, 1)
// CHECK:       aie.dma_bd(%[[BUFF0]]
// CHECK:       aie.use_lock(%[[CONS_CONS]], Release, 1)
// CHECK-NOT: conduit.create
// CHECK-NOT: conduit.acquire_async
// CHECK-NOT: conduit.wait_window
// CHECK-NOT: conduit.subview_access
// CHECK-NOT: conduit.release

module @async_window_path {
  aie.device(npu1_1col) {
    func.func @process(%buf: memref<8xi32>) -> () {
      return
    }

    %tile_0_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)

    // Hand-written Conduit IR that the async path would produce.
    // conduit.create declares the channel metadata for Pass C.
    conduit.create {name = "fifo_async", capacity = 8 : i64,
                    producer_tile = array<i64: 0, 0>,
                    consumer_tiles = array<i64: 0, 2>,
                    element_type = memref<8xi32>,
                    depth = 1 : i64}

    %core_0_2 = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c4 = arith.constant 4 : index

      scf.for %arg0 = %c0 to %c4 step %c1 {
        // Async path: acquire_async issues the lock request (no hardware op
        // emitted here — deferred to wait_window).
        %tok = conduit.acquire_async {name = "fifo_async", count = 1 : i64}
                   : !conduit.window.token

        // wait_window blocks until the lock is granted and produces the window.
        // Pass C emits use_lock(consLock, AcquireGreaterEqual, 1) here.
        %win = conduit.wait_window %tok for "fifo_async"
                   : !conduit.window.token -> !conduit.window<memref<8xi32>>

        // subview_access resolves to the allocated aie.buffer.
        // Fix NF3: this used to crash because the defining op was WaitWindow,
        // not Acquire.
        %elem = conduit.subview_access %win {index = 0 : i64}
                    : !conduit.window<memref<8xi32>> -> memref<8xi32>

        func.call @process(%elem) : (memref<8xi32>) -> ()

        // release emits use_lock(prodLock, Release, 1).
        conduit.release %win {count = 1 : i64, port = #conduit.port<Consume>}
            : !conduit.window<memref<8xi32>>
      }
      aie.end
    } {dynamic_objfifo_lowering = true}
  }
}
