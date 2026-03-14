// RUN: aie-opt --conduit-to-dma %s | FileCheck %s
//
// Pass C test: conduit.wait_all_async Phase 7 erasure, plus erasure of
// put_memref_async / get_memref_async / put_memref / get_memref.
//
// This test exercises five patterns to verify that Phase 7 erases all
// Conduit token-carrying ops cleanly:
//
//   Pattern 1 (simple wait_all_async): release_async → wait_all_async →
//     conduit.wait.  All three ops erased; release_async still emits
//     aie.use_lock via Step 8d.
//
//   Pattern 2 (chained wait_all_async): release_async → waa_A → waa_B →
//     conduit.wait.  MLIR walk() visits ops in pre-order (defs before uses),
//     so inline-erasing waa_A leaves waa_B holding a deleted SSA value.
//     The collect-then-erase fix avoids this crash.
//
//   Pattern 3 (put_memref_async → conduit.wait): put_memref_async produces a
//     !conduit.dma.token consumed by conduit.wait.  Phase 7 erases wait first
//     (leaving put_memref_async result dead), then erases put_memref_async.
//     Proper erasure order: consumers before producers.
//
//   Pattern 4 (get_memref_async, dead result): get_memref_async whose result
//     token has no consumer (never waited on).  Phase 7 must erase the op
//     even though no conduit.wait references it.
//
//   Pattern 5 (blocking put_memref / get_memref): blocking DMA ops with no
//     result token.  Phase 7 must erase them.
//
// In all patterns: no conduit.* op should survive into the output IR.
//
// Resources expected (depth=1, shim tile_0_0 → compute tile_0_2):
//   aie.buffer:  1 (fifo_waa_cons_buff_0 on tile_0_2)
//   aie.lock:    4 (cons prod_lock init=1, cons cons_lock init=0 on tile_0_2;
//                   prod_lock, cons_lock on shim tile_0_0)
//   aie.flow:    1 (shim DMA:0 → tile_0_2 DMA:0)

// CHECK-LABEL: module @wait_all_async_erasure
// CHECK:   aie.device(npu1_1col) {
// CHECK:     aie.tile(0, 0)
// CHECK:     aie.tile(0, 2)
// CHECK:     %[[BUFF0:.*]] = aie.buffer(%{{.*}}tile_0_2)
// CHECK-SAME:   sym_name = "fifo_waa_cons_buff_0"
// CHECK:     %[[CONS_PROD:.*]] = aie.lock(%{{.*}}tile_0_2
// CHECK-SAME:   init = 1
// CHECK-SAME:   sym_name = "fifo_waa_cons_prod_lock_0"
// CHECK:     %[[CONS_CONS:.*]] = aie.lock(%{{.*}}tile_0_2
// CHECK-SAME:   init = 0
// CHECK-SAME:   sym_name = "fifo_waa_cons_cons_lock_0"
// CHECK:     aie.core(%{{.*}}tile_0_2) {
// CHECK:       scf.for
//
// --- Pattern 1: release_async emits use_lock; wait_all_async + wait erased ---
// CHECK:         aie.use_lock(%[[CONS_CONS]], AcquireGreaterEqual, 1)
// CHECK:         aie.use_lock(%[[CONS_PROD]], Release, 1)
//
// --- Pattern 2: chained wait_all_async; both chains erased without crash ---
// CHECK:         aie.use_lock(%[[CONS_CONS]], AcquireGreaterEqual, 1)
// CHECK:         aie.use_lock(%[[CONS_PROD]], Release, 1)
//
// Patterns 3–5 emit no hardware ops: put_memref_async, get_memref_async,
// put_memref, get_memref are all erased without emitting aie hardware ops.
//
// --- No surviving Conduit ops of any kind ---
// CHECK-NOT: conduit.wait_all_async
// CHECK-NOT: conduit.wait
// CHECK-NOT: conduit.release_async
// CHECK-NOT: conduit.acquire
// CHECK-NOT: conduit.put_memref_async
// CHECK-NOT: conduit.get_memref_async
// CHECK-NOT: conduit.put_memref
// CHECK-NOT: conduit.get_memref
// CHECK-NOT: conduit.create

module @wait_all_async_erasure {
  aie.device(npu1_1col) {
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)

    // conduit.create: shim (row=0) produces; tile_0_2 consumes.
    // Pass C allocates aie.buffer + aie.lock on tile_0_2.
    conduit.create {name = "fifo_waa", capacity = 8 : i64,
                    producer_tile = array<i64: 0, 0>,
                    consumer_tiles = array<i64: 0, 2>,
                    element_type = memref<8xi32>,
                    depth = 1 : i64}

    %core_0_2 = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c4 = arith.constant 4 : index

      scf.for %arg0 = %c0 to %c4 step %c1 {

        // ----------------------------------------------------------------
        // Pattern 1: simple wait_all_async — result feeds conduit.wait.
        //
        // acquire emits use_lock(consLock, AcquireGreaterEqual, 1).
        %win1 = conduit.acquire {name = "fifo_waa", count = 1 : i64,
                                 port = #conduit.port<Consume>}
                    : !conduit.window<memref<8xi32>>

        // release_async (Step 8d) emits use_lock(prodLock, Release, 1).
        %rel_tok1 = conduit.release_async {name = "fifo_waa", count = 1 : i64, port = #conduit.port<Consume>}
                        : !conduit.window.token

        // wait_all_async: fan-in of a single window token.
        // No hardware op — erased in Phase 7.
        %merged1 = conduit.wait_all_async %rel_tok1 :
            (!conduit.window.token) -> !conduit.dma.token

        // conduit.wait: no hardware op — erased in Phase 7 before WaitAllAsync.
        conduit.wait %merged1 : !conduit.dma.token

        // ----------------------------------------------------------------
        // Pattern 2: chained wait_all_async.
        //
        // Second iteration in same loop body exercises chaining:
        //   waa_A's result feeds waa_B, whose result feeds conduit.wait.
        // All three Conduit ops must be erased without use-after-erase.
        //
        // acquire emits use_lock(consLock, AcquireGreaterEqual, 1).
        %win2 = conduit.acquire {name = "fifo_waa", count = 1 : i64,
                                 port = #conduit.port<Consume>}
                    : !conduit.window<memref<8xi32>>

        // release_async (Step 8d) emits use_lock(prodLock, Release, 1).
        %rel_tok2 = conduit.release_async {name = "fifo_waa", count = 1 : i64, port = #conduit.port<Consume>}
                        : !conduit.window.token

        // First wait_all_async: erased in Phase 7 walk.
        %merged2a = conduit.wait_all_async %rel_tok2 :
            (!conduit.window.token) -> !conduit.dma.token

        // Second (chained) wait_all_async: consumes merged2a.
        // MLIR walk() visits in pre-order; merged2a's defining op is erased
        // first.  The implementation must not crash on a dangling use here —
        // collecting ops then erasing in a second pass avoids this.
        %merged2b = conduit.wait_all_async %merged2a :
            (!conduit.dma.token) -> !conduit.dma.token

        // conduit.wait on the chained result — also erased in Phase 7.
        conduit.wait %merged2b : !conduit.dma.token

        // ----------------------------------------------------------------
        // Pattern 3: put_memref_async → conduit.wait.
        //
        // Phase 7 must erase conduit.wait FIRST (it uses the token), then
        // erase put_memref_async (now its result has no users).  Erasing in
        // the wrong order — put_memref_async first — would try to erase an
        // op that still has live SSA uses and crash in debug builds.
        // No hardware op is emitted for either: DMA descriptor lowering for
        // put_memref_async is a separate future gap.
        %dma_tok = conduit.put_memref_async {name = "fifo_waa",
                       num_elems = 8 : i64,
                       offsets = array<i64: 0>,
                       sizes = array<i64: 8>,
                       strides = array<i64: 1>}
                       : !conduit.dma.token

        conduit.wait %dma_tok : !conduit.dma.token

        // ----------------------------------------------------------------
        // Pattern 4: get_memref_async with unused (dead) result.
        //
        // The result token has no consumers — Phase 7 must erase the op
        // even though no conduit.wait holds a reference to it.
        %_unused = conduit.get_memref_async {name = "fifo_waa",
                       num_elems = 8 : i64,
                       offsets = array<i64: 0>,
                       sizes = array<i64: 8>,
                       strides = array<i64: 1>}
                       : !conduit.dma.token

        // ----------------------------------------------------------------
        // Pattern 5: blocking put_memref and get_memref (no result token).
        //
        // These ops have no SSA result; they must be erased by a dedicated
        // walk so they do not appear as dangling Conduit ops in the output.
        conduit.put_memref {name = "fifo_waa", num_elems = 8 : i64,
                            offsets = array<i64: 0>,
                            sizes = array<i64: 8>,
                            strides = array<i64: 1>}

        conduit.get_memref {name = "fifo_waa", num_elems = 8 : i64,
                            offsets = array<i64: 0>,
                            sizes = array<i64: 8>,
                            strides = array<i64: 1>}
      }
      aie.end
    } {dynamic_objfifo_lowering = true}
  }
}
