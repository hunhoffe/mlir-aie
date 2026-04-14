// RUN: aie-opt --conduit-to-dma %s | FileCheck %s
//
// Pass C test: conduit.release_async Phase 7 erasure, plus erasure of
// put_memref_async / get_memref_async / put_memref / get_memref.
//
// This test exercises patterns to verify that Phase 7 erases all
// Conduit token-carrying ops cleanly:
//
//   Pattern 1: release_async → conduit.wait_all.
//     release_async emits aie.use_lock via Step 8d.
//     conduit.wait_all is erased.
//
//   Pattern 2: second release_async in same loop body (regression for
//     Phase 7 walk correctness: multiple erased ops in same block).
//
//   Pattern 3 (put_memref_async → conduit.wait_all): put_memref_async produces a
//     !conduit.dma.token consumed by conduit.wait_all.  Phase 7 erases wait first
//     (leaving put_memref_async result dead), then erases put_memref_async.
//
//   Pattern 4 (get_memref_async, dead result): get_memref_async whose result
//     token has no consumer (never waited on).  Phase 7 must erase the op
//     even though no conduit.wait_all references it.
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
// --- Pattern 1: release_async emits use_lock; wait_all erased ---
// CHECK:         aie.use_lock(%[[CONS_CONS]], AcquireGreaterEqual, 1)
// CHECK:         aie.use_lock(%[[CONS_PROD]], Release, 1)
//
// --- Pattern 2: second release_async in same loop body ---
// CHECK:         aie.use_lock(%[[CONS_CONS]], AcquireGreaterEqual, 1)
// CHECK:         aie.use_lock(%[[CONS_PROD]], Release, 1)
//
// Patterns 3–5 emit no hardware ops: put_memref_async, get_memref_async,
// put_memref, get_memref are all erased without emitting aie hardware ops.
//
// --- No surviving Conduit ops of any kind ---
// CHECK-NOT: conduit.wait_all
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
    // Consumer tile inferred from conduit.acquire(Consume) inside aie.core.
    // Producer tile declared via producer_tile attr (shim side).
    // Pass C allocates aie.buffer + aie.lock on tile_0_2.
    conduit.create @fifo_waa {slot_elems = 8 : i64,
                    element_type = memref<8xi32>,
                    depth = 1 : i64}

    %core_0_2 = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c4 = arith.constant 4 : index

      scf.for %arg0 = %c0 to %c4 step %c1 {

        // ----------------------------------------------------------------
        // Pattern 1: release_async → conduit.wait_all.
        //
        // acquire emits use_lock(consLock, AcquireGreaterEqual, 1).
        %win1 = conduit.acquire {name = @fifo_waa, count = 1 : i64,
                                 port = #conduit.port<Consume>}
                    : !conduit.window<memref<8xi32>>

        // release_async (Step 8d) emits use_lock(prodLock, Release, 1).
        %rel_tok1 = conduit.release_async {name = @fifo_waa, count = 1 : i64, port = #conduit.port<Consume>}
                        : !conduit.window.token

        // conduit.wait_all: erased in Phase 7.
        conduit.wait_all %rel_tok1 : !conduit.window.token

        // ----------------------------------------------------------------
        // Pattern 2: second acquire/release_async in same loop body.
        //
        // Verifies Phase 7 handles multiple ops in a single block correctly.
        %win2 = conduit.acquire {name = @fifo_waa, count = 1 : i64,
                                 port = #conduit.port<Consume>}
                    : !conduit.window<memref<8xi32>>

        // release_async (Step 8d) emits use_lock(prodLock, Release, 1).
        %rel_tok2 = conduit.release_async {name = @fifo_waa, count = 1 : i64, port = #conduit.port<Consume>}
                        : !conduit.window.token

        conduit.wait_all %rel_tok2 : !conduit.window.token

        // ----------------------------------------------------------------
        // Pattern 3: put_memref_async → conduit.wait_all.
        //
        // Phase 7 must erase conduit.wait_all FIRST (it uses the token), then
        // erase put_memref_async (now its result has no users).  Erasing in
        // the wrong order — put_memref_async first — would try to erase an
        // op that still has live SSA uses and crash in debug builds.
        // No hardware op is emitted for either: DMA descriptor lowering for
        // put_memref_async is a separate future gap.
        %dma_tok = conduit.put_memref_async {name = @fifo_waa,
                       num_elems = 8 : i64,
                       offsets = array<i64: 0>,
                       sizes = array<i64: 8>,
                       strides = array<i64: 1>}
                       : !conduit.dma.token

        conduit.wait_all %dma_tok : !conduit.dma.token

        // ----------------------------------------------------------------
        // Pattern 4: get_memref_async with unused (dead) result.
        //
        // The result token has no consumers — Phase 7 must erase the op
        // even though no conduit.wait_all references it.
        %_unused = conduit.get_memref_async {name = @fifo_waa,
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
        conduit.put_memref {name = @fifo_waa, num_elems = 8 : i64,
                            offsets = array<i64: 0>,
                            sizes = array<i64: 8>,
                            strides = array<i64: 1>}

        conduit.get_memref {name = @fifo_waa, num_elems = 8 : i64,
                            offsets = array<i64: 0>,
                            sizes = array<i64: 8>,
                            strides = array<i64: 1>}
      }
      aie.end
    } {dynamic_objfifo_lowering = true}
  }
}
