// RUN: aie-opt --conduit-check-channels --conduit-to-dma %s | FileCheck %s
//
// Regression test: B-1 — Produce-port cross-block heldCount reset for
// non-uniform partial-release patterns.
//
// Bug scenario: parent does acquire{count=2}/release{count=1}, leaving
// heldCount=1 in the parent scope. Before the B-1 fix, the child loop
// body inherited heldCount=1 (not 0), so the first Produce acquire in the
// loop computed delta = 1 - 1 = 0 → no AcquireGreaterEqual emitted →
// core writes without owning the lock → hardware deadlock.
//
// After the B-1 fix: when entering a nested block, Produce-port heldCount
// is reset to 0. The child acquire{count=1} computes delta = 1 - 0 = 1
// → AcquireGreaterEqual(1) correctly emitted.
//
// State trace (correct, post-fix):
//   Parent acquire(2): heldCount=0 → delta=2 → AGE(2). held=2, last=2.
//   Parent release(1): held=2-1=1. last=2.
//   Enter scf.for:     Produce: childState.heldCount = 0 (B-1 reset, NOT 1).
//     Loop acquire(1): held=0 → delta=1-0=1 → AGE(1). held=1, last=1.
//     Loop release(1): held=0.
//
// Topology: compute(0,2) → shim(0,0), depth=2.
//
// CHECK-LABEL: module @passC_produce_crossblock_heldcount_reset
// CHECK: aie.core
// Parent preamble: AcquireGreaterEqual(2) for prod_lock (acquire count=2).
// CHECK:   aie.use_lock(%{{.*}}_prod_lock_0, AcquireGreaterEqual, 2)
// Parent release(1): Release(1) for cons_lock.
// CHECK:   aie.use_lock(%{{.*}}_cons_lock_0, Release, 1)
// Inside the loop: must emit AcquireGreaterEqual(1) — NOT subsumed by partial hold.
// CHECK:   scf.for
// CHECK:     aie.use_lock(%{{.*}}_prod_lock_0, AcquireGreaterEqual, 1)
// CHECK:     aie.use_lock(%{{.*}}_cons_lock_0, Release, 1)
// CHECK:   }
// CHECK:   aie.end

module @passC_produce_crossblock_heldcount_reset {
  aie.device(npu1_1col) {
    %shim = aie.tile(0, 0)
    %tile = aie.tile(0, 2)

    // @outChan: compute tile produces → shim consumes.
    conduit.create @outChan {slot_elems = 64 : i64, depth = 2 : i64,
                    element_type = memref<32xi32>,
                    shim_consumer_tiles = array<i64: 0, 0>}

    aie.shim_dma_allocation @outChan_shim_alloc(%shim, S2MM, 0)

    %core = aie.core(%tile) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c4 = arith.constant 4 : index
      %val = arith.constant 42 : i32

      // Parent preamble: acquire 2 output slots (non-uniform acquire).
      %pre_win = conduit.acquire {name = @outChan, count = 2 : i64,
                                   port = #conduit.port<Produce>}
                     : !conduit.window<memref<32xi32>>
      %pre_buf0 = conduit.subview_access %pre_win {index = 0 : i64}
                      : !conduit.window<memref<32xi32>> -> memref<32xi32>
      memref.store %val, %pre_buf0[%c0] : memref<32xi32>

      // Partial release: release only 1 of the 2 acquired slots.
      // After release: heldCount = 2 - 1 = 1 in parent scope.
      conduit.release %pre_win {count = 1 : i64, port = #conduit.port<Produce>}
          : !conduit.window<memref<32xi32>>

      // Inner loop: each iteration acquires 1 slot.
      // B-1 fix: child inherits heldCount=0 (not the parent's heldCount=1).
      // Without the fix, the first loop acquire would get delta=0 → deadlock.
      scf.for %i = %c0 to %c4 step %c1 {
        %loop_win = conduit.acquire {name = @outChan, count = 1 : i64,
                                      port = #conduit.port<Produce>}
                        : !conduit.window<memref<32xi32>>
        %loop_buf = conduit.subview_access %loop_win {index = 0 : i64}
                        : !conduit.window<memref<32xi32>> -> memref<32xi32>
        memref.store %val, %loop_buf[%c0] : memref<32xi32>

        conduit.release %loop_win {count = 1 : i64, port = #conduit.port<Produce>}
            : !conduit.window<memref<32xi32>>
      }

      aie.end
    }

    aie.runtime_sequence(%out: memref<160xi32>) {
      aiex.npu.dma_memcpy_nd (%out[0,0,0,0][1,1,1,160][0,0,0,1])
          {metadata = @outChan_shim_alloc, id = 0 : i64} : memref<160xi32>
    }
  }
}
