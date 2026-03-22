// RUN: aie-opt --conduit-check-channels --conduit-to-dma %s | FileCheck %s
//
// Regression test: Pass C delta inference for Produce acquires in loop bodies.
//
// Bug: when a Produce acquire in a parent block has count=1 and is released
// (heldCount=0), the child loop body's Produce acquire of count=1 should
// get delta=1 (AcquireGreaterEqual, 1).  Before the fix, the cross-block
// state reset propagated lastAcquireCount=1 as heldCount into the child block
// (the same reset used for Consume channels where DMA pre-fills slots).
// For Produce, there is no DMA pre-fill, so heldCount after release is truly 0.
// The incorrect reset made delta=1-1=0 → no AcquireGreaterEqual emitted →
// core writes output without owning the lock → hardware deadlock.
//
// State trace (correct, post-fix):
//   Preamble acquire(1):  heldCount=0 → delta=1 → AGE(1). held=1, last=1.
//   Preamble release(1):  held=1-1=0. last=1.
//   Enter scf.for:        Produce: childState.heldCount = heldCount = 0 (NOT last=1).
//     Loop acquire(1):    held=0 → delta=1-0=1 → AGE(1). held=1, last=1.
//     Loop release(1):    held=0.
//
// Topology: compute(0,2) → shim(0,0), depth=2, element=memref<32xi32>.
//
// CHECK-LABEL: module @passC_produce_acquire_loop_delta
// CHECK: aie.core
// Preamble: AcquireGreaterEqual(1) for outRows_prod_lock.
// CHECK:   aie.use_lock(%{{.*}}_prod_lock_0, AcquireGreaterEqual, 1)
// CHECK:   aie.use_lock(%{{.*}}_cons_lock_0, Release, 1)
// Inside the loop: must also emit AcquireGreaterEqual(1) — Produce is not subsumed.
// CHECK:   scf.for
// CHECK:     aie.use_lock(%{{.*}}_prod_lock_0, AcquireGreaterEqual, 1)
// CHECK:     aie.use_lock(%{{.*}}_cons_lock_0, Release, 1)
// CHECK:   }
// CHECK:   aie.end

module @passC_produce_acquire_loop_delta {
  aie.device(npu1_1col) {
    %shim = aie.tile(0, 0)
    %tile = aie.tile(0, 2)

    // @outRows: compute tile produces → shim consumes.
    conduit.create {name = "outRows", capacity = 64 : i64, depth = 2 : i64,
                    element_type = memref<32xi32>,
                    producer_tile = array<i64: 0, 2>,
                    consumer_tiles = array<i64>,
                    shim_consumer_tiles = array<i64: 0, 0>}

    aie.shim_dma_allocation @outRows_shim_alloc(%shim, S2MM, 0)

    %core = aie.core(%tile) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c4 = arith.constant 4 : index
      %val = arith.constant 42 : i32

      // Preamble: acquire 1 output slot, write it, release it.
      %pre_win = conduit.acquire {name = "outRows", count = 1 : i64,
                                   port = #conduit.port<Produce>}
                     : !conduit.window<memref<32xi32>>
      %pre_buf = conduit.subview_access %pre_win {index = 0 : i64}
                     : !conduit.window<memref<32xi32>> -> memref<32xi32>
      memref.store %val, %pre_buf[%c0] : memref<32xi32>

      // Release preamble window — heldCount → 0.
      conduit.release %pre_win {count = 1 : i64, port = #conduit.port<Produce>}
          : !conduit.window<memref<32xi32>>

      // Inner loop: each iteration acquires a fresh output slot.
      // This acquire must NOT be subsumed by the released preamble window.
      scf.for %i = %c0 to %c4 step %c1 {
        %loop_win = conduit.acquire {name = "outRows", count = 1 : i64,
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
          {metadata = @outRows_shim_alloc, id = 0 : i64} : memref<160xi32>
    }
  }
}
