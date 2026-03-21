// RUN: aie-opt --conduit-check-channels --conduit-to-dma %s | FileCheck %s
//
// Cross-block delta inference: child inherits parent's lastAcquireCount.
//
// Parent entry block: acquire(2), release(1) → heldCount=1, lastAcquireCount=2.
// scf.for loop body: acquire(3) → delta = 3 - lastAcquireCount(2) = 1 → AGE(1).
//
// KEY INVARIANT: The child block uses lastAcquireCount=2 (NOT heldCount=1).
// Rationale: the DMA eagerly pre-fills slots up to lastAcquireCount.  Those
// DMA-delivered elements are already in the buffer when the child block runs,
// so the delta is relative to the number of DMA slots claimed, not the number
// of logically held elements.
//
// If the implementation incorrectly used heldCount=1 instead, the delta would
// be 3-1=2, which would cause a stall (waiting for 2 new elements when only 1
// DMA slot is available beyond the pre-fill).
//
// XFAIL: *
//
// Topology: shim(0,0) → compute(0,2), depth=4, element=memref<128xi32>.
//
// CHECK-LABEL: module @passC_delta_cross_block_inherits
//
// Preamble: acquire 2.
// CHECK: aie.core(%tile_0_2)
// CHECK: use_lock(%{{.*}}_cons_lock_0, AcquireGreaterEqual, 2)
//
// Loop body: delta = 3 - lastAcquireCount(2) = 1 → AGE(1).
// CHECK: scf.for
// CHECK-NOT: AcquireGreaterEqual, 2
// CHECK: use_lock(%{{.*}}_cons_lock_0, AcquireGreaterEqual, 1)

module @passC_delta_cross_block_inherits {
  aie.device(npu1_1col) {
    %shim = aie.tile(0, 0)
    %tile = aie.tile(0, 2)

    conduit.create {name = "fifo", capacity = 512 : i64, depth = 4 : i64,
                    element_type = memref<128xi32>,
                    producer_tile = array<i64: 0, 0>,
                    consumer_tiles = array<i64: 0, 2>}

    aie.shim_dma_allocation @fifo_shim_alloc(%shim, MM2S, 0)

    %core = aie.core(%tile) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c4 = arith.constant 4 : index
      %val = arith.constant 42 : i32

      // Preamble: acquire(2), use, release(1).
      // After: heldCount=1, lastAcquireCount=2.
      %win_pre = conduit.acquire {name = "fifo", count = 2 : i64,
                                   port = #conduit.port<Consume>}
                     : !conduit.window<memref<128xi32>>
      %pre0 = conduit.subview_access %win_pre {index = 0 : i64}
                  : !conduit.window<memref<128xi32>> -> memref<128xi32>
      %pre1 = conduit.subview_access %win_pre {index = 1 : i64}
                  : !conduit.window<memref<128xi32>> -> memref<128xi32>
      memref.store %val, %pre0[%c0] : memref<128xi32>

      conduit.release %win_pre {count = 1 : i64, port = #conduit.port<Consume>}
          : !conduit.window<memref<128xi32>>

      // Loop body: acquire(3).
      // Cross-block rule: inherits lastAcquireCount=2 (not heldCount=1).
      // delta = 3 - 2 = 1 → AGE(1).
      scf.for %i = %c0 to %c4 step %c1 {
        %win_mid = conduit.acquire {name = "fifo", count = 3 : i64,
                                     port = #conduit.port<Consume>}
                       : !conduit.window<memref<128xi32>>
        %mid0 = conduit.subview_access %win_mid {index = 0 : i64}
                    : !conduit.window<memref<128xi32>> -> memref<128xi32>
        %mid1 = conduit.subview_access %win_mid {index = 1 : i64}
                    : !conduit.window<memref<128xi32>> -> memref<128xi32>
        %mid2 = conduit.subview_access %win_mid {index = 2 : i64}
                    : !conduit.window<memref<128xi32>> -> memref<128xi32>
        memref.store %val, %mid0[%c0] : memref<128xi32>

        conduit.release %win_mid {count = 1 : i64, port = #conduit.port<Consume>}
            : !conduit.window<memref<128xi32>>
      }

      aie.end
    }

    aie.runtime_sequence(%in: memref<512xi32>) {
      aiex.npu.dma_memcpy_nd (%in[0,0,0,0][1,1,1,512][0,0,0,1])
          {metadata = @fifo_shim_alloc, id = 0 : i64} : memref<512xi32>
    }
  }
}
