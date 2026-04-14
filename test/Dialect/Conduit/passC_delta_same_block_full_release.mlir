// RUN: aie-opt --conduit-check-channels --conduit-to-dma %s | FileCheck %s
//
// Same-block full release: acquire(2), release(2) → heldCount=0, window fully
// closed.  Then acquire(3) → delta = 3 - 0 = 3 → AcquireGreaterEqual(3).
//
// After a full release, all elements are returned to the producer.  The next
// acquire is effectively "fresh" and should wait for all requested elements.
//
// State trace:
//   acquire(2):  heldCount=0 → delta=2 → AGE(2).  Update: held=2, last=2.
//   release(2):  heldCount=2-2=0.  lastAcquireCount unchanged (still 2).
//   acquire(3):  heldCount=0 → delta=3-0=3 → AGE(3).
//
// Topology: shim(0,0) → compute(0,2), depth=4, element=memref<128xi32>.
//
// CHECK-LABEL: module @passC_delta_same_block_full_release
// CHECK: aie.core(%tile_0_2)
//
// First acquire: AGE(2) — fresh.
// CHECK: use_lock(%{{.*}}_cons_lock_0, AcquireGreaterEqual, 2)
//
// Second acquire: AGE(3) — fully fresh after complete release.
// CHECK: use_lock(%{{.*}}_cons_lock_0, AcquireGreaterEqual, 3)

module @passC_delta_same_block_full_release {
  aie.device(npu1_1col) {
    %shim = aie.tile(0, 0)
    %tile = aie.tile(0, 2)

    conduit.create @fifo {slot_elems = 512 : i64, depth = 4 : i64,
                    element_type = memref<128xi32>
                    }

    aie.shim_dma_allocation @fifo_shim_alloc(%shim, MM2S, 0)

    %core = aie.core(%tile) {
      %c0 = arith.constant 0 : index
      %val = arith.constant 42 : i32

      // First acquire: count=2 → AGE(2).
      %win1 = conduit.acquire {name = @fifo, count = 2 : i64,
                                port = #conduit.port<Consume>}
                  : !conduit.window<memref<128xi32>>
      %e0 = conduit.subview_access %win1 {index = 0 : i64}
                : !conduit.window<memref<128xi32>> -> memref<128xi32>
      memref.store %val, %e0[%c0] : memref<128xi32>

      // Full release: 2 elements → heldCount=0.
      conduit.release %win1 {count = 2 : i64, port = #conduit.port<Consume>}
          : !conduit.window<memref<128xi32>>

      // Second acquire: count=3, held=0 → delta=3 → AGE(3).
      %win2 = conduit.acquire {name = @fifo, count = 3 : i64,
                                port = #conduit.port<Consume>}
                  : !conduit.window<memref<128xi32>>
      %f0 = conduit.subview_access %win2 {index = 0 : i64}
                : !conduit.window<memref<128xi32>> -> memref<128xi32>
      %f1 = conduit.subview_access %win2 {index = 1 : i64}
                : !conduit.window<memref<128xi32>> -> memref<128xi32>
      %f2 = conduit.subview_access %win2 {index = 2 : i64}
                : !conduit.window<memref<128xi32>> -> memref<128xi32>
      memref.store %val, %f0[%c0] : memref<128xi32>

      conduit.release %win2 {count = 3 : i64, port = #conduit.port<Consume>}
          : !conduit.window<memref<128xi32>>

      aie.end
    }

    aie.runtime_sequence(%in: memref<512xi32>) {
      aiex.npu.dma_memcpy_nd (%in[0,0,0,0][1,1,1,512][0,0,0,1])
          {metadata = @fifo_shim_alloc, id = 0 : i64} : memref<512xi32>
    }
  }
}
