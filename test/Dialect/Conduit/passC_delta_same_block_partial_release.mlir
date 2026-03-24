// RUN: aie-opt --conduit-check-channels --conduit-to-dma %s | FileCheck %s
//
// Same-block partial release: acquire(2), release(1) → heldCount=1.
// Then acquire(2) → delta = 2 - 1 = 1 → AcquireGreaterEqual(1).
//
// This is the TAIL pattern from the bottleneck sliding window: after a
// partial release, the next acquire should only wait for the newly needed
// elements, not re-acquire the ones still held.
//
// State trace:
//   acquire(2):  heldCount=0 → delta=2 → AGE(2).  Update: held=2, last=2.
//   release(1):  heldCount=2-1=1.  lastAcquireCount unchanged (still 2).
//   acquire(2):  heldCount=1 → delta=2-1=1 → AGE(1).
//
// Topology: shim(0,0) → compute(0,2), depth=4, element=memref<128xi32>.
//
// CHECK-LABEL: module @passC_delta_same_block_partial_release
// CHECK: aie.core(%tile_0_2)
//
// First acquire: AGE(2) — fresh.
// CHECK: use_lock(%{{.*}}_cons_lock_0, AcquireGreaterEqual, 2)
//
// Second acquire: AGE(1) — only 1 new element needed.
// CHECK: use_lock(%{{.*}}_cons_lock_0, AcquireGreaterEqual, 1)

module @passC_delta_same_block_partial_release {
  aie.device(npu1_1col) {
    %shim = aie.tile(0, 0)
    %tile = aie.tile(0, 2)

    conduit.create @fifo {capacity = 512 : i64, depth = 4 : i64,
                    element_type = memref<128xi32>,
                    producer_tile = array<i64: 0, 0>,
                    consumer_tiles = array<i64: 0, 2>}

    aie.shim_dma_allocation @fifo_shim_alloc(%shim, MM2S, 0)

    %core = aie.core(%tile) {
      %c0 = arith.constant 0 : index
      %val = arith.constant 42 : i32

      // First acquire: count=2, fresh → AGE(2).
      %win1 = conduit.acquire {name = @fifo, count = 2 : i64,
                                port = #conduit.port<Consume>}
                  : !conduit.window<memref<128xi32>>
      %e0 = conduit.subview_access %win1 {index = 0 : i64}
                : !conduit.window<memref<128xi32>> -> memref<128xi32>
      %e1 = conduit.subview_access %win1 {index = 1 : i64}
                : !conduit.window<memref<128xi32>> -> memref<128xi32>
      memref.store %val, %e0[%c0] : memref<128xi32>

      // Partial release: 1 element returned → heldCount=1.
      conduit.release %win1 {count = 1 : i64, port = #conduit.port<Consume>}
          : !conduit.window<memref<128xi32>>

      // Second acquire: count=2, held=1 → delta=1 → AGE(1).
      %win2 = conduit.acquire {name = @fifo, count = 2 : i64,
                                port = #conduit.port<Consume>}
                  : !conduit.window<memref<128xi32>>
      %f0 = conduit.subview_access %win2 {index = 0 : i64}
                : !conduit.window<memref<128xi32>> -> memref<128xi32>
      %f1 = conduit.subview_access %win2 {index = 1 : i64}
                : !conduit.window<memref<128xi32>> -> memref<128xi32>
      memref.store %val, %f0[%c0] : memref<128xi32>

      conduit.release %win2 {count = 2 : i64, port = #conduit.port<Consume>}
          : !conduit.window<memref<128xi32>>

      aie.end
    }

    aie.runtime_sequence(%in: memref<512xi32>) {
      aiex.npu.dma_memcpy_nd (%in[0,0,0,0][1,1,1,512][0,0,0,1])
          {metadata = @fifo_shim_alloc, id = 0 : i64} : memref<512xi32>
    }
  }
}
