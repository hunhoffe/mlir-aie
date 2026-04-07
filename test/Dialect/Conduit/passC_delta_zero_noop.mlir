// RUN: aie-opt --conduit-check-channels --conduit-to-dma %s | FileCheck %s
//
// Delta-zero noop: parent acquires 3, no release; loop body acquires 3.
// delta = 3 - lastAcquireCount(3) = 0 → NO AcquireGreaterEqual emitted in the
// loop body.
//
// This tests the read-only re-access pattern: the parent acquires a window
// and the loop body re-acquires the same size without releasing.  Since the
// DMA already delivered all 3 elements, the child acquire needs zero
// additional elements — the existing window is large enough.
//
// State trace:
//   Parent acquire(3): heldCount=0 → delta=3 → AGE(3). Update: held=3, last=3.
//   (no release)
//   Enter scf.for: childState.heldCount = lastAcquireCount = 3.
//   Child acquire(3): heldCount=3 → delta=3-3=0 → no AGE emitted.
//
// Topology: shim(0,0) → compute(0,2), depth=4, element=memref<128xi32>.
//
// CHECK-LABEL: module @passC_delta_zero_noop
// CHECK: aie.core(%tile_0_2)
//
// Preamble: AGE(3) — fresh acquire.
// CHECK: use_lock(%{{.*}}_cons_lock_0, AcquireGreaterEqual, 3)
//
// Loop body: delta=0 → no AcquireGreaterEqual.
// CHECK: scf.for
// CHECK-NOT: AcquireGreaterEqual
// CHECK: aie.end

module @passC_delta_zero_noop {
  aie.device(npu1_1col) {
    %shim = aie.tile(0, 0)
    %tile = aie.tile(0, 2)

    conduit.create @fifo {slot_elems = 512 : i64, depth = 4 : i64,
                    element_type = memref<128xi32>,
                    producer_tile = array<i64: 0, 0>,
                    consumer_tiles = array<i64: 0, 2>}

    aie.shim_dma_allocation @fifo_shim_alloc(%shim, MM2S, 0)

    %core = aie.core(%tile) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c4 = arith.constant 4 : index
      %val = arith.constant 42 : i32

      // Parent: acquire(3), no release → heldCount=3, lastAcquireCount=3.
      %win_pre = conduit.acquire {name = @fifo, count = 3 : i64,
                                   port = #conduit.port<Consume>}
                     : !conduit.window<memref<128xi32>>
      %pre0 = conduit.subview_access %win_pre {index = 0 : i64}
                  : !conduit.window<memref<128xi32>> -> memref<128xi32>
      %pre1 = conduit.subview_access %win_pre {index = 1 : i64}
                  : !conduit.window<memref<128xi32>> -> memref<128xi32>
      %pre2 = conduit.subview_access %win_pre {index = 2 : i64}
                  : !conduit.window<memref<128xi32>> -> memref<128xi32>
      memref.store %val, %pre0[%c0] : memref<128xi32>

      // Loop body: acquire(3), delta=0 → no AcquireGreaterEqual.
      // Re-accesses the same 3 elements already in the buffer.
      scf.for %i = %c0 to %c4 step %c1 {
        %win_mid = conduit.acquire {name = @fifo, count = 3 : i64,
                                     port = #conduit.port<Consume>}
                       : !conduit.window<memref<128xi32>>
        %mid0 = conduit.subview_access %win_mid {index = 0 : i64}
                    : !conduit.window<memref<128xi32>> -> memref<128xi32>
        %mid1 = conduit.subview_access %win_mid {index = 1 : i64}
                    : !conduit.window<memref<128xi32>> -> memref<128xi32>
        %mid2 = conduit.subview_access %win_mid {index = 2 : i64}
                    : !conduit.window<memref<128xi32>> -> memref<128xi32>
        memref.store %val, %mid0[%c0] : memref<128xi32>
      }

      conduit.release %win_pre {count = 3 : i64, port = #conduit.port<Consume>}
          : !conduit.window<memref<128xi32>>

      aie.end
    }

    aie.runtime_sequence(%in: memref<512xi32>) {
      aiex.npu.dma_memcpy_nd (%in[0,0,0,0][1,1,1,512][0,0,0,1])
          {metadata = @fifo_shim_alloc, id = 0 : i64} : memref<512xi32>
    }
  }
}
