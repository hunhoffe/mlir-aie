// RUN: aie-opt --conduit-check-channels --conduit-to-dma %s | FileCheck %s
//
// Full preamble/middle/tail sliding-window pattern with delta inference.
//
// This is the canonical sliding-window pattern from the bottleneck benchmark,
// expressed directly in Conduit IR (no ObjectFIFO or air.channel source).
//
//   Preamble: acquire(2), release(1)  → AGE(2)
//   Middle:   scf.for { acquire(3), release(1) }  → AGE(1) per iter
//   Tail:     acquire(2), release(2)  → NO AcquireGreaterEqual (delta=0)
//
// State trace (with cross-block held-count propagation fix):
//   Preamble acquire(2):  heldCount=0 → delta=2 → AGE(2). held=2, last=2.
//   Preamble release(1):  held=2-1=1. last=2 (unchanged).
//   Enter scf.for:        childState.heldCount = last = 2.
//     Middle acquire(3):  held=2 → delta=3-2=1 → AGE(1). held=3, last=3.
//     Middle release(1):  held=3-1=2. last=3.
//     (next iter: acquire(3), held=2, delta=1 → AGE(1). Consistent.)
//   Child exits with:     held=2, last=3.
//   Back in parent:       held updated to 2 (child exit held propagated back).
//   Tail acquire(2):      held=2 → delta=2-2=0 → NO AcquireGreaterEqual.
//   Tail release(2):      held=2-2=0.
//
// Expected: AGE(2), AGE(1) inside loop, NO AGE for tail.
// No AGE(3) anywhere.  No AGE(2) after the first one.
//
// Topology: shim(0,0) → compute(0,2), depth=4, element=memref<128xi32>.
//
// CHECK-LABEL: module @passC_delta_sliding_window
//
// Preamble: AGE(2).
// CHECK: aie.core(%tile_0_2)
// CHECK: use_lock(%{{.*}}_cons_lock_0, AcquireGreaterEqual, 2)
//
// Between preamble and loop: no more AGE(2) or AGE(3).
// CHECK-NOT: AcquireGreaterEqual, 2
// CHECK-NOT: AcquireGreaterEqual, 3
//
// Middle: AGE(1) inside loop.
// CHECK: scf.for
// CHECK-NOT: AcquireGreaterEqual, 2
// CHECK-NOT: AcquireGreaterEqual, 3
// CHECK: use_lock(%{{.*}}_cons_lock_0, AcquireGreaterEqual, 1)
//
// Tail: NO AcquireGreaterEqual (delta=0 after cross-block held-count fix).
// The Release(2) must be the next use_lock after the loop's Release(1).
// CHECK-NOT: AcquireGreaterEqual, 2
// CHECK-NOT: AcquireGreaterEqual, 1
// CHECK: aie.end

module @passC_delta_sliding_window {
  aie.device(npu1_1col) {
    %shim = aie.tile(0, 0)
    %tile = aie.tile(0, 2)

    conduit.create @fifo {slot_elems = 512 : i64, depth = 4 : i64,
                    element_type = memref<128xi32>
                    }

    aie.shim_dma_allocation @fifo_shim_alloc(%shim, MM2S, 0)

    %core = aie.core(%tile) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c4 = arith.constant 4 : index
      %val = arith.constant 42 : i32

      // === Preamble: acquire(2), release(1) ===
      %win_pre = conduit.acquire {name = @fifo, count = 2 : i64,
                                   port = #conduit.port<Consume>}
                     : !conduit.window<memref<128xi32>>
      %pre0 = conduit.subview_access %win_pre {index = 0 : i64}
                  : !conduit.window<memref<128xi32>> -> memref<128xi32>
      %pre1 = conduit.subview_access %win_pre {index = 1 : i64}
                  : !conduit.window<memref<128xi32>> -> memref<128xi32>
      memref.store %val, %pre0[%c0] : memref<128xi32>

      conduit.release %win_pre {count = 1 : i64, port = #conduit.port<Consume>}
          : !conduit.window<memref<128xi32>>

      // === Middle: scf.for { acquire(3), release(1) } ===
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

        conduit.release %win_mid {count = 1 : i64, port = #conduit.port<Consume>}
            : !conduit.window<memref<128xi32>>
      }

      // === Tail: acquire(2), release(2) ===
      // After the loop, parent held is updated to 2 (child exit held).
      // acquire(2) → delta = 2-2 = 0 → NO AcquireGreaterEqual.
      %win_tail = conduit.acquire {name = @fifo, count = 2 : i64,
                                    port = #conduit.port<Consume>}
                      : !conduit.window<memref<128xi32>>
      %tail0 = conduit.subview_access %win_tail {index = 0 : i64}
                   : !conduit.window<memref<128xi32>> -> memref<128xi32>
      %tail1 = conduit.subview_access %win_tail {index = 1 : i64}
                   : !conduit.window<memref<128xi32>> -> memref<128xi32>
      memref.store %val, %tail0[%c0] : memref<128xi32>

      conduit.release %win_tail {count = 2 : i64, port = #conduit.port<Consume>}
          : !conduit.window<memref<128xi32>>

      aie.end
    }

    aie.runtime_sequence(%in: memref<896xi32>) {
      aiex.npu.dma_memcpy_nd (%in[0,0,0,0][1,1,1,896][0,0,0,1])
          {metadata = @fifo_shim_alloc, id = 0 : i64} : memref<896xi32>
    }
  }
}
