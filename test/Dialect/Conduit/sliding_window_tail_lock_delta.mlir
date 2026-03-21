// RUN: aie-opt --objectfifo-to-conduit --conduit-to-dma %s | FileCheck %s
//
// Verifies Pass C correctly infers the lock delta for the sliding-window
// tail acquire using same-block held-count tracking.
//
// In a preamble(2)/middle(3,release 1)/tail(2) sliding window with depth=4:
//   - After N middle iterations each releasing 1, heldCount = 1 at the tail.
//   - Pass C infers delta = count(2) - heldCount(1) = 1.
//   - AcquireGreaterEqual(1) is emitted — correct.
//
// Regression guard: neither the middle nor the tail acquires 2.
//
// Topology: shim(0,0) → compute(0,2), depth=4, 7 input rows → 6 output rows.
// Simplified from the bottleneck benchmark (fewer rows for test brevity).
//
// CHECK-LABEL: module @sliding_window_tail_delta

// Preamble: acquire 2.
// CHECK: aie.core(%tile_0_2)
// CHECK: use_lock(%{{.*}}_cons_lock_0, AcquireGreaterEqual, 2)

// Middle (4 iters): acquire delta 1 each.
// CHECK: scf.for
// CHECK-NEXT: use_lock(%{{.*}}_cons_lock_0, AcquireGreaterEqual, 1)

// Tail: must acquire only 1 (1 row still held from last middle window).
// BUG: currently emits AcquireGreaterEqual(2); correct value is 1.
// CHECK: use_lock(%{{.*}}_cons_lock_0, AcquireGreaterEqual, 1)

// Regression guard: tail must never acquire 2 — that exhausts the DMA supply.
// CHECK-NOT: use_lock(%{{.*}}_cons_lock_0, AcquireGreaterEqual, 2)

module @sliding_window_tail_delta {
  aie.device(npu1_1col) {
    %shim = aie.tile(0, 0)
    %tile = aie.tile(0, 2)

    // depth=4: fits max sliding window of 3 + 1 overlap.
    aie.objectfifo @fifo(%shim, {%tile}, 4 : i32)
        : !aie.objectfifo<memref<128xi32>>

    aie.objectfifo @out(%tile, {%shim}, 2 : i32)
        : !aie.objectfifo<memref<64xi32>>

    %core = aie.core(%tile) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c4 = arith.constant 4 : index

      // Preamble: 2 rows (top border, duplicate row 0).
      %sv_pre = aie.objectfifo.acquire @fifo(Consume, 2)
                    : !aie.objectfifosubview<memref<128xi32>>
      %pre0 = aie.objectfifo.subview.access %sv_pre[0]
                  : !aie.objectfifosubview<memref<128xi32>> -> memref<128xi32>
      %pre1 = aie.objectfifo.subview.access %sv_pre[1]
                  : !aie.objectfifosubview<memref<128xi32>> -> memref<128xi32>
      %sv_out_pre = aie.objectfifo.acquire @out(Produce, 1)
                        : !aie.objectfifosubview<memref<64xi32>>
      %out_pre = aie.objectfifo.subview.access %sv_out_pre[0]
                     : !aie.objectfifosubview<memref<64xi32>> -> memref<64xi32>
      aie.objectfifo.release @out(Produce, 1)
      aie.objectfifo.release @fifo(Consume, 1)

      // Middle: 4 rows, acquire(3)/release(1) sliding window.
      scf.for %i = %c0 to %c4 step %c1 {
        %sv_mid = aie.objectfifo.acquire @fifo(Consume, 3)
                      : !aie.objectfifosubview<memref<128xi32>>
        %mid0 = aie.objectfifo.subview.access %sv_mid[0]
                    : !aie.objectfifosubview<memref<128xi32>> -> memref<128xi32>
        %mid1 = aie.objectfifo.subview.access %sv_mid[1]
                    : !aie.objectfifosubview<memref<128xi32>> -> memref<128xi32>
        %mid2 = aie.objectfifo.subview.access %sv_mid[2]
                    : !aie.objectfifosubview<memref<128xi32>> -> memref<128xi32>
        %sv_out_mid = aie.objectfifo.acquire @out(Produce, 1)
                          : !aie.objectfifosubview<memref<64xi32>>
        %out_mid = aie.objectfifo.subview.access %sv_out_mid[0]
                       : !aie.objectfifosubview<memref<64xi32>> -> memref<64xi32>
        aie.objectfifo.release @fifo(Consume, 1)
        aie.objectfifo.release @out(Produce, 1)
      }

      // Tail: acquire(2), bottom border.
      // heldCount=1 after partial preamble release → delta=1 → AcquireGreaterEqual(1).
      %sv_tail = aie.objectfifo.acquire @fifo(Consume, 2)
                     : !aie.objectfifosubview<memref<128xi32>>
      %tail0 = aie.objectfifo.subview.access %sv_tail[0]
                   : !aie.objectfifosubview<memref<128xi32>> -> memref<128xi32>
      %tail1 = aie.objectfifo.subview.access %sv_tail[1]
                   : !aie.objectfifosubview<memref<128xi32>> -> memref<128xi32>
      %sv_out_tail = aie.objectfifo.acquire @out(Produce, 1)
                         : !aie.objectfifosubview<memref<64xi32>>
      %out_tail = aie.objectfifo.subview.access %sv_out_tail[0]
                      : !aie.objectfifosubview<memref<64xi32>> -> memref<64xi32>
      aie.objectfifo.release @out(Produce, 1)
      aie.objectfifo.release @fifo(Consume, 1)
      aie.objectfifo.release @fifo(Consume, 1)

      aie.end
    }
  }
}
