// RUN: aie-opt --objectfifo-to-conduit --conduit-to-dma %s | FileCheck %s
//
// Non-uniform cross-block acquire: preamble(2) / loop-body(3) / tail(2).
//
// This is the sliding-window pattern from the bottleneck benchmark.
// The core entry block holds preamble and tail acquires; the loop body
// is a separate block (scf.for body) with a strictly larger acquire count.
//
// Bug 1 (FIXED): blockWindowMap["fifo"] was overwritten by the tail window,
// causing the loop body's cross-block lookup to find the tail (which does not
// dominate the scf.for), fail the SSA check, and emit conduit.acquire{count=3}
// causing the loop body to emit AcquireGreaterEqual(3) → hardware deadlock.
// Fix: blockWindowMap stores a vector; lookup picks the latest dominating one.
// Result: Pass C infers delta=1 for the loop body (lastAcquireCount=2).
//
// Bug 2 (FIXED): the tail acquire(2) after the scf.for generated
// AcquireGreaterEqual(2), but only 1 new row is needed (the other is still
// in the buffer from the last middle window). Pass C now infers delta from
// same-block heldCount: after preamble release(1), heldCount=1, so
// tail delta = 2-1 = 1 → AcquireGreaterEqual(1).
//
// Lock budget per pass: preamble(2) + N×middle(1) + tail(1) = 2+N+1 = N+3.
// For this test: N=5 middle iters, total rows = 5+3 = 8. ✓
//
// Topology: shim(0,0) → compute(0,2), depth=4 fifo, 1D DMA.
//
// CHECK-LABEL: module @sliding_window_nonuniform

// Preamble: acquire 2 (before scf.for, in core entry block).
// CHECK: aie.core(%tile_0_2)
// CHECK: use_lock(%{{.*}}_cons_lock_0, AcquireGreaterEqual, 2)

// Loop body: acquire only 1 (delta = 3 - lastAcquireCount(2) = 1).
// Regression: the original bug emitted AcquireGreaterEqual(3) here.
// CHECK: scf.for
// CHECK-NEXT: use_lock(%{{.*}}_cons_lock_0, AcquireGreaterEqual, 1)

// Tail: acquire only 1 (delta = 2 - heldCount(1) = 1; 1 row still held).
// Regression: the original bug emitted AcquireGreaterEqual(2) here.
// CHECK: use_lock(%{{.*}}_cons_lock_0, AcquireGreaterEqual, 1)

// Neither the middle nor the tail must ever acquire 2 or 3 in full.
// CHECK-NOT: use_lock(%{{.*}}_cons_lock_0, AcquireGreaterEqual, 2)
// CHECK-NOT: use_lock(%{{.*}}, AcquireGreaterEqual, 3)

module @sliding_window_nonuniform {
  aie.device(npu1_1col) {
    %shim = aie.tile(0, 0)
    %tile = aie.tile(0, 2)

    // depth=4 to fit max sliding window of 3 + 1 overlap
    aie.objectfifo @fifo(%shim, {%tile}, 4 : i32)
        : !aie.objectfifo<memref<128xi32>>

    aie.objectfifo @out(%tile, {%shim}, 2 : i32)
        : !aie.objectfifo<memref<64xi32>>

    %core = aie.core(%tile) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c5 = arith.constant 5 : index

      // Preamble: acquire 2 (top border row — needs 2 input lines).
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

      // Simulate compute using pre0, pre1 → out_pre (omitted for brevity).

      aie.objectfifo.release @out(Produce, 1)
      aie.objectfifo.release @fifo(Consume, 1)

      // Middle rows: sliding window of 3 — acquire(3)/release(1) per row.
      scf.for %i = %c0 to %c5 step %c1 {
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

      // Tail: acquire 2 (bottom border row — only 2 input lines needed).
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
