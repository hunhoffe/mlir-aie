// RUN: aie-opt --objectfifo-to-conduit --conduit-to-dma %s | FileCheck %s
// XFAIL: *
//
// Pass A + Pass C end-to-end test: depth-2 single-consumer objectfifo (double-buffering).
//
// Known gap: Pass C currently generates only a depth-1 BD chain regardless of
// the declared depth.  The depth-N BD ring (buff_0 -> buff_1 -> buff_0 ...)
// is a known unimplemented feature; see CLAUDE.md "Known gaps in Pass C".
//
// This test defines the EXPECTED output once the depth-2 BD chain is fixed.
// Until that fix lands, this test is marked XFAIL so the lit suite stays green.
//
// Ground truth (from --aie-objectFifo-stateful-transform on the same input):
//   The BD chain lives in aie.mem(%tile_0_2), NOT aie.memtile_dma — there
//   is no MemTile relay here, only a direct shim-to-tile-DMA path.
//   The ring is: ^ingest_0 -> ^ingest_1 -> ^ingest_0 (two BD blocks, ring closure).
//   Names follow stateful-transform convention: input_fifo_cons_buff_{0,1}.
//
// Expected resource counts for depth=2 (double-buffering):
//   aie.buffer:   2  (input_fifo_cons_buff_0, input_fifo_cons_buff_1 on tile_0_2)
//   aie.lock:     4  (cons prod_lock init=2, cons cons_lock init=0 on tile_0_2;
//                     prod_lock init=0, cons_lock init=0 on shim tile_0_0)
//   aie.flow:     1  (shim tile 0,0 -> tile 0,2)
//   aie.dma_bd:   2  (two BD blocks in the BD ring, one per buffer)
//   aie.next_bd:  2  (buff_0->buff_1 and buff_1->buff_0 completing the ring)
//
// Naming follows --aie-objectFifo-stateful-transform convention:
//   Consumer-side: input_fifo_cons_buff_N, input_fifo_cons_prod_lock_0,
//                  input_fifo_cons_cons_lock_0
//   Shim-side: input_fifo_prod_lock_0, input_fifo_cons_lock_0

// CHECK-LABEL: module
// CHECK:   aie.device(npu1_1col) {
// CHECK:     %{{.*}}tile_0_0 = aie.tile(0, 0)
// CHECK:     %{{.*}}tile_0_2 = aie.tile(0, 2)
// --- Two buffers on the consumer tile (depth=2) ---
// CHECK:     %[[BUFF0:.*]] = aie.buffer(%{{.*}}tile_0_2)
// CHECK-SAME:   sym_name = "input_fifo_cons_buff_0"
// CHECK:     %[[BUFF1:.*]] = aie.buffer(%{{.*}}tile_0_2)
// CHECK-SAME:   sym_name = "input_fifo_cons_buff_1"
// --- Consumer prod_lock (init=2: two free slots) and cons_lock (init=0) ---
// CHECK:     %[[CONS_PROD:.*]] = aie.lock(%{{.*}}tile_0_2
// CHECK-SAME:   init = 2
// CHECK-SAME:   sym_name = "input_fifo_cons_prod_lock_0"
// CHECK:     %[[CONS_CONS:.*]] = aie.lock(%{{.*}}tile_0_2
// CHECK-SAME:   init = 0
// CHECK-SAME:   sym_name = "input_fifo_cons_cons_lock_0"
// CHECK:     aie.shim_dma_allocation @{{.*}}shim_alloc
// CHECK:     aie.flow(%{{.*}}tile_0_0, DMA : 0, %{{.*}}tile_0_2, DMA : 0)
// --- Tile DMA (aie.mem, not aie.memtile_dma) holds the BD ring ---
// The BD ring must contain exactly TWO dma_bd blocks and TWO next_bd ops.
// CHECK:     aie.mem(%{{.*}}tile_0_2) {
// CHECK:       aie.dma_start(S2MM
// CHECK:       aie.dma_bd(%[[BUFF0]]
// CHECK:       aie.use_lock(%[[CONS_CONS]], Release, 1)
// CHECK:       aie.next_bd
// CHECK:       aie.dma_bd(%[[BUFF1]]
// CHECK:       aie.use_lock(%[[CONS_CONS]], Release, 1)
// CHECK:       aie.next_bd
// CHECK:       aie.end
// CHECK:     }
// Verify exactly 2 BD blocks and 2 next_bd ops in total (ring has no extras).
// CHECK-COUNT-2: aie.dma_bd(
// CHECK-COUNT-2: aie.next_bd
// --- Core body: acquire/release use the consumer locks ---
// CHECK:     aie.core(%{{.*}}tile_0_2) {
// CHECK:       scf.for
// CHECK:         aie.use_lock(%[[CONS_CONS]], AcquireGreaterEqual, 1)
// CHECK:         aie.use_lock(%[[CONS_PROD]], Release, 1)
// CHECK:     }
// CHECK-NOT: conduit.create
// CHECK-NOT: conduit.acquire
// CHECK-NOT: conduit.release

module {
  aie.device(npu1_1col) {
    func.func @process_10_i32(%line_in: memref<10xi32>) -> () {
      return
    }

    %tile_0_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)
    // depth=2 declares double-buffering: two ping-pong buffers
    aie.objectfifo @input_fifo(%tile_0_0, {%tile_0_2}, 2 : i32) : !aie.objectfifo<memref<10xi32>>

    %core_0_2 = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c10 = arith.constant 10 : index

      scf.for %arg0 = %c0 to %c10 step %c1 {
        %0 = aie.objectfifo.acquire @input_fifo(Consume, 1) : !aie.objectfifosubview<memref<10xi32>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<10xi32>> -> memref<10xi32>
        func.call @process_10_i32(%1) : (memref<10xi32>) -> ()
        aie.objectfifo.release @input_fifo(Consume, 1)
      }

      aie.end
    } {dynamic_objfifo_lowering = true}
  }
}
