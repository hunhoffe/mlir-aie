// RUN: aie-opt --objectfifo-to-conduit --conduit-to-dma %s | FileCheck %s
// XFAIL: *
//
// Pass A + Pass C end-to-end test: depth-3 single-consumer objectfifo (triple-buffering).
//
// Known gap: Pass C only generates a depth-1 BD chain.  This test verifies
// the N-general depth-N BD ring once the fix lands.  Depth-3 is the minimal
// case that proves N-generality beyond the ping-pong (depth-2) case.
//
// Ground truth (from --aie-objectFifo-stateful-transform on the same input):
//   BD chain lives in aie.mem(%tile_0_2), NOT aie.memtile_dma.
//   Ring: ^bd0 -> ^bd1 -> ^bd2 -> ^bd0 (three BD blocks, ring closure at buff_2).
//   Names: win_fifo_cons_buff_{0,1,2}, win_fifo_cons_prod_lock_0 (init=3),
//          win_fifo_cons_cons_lock_0 (init=0).
//
// Expected resource counts for depth=3:
//   aie.buffer:   3  (win_fifo_cons_buff_0, _buff_1, _buff_2 on tile_0_2)
//   aie.lock:     4  (cons prod_lock init=3, cons cons_lock init=0 on tile_0_2;
//                     prod_lock, cons_lock on shim tile_0_0)
//   aie.flow:     1
//   aie.dma_bd:   3  (three blocks in the BD ring)
//   aie.next_bd:  3  (ring: 0->1->2->0)
//
// The core side uses a sliding-window acquire-2 pattern to exercise the
// depth-3 fifo meaningfully (acquires 2 of 3 slots simultaneously).

// CHECK-LABEL: module @depth_three_fifo
// CHECK:   aie.device(npu1_1col) {
// CHECK:     %{{.*}}tile_0_0 = aie.tile(0, 0)
// CHECK:     %{{.*}}tile_0_2 = aie.tile(0, 2)
// --- Three buffers on the consumer tile (depth=3) ---
// CHECK:     %[[BUFF0:.*]] = aie.buffer(%{{.*}}tile_0_2)
// CHECK-SAME:   sym_name = "win_fifo_cons_buff_0"
// CHECK:     %[[BUFF1:.*]] = aie.buffer(%{{.*}}tile_0_2)
// CHECK-SAME:   sym_name = "win_fifo_cons_buff_1"
// CHECK:     %[[BUFF2:.*]] = aie.buffer(%{{.*}}tile_0_2)
// CHECK-SAME:   sym_name = "win_fifo_cons_buff_2"
// --- prod_lock init=3 (three free slots), cons_lock init=0 ---
// CHECK:     %[[CONS_PROD:.*]] = aie.lock(%{{.*}}tile_0_2
// CHECK-SAME:   init = 3
// CHECK-SAME:   sym_name = "win_fifo_cons_prod_lock_0"
// CHECK:     %[[CONS_CONS:.*]] = aie.lock(%{{.*}}tile_0_2
// CHECK-SAME:   init = 0
// CHECK-SAME:   sym_name = "win_fifo_cons_cons_lock_0"
// CHECK:     aie.shim_dma_allocation @{{.*}}shim_alloc
// CHECK:     aie.flow(%{{.*}}tile_0_0, DMA : 0, %{{.*}}tile_0_2, DMA : 0)
// --- Tile DMA (aie.mem, not aie.memtile_dma) holds the BD ring ---
// The BD ring must contain exactly THREE dma_bd blocks in order, plus ring closure.
// CHECK:     aie.mem(%{{.*}}tile_0_2) {
// CHECK:       aie.dma_start(S2MM
// CHECK:       aie.dma_bd(%[[BUFF0]]
// CHECK:       aie.use_lock(%[[CONS_CONS]], Release, 1)
// CHECK:       aie.next_bd
// CHECK:       aie.dma_bd(%[[BUFF1]]
// CHECK:       aie.use_lock(%[[CONS_CONS]], Release, 1)
// CHECK:       aie.next_bd
// CHECK:       aie.dma_bd(%[[BUFF2]]
// CHECK:       aie.use_lock(%[[CONS_CONS]], Release, 1)
// CHECK:       aie.next_bd
// CHECK:       aie.end
// CHECK:     }
// Verify exactly 3 BD blocks and 3 next_bd ops (ring closure, not linear chain).
// CHECK-COUNT-3: aie.dma_bd(
// CHECK-COUNT-3: aie.next_bd
// --- Core body ---
// CHECK:     aie.core(%{{.*}}tile_0_2) {
// CHECK:       scf.for
// CHECK:         aie.use_lock(%[[CONS_CONS]], AcquireGreaterEqual
// CHECK:         aie.use_lock(%[[CONS_PROD]], Release
// CHECK:     }
// CHECK-NOT: conduit.create
// CHECK-NOT: conduit.acquire
// CHECK-NOT: conduit.release

module @depth_three_fifo {
  aie.device(npu1_1col) {
    func.func @sliding_window(%a: memref<10xi32>, %b: memref<10xi32>) -> () {
      return
    }

    %tile_0_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)
    // depth=3: triple-buffering for sliding window access patterns
    aie.objectfifo @win_fifo(%tile_0_0, {%tile_0_2}, 3 : i32) : !aie.objectfifo<memref<10xi32>>

    %core_0_2 = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c8 = arith.constant 8 : index

      scf.for %arg0 = %c0 to %c8 step %c1 {
        // Sliding window: acquire 2 consecutive slots simultaneously
        %0 = aie.objectfifo.acquire @win_fifo(Consume, 2) : !aie.objectfifosubview<memref<10xi32>>
        %elem0 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<10xi32>> -> memref<10xi32>
        %elem1 = aie.objectfifo.subview.access %0[1] : !aie.objectfifosubview<memref<10xi32>> -> memref<10xi32>
        func.call @sliding_window(%elem0, %elem1) : (memref<10xi32>, memref<10xi32>) -> ()
        aie.objectfifo.release @win_fifo(Consume, 1)
      }

      aie.end
    } {dynamic_objfifo_lowering = true}
  }
}
