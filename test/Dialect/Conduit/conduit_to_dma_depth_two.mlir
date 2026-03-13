// RUN: aie-opt --objectfifo-to-conduit --conduit-to-dma %s | FileCheck %s
//
// Pass A + Pass C end-to-end test: depth-2 single-consumer objectfifo (double-buffering).
//
// Verifies the full depth-2 lowering including:
//   - Two ping-pong buffers on the consumer tile
//   - Rotation counter buffer for dynamic buffer selection in the core body
//   - Counter initialized to 0 at top of core, incremented mod 2 after each release
//   - scf.index_switch in core body to alternate between buff_0 and buff_1
//   - Two-block BD ring in aie.mem (not aie.memtile_dma)
//   - Shim-side locks and flow
//
// Resource counts:
//   aie.buffer:  3  (input_fifo_cons_buff_0, input_fifo_cons_buff_1, rotation counter)
//   aie.lock:    4  (tile_0_2: cons_prod_lock init=2, cons_cons_lock init=0;
//                    shim: prod_lock init=0, cons_lock init=0)
//   aie.flow:    1
//   aie.dma_bd:  2  (depth-2 BD ring)
//   aie.next_bd: 2

// CHECK-LABEL: module
// CHECK:   aie.device(npu1_1col) {
// CHECK:     %{{.*}}tile_0_0 = aie.tile(0, 0)
// CHECK:     %{{.*}}tile_0_2 = aie.tile(0, 2)
// --- Two data buffers on the consumer tile ---
// CHECK:     %[[BUFF0:.*]] = aie.buffer(%{{.*}}tile_0_2)
// CHECK-SAME:   sym_name = "input_fifo_cons_buff_0"
// CHECK:     %[[BUFF1:.*]] = aie.buffer(%{{.*}}tile_0_2)
// CHECK-SAME:   sym_name = "input_fifo_cons_buff_1"
// --- Consumer locks: prod_lock (init=2 free slots), cons_lock (init=0) ---
// CHECK:     %[[CONS_PROD:.*]] = aie.lock(%{{.*}}tile_0_2
// CHECK-SAME:   init = 2
// CHECK-SAME:   sym_name = "input_fifo_cons_prod_lock_0"
// CHECK:     %[[CONS_CONS:.*]] = aie.lock(%{{.*}}tile_0_2
// CHECK-SAME:   init = 0
// CHECK-SAME:   sym_name = "input_fifo_cons_cons_lock_0"
// --- Rotation counter buffer (no sym_name, anonymous) ---
// CHECK:     aie.buffer(%{{.*}}tile_0_2) : memref<1xi32>
// --- Core body: counter init, scf.index_switch, and counter increment ---
// CHECK:     aie.core(%{{.*}}tile_0_2) {
// CHECK:       memref.store {{.*}} : memref<1xi32>
// CHECK:       scf.for
// CHECK:         aie.use_lock(%[[CONS_CONS]], AcquireGreaterEqual, 1)
// CHECK:         scf.index_switch
// CHECK:           scf.yield %[[BUFF0]]
// CHECK:           scf.yield %[[BUFF1]]
// CHECK:           scf.yield %[[BUFF0]]
// CHECK:         func.call @process_10_i32
// CHECK:         aie.use_lock(%[[CONS_PROD]], Release, 1)
// CHECK:         memref.store {{.*}} : memref<1xi32>
// CHECK:     }
// --- Shim DMA and flow ---
// CHECK:     aie.shim_dma_allocation @{{.*}}shim_alloc
// CHECK:     aie.flow(%{{.*}}tile_0_0, DMA : 0, %{{.*}}tile_0_2, DMA : 0)
// --- Tile DMA: depth-2 BD ring (aie.mem, not aie.memtile_dma) ---
// CHECK:     aie.mem(%{{.*}}tile_0_2) {
// CHECK:       aie.dma_start(S2MM
// CHECK:       aie.use_lock(%[[CONS_PROD]], AcquireGreaterEqual, 1)
// CHECK:       aie.dma_bd(%[[BUFF0]]
// CHECK:       aie.use_lock(%[[CONS_CONS]], Release, 1)
// CHECK:       aie.next_bd
// CHECK:       aie.use_lock(%[[CONS_PROD]], AcquireGreaterEqual, 1)
// CHECK:       aie.dma_bd(%[[BUFF1]]
// CHECK:       aie.use_lock(%[[CONS_CONS]], Release, 1)
// CHECK:       aie.next_bd
// CHECK:       aie.end
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
