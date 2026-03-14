// RUN: aie-opt --objectfifo-to-conduit --conduit-to-dma %s | FileCheck %s
//
// Pass A + Pass C end-to-end test: depth-2 double-buffered objectfifo on AIE1
// (xcvc1902).
//
// AIE1 per-slot lock semantics for depth-2:
//   Two locks (lock_0, lock_1), each init=0 (empty).
//   BD block 0 uses lock_0; BD block 1 uses lock_1.
//   Core uses rotation counter (arith.remui) to advance through slots.
//   Fix 1b verified: rotation counter uses arith.remui (true modulo, not single subtract).

// CHECK: aie.device(xcvc1902)

// --- Two data buffers on the consumer tile ---
// CHECK: aie.buffer({{.*}}) {sym_name = "data_fifo{{.*}}buff_0"
// CHECK: aie.buffer({{.*}}) {sym_name = "data_fifo{{.*}}buff_1"

// --- Two AIE1 per-slot locks (init=0 each) ---
// CHECK: aie.lock({{.*}}) {init = 0 : i32, sym_name = "data_fifo{{.*}}lock_0"
// CHECK: aie.lock({{.*}}) {init = 0 : i32, sym_name = "data_fifo{{.*}}lock_1"

// --- Rotation counter buffer (memref<1xi32>) ---
// CHECK: aie.buffer(%{{.*}}) : memref<1xi32>

// --- Core body: rotation counter init, scf.index_switch, arith.remui (fix 1b) ---
// CHECK: aie.core(
// CHECK: memref.store
// CHECK: scf.for
// CHECK: aie.use_lock(%{{.*}}, Acquire, 1)
// CHECK: scf.index_switch
// CHECK: aie.use_lock(%{{.*}}, Release, 0)
// CHECK: arith.remui

// --- Shim DMA and flow ---
// CHECK: aie.shim_dma_allocation @{{.*}}shim_alloc
// CHECK: aie.flow(%{{.*}}, DMA : 0, %{{.*}}, DMA : 0)

// --- Tile DMA: two-block BD ring, one lock per slot ---
// CHECK: aie.mem(
// CHECK: aie.dma_start(S2MM
// CHECK: aie.use_lock(%{{.*}}, Acquire, 0)
// CHECK: aie.dma_bd(
// CHECK: aie.use_lock(%{{.*}}, Release, 1)
// CHECK: aie.next_bd
// CHECK: aie.use_lock(%{{.*}}, Acquire, 0)
// CHECK: aie.dma_bd(
// CHECK: aie.use_lock(%{{.*}}, Release, 1)
// CHECK: aie.next_bd
// CHECK: aie.end

// --- No residual Conduit ops ---
// CHECK-NOT: conduit.create
// CHECK-NOT: conduit.acquire
// CHECK-NOT: conduit.release

module {
  aie.device(xcvc1902) {
    func.func @process_8_i32(%buf: memref<8xi32>) -> () {
      return
    }

    %tile00 = aie.tile(0, 0)
    %tile02 = aie.tile(0, 2)
    // depth=2: double-buffering; AIE1 per-slot lock model
    aie.objectfifo @data_fifo(%tile00, {%tile02}, 2 : i32) : !aie.objectfifo<memref<8xi32>>

    %core02 = aie.core(%tile02) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c8 = arith.constant 8 : index

      scf.for %arg0 = %c0 to %c8 step %c1 {
        %0 = aie.objectfifo.acquire @data_fifo(Consume, 1) : !aie.objectfifosubview<memref<8xi32>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<8xi32>> -> memref<8xi32>
        func.call @process_8_i32(%1) : (memref<8xi32>) -> ()
        aie.objectfifo.release @data_fifo(Consume, 1)
      }

      aie.end
    } {dynamic_objfifo_lowering = true}
  }
}
