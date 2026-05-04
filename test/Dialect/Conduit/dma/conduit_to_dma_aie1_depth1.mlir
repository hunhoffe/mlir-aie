// RUN: aie-opt --objectfifo-to-conduit --conduit-to-dma %s | FileCheck %s
//
// Pass A + Pass C end-to-end test: depth-1 single-consumer objectfifo on AIE1
// (xcvc1902).
//
// Key difference from npu1_1col (AIE2):
//   - AIE2 uses semaphore-based AcquireGreaterEqual locks
//   - AIE1 uses value-based Acquire/Release locks with explicit values
//     One lock per buffer slot (init=0 = empty).
//     Core: Acquire(1)=wait-full, Release(0)=mark-empty
//     DMA:  Acquire(0)=wait-empty, Release(1)=mark-full

// Structural CHECK patterns using CHECK-DAG where order is not guaranteed.

// CHECK: aie.device(xcvc1902)

// --- One buffer on the consumer tile ---
// CHECK: aie.buffer({{.*}}) {sym_name = "input_fifo{{.*}}buff_0"

// --- One AIE1 per-slot lock (init=0 = empty) ---
// CHECK: aie.lock({{.*}}) {init = 0 : i32, sym_name = "input_fifo{{.*}}lock_0"

// --- Core body: acquire with value 1 (full), release with value 0 (empty) ---
// CHECK: aie.core(
// CHECK: aie.use_lock(%{{.*}}, Acquire, 1)
// CHECK: aie.use_lock(%{{.*}}, Release, 0)

// --- Shim DMA allocation ---
// CHECK: aie.shim_dma_allocation @{{.*}}shim_alloc

// --- Flow: shim -> tile ---
// CHECK: aie.flow(%{{.*}}, DMA : 0, %{{.*}}, DMA : 0)

// --- Tile DMA: depth-1 BD ring ---
// CHECK: aie.mem(
// CHECK: aie.dma_start(S2MM
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
    func.func @process_16_i32(%buf: memref<16xi32>) -> () {
      return
    }

    %tile70 = aie.tile(7, 0)
    %tile71 = aie.tile(7, 1)
    aie.objectfifo @input_fifo(%tile70, {%tile71}, 1 : i32) : !aie.objectfifo<memref<16xi32>>

    %core71 = aie.core(%tile71) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c8 = arith.constant 8 : index

      scf.for %arg0 = %c0 to %c8 step %c1 {
        %0 = aie.objectfifo.acquire @input_fifo(Consume, 1) : !aie.objectfifosubview<memref<16xi32>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>
        func.call @process_16_i32(%1) : (memref<16xi32>) -> ()
        aie.objectfifo.release @input_fifo(Consume, 1)
      }

      aie.end
    } {dynamic_objfifo_lowering = true}
  }
}
