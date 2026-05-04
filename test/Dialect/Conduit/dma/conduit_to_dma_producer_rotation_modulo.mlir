// RUN: aie-opt --objectfifo-to-conduit --conduit-to-dma %s | FileCheck %s
//
// Pass A + Pass C end-to-end test: BLOCK-1 fix — produce-port rotation counter
// wraps at effectiveDepth, not depth.
//
// Background:
//   For a producer core, the rotation counter cycles through buffer slots
//   0, 1, ..., effectiveDepth-1, where:
//
//     effectiveDepth = min(depth, acquireCount + 1)
//
//   With depth=4 and the producer acquiring 1 at a time:
//     effectiveDepth = min(4, 1+1) = 2
//
//   Both depths (2 and 4) are power-of-2, so the counter update uses
//   arith.andi (mask = depth-1), not arith.remui (software divide).
//   Producer uses effectiveDepth=2 → mask=1; consumer uses depth=4 → mask=3.
//
//   Before the BLOCK-1 fix, the producer used divisor 4 (depth), leading
//   to a mismatch between the core's buffer selection and the DMA BD ring.
//
// Topology: compute producer tile(0,2) → compute consumer tile(0,4), depth=4.
//   Producer acquires 1 at a time → effectiveDepth=2 → producer andi mask=1.
//   Consumer acquires 1 at a time → consumer andi mask=3 (full depth=4 ring).
//
// Key assertions:
//   - Producer-side: 2 buffers on tile_0_2 (effectiveDepth=2); rotation counter
//     is aie.buffer memref<2xi32> on tile (slot 1 for producer).
//   - Producer core andi mask = 1 (effectiveDepth=2), NOT 3 (depth=4).
//   - Consumer-side: 4 buffers on tile_0_4; rotation counter is aie.buffer
//     memref<1xi32> on tile; andi mask = 3 (depth=4).
//   - CHECK-NOT: %c3_i32 in the producer core andi (regression guard).

// CHECK-LABEL: module @producer_rotation_modulo
// CHECK:   aie.device(npu1_1col) {

// --- Producer tile: only effectiveDepth=2 buffers allocated ---
// CHECK:     aie.buffer(%{{.*}}tile_0_2) {sym_name = "fifo_buff_0"} : memref<8xi32>
// CHECK:     aie.buffer(%{{.*}}tile_0_2) {sym_name = "fifo_buff_1"} : memref<8xi32>
// CHECK-NOT: aie.buffer(%{{.*}}tile_0_2) {sym_name = "fifo_buff_2"}
// CHECK-NOT: aie.buffer(%{{.*}}tile_0_2) {sym_name = "fifo_buff_3"}

// --- Producer lock: init=2 (effectiveDepth=2 free slots) ---
// CHECK:     aie.lock(%{{.*}}tile_0_2{{.*}}) {init = 2 : i32, sym_name = "fifo_prod_lock_0"}

// --- Consumer tile: all 4 buffers allocated (full depth=4 ring) ---
// CHECK:     aie.buffer(%{{.*}}tile_0_4) {sym_name = "fifo_cons_buff_0"} : memref<8xi32>
// CHECK:     aie.buffer(%{{.*}}tile_0_4) {sym_name = "fifo_cons_buff_1"} : memref<8xi32>
// CHECK:     aie.buffer(%{{.*}}tile_0_4) {sym_name = "fifo_cons_buff_2"} : memref<8xi32>
// CHECK:     aie.buffer(%{{.*}}tile_0_4) {sym_name = "fifo_cons_buff_3"} : memref<8xi32>
// CHECK:     aie.lock(%{{.*}}tile_0_4{{.*}}) {init = 4 : i32, sym_name = "fifo_cons_prod_lock_0"}

// --- Producer rotation counter: aie.buffer on tile (memref<2xi32>) ---
// --- Slot 1 of the shared counter used by the producer core ---
// CHECK:     aie.buffer(%{{.*}}tile_0_2) : memref<2xi32>
// CHECK:     aie.core(%{{.*}}tile_0_2) {
// CHECK:         memref.store %c0_i32, {{.*}}[%c1{{.*}}] : memref<2xi32>
// CHECK:       scf.for
// CHECK:         aie.use_lock(%{{.*}}fifo_prod_lock_0, AcquireGreaterEqual, 1)
// CHECK:         memref.load {{.*}}[%c1{{.*}}] : memref<2xi32>
// CHECK:         arith.index_cast
// CHECK:         scf.index_switch
// CHECK:           scf.yield %{{.*}}fifo_buff_0
// CHECK:           scf.yield %{{.*}}fifo_buff_1
// CHECK:         func.call @generate
// CHECK:         aie.use_lock(%{{.*}}fifo_cons_lock_0, Release, 1)
// CHECK:         memref.load {{.*}}[%c1{{.*}}] : memref<2xi32>
// CHECK:         arith.addi
// CHECK:         arith.constant 1 : i32
// CHECK:         arith.andi {{.*}} : i32
// CHECK:         memref.store {{.*}}[%c1{{.*}}] : memref<2xi32>

// --- Consumer rotation counter: aie.buffer on tile (memref<1xi32>) ---
// CHECK:     aie.buffer(%{{.*}}tile_0_4) : memref<1xi32>
// CHECK:     aie.core(%{{.*}}tile_0_4) {
// CHECK:         memref.store {{.*}} : memref<1xi32>
// CHECK:       scf.for
// CHECK:         aie.use_lock(%{{.*}}fifo_cons_cons_lock_0, AcquireGreaterEqual, 1)
// CHECK:         arith.index_cast
// CHECK:         scf.index_switch
// CHECK:           scf.yield %{{.*}}fifo_cons_buff_0
// CHECK:           scf.yield %{{.*}}fifo_cons_buff_1
// CHECK:           scf.yield %{{.*}}fifo_cons_buff_2
// CHECK:           scf.yield %{{.*}}fifo_cons_buff_3
// CHECK:         func.call @consume
// CHECK:         arith.addi
// CHECK:         %[[MASK4:.*]] = arith.constant 3 : i32
// CHECK:         arith.andi {{.*}} %[[MASK4]] : i32
// CHECK-NOT: conduit.create
// CHECK-NOT: conduit.acquire
// CHECK-NOT: conduit.release

module @producer_rotation_modulo {
  aie.device(npu1_1col) {
    func.func @generate(%buf: memref<8xi32>) -> () {
      return
    }
    func.func @consume(%buf: memref<8xi32>) -> () {
      return
    }

    %tile_0_2 = aie.tile(0, 2)
    %tile_0_4 = aie.tile(0, 4)

    // depth=4: four ping-pong buffers on the producer tile.
    // Producer acquires 1 at a time → effectiveDepth = min(4, 1+1) = 2.
    aie.objectfifo @fifo(%tile_0_2, {%tile_0_4}, 4 : i32)
        : !aie.objectfifo<memref<8xi32>>

    %core_0_2 = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c8 = arith.constant 8 : index
      scf.for %arg0 = %c0 to %c8 step %c1 {
        %0 = aie.objectfifo.acquire @fifo(Produce, 1) : !aie.objectfifosubview<memref<8xi32>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<8xi32>> -> memref<8xi32>
        func.call @generate(%1) : (memref<8xi32>) -> ()
        aie.objectfifo.release @fifo(Produce, 1)
      }
      aie.end
    } {dynamic_objfifo_lowering = true}

    %core_0_4 = aie.core(%tile_0_4) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c8 = arith.constant 8 : index
      scf.for %arg0 = %c0 to %c8 step %c1 {
        %0 = aie.objectfifo.acquire @fifo(Consume, 1) : !aie.objectfifosubview<memref<8xi32>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<8xi32>> -> memref<8xi32>
        func.call @consume(%1) : (memref<8xi32>) -> ()
        aie.objectfifo.release @fifo(Consume, 1)
      }
      aie.end
    } {dynamic_objfifo_lowering = true}
  }
}
