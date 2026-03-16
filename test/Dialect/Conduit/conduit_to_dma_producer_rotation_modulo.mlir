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
//   The producer counter must therefore use arith.remui with divisor 2
//   (not 4), while the consumer counter (which cycles over all depth slots
//   to match the DMA BD ring) must use arith.remui with divisor 4.
//
//   Before the BLOCK-1 fix, the producer used divisor 4 (depth), leading
//   to a mismatch between the core's buffer selection and the DMA BD ring.
//
// Topology: compute producer tile(0,2) → compute consumer tile(0,4), depth=4.
//   Producer acquires 1 at a time → effectiveDepth=2 → producer remui uses %c2.
//   Consumer acquires 1 at a time → effectiveDepth=min(4,1+1)=2 → consumer
//   remui also uses %c2 in this topology (consumer side), but the DMA BD ring
//   has 4 entries.
//
// Key assertions:
//   - Producer-side: 2 buffers on tile_0_2 (effectiveDepth=2), shared counter
//     buffer memref<2xi32> (slot 1 for producer, slot 0 stays for any consumer).
//   - Producer core remui divisor = 2 (effectiveDepth), NOT 4 (depth).
//   - Consumer-side: 4 buffers on tile_0_4, counter memref<1xi32>, remui divisor = 4.
//   - CHECK-NOT: %c4_i32 in the producer core remui (regression guard).

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

// --- Rotation counter: consumer tile gets memref<1xi32>, producer tile gets memref<2xi32> ---
// (tile_0_2 buffer has 2 slots: slot 0 unused here, slot 1 = producer counter)
// CHECK:     aie.buffer(%{{.*}}tile_0_4) {sym_name = "_conduit_rot_ctr_tile_0_4"} : memref<1xi32>
// CHECK:     aie.buffer(%{{.*}}tile_0_2) {sym_name = "_conduit_rot_ctr_tile_0_2"} : memref<2xi32>

// --- Producer core: counter uses remui with divisor 2 (effectiveDepth), NOT 4 (depth) ---
// CHECK:     aie.core(%{{.*}}tile_0_2) {
// CHECK:       memref.store %c0_i32, %{{.*}}_conduit_rot_ctr_tile_0_2[%c1{{.*}}] : memref<2xi32>
// CHECK:       scf.for
// CHECK:         aie.use_lock(%{{.*}}fifo_prod_lock_0, AcquireGreaterEqual, 1)
// CHECK:         memref.load %{{.*}}_conduit_rot_ctr_tile_0_2[%c1{{.*}}] : memref<2xi32>
// CHECK:         scf.index_switch
// CHECK:           scf.yield %{{.*}}fifo_buff_0
// CHECK:           scf.yield %{{.*}}fifo_buff_1
// CHECK:         func.call @generate
// CHECK:         aie.use_lock(%{{.*}}fifo_cons_lock_0, Release, 1)
// CHECK:         memref.load %{{.*}}_conduit_rot_ctr_tile_0_2[%c1{{.*}}] : memref<2xi32>
// CHECK:         arith.addi
// CHECK:         %c2_i32 = arith.constant 2 : i32
// CHECK:         arith.remui {{.*}} %c2_i32 : i32
// CHECK:         memref.store {{.*}} %{{.*}}_conduit_rot_ctr_tile_0_2[%c1{{.*}}] : memref<2xi32>

// --- Consumer core: counter uses remui with divisor 4 (full depth) ---
// CHECK:     aie.core(%{{.*}}tile_0_4) {
// CHECK:       scf.for
// CHECK:         aie.use_lock(%{{.*}}fifo_cons_cons_lock_0, AcquireGreaterEqual, 1)
// CHECK:         scf.index_switch
// CHECK:           scf.yield %{{.*}}fifo_cons_buff_0
// CHECK:           scf.yield %{{.*}}fifo_cons_buff_1
// CHECK:           scf.yield %{{.*}}fifo_cons_buff_2
// CHECK:           scf.yield %{{.*}}fifo_cons_buff_3
// CHECK:         arith.addi
// CHECK:         %c4_i32 = arith.constant 4 : i32
// CHECK:         arith.remui {{.*}} %c4_i32 : i32
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
