// RUN: aie-opt --objectfifo-to-conduit --conduit-to-dma %s | FileCheck %s
//
// Pass A + Pass C end-to-end test: BLOCK-3 fix — two depth-2 shim→compute
// conduits on the SAME compute tile share one rotation counter buffer.
//
// Background:
//   When multiple conduits target the same compute tile, Pass C must allocate
//   a single shared rotation counter buffer (memref<Nxi32>) on that tile, with
//   non-overlapping slot indices (0 and 1 for two conduits).  Before the BLOCK-3
//   fix, each conduit allocated its own memref<1xi32> buffer, wasting buffer
//   resources and potentially causing allocation conflicts.
//
// Topology: Two objectfifos, fifo_a and fifo_b, both from shim tile(0,0) to
//   compute tile(0,2), both depth=2.  The core consumes from both.
//
// Key assertions:
//   - Exactly ONE aie.buffer "_conduit_rot_ctr_tile_0_2" of type memref<2xi32>.
//   - No per-conduit memref<1xi32> rotation counter buffers on tile_0_2.
//   - fifo_a uses slot 0 (index %c0), fifo_b uses slot 1 (index %c1).
//   - Both conduits wrap their counter at modulus 2 (depth=2).

// CHECK-LABEL: module @shared_rotation_buf
// CHECK:   aie.device(npu1_1col) {

// --- Data buffers for both conduits on tile_0_2 ---
// CHECK:     aie.buffer(%{{.*}}tile_0_2) {sym_name = "fifo_b_cons_buff_0"} : memref<8xi32>
// CHECK:     aie.buffer(%{{.*}}tile_0_2) {sym_name = "fifo_b_cons_buff_1"} : memref<8xi32>
// CHECK:     aie.buffer(%{{.*}}tile_0_2) {sym_name = "fifo_a_cons_buff_0"} : memref<8xi32>
// CHECK:     aie.buffer(%{{.*}}tile_0_2) {sym_name = "fifo_a_cons_buff_1"} : memref<8xi32>

// --- ONE shared rotation counter allocated inside core body (memref<2xi32>) ---
// CHECK:     aie.core(%{{.*}}tile_0_2) {
// CHECK:       %[[ALLOCA:.*]] = memref.alloca() : memref<2xi32>
// CHECK-NOT:   memref.alloca() : memref<1xi32>

// --- Core init: slot 1 for fifo_b, slot 0 for fifo_a (or reverse) ---
// CHECK:       memref.store %c0_i32{{.*}}, %[[ALLOCA]][%c{{[01]}}{{.*}}] : memref<2xi32>
// CHECK:       memref.store %c0_i32{{.*}}, %[[ALLOCA]][%c{{[01]}}{{.*}}] : memref<2xi32>

// --- fifo_a acquire: loads from slot 0 ---
// CHECK:       aie.use_lock(%{{.*}}fifo_a_cons_cons_lock_0, AcquireGreaterEqual, 1)
// CHECK:       %c0{{.*}} = arith.constant 0 : index
// CHECK:       memref.load %[[ALLOCA]][%c0{{.*}}] : memref<2xi32>
// CHECK:       scf.index_switch
// CHECK:         scf.yield %{{.*}}fifo_a_cons_buff_0
// CHECK:         scf.yield %{{.*}}fifo_a_cons_buff_1
// CHECK:       func.call @process_a
// CHECK:       aie.use_lock(%{{.*}}fifo_a_cons_prod_lock_0, Release, 1)
// CHECK:       arith.remui {{.*}} %c2_i32{{.*}} : i32
// CHECK:       memref.store {{.*}} %[[ALLOCA]][%c0{{.*}}] : memref<2xi32>

// --- fifo_b acquire: loads from slot 1 ---
// CHECK:       aie.use_lock(%{{.*}}fifo_b_cons_cons_lock_0, AcquireGreaterEqual, 1)
// CHECK:       %c1{{.*}} = arith.constant 1 : index
// CHECK:       memref.load %[[ALLOCA]][%c1{{.*}}] : memref<2xi32>
// CHECK:       scf.index_switch
// CHECK:         scf.yield %{{.*}}fifo_b_cons_buff_0
// CHECK:         scf.yield %{{.*}}fifo_b_cons_buff_1
// CHECK:       func.call @process_b
// CHECK:       aie.use_lock(%{{.*}}fifo_b_cons_prod_lock_0, Release, 1)
// CHECK:       arith.remui {{.*}} %c2_i32{{.*}} : i32
// CHECK:       memref.store {{.*}} %[[ALLOCA]][%c1{{.*}}] : memref<2xi32>
// CHECK-NOT: conduit.create
// CHECK-NOT: conduit.acquire
// CHECK-NOT: conduit.release

module @shared_rotation_buf {
  aie.device(npu1_1col) {
    func.func @process_a(%buf: memref<8xi32>) -> () {
      return
    }
    func.func @process_b(%buf: memref<8xi32>) -> () {
      return
    }

    %tile_0_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)

    // Two shim→compute conduits targeting the same compute tile.
    aie.objectfifo @fifo_a(%tile_0_0, {%tile_0_2}, 2 : i32)
        : !aie.objectfifo<memref<8xi32>>
    aie.objectfifo @fifo_b(%tile_0_0, {%tile_0_2}, 2 : i32)
        : !aie.objectfifo<memref<8xi32>>

    %core_0_2 = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c4 = arith.constant 4 : index
      scf.for %arg0 = %c0 to %c4 step %c1 {
        %0 = aie.objectfifo.acquire @fifo_a(Consume, 1) : !aie.objectfifosubview<memref<8xi32>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<8xi32>> -> memref<8xi32>
        func.call @process_a(%1) : (memref<8xi32>) -> ()
        aie.objectfifo.release @fifo_a(Consume, 1)

        %2 = aie.objectfifo.acquire @fifo_b(Consume, 1) : !aie.objectfifosubview<memref<8xi32>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<8xi32>> -> memref<8xi32>
        func.call @process_b(%3) : (memref<8xi32>) -> ()
        aie.objectfifo.release @fifo_b(Consume, 1)
      }
      aie.end
    } {dynamic_objfifo_lowering = true}
  }
}
