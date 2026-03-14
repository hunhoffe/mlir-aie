// RUN: aie-opt --objectfifo-to-conduit --conduit-to-dma %s | FileCheck %s
//
// P1-H regression test: rotation counter must use arith.remui (true modulo).
//
// Background (Fix 1b):
//   For depth>1 conduits, Pass C emits a rotation counter that cycles through
//   buffer slots 0, 1, ..., depth-1 on each acquire/release iteration.
//   The counter update is:
//
//     new_counter = (old_counter + 1) % depth
//
//   Before Fix 1b, this was implemented as a conditional subtract:
//
//     new_counter = old_counter + 1
//     if (new_counter >= depth)  new_counter = new_counter - depth
//
//   This is incorrect when the CSDF acquire count can exceed depth across
//   multiple iterations.  The conditional subtract only handles the case where
//   old_counter+1 == depth; if old_counter+1 > depth (which happens when the
//   counter is advanced multiple times per iteration), the subtract produces
//   a value > 0 that is still >= depth, corrupting the buffer index.
//
//   Fix 1b replaced the conditional-subtract with arith.remui, which handles
//   all values correctly:
//
//     %new = arith.addi %old, %c1 : i32
//     %mod = arith.remui %new, %depth : i32
//
// This test verifies that arith.remui appears in the rotation counter update
// logic for a depth-2 consumer.  If someone reverts to the conditional-subtract
// pattern, this test will fail.
//
// Topology: depth-2 shim-to-compute (shim tile [0,0] → compute tile [0,2]).
// Target: npu1_1col (AIE2).

// CHECK-LABEL: module @rotation_modulo_test
// CHECK:   aie.device(npu1_1col) {

// --- Rotation counter buffer (memref<1xi32>) ---
// CHECK:     aie.buffer(%{{.*}}tile_0_2) : memref<1xi32>

// --- Core body: counter init, loop, and rotation counter update ---
// CHECK:     aie.core(%{{.*}}tile_0_2) {
// --- Counter initialized to 0 ---
// CHECK:       memref.store {{.*}} : memref<1xi32>
// CHECK:       scf.for
// --- Counter loaded, used for index_switch, then incremented with remui ---
// CHECK:         memref.load {{.*}} : memref<1xi32>
// CHECK:         scf.index_switch
// --- Fix 1b: rotation counter update uses arith.remui (NOT conditional subtract) ---
// CHECK:         memref.load {{.*}} : memref<1xi32>
// CHECK:         arith.addi
// CHECK:         arith.remui
// CHECK:         memref.store {{.*}} : memref<1xi32>

// --- No residual Conduit ops ---
// CHECK-NOT: conduit.create
// CHECK-NOT: conduit.acquire
// CHECK-NOT: conduit.release

module @rotation_modulo_test {
  aie.device(npu1_1col) {
    func.func @compute(%buf: memref<16xi32>) -> () {
      return
    }

    %tile_0_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)
    // depth=2: triggers rotation counter in consumer core
    aie.objectfifo @rot_fifo(%tile_0_0, {%tile_0_2}, 2 : i32) : !aie.objectfifo<memref<16xi32>>

    %core_0_2 = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c8 = arith.constant 8 : index

      scf.for %arg0 = %c0 to %c8 step %c1 {
        %0 = aie.objectfifo.acquire @rot_fifo(Consume, 1) : !aie.objectfifosubview<memref<16xi32>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>
        func.call @compute(%1) : (memref<16xi32>) -> ()
        aie.objectfifo.release @rot_fifo(Consume, 1)
      }

      aie.end
    } {dynamic_objfifo_lowering = true}
  }
}
