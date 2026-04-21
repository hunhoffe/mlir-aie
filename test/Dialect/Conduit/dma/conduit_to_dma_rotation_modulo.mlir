// RUN: aie-opt --objectfifo-to-conduit --conduit-to-dma %s | FileCheck %s
//
// Rotation counter update test: depth-2 counter uses arith.andi (power-of-2
// fast path), not arith.remui (software divide, no hardware divide on AIE2).
//
// Background:
//   For depth>1 conduits, Pass C emits a rotation counter that cycles through
//   buffer slots 0, 1, ..., depth-1 on each acquire/release iteration.
//   The counter update is:
//
//     new_counter = (old_counter + count) % depth
//
//   Since counter ∈ [0, depth-1] and count ≤ depth, the sum is < 2*depth, so
//   one subtract (or AND for power-of-2) always suffices:
//
//     Power-of-2 depth:   new = arith.andi(old + count, depth - 1)
//     General depth:      new = (old + count) >= depth
//                               ? (old + count) - depth : (old + count)
//
//   arith.remui is avoided because AIE2 has no hardware divide instruction.
//
// This test verifies the power-of-2 fast path (arith.andi) for depth=2.
//
// Topology: depth-2 shim-to-compute (shim tile [0,0] → compute tile [0,2]).
// Target: npu1_1col (AIE2).
// Note: the rotation counter is allocated as aie.buffer on the tile
// (device-level buffer, no sym_name).

// CHECK-LABEL: module @rotation_modulo_test
// CHECK:   aie.device(npu1_1col) {

// --- Rotation counter allocated as aie.buffer on tile ---
// CHECK:     aie.buffer(%{{.*}}tile_0_2) : memref<1xi32>
// CHECK:     aie.core(%{{.*}}tile_0_2) {
// --- Counter initialized to 0 at top of core body ---
// CHECK:         memref.store {{.*}} : memref<1xi32>
// CHECK:       scf.for
// --- Counter loaded, used for scf.if chain buffer selection, then incremented ---
// CHECK:         memref.load {{.*}} : memref<1xi32>
// CHECK:         arith.index_cast
// CHECK:         arith.cmpi eq
// CHECK:         scf.if
// --- Rotation counter update: arith.andi for power-of-2 depth (NOT arith.remui) ---
// CHECK:         memref.load {{.*}} : memref<1xi32>
// CHECK:         arith.addi
// CHECK:         arith.andi
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
