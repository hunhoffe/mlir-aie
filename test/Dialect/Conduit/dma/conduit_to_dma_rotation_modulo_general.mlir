// RUN: aie-opt --objectfifo-to-conduit --conduit-to-dma %s | FileCheck %s
//
// Rotation counter fast-modulo test — GENERAL (non-power-of-2) depth path.
//
// For power-of-2 depths (2, 4, 8, ...) the counter update uses:
//   arith.andi(counter + delta, depth - 1)
//
// For non-power-of-2 depths (3, 5, 6, ...) there is no single-AND fast path.
// The bound counter + delta < 2*depth still holds (counter < depth, delta <= depth),
// so one branchless conditional subtract is sufficient — no arith.remui needed:
//
//   %sum  = arith.addi %counter, %delta
//   %cond = arith.cmpi uge, %sum, %depth
//   %sub  = arith.subi %sum, %depth
//   %new  = arith.select %cond, %sub, %sum
//
// This test verifies the general path for depth=3 (non-power-of-2).
// Regression guard: arith.remui must NOT appear.

// CHECK-LABEL: module @rotation_modulo_general
// CHECK: aie.device(npu1_1col) {
// CHECK:   aie.core(
// CHECK:     scf.for
// --- General-depth counter update: cmpi uge + subi + select (NOT andi, NOT remui) ---
// CHECK:         arith.addi
// CHECK:         %[[DEPTH:.*]] = arith.constant 3 : i32
// CHECK:         arith.cmpi uge, {{.*}}, %[[DEPTH]] : i32
// CHECK:         arith.subi
// CHECK:         arith.select
// CHECK-NOT:     arith.remui
// CHECK-NOT:     arith.andi {{.*}} %[[DEPTH]]

module @rotation_modulo_general {
  aie.device(npu1_1col) {
    func.func @compute(%buf: memref<10xi32>) -> () {
      return
    }

    %tile_0_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)

    // depth=3: non-power-of-2, exercises the general cmpi+subi+select path.
    aie.objectfifo @fifo3(%tile_0_0, {%tile_0_2}, 3 : i32)
        : !aie.objectfifo<memref<10xi32>>

    %core_0_2 = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c9 = arith.constant 9 : index

      scf.for %arg0 = %c0 to %c9 step %c1 {
        %0 = aie.objectfifo.acquire @fifo3(Consume, 1)
            : !aie.objectfifosubview<memref<10xi32>>
        %1 = aie.objectfifo.subview.access %0[0]
            : !aie.objectfifosubview<memref<10xi32>> -> memref<10xi32>
        func.call @compute(%1) : (memref<10xi32>) -> ()
        aie.objectfifo.release @fifo3(Consume, 1)
      }

      aie.end
    } {dynamic_objfifo_lowering = true}
  }
}
