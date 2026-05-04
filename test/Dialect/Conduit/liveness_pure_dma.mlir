// RUN: aie-opt --conduit-check-liveness %s 2>&1 | FileCheck %s
//
// P2-B: Mixed-mode liveness check — pure DMA core (no cascade, no error).
//
// A core body that uses only DMA (aie.use_lock + memref.load/store) with
// no aie.put_cascade calls.  The mixed-mode check is skipped entirely.
// No diagnostics should be emitted.
//
// CHECK-NOT: error
// CHECK-NOT: warning

module {
  aie.device(npu1) {
    %tile03 = aie.tile(0, 3)
    %tile13 = aie.tile(1, 3)
    %buf = aie.buffer(%tile03) : memref<16xi32>
    %lock = aie.lock(%tile03, 0) { init = 0 : i32 }

    aie.core(%tile03) {
      aie.use_lock(%lock, Acquire, 1)
      %c0 = arith.constant 0 : index
      %v = memref.load %buf[%c0] : memref<16xi32>
      // Process and store result — pure DMA path, no cascade.
      %v2 = arith.addi %v, %v : i32
      memref.store %v2, %buf[%c0] : memref<16xi32>
      aie.use_lock(%lock, Release, 0)
      aie.end
    }
  }
}
