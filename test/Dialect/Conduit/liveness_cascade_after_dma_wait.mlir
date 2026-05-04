// RUN: aie-opt --conduit-check-liveness %s 2>&1 | FileCheck %s
//
// P2-B: Mixed-mode liveness check — correct ordering (no error).
//
// The aie.use_lock(Acquire, 1) dominates the aie.put_cascade in the same
// block.  The cascade value is loaded from a buffer (DMA-dependent).
// No error should be emitted.
//
// CHECK-NOT: error
// CHECK-NOT: put_cascade may fire before DMA transfer completes

module {
  aie.device(npu1) {
    %tile03 = aie.tile(0, 3)
    %tile13 = aie.tile(1, 3)
    %buf = aie.buffer(%tile03) : memref<16xi32>
    %lock = aie.lock(%tile03, 0) { init = 0 : i32 }

    aie.core(%tile03) {
      // DMA wait: lock acquisition signals data is ready in %buf.
      aie.use_lock(%lock, Acquire, 1)
      // Load from DMA-filled buffer, then send on cascade.
      %c0 = arith.constant 0 : index
      %v = memref.load %buf[%c0] : memref<16xi32>
      %vec = vector.broadcast %v : i32 to vector<16xi32>
      // put_cascade dominates AFTER the use_lock — correct ordering.
      aie.put_cascade(%vec : vector<16xi32>)
      aie.use_lock(%lock, Release, 0)
      aie.end
    }
  }
}
