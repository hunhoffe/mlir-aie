// RUN: aie-opt --conduit-check-liveness %s 2>&1 | FileCheck %s
//
// P2-B: Mixed-mode liveness check — cascade uses independent data (no error).
//
// The core has a DMA consumer lock but the cascade value comes from a
// constant (not a memref.load), so the put_cascade is NOT DMA-dependent.
// No error should be emitted even though the put_cascade precedes the
// use_lock.
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
      // Cascade value is a constant — independent of DMA data.
      %vec = arith.constant dense<99> : vector<16xi32>
      // put_cascade is before use_lock, but the value is DMA-independent.
      aie.put_cascade(%vec : vector<16xi32>)
      // DMA wait for the local buffer (unrelated to the cascade send).
      aie.use_lock(%lock, Acquire, 1)
      aie.use_lock(%lock, Release, 0)
      aie.end
    }
  }
}
