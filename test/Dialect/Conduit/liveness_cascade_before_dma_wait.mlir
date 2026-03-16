// RUN: aie-opt --conduit-check-liveness --verify-diagnostics %s
//
// P2-B: Mixed-mode liveness check — incorrect ordering (error).
//
// The aie.put_cascade fires before aie.use_lock(Acquire, 1) in program
// order.  The cascade value comes from a memref.load (DMA-dependent).
// The lock acquisition does NOT dominate the put_cascade.
// An error should be emitted on the put_cascade.

module {
  aie.device(npu1) {
    %tile03 = aie.tile(0, 3)
    %buf = aie.buffer(%tile03) : memref<16xi32>
    %lock = aie.lock(%tile03, 0) { init = 0 : i32 }

    aie.core(%tile03) {
      // Load from buffer BEFORE waiting for DMA to complete — stale data.
      %c0 = arith.constant 0 : index
      %v = memref.load %buf[%c0] : memref<16xi32>
      %vec = vector.broadcast %v : i32 to vector<16xi32>
      // expected-error@+1 {{put_cascade may fire before DMA transfer completes}}
      aie.put_cascade(%vec : vector<16xi32>)
      // DMA wait comes AFTER put_cascade — too late.
      aie.use_lock(%lock, Acquire, 1)
      aie.use_lock(%lock, Release, 0)
      aie.end
    }
  }
}
