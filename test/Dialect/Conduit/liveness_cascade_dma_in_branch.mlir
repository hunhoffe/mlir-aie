// RUN: aie-opt --conduit-check-liveness --verify-diagnostics %s
//
// P2-B: Mixed-mode liveness check — DMA wait in one branch only (error).
//
// The use_lock(Acquire) is inside a conditional branch.  The put_cascade is
// in the merge block (after the conditional), where the lock acquisition may
// not have executed.  The lock does NOT dominate the put_cascade.
// An error should be emitted.

module {
  aie.device(npu1) {
    %tile03 = aie.tile(0, 3)
    %buf = aie.buffer(%tile03) : memref<16xi32>
    %lock = aie.lock(%tile03, 0) { init = 0 : i32 }

    aie.core(%tile03) {
      %c0 = arith.constant 0 : index
      %cond = arith.constant 1 : i1

      // DMA wait is inside the 'then' branch — does NOT dominate code after.
      scf.if %cond {
        aie.use_lock(%lock, Acquire, 1)
        aie.use_lock(%lock, Release, 0)
      }

      // Load from DMA buffer and send on cascade.
      // Lock may or may not have been acquired — ordering not guaranteed.
      %v = memref.load %buf[%c0] : memref<16xi32>
      %vec = vector.broadcast %v : i32 to vector<16xi32>
      // expected-error@+1 {{put_cascade may fire before DMA transfer completes}}
      aie.put_cascade(%vec : vector<16xi32>)
      aie.end
    }
  }
}
