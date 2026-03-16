// RUN: aie-opt --conduit-check-liveness %s 2>&1 | FileCheck %s
//
// P2-B: Mixed-mode liveness check — pure cascade core (no DMA, no error).
//
// A core body that uses only cascade (aie.put_cascade) with no
// aie.use_lock calls.  The mixed-mode check is skipped entirely.
// No diagnostics should be emitted.
//
// CHECK-NOT: error
// CHECK-NOT: warning

module {
  aie.device(npu1) {
    %tile03 = aie.tile(0, 3)

    aie.core(%tile03) {
      %vec = arith.constant dense<1> : vector<16xi32>
      aie.put_cascade(%vec : vector<16xi32>)
      aie.end
    }
  }
}
