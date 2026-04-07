// RUN: not aie-opt --conduit-to-dma %s 2>&1 | FileCheck %s
//
// Regression test: B-11 — conduit.acquire placed outside aie.core but inside
// aie.device is handled gracefully by resolveForTile() (no crash/assertion).
//
// The B-11 fix adds a DeviceOp sentinel in resolveForTile(): when the parent
// walk reaches aie.device without finding an aie.core, it returns coreOp=null
// instead of continuing past DeviceOp to ModuleOp.
//
// In this test the conduit.acquire/release are in the device body (not inside
// any aie.core), so resolveForTile() returns coreOp=null. The locks are still
// found (from conduit.create info), so aie.use_lock is still emitted in the
// wrong context, which the AIE verifier catches with a clear error — but the
// pass does NOT crash with an assertion failure or segfault.
//
// The critical property being tested: aie-opt exits with a diagnostic error
// message rather than an uncaught assertion.
//
// CHECK: aie.use_lock{{.*}}must be used in a core or memory operation

module @passC_conduit_outside_core {
  aie.device(npu1_1col) {
    %shim = aie.tile(0, 0)
    %tile = aie.tile(0, 2)

    conduit.create @myChan {slot_elems = 32 : i64, depth = 1 : i64,
                    element_type = memref<32xi32>,
                    producer_tile = array<i64: 0, 0>,
                    consumer_tiles = array<i64: 0, 2>,
                    shim_consumer_tiles = array<i64>}

    // conduit.acquire/release placed OUTSIDE aie.core — in the device body.
    // B-11: resolveForTile() should NOT walk past DeviceOp (sentinel stop).
    // The pass should produce a clear verifier diagnostic, NOT crash.
    %win = conduit.acquire {name = @myChan, count = 1 : i64,
                             port = #conduit.port<Consume>}
               : !conduit.window<memref<32xi32>>
    conduit.release %win {count = 1 : i64, port = #conduit.port<Consume>}
        : !conduit.window<memref<32xi32>>

    %core = aie.core(%tile) {
      aie.end
    }
  }
}
