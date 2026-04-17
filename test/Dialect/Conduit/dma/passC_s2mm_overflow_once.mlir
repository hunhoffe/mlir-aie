// RUN: aie-opt --objectfifo-to-conduit --conduit-to-dma --verify-diagnostics %s
//
// Regression test: B-3 — passFailed+continue → passFailed+return.
//
// Phase 4b: when a compute tile's objectfifo sends to a shim consumer and the
// shim tile's S2MM channels are exhausted, the old code did:
//   state.passFailed = true; continue;
// This continued the outer conduit loop, processing the next conduit which
// would emit a SECOND "S2MM DMA channel exhausted" error. With --verify-
// diagnostics, that unexpected second error causes the test to FAIL — correctly
// catching the regression. After B-3 fix (continue → return), the phase exits
// after the first error; conduit of4 is never processed.
//
// Setup (xcve2302): compute tile(2,3) → shim tile(2,0).
// Shim tiles have max 2 S2MM channels (getNumDestShimMuxConnections).
//   of1: claims shim S2MM channel 0  ok
//   of2: claims shim S2MM channel 1  ok
//   of3: needs channel 2 → OVERFLOW (expected-error)
//   of4: would need channel 3 → second error IF loop continued (unexpected)
//
// With `return` (correct): one error, test passes.
// With `continue` (buggy): two errors, unexpected second error seen by
//   --verify-diagnostics → test fails.
//
// NOTE: The aie.core block must use the objectfifos so that structural tile
// inference (inferAllTiles) can determine the producer tile.

module @passC_s2mm_overflow_once {
  // expected-error @below {{S2MM DMA channel exhausted on shim tile (2,0)}}
  aie.device(xcve2302) {
    %shim = aie.tile(2, 0)
    %comp = aie.tile(2, 3)

    // of1 and of2 succeed: claim shim S2MM channels 0 and 1.
    aie.objectfifo @of1(%comp, {%shim}, 2 : i32) : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo @of2(%comp, {%shim}, 2 : i32) : !aie.objectfifo<memref<16xi32>>

    // of3 triggers shim S2MM overflow (needs channel 2, max is 2).
    aie.objectfifo @of3(%comp, {%shim}, 2 : i32) : !aie.objectfifo<memref<16xi32>>

    // of4: with continue (buggy) this also fires an overflow error (unexpected).
    // With return (correct) this is never processed — no second error.
    aie.objectfifo @of4(%comp, {%shim}, 2 : i32) : !aie.objectfifo<memref<16xi32>>

    %core = aie.core(%comp) {
      %sv1 = aie.objectfifo.acquire @of1 (Produce, 1) : !aie.objectfifosubview<memref<16xi32>>
      aie.objectfifo.release @of1 (Produce, 1)
      %sv2 = aie.objectfifo.acquire @of2 (Produce, 1) : !aie.objectfifosubview<memref<16xi32>>
      aie.objectfifo.release @of2 (Produce, 1)
      %sv3 = aie.objectfifo.acquire @of3 (Produce, 1) : !aie.objectfifosubview<memref<16xi32>>
      aie.objectfifo.release @of3 (Produce, 1)
      %sv4 = aie.objectfifo.acquire @of4 (Produce, 1) : !aie.objectfifosubview<memref<16xi32>>
      aie.objectfifo.release @of4 (Produce, 1)
      aie.end
    }
  }
}
