// RUN: aie-opt --conduit-canonicalize-channel-puts %s | FileCheck %s
//
// HISTORICAL: this fixture originally pinned ArithProgressionPattern's
// dim-cap refuse path — 2 puts already at 4-dim producer_dimensions
// (compute cap=3, MemTile/Shim cap=4), so prepending the outer wrap dim
// would push to 5 dims and trip AIEDialect.cpp:2233-2236 dma_bd verifier
// + AIEDMATasksToNPU.cpp:347-350 runtime-sequence cap.  Canon emitted
// "refusing to collapse 2 puts on @C_L2L3" via emitWarning; pinned with
// --verify-diagnostics + expected-warning.
//
// FLIPPED 2026-05-03 (canon refuse-to-collapse-on-await predicate, this
// commit): the 2 puts each carry `wait_all{token=true}` +
// `wait_all{token=false}` (chain shape `[true, false]`).  Per the new
// `chainHasAwait` predicate, ArithProgressionPattern now refuses
// EARLIER — before the dim-cap refuse fires — so the dim-cap warning no
// longer reaches the user for this fixture's chain shape.  The
// --verify-diagnostics + expected-warning directive is removed.
//
// Same root-cause class as the canon link-refusal landed in commit
// 375b0e5233; per CLAUDE.md USER-LOCKED 2026-04-28 "wrong is right"
// anti-pattern, the prior pinned shape was correct in OUTCOME (canon
// refused) but the REASON was the wrong gate — canon should have
// refused on chain-await first, not on dim-cap.  Empirical HW backing
// for the new gate:
// `test/npu-xrt/conduit_canon_no_collapse_on_puts_with_await/`.
//
// Dim-cap-refuse coverage on chain `[false]` only (the LEGITIMATE refuse
// case) is preserved by sibling pins / future cap-stress fixtures.

// CHECK-LABEL: aie.device(npu1)

// Channel must NOT gain an outer wrap dimension on producer_dimensions
// (canon refused — chain has token=true); the existing 4-dim attr is
// preserved verbatim.
// CHECK: conduit.create @C_L2L3
// CHECK-NOT: <size = 2, stride = 524288>

// Both puts survive on @C_L2L3 (canon left the IR alone).
// CHECK-COUNT-2: conduit.put_memref_async {{.*}}name = @C_L2L3

module @canon_arith_progression_dim_cap_refuse {
  aie.device(npu1) {
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)

    conduit.create @C_L2L3 {
      element_type = memref<64xbf16>,
      depth = 2 : i64
    }

    aie.shim_dma_allocation @C_L2L3_shim_alloc(%tile_0_0, MM2S, 0) {conduit_channel = @C_L2L3}

    %core_0_2 = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c2 = arith.constant 2 : index
      %c1 = arith.constant 1 : index
      scf.for %i = %c0 to %c2 step %c1 {
        %g = conduit.get_memref_async {name = @C_L2L3,
                  num_elems = 64 : i64,
                  offsets = array<i64: 0>,
                  sizes = array<i64: 64>,
                  strides = array<i64: 1>} : !conduit.dma.token
        conduit.wait_all %g : !conduit.dma.token
      }
      aie.end
    } {dynamic_objfifo_lowering = true}

    func.func @sequence(%arg0: memref<1048576xbf16>) {
      // 2 sliding-offset puts (delta = 524288), each already at 4-dim
      // producer_dimensions (3 padding + 1 data) — op7-shape input.
      %t0 = conduit.put_memref_async {name = @C_L2L3,
            num_elems = 64 : i64,
            offsets = array<i64: 0>,
            sizes = array<i64: 64>,
            strides = array<i64: 1>,
            producer_dimensions = #aie<bd_dim_layout_array[
              <size = 1, stride = 0>,
              <size = 1, stride = 0>,
              <size = 1, stride = 0>,
              <size = 64, stride = 1>]>} : !conduit.dma.token
      conduit.wait_all %t0 {token = true} : !conduit.dma.token
      conduit.wait_all %t0 {token = false} : !conduit.dma.token

      %t1 = conduit.put_memref_async {name = @C_L2L3,
            num_elems = 64 : i64,
            offsets = array<i64: 524288>,
            sizes = array<i64: 64>,
            strides = array<i64: 1>,
            producer_dimensions = #aie<bd_dim_layout_array[
              <size = 1, stride = 0>,
              <size = 1, stride = 0>,
              <size = 1, stride = 0>,
              <size = 64, stride = 1>]>} : !conduit.dma.token
      conduit.wait_all %t1 {token = true} : !conduit.dma.token
      conduit.wait_all %t1 {token = false} : !conduit.dma.token
      return
    }
  }
}
