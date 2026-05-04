// RUN: aie-opt --conduit-canonicalize-channel-puts %s | FileCheck %s
// RUN: aie-opt --conduit-canonicalize-channel-puts --conduit-depth-promote --conduit-to-dma --aie-substitute-shim-dma-allocations --aie-assign-runtime-sequence-bd-ids %s
//
// HISTORICAL: this fixture originally pinned the basic arith-progression
// collapse — 4 sliding-offset puts → 1 surviving put + outer wrap+stride
// dim <size=4, stride=8> on producer_dimensions.
//
// FLIPPED 2026-05-03 (canon refuse-to-collapse-on-await predicate, this
// commit): the 4 puts each carry `wait_all{token=true}` +
// `wait_all{token=false}` (chain shape `[true, false]`).  Per the new
// `chainHasAwait` predicate, ArithProgressionPattern now REFUSES to
// collapse such chains because the consolidated single-configure form
// (with outer wrap+stride dim encoding the per-cycle variation) starves
// the per-chunk consumer-side ack and stalls HW.  Same root-cause class
// as the canon link-refusal landed in commit 375b0e5233; per CLAUDE.md
// USER-LOCKED 2026-04-28 "wrong is right" anti-pattern, the prior
// collapse-asserting CHECKs encoded HW-broken behavior.  Empirical HW
// backing: `test/npu-xrt/conduit_canon_no_collapse_on_puts_with_await/`.
//
// Geometry (unchanged): shim(0,0) producer → compute(0,2) consumer,
//           depth=2, memref<8xi32> per put, 4 host dispatches at offsets
//           [0,8,16,24] with per-issue-await chain → canon refuses.
//           Second RUN line is the metafix-aiecc-smoke convention.

// CHECK-LABEL: aie.device(npu1)

// Channel must NOT carry the canon-introduced outer wrap dim
// (canon refused — chain has token=true).
// CHECK: conduit.create @chan
// CHECK-NOT: producer_dimensions

// All 4 puts survive on @chan at original offsets (canon left the IR alone).
// CHECK-COUNT-4: conduit.put_memref_async {{.*}}name = @chan

module @arith_progression_1d_collapse {
  aie.device(npu1) {
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)

    conduit.create @chan {
      element_type = memref<8xi32>,
      depth = 2 : i64
    }

    aie.shim_dma_allocation @chan_shim_alloc(%tile_0_0, MM2S, 0) {conduit_channel = @chan}

    %core_0_2 = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c4 = arith.constant 4 : index
      %c1 = arith.constant 1 : index
      scf.for %i = %c0 to %c4 step %c1 {
        %g = conduit.get_memref_async {name = @chan,
                  num_elems = 8 : i64,
                  offsets = array<i64: 0>,
                  sizes = array<i64: 8>,
                  strides = array<i64: 1>} : !conduit.dma.token
        conduit.wait_all %g : !conduit.dma.token
      }
      aie.end
    } {dynamic_objfifo_lowering = true}

    func.func @sequence(%arg0: memref<32xi32>) {
      // 4 sliding-offset puts, each with await + free chain — IRON's
      // host-Python-unroll pattern with per-batch base-offset increment.
      %t0 = conduit.put_memref_async {name = @chan, num_elems = 8 : i64,
            offsets = array<i64: 0>, sizes = array<i64: 8>,
            strides = array<i64: 1>} : !conduit.dma.token
      conduit.wait_all %t0 {token = true} : !conduit.dma.token
      conduit.wait_all %t0 {token = false} : !conduit.dma.token

      %t1 = conduit.put_memref_async {name = @chan, num_elems = 8 : i64,
            offsets = array<i64: 8>, sizes = array<i64: 8>,
            strides = array<i64: 1>} : !conduit.dma.token
      conduit.wait_all %t1 {token = true} : !conduit.dma.token
      conduit.wait_all %t1 {token = false} : !conduit.dma.token

      %t2 = conduit.put_memref_async {name = @chan, num_elems = 8 : i64,
            offsets = array<i64: 16>, sizes = array<i64: 8>,
            strides = array<i64: 1>} : !conduit.dma.token
      conduit.wait_all %t2 {token = true} : !conduit.dma.token
      conduit.wait_all %t2 {token = false} : !conduit.dma.token

      %t3 = conduit.put_memref_async {name = @chan, num_elems = 8 : i64,
            offsets = array<i64: 24>, sizes = array<i64: 8>,
            strides = array<i64: 1>} : !conduit.dma.token
      conduit.wait_all %t3 {token = true} : !conduit.dma.token
      conduit.wait_all %t3 {token = false} : !conduit.dma.token
      return
    }
  }
}
