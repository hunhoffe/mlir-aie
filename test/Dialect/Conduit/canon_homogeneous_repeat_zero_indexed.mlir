// RUN: aie-opt --conduit-canonicalize-channel-puts %s | FileCheck %s
//
// HISTORICAL: this fixture originally pinned the 0-INDEXED dma_repeat
// stamp (Bug #98 / Task #39) — 4 IRON puts collapsed to 1 surviving put +
// `dma_repeat = 3` (= 4 total fires).  Bug #98's source-side fix is still
// live; the collapse-stamp coverage is preserved by other regression
// paths (HomogeneousRepeatPattern unit + Pass C's verbatim surface).
//
// FLIPPED 2026-05-03 (canon refuse-to-collapse-on-await predicate, this
// commit): the 4 puts here each carry a `wait_all{token=true}` plus a
// `wait_all{token=false}` (chain shape `[true, false]`).  Per the new
// `chainHasAwait` predicate in `CanonicalizeChannelPutsUtils.{h,cpp}`,
// canon now REFUSES to collapse channels whose IR carries any per-issue
// ack request, because the consolidated `1 configure × dma_repeat=N-1`
// form starves the per-chunk consumer-side ack and stalls HW
// (root-cause class shared with the canon link-refusal landed in
// commit 375b0e5233).  Per CLAUDE.md USER-LOCKED 2026-04-28 "wrong is
// right" anti-pattern: this fixture's prior collapse-asserting CHECKs
// encoded behavior that is HW-broken; flipping is correct.  Empirical
// HW backing: `test/npu-xrt/conduit_canon_no_collapse_on_puts_with_await/`
// + `test/npu-xrt/conduit_canon_no_collapse_on_gets_with_await/`.
//
// Geometry (unchanged): shim(0,0) producer → compute(0,2) consumer,
//           depth=2, memref<16xi32>, 4 host dispatches with
//           per-issue-await chain → canon refuses; all 4 puts survive.

// CHECK-LABEL: aie.device(npu1)

// Channel must NOT carry dma_repeat (canon refused — chain has token=true).
// CHECK: conduit.create @chan
// CHECK-NOT: dma_repeat

// All 4 puts survive on @chan (canon left the IR alone).
// CHECK-COUNT-4: conduit.put_memref_async {{.*}}name = @chan

module @canon_homogeneous_repeat_zero_indexed {
  aie.device(npu1) {
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)

    conduit.create @chan {
      element_type = memref<16xi32>,
      depth = 2 : i64
    }

    aie.shim_dma_allocation @chan_shim_alloc(%tile_0_0, MM2S, 0) {conduit_channel = @chan}

    %core_0_2 = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c4 = arith.constant 4 : index
      %c1 = arith.constant 1 : index
      scf.for %i = %c0 to %c4 step %c1 {
        %g = conduit.get_memref_async {name = @chan,
                  num_elems = 16 : i64,
                  offsets = array<i64: 0>,
                  sizes = array<i64: 16>,
                  strides = array<i64: 1>} : !conduit.dma.token
        conduit.wait_all %g : !conduit.dma.token
      }
      aie.end
    } {dynamic_objfifo_lowering = true}

    func.func @sequence(%arg0: memref<16xi32>) {
      // 4 IRON-pattern identical puts.  Canon collapses to:
      //   1 surviving put + dma_repeat = 3 (= 4 total fires).
      %t0 = conduit.put_memref_async {name = @chan, num_elems = 16 : i64,
            offsets = array<i64: 0>, sizes = array<i64: 16>,
            strides = array<i64: 1>} : !conduit.dma.token
      conduit.wait_all %t0 {token = true} : !conduit.dma.token
      conduit.wait_all %t0 {token = false} : !conduit.dma.token

      %t1 = conduit.put_memref_async {name = @chan, num_elems = 16 : i64,
            offsets = array<i64: 0>, sizes = array<i64: 16>,
            strides = array<i64: 1>} : !conduit.dma.token
      conduit.wait_all %t1 {token = true} : !conduit.dma.token
      conduit.wait_all %t1 {token = false} : !conduit.dma.token

      %t2 = conduit.put_memref_async {name = @chan, num_elems = 16 : i64,
            offsets = array<i64: 0>, sizes = array<i64: 16>,
            strides = array<i64: 1>} : !conduit.dma.token
      conduit.wait_all %t2 {token = true} : !conduit.dma.token
      conduit.wait_all %t2 {token = false} : !conduit.dma.token

      %t3 = conduit.put_memref_async {name = @chan, num_elems = 16 : i64,
            offsets = array<i64: 0>, sizes = array<i64: 16>,
            strides = array<i64: 1>} : !conduit.dma.token
      conduit.wait_all %t3 {token = true} : !conduit.dma.token
      conduit.wait_all %t3 {token = false} : !conduit.dma.token
      return
    }
  }
}
