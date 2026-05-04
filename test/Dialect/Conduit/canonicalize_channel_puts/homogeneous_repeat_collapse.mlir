// RUN: aie-opt --conduit-canonicalize-channel-puts %s | FileCheck %s
//
// HISTORICAL: this fixture originally pinned the basic collapse —
// 4 structurally-identical IRON puts → 1 surviving put + dma_repeat = 3.
//
// FLIPPED 2026-05-03 (canon refuse-to-collapse-on-await predicate, this
// commit): the 4 puts each carry `wait_all{token=true}` (await) +
// `wait_all{token=false}` (free) → chain shape `[true, false]`.  Per the
// new `chainHasAwait` predicate, canon REFUSES to collapse such chains
// because the consolidated `1 configure × dma_repeat=N-1` form starves
// the per-chunk consumer-side ack and stalls HW.  Same root-cause class
// as the canon link-refusal landed in commit 375b0e5233; per CLAUDE.md
// USER-LOCKED 2026-04-28 "wrong is right" anti-pattern, the prior
// collapse-asserting CHECKs encoded HW-broken behavior.  Empirical HW
// backing: `test/npu-xrt/conduit_canon_no_collapse_on_puts_with_await/`
// + `test/npu-xrt/conduit_canon_no_collapse_on_gets_with_await/`.
//
// To restore lit coverage of the collapse-stamp itself for the
// chain-without-await shape (the LEGITIMATE collapse case — IRON's
// MM2S puts that emit only `dma_free_task` without `dma_await_task`),
// see `conduit_to_dma_b_channel_consolidation.mlir`'s history (b-channel
// path emits chain `[false]` only and exercises the linked-refusal
// branch instead).
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

module @conduit_canonicalize_loop_unroll_puts_collapse {
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
      // 4 IRON-pattern identical puts, each with await + free.
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
