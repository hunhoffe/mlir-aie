// RUN: aie-opt --conduit-canonicalize-channel-puts %s | FileCheck %s
//
// Direct lit pin for the canon refuse-to-collapse-on-await predicate
// (`chainHasAwait` in CanonicalizeChannelPutsUtils.{h,cpp}, landed
// 2026-05-03 in this commit).  Distinct from the 9 flipped sibling
// pins (which document the predicate as a side-effect of their original
// collapse-asserting CHECKs being inverted) by being the AT-SITE pin
// for the predicate itself.
//
// Geometry: shim(0,0) MM2S → compute(0,2) consumer.  Two channels:
//
//   (a) @chan_no_await — 4 IRON-pattern identical puts whose chain
//       shape is `[false]` only (one `wait_all{token=false}` per put,
//       NO `wait_all{token=true}`).  This is the LEGITIMATE collapse
//       case — chain has no per-issue ack request, so canon is free to
//       fold N puts → 1 + dma_repeat=N-1 (0-indexed convention,
//       Bug #98).  Pinned to verify the predicate doesn't over-refuse.
//
//   (b) @chan_with_await — 4 IRON-pattern identical puts whose chain
//       shape is `[true, false]` (per-issue ack + free).  Per the new
//       predicate, canon REFUSES to collapse — the consolidated
//       1-configure × dma_repeat=N-1 form starves the per-chunk
//       consumer-side ack and stalls HW (root-cause class shared with
//       the canon link-refusal landed in commit 375b0e5233; empirical
//       HW backing: `test/npu-xrt/conduit_canon_no_collapse_on_puts_with_await/`).
//
// The two channels share the same enclosing device so a single canon
// invocation exercises BOTH branches in one IR.  Each channel uses its
// own shim_dma_allocation + own consumer scf.for to keep them
// structurally independent.

// CHECK-LABEL: aie.device(npu1)

// Post-canon IR layout: both `conduit.create` ops precede `func.func @sequence`
// (creates live in the device body; puts live in the runtime sequence).
// FileCheck scans forward only — anchor the create-side CHECKs first, then
// the func.func body, then the surviving put counts.

// (a) Legitimate collapse on @chan_no_await: chain shape `[false]` only — canon
// collapses 4 → 1 + dma_repeat=3 (0-indexed; Bug #98).  dma_repeat lands on the
// `conduit.create` op (NOT on the put), per Pass C's emit shape.
// CHECK: conduit.create @chan_no_await
// CHECK-SAME: dma_repeat = 3

// (b) Refused collapse on @chan_with_await: chain shape `[true, false]` — canon
// refuses via chainHasAwait.  No dma_repeat ever stamped on the create.
// CHECK: conduit.create @chan_with_await
// CHECK-NOT: dma_repeat

// Runtime sequence: 1 surviving put on @chan_no_await (3 of 4 collapsed) + all
// 4 puts on @chan_with_await preserved (canon left them alone).
// CHECK: func.func @sequence
// CHECK-COUNT-1: conduit.put_memref_async {{.*}}name = @chan_no_await
// CHECK-COUNT-4: conduit.put_memref_async {{.*}}name = @chan_with_await

module @conduit_canon_no_collapse_on_token_true {
  aie.device(npu1) {
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)
    %tile_0_3 = aie.tile(0, 3)

    // (a) Channel with NO per-issue ack request — collapse-eligible.
    conduit.create @chan_no_await {
      element_type = memref<8xi32>,
      depth = 2 : i64
    }
    aie.shim_dma_allocation @chan_no_await_shim_alloc(%tile_0_0, MM2S, 0)
        {conduit_channel = @chan_no_await}

    // (b) Channel with per-issue ack request — collapse-refused.
    conduit.create @chan_with_await {
      element_type = memref<8xi32>,
      depth = 2 : i64
    }
    aie.shim_dma_allocation @chan_with_await_shim_alloc(%tile_0_0, MM2S, 1)
        {conduit_channel = @chan_with_await}

    // Consumer for @chan_no_await — single scf.for over 4 iterations.
    %core_0_2 = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c4 = arith.constant 4 : index
      %c1 = arith.constant 1 : index
      scf.for %i = %c0 to %c4 step %c1 {
        %g = conduit.get_memref_async {name = @chan_no_await,
                  num_elems = 8 : i64,
                  offsets = array<i64: 0>,
                  sizes = array<i64: 8>,
                  strides = array<i64: 1>} : !conduit.dma.token
        conduit.wait_all %g : !conduit.dma.token
      }
      aie.end
    } {dynamic_objfifo_lowering = true}

    // Consumer for @chan_with_await.
    %core_0_3 = aie.core(%tile_0_3) {
      %c0 = arith.constant 0 : index
      %c4 = arith.constant 4 : index
      %c1 = arith.constant 1 : index
      scf.for %i = %c0 to %c4 step %c1 {
        %g = conduit.get_memref_async {name = @chan_with_await,
                  num_elems = 8 : i64,
                  offsets = array<i64: 0>,
                  sizes = array<i64: 8>,
                  strides = array<i64: 1>} : !conduit.dma.token
        conduit.wait_all %g : !conduit.dma.token
      }
      aie.end
    } {dynamic_objfifo_lowering = true}

    func.func @sequence(%arg0: memref<32xi32>) {
      // (a) 4 puts on @chan_no_await — chain shape `[false]` only.
      %a0 = conduit.put_memref_async {name = @chan_no_await, num_elems = 8 : i64,
            offsets = array<i64: 0>, sizes = array<i64: 8>,
            strides = array<i64: 1>} : !conduit.dma.token
      conduit.wait_all %a0 {token = false} : !conduit.dma.token
      %a1 = conduit.put_memref_async {name = @chan_no_await, num_elems = 8 : i64,
            offsets = array<i64: 0>, sizes = array<i64: 8>,
            strides = array<i64: 1>} : !conduit.dma.token
      conduit.wait_all %a1 {token = false} : !conduit.dma.token
      %a2 = conduit.put_memref_async {name = @chan_no_await, num_elems = 8 : i64,
            offsets = array<i64: 0>, sizes = array<i64: 8>,
            strides = array<i64: 1>} : !conduit.dma.token
      conduit.wait_all %a2 {token = false} : !conduit.dma.token
      %a3 = conduit.put_memref_async {name = @chan_no_await, num_elems = 8 : i64,
            offsets = array<i64: 0>, sizes = array<i64: 8>,
            strides = array<i64: 1>} : !conduit.dma.token
      conduit.wait_all %a3 {token = false} : !conduit.dma.token

      // (b) 4 puts on @chan_with_await — chain shape `[true, false]`.
      %b0 = conduit.put_memref_async {name = @chan_with_await, num_elems = 8 : i64,
            offsets = array<i64: 0>, sizes = array<i64: 8>,
            strides = array<i64: 1>} : !conduit.dma.token
      conduit.wait_all %b0 {token = true} : !conduit.dma.token
      conduit.wait_all %b0 {token = false} : !conduit.dma.token
      %b1 = conduit.put_memref_async {name = @chan_with_await, num_elems = 8 : i64,
            offsets = array<i64: 0>, sizes = array<i64: 8>,
            strides = array<i64: 1>} : !conduit.dma.token
      conduit.wait_all %b1 {token = true} : !conduit.dma.token
      conduit.wait_all %b1 {token = false} : !conduit.dma.token
      %b2 = conduit.put_memref_async {name = @chan_with_await, num_elems = 8 : i64,
            offsets = array<i64: 0>, sizes = array<i64: 8>,
            strides = array<i64: 1>} : !conduit.dma.token
      conduit.wait_all %b2 {token = true} : !conduit.dma.token
      conduit.wait_all %b2 {token = false} : !conduit.dma.token
      %b3 = conduit.put_memref_async {name = @chan_with_await, num_elems = 8 : i64,
            offsets = array<i64: 0>, sizes = array<i64: 8>,
            strides = array<i64: 1>} : !conduit.dma.token
      conduit.wait_all %b3 {token = true} : !conduit.dma.token
      conduit.wait_all %b3 {token = false} : !conduit.dma.token
      return
    }
  }
}
