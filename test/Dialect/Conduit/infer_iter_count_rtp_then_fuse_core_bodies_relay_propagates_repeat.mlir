// RUN: aie-opt --objectfifo-to-conduit --conduit-fuse-core-bodies %s | FileCheck %s
//
// Pattern D (Task #16) × `--conduit-fuse-core-bodies` MemTile-relay
// cross-pass test.  PINS CURRENT BEHAVIOR (Pattern D × fuse-core-bodies
// interaction gap, NOT a Pattern D source bug).
//
// Variant of `infer_iter_count_then_fuse_core_bodies_relay_propagates_dma_repeat.mlir`:
// the producer + consumer outer scf.for trips come from the Pattern D
// RTP-constant fold (`arith.index_cast (memref.load %my_rtp[%c0])`)
// instead of an `arith.constant 16 : index`.
//
// CURRENT BEHAVIOR (today):
//   * Pass A successfully RTP-folds the trip → 16 and stamps
//     `dma_repeat = 16` on @input, @intermediate, and @output.
//   * `--conduit-fuse-core-bodies` does NOT fuse the two cores.
//     `ConduitFuseCoreBodyPass.cpp:493-518::hasMatchingLoopStructure`
//     requires the loop UB to be an `arith.ConstantIndexOp`; the
//     RTP-folded UB is `arith.index_cast (memref.load %my_rtp[%c0])`,
//     which fails the cast check → fusion skipped → no MemTile relay
//     emitted → no `@intermediate_relay` rename.
//
// TODO (Task #88, gating workstream): port `air::evaluateConstantsInMap`
// into `hasMatchingLoopStructure` so RTP-folded UBs are recognized as
// constant-equivalent.  When that lands, the negative directives below
// flip back to a positive match on `@intermediate_relay` (mirroring the
// literal-trip sibling), and the two-core count drops back to one
// surviving core.
//
// This test pins Pattern D's dma_repeat layer (verify dma_repeat=16 on
// @input/@intermediate/@output) but does NOT exercise the fuse-core-bodies
// relay path on RTP-folded trips today.  This is NOT an expected-failure
// (the test SHOULD pass as written); the gap it documents is in
// fuse-core-bodies, not in Pattern D source.

// CHECK-LABEL: module @infer_rtp_then_fuse_core_bodies_relay
//
// Pass A still stamps RTP-folded dma_repeat = 16 on every channel.
// CHECK-NOT:   @intermediate_relay
// CHECK:       conduit.create @input
// CHECK-SAME:  dma_repeat = 16
// CHECK-NOT:   @intermediate_relay
// CHECK:       conduit.create @intermediate
// CHECK-SAME:  dma_repeat = 16
// CHECK-NOT:   @intermediate_relay
// CHECK:       conduit.create @output
// CHECK-SAME:  dma_repeat = 16
// CHECK-NOT:   @intermediate_relay
//
// Both cores remain (fusion did NOT occur — RTP-folded UB fails the
// constant-UB check in hasMatchingLoopStructure).
// CHECK-COUNT-2: aie.core
// CHECK-NOT:   @intermediate_relay

module @infer_rtp_then_fuse_core_bodies_relay {
  aie.device(npu2) {
    %shim_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)

    %my_rtp = aie.buffer(%tile_0_2) {sym_name = "my_rtp", use_write_rtp = true} : memref<2xi32>

    aie.objectfifo @input(%shim_0, {%tile_0_2}, 2 : i32)
        : !aie.objectfifo<memref<131072xbf16>>
    // Intermediate is too large for L1 (256 KiB > 64 KiB npu2 L1) →
    // decideRoute() picks MemTile relay.
    aie.objectfifo @intermediate(%tile_0_2, {%tile_0_2}, 2 : i32)
        : !aie.objectfifo<memref<131072xbf16>>
    aie.objectfifo @output(%tile_0_2, {%shim_0}, 2 : i32)
        : !aie.objectfifo<memref<131072xbf16>>

    func.func private @produce_kernel(memref<131072xbf16>, memref<131072xbf16>)
    func.func private @consume_kernel(memref<131072xbf16>, memref<131072xbf16>)

    // Core A: producer.  Outer trip RTP-folded from %my_rtp[0].
    aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %i0 = arith.constant 0 : index
      %va = memref.load %my_rtp[%i0] : memref<2xi32>
      %ub_a = arith.index_cast %va : i32 to index
      scf.for %i = %c0 to %ub_a step %c1 {
        %in = aie.objectfifo.acquire @input(Consume, 1)
            : !aie.objectfifosubview<memref<131072xbf16>>
        %in_buf = aie.objectfifo.subview.access %in[0]
            : !aie.objectfifosubview<memref<131072xbf16>> -> memref<131072xbf16>
        %ix = aie.objectfifo.acquire @intermediate(Produce, 1)
            : !aie.objectfifosubview<memref<131072xbf16>>
        %ix_buf = aie.objectfifo.subview.access %ix[0]
            : !aie.objectfifosubview<memref<131072xbf16>> -> memref<131072xbf16>
        func.call @produce_kernel(%in_buf, %ix_buf)
            : (memref<131072xbf16>, memref<131072xbf16>) -> ()
        aie.objectfifo.release @intermediate(Produce, 1)
        aie.objectfifo.release @input(Consume, 1)
      }
      aie.end
    } {link_with = "producer.o"}

    // Core B: consumer (same tile).  Outer trip RTP-folded from same slot.
    aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %i0 = arith.constant 0 : index
      %vb = memref.load %my_rtp[%i0] : memref<2xi32>
      %ub_b = arith.index_cast %vb : i32 to index
      scf.for %i = %c0 to %ub_b step %c1 {
        %ix = aie.objectfifo.acquire @intermediate(Consume, 1)
            : !aie.objectfifosubview<memref<131072xbf16>>
        %ix_buf = aie.objectfifo.subview.access %ix[0]
            : !aie.objectfifosubview<memref<131072xbf16>> -> memref<131072xbf16>
        %out = aie.objectfifo.acquire @output(Produce, 1)
            : !aie.objectfifosubview<memref<131072xbf16>>
        %out_buf = aie.objectfifo.subview.access %out[0]
            : !aie.objectfifosubview<memref<131072xbf16>> -> memref<131072xbf16>
        func.call @consume_kernel(%ix_buf, %out_buf)
            : (memref<131072xbf16>, memref<131072xbf16>) -> ()
        aie.objectfifo.release @output(Produce, 1)
        aie.objectfifo.release @intermediate(Consume, 1)
      }
      aie.end
    } {link_with = "consumer.o"}

    // Host writes RTP slot 0 = 16, the trip Pattern D should fold.
    aie.runtime_sequence() {
      aiex.npu.rtp_write(@my_rtp, 0, 16)
    }
  }
}
