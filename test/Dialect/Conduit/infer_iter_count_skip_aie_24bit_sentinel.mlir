// RUN: aie-opt --objectfifo-to-conduit %s | FileCheck %s
//
// Pass A — sentinel-detection threshold fix (2026-05-03).
//
// IRON's documented "loop forever" idiom emits an outer-loop bound of
// `0xFFFFFE = 16,777,214` (= 2^24 - 2, the AIE 24-bit BD-loop saturation
// register width).  Pass A's silent-skip predicate previously gated only on
// `kTripCountUnboundedSentinel = 1 << 30 = 1,073,741,824`, which is
// LARGER than 0xFFFFFE — so the IRON-shaped sentinel slipped past and Pass
// A folded `dma_repeat = 16777214` onto the channel.  The bogus value then
// silently propagated through canon + depth-promote.
//
// Empirical context: this latent gap was masked for an entire sprint by the
// canon-fix-implementer's HomogeneousRepeatPattern over-collapse (which
// over-fired the shim BD via a different mechanism — see commit 375b0e5233
// + the `conduit_canon_no_collapse_on_link` smoke).  When that fix landed,
// multi-invocation runs began wedging post-invocation-1; IR diff vs
// stateful surfaced the `0xFFFFFE` core-loop bound on the conduit side
// vs the actual finite NUM_INVOCATIONS bound on the stateful side.
//
// Pass A is now expected to silently skip dma_repeat stamping when the
// agreed trip count == kAie24bitBdLoopSentinel (0xFFFFFE), matching the
// pre-existing legacy `cmax = i64::MAX` skip behavior.
//
// Geometry (mirrors infer_iter_count_compute_to_compute_no_rt_seq.mlir):
//   compute-to-compute objectfifo on npu1, no aie.runtime_sequence
//   producer trip = 16777214 (0xFFFFFE), consumer trip = 16777214
//   With the fix in place: trip == kAie24bitBdLoopSentinel → silent skip
//   → conduit.create gets NO dma_repeat attribute.
//
// Without the fix, Pass A would stamp `dma_repeat = 16777214` here.

// CHECK-LABEL: module @skip_aie_24bit_sentinel
// CHECK: conduit.create @chan
// CHECK-NOT: dma_repeat
// CHECK: aie.core
// (CHECK-NOT scans from the conduit.create line up to the first aie.core;
//  no dma_repeat attribute may appear in that window.)

module @skip_aie_24bit_sentinel {
  aie.device(npu1) {
    %tile_0_2 = aie.tile(0, 2)
    %tile_0_3 = aie.tile(0, 3)

    aie.objectfifo @chan(%tile_0_2, {%tile_0_3}, 2 : i32)
        : !aie.objectfifo<memref<64xbf16>>

    // Producer core: outer loop bound = 0xFFFFFE (AIE 24-bit BD-loop
    // saturation sentinel; IRON's "loop forever" idiom).
    aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %sentinel = arith.constant 16777214 : index   // 0xFFFFFE = 2^24 - 2
      scf.for %i = %c0 to %sentinel step %c1 {
        %sub = aie.objectfifo.acquire @chan (Produce, 1)
            : !aie.objectfifosubview<memref<64xbf16>>
        %elem = aie.objectfifo.subview.access %sub[0]
            : !aie.objectfifosubview<memref<64xbf16>> -> memref<64xbf16>
        aie.objectfifo.release @chan (Produce, 1)
      }
      aie.end
    }

    // Consumer core: same sentinel bound (must match producer to avoid the
    // mismatch-skip path so the sentinel-skip path is what's exercised).
    aie.core(%tile_0_3) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %sentinel = arith.constant 16777214 : index   // 0xFFFFFE = 2^24 - 2
      scf.for %i = %c0 to %sentinel step %c1 {
        %sub = aie.objectfifo.acquire @chan (Consume, 1)
            : !aie.objectfifosubview<memref<64xbf16>>
        %elem = aie.objectfifo.subview.access %sub[0]
            : !aie.objectfifosubview<memref<64xbf16>> -> memref<64xbf16>
        aie.objectfifo.release @chan (Consume, 1)
      }
      aie.end
    }
    // NOTE: NO aie.runtime_sequence — keeps the geometry compute-to-compute,
    // so no shim-skip pre-empts the sentinel-skip we want to exercise.
  }
}
