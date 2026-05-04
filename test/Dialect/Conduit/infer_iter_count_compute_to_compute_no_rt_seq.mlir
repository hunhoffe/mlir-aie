// RUN: aie-opt --objectfifo-to-conduit %s | FileCheck %s
//
// Task #32 — Pass A dma_repeat inference on a compute-to-compute objectfifo
// with NO aiex.runtime_sequence at all.
//
// inspectChannelEmissions returns count=0, bdLen=nullopt for this geometry.
// The helper then takes:
//   emissions = (count > 0) ? count : 1   // → 1
//   acquiresPerBD = 1                     // bdLen unset → default
// so dma_repeat = trip / 1 / 1 = trip.
//
// Both producer and consumer cores wrap the acquire in scf.for trip=12 (they
// must agree to avoid the mismatch-skip path).
//
// Geometry:
//   producer trip = 12, consumer trip = 12 → agreed trip = 12
//   no rt seq → emissions = 1, acquires_per_BD = 1
//   → dma_repeat = 12

// CHECK-LABEL: module @infer_compute_to_compute_no_rt_seq
// CHECK: conduit.create @chan
// CHECK-SAME: dma_repeat = 12

module @infer_compute_to_compute_no_rt_seq {
  aie.device(npu1) {
    %tile_0_2 = aie.tile(0, 2)
    %tile_0_3 = aie.tile(0, 3)

    aie.objectfifo @chan(%tile_0_2, {%tile_0_3}, 2 : i32)
        : !aie.objectfifo<memref<64xbf16>>

    // Producer core: trip = 12.
    aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c12 = arith.constant 12 : index
      scf.for %i = %c0 to %c12 step %c1 {
        %sub = aie.objectfifo.acquire @chan (Produce, 1)
            : !aie.objectfifosubview<memref<64xbf16>>
        %elem = aie.objectfifo.subview.access %sub[0]
            : !aie.objectfifosubview<memref<64xbf16>> -> memref<64xbf16>
        aie.objectfifo.release @chan (Produce, 1)
      }
      aie.end
    }

    // Consumer core: trip = 12 (must match producer to avoid mismatch skip).
    aie.core(%tile_0_3) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c12 = arith.constant 12 : index
      scf.for %i = %c0 to %c12 step %c1 {
        %sub = aie.objectfifo.acquire @chan (Consume, 1)
            : !aie.objectfifosubview<memref<64xbf16>>
        %elem = aie.objectfifo.subview.access %sub[0]
            : !aie.objectfifosubview<memref<64xbf16>> -> memref<64xbf16>
        aie.objectfifo.release @chan (Consume, 1)
      }
      aie.end
    }
    // NOTE: NO aie.runtime_sequence at all — pinning the
    // emissions=0 → emissions=1 fallback path.
  }
}
