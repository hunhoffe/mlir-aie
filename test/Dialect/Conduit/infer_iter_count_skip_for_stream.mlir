// RUN: aie-opt --objectfifo-to-conduit -split-input-file %s | FileCheck %s
//
// Task #29 — streaming-mode hardening: Pass A's dma_repeat inference must
// SKIP when the source aie.objectfifo is explicitly routed via Stream or
// Cascade.  Those routing modes emit no shim DMA BD chain, so a stamped
// dma_repeat is wrong-by-construction (Pass C currently ignores it on
// Stream channels via skip-branches, but we should not stamp it at all).
//
// Three split-input-file scenarios:
//   (1) explicit routing_mode = "stream"  → no dma_repeat
//   (2) via_cascade = true                → no dma_repeat
//   (3) default (no routing_mode override) + finite scf.for trip=8
//       → dma_repeat = 8 (positive control: skip is targeted, not blanket).

// (1) Explicit Stream routing — inference must skip even though the
//     consumer wraps its acquire in a static scf.for of trip=8.
// CHECK-LABEL: module @skip_for_explicit_stream
// CHECK: conduit.create @chan_stream
// CHECK-SAME: routing_mode = #conduit.routing_mode<stream>
// CHECK-NOT: dma_repeat
module @skip_for_explicit_stream {
  aie.device(npu1) {
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)

    aie.objectfifo @chan_stream(%tile_0_0, {%tile_0_2}, 2 : i32)
        {routing_mode = "stream"}
        : !aie.objectfifo<memref<128xbf16>>

    aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.for %i = %c0 to %c8 step %c1 {
        %sub = aie.objectfifo.acquire @chan_stream (Consume, 1)
            : !aie.objectfifosubview<memref<128xbf16>>
        %elem = aie.objectfifo.subview.access %sub[0]
            : !aie.objectfifosubview<memref<128xbf16>> -> memref<128xbf16>
        aie.objectfifo.release @chan_stream (Consume, 1)
      }
      aie.end
    }
  }
}

// -----

// (2) via_cascade=true — inference must skip.  Cascade requires depth=1
//     and SDF rate (1,1); the consumer scf.for of trip=8 satisfies that
//     per-iteration.  The producer/consumer cores are rewritten to
//     put_cascade/get_cascade in Phase 4, but inference (Phase 2) sees
//     the raw acquires/releases — without the routingSkipsDma guard it
//     would stamp dma_repeat = 8 on the cascade conduit.create.
// CHECK-LABEL: module @skip_for_via_cascade
// CHECK: conduit.create @chan_cascade
// CHECK-SAME: routing_mode = #conduit.routing_mode<cascade>
// CHECK-NOT: dma_repeat
module @skip_for_via_cascade {
  aie.device(npu1) {
    %tile03 = aie.tile(0, 3)
    %tile13 = aie.tile(1, 3)

    aie.objectfifo @chan_cascade(%tile03, {%tile13}, 1 : i32)
        {via_cascade = true}
        : !aie.objectfifo<memref<1xvector<16xi32>>>

    aie.core(%tile03) {
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.for %i = %c0 to %c8 step %c1 {
        %subview = aie.objectfifo.acquire @chan_cascade(Produce, 1)
            : !aie.objectfifosubview<memref<1xvector<16xi32>>>
        %elem0 = aie.objectfifo.subview.access %subview[0]
            : !aie.objectfifosubview<memref<1xvector<16xi32>>> -> memref<1xvector<16xi32>>
        %cz = arith.constant 0 : index
        %v = arith.constant dense<42> : vector<16xi32>
        memref.store %v, %elem0[%cz] : memref<1xvector<16xi32>>
        aie.objectfifo.release @chan_cascade(Produce, 1)
      }
      aie.end
    }

    aie.core(%tile13) {
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.for %i = %c0 to %c8 step %c1 {
        %subview = aie.objectfifo.acquire @chan_cascade(Consume, 1)
            : !aie.objectfifosubview<memref<1xvector<16xi32>>>
        %elem0 = aie.objectfifo.subview.access %subview[0]
            : !aie.objectfifosubview<memref<1xvector<16xi32>>> -> memref<1xvector<16xi32>>
        %cz = arith.constant 0 : index
        %r = memref.load %elem0[%cz] : memref<1xvector<16xi32>>
        vector.print %r : vector<16xi32>
        aie.objectfifo.release @chan_cascade(Consume, 1)
      }
      aie.end
    }
  }
}

// -----

// (3) Positive control: default-routed objectfifo (no routing_mode override,
//     no via_cascade, no aie_stream) with the SAME consumer scf.for of
//     trip=8 — inference fires and stamps dma_repeat = 8.  This pins that
//     the skip in (1) and (2) is targeted at Stream/Cascade specifically,
//     not a blanket regression.
// CHECK-LABEL: module @stamp_for_default_routing
// CHECK: conduit.create @chan_default
// CHECK-SAME: dma_repeat = 8
module @stamp_for_default_routing {
  aie.device(npu1) {
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)

    aie.objectfifo @chan_default(%tile_0_0, {%tile_0_2}, 2 : i32)
        : !aie.objectfifo<memref<128xbf16>>

    aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c8 = arith.constant 8 : index
      %c1 = arith.constant 1 : index
      scf.for %i = %c0 to %c8 step %c1 {
        %sub = aie.objectfifo.acquire @chan_default (Consume, 1)
            : !aie.objectfifosubview<memref<128xbf16>>
        %elem = aie.objectfifo.subview.access %sub[0]
            : !aie.objectfifosubview<memref<128xbf16>> -> memref<128xbf16>
        aie.objectfifo.release @chan_default (Consume, 1)
      }
      aie.end
    }
  }
}
