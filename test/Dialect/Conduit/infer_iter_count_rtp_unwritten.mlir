// RUN: aie-opt --objectfifo-to-conduit %s -verify-diagnostics 2>&1 | FileCheck %s
//
// Pattern D (Task #16) — negative case: the consumer core loads from an
// RTP buffer that was DECLARED but never written by the runtime_sequence
// (no `aiex.npu.rtp_write` op anywhere in the device).
//
// `tryFoldRtpBound` looks up `(@my_rtp, 0)` in `rtpConstantMap`, misses
// (the map is empty for this device), returns nullopt, and the inference
// falls through to the existing dynamic-loop skip.
//
// This is the conservative default: an RTP slot whose host-side value is
// invisible to the compiler MUST NOT be treated as if it were a known
// constant.  Pre-Pattern-D this case was already Dynamic-skipped because
// `getStaticTripCount` returned nullopt; Pattern D preserves the same
// behavior on miss instead of degrading it.

// CHECK-LABEL: module @infer_rtp_unwritten
// CHECK: conduit.create @chan
// CHECK-NOT: dma_repeat
// CHECK-NEXT: aie.core

module @infer_rtp_unwritten {
  aie.device(npu1) {
    %tile_0_2 = aie.tile(0, 2)
    %tile_0_3 = aie.tile(0, 3)

    %my_rtp = aie.buffer(%tile_0_3) {sym_name = "my_rtp", use_write_rtp = true} : memref<2xi32>

    // expected-remark@+1 {{conduit-objectfifo: dma_repeat inference skipped: dynamic loop bounds in producer or consumer core}}
    aie.objectfifo @chan(%tile_0_2, {%tile_0_3}, 2 : i32)
        : !aie.objectfifo<memref<8xbf16>>

    // Producer core: literal trip = 8.
    aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c8 = arith.constant 8 : index
      scf.for %i = %c0 to %c8 step %c1 {
        %s = aie.objectfifo.acquire @chan (Produce, 1)
            : !aie.objectfifosubview<memref<8xbf16>>
        %b = aie.objectfifo.subview.access %s[0]
            : !aie.objectfifosubview<memref<8xbf16>> -> memref<8xbf16>
        aie.objectfifo.release @chan (Produce, 1)
      }
      aie.end
    }

    // Consumer core: RTP-load upper bound; fold misses because no
    // `aiex.npu.rtp_write` exists for any slot of `@my_rtp`.
    aie.core(%tile_0_3) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %i0 = arith.constant 0 : index
      %v = memref.load %my_rtp[%i0] : memref<2xi32>
      %ub = arith.index_cast %v : i32 to index
      scf.for %k = %c0 to %ub step %c1 {
        %s = aie.objectfifo.acquire @chan (Consume, 1)
            : !aie.objectfifosubview<memref<8xbf16>>
        %b = aie.objectfifo.subview.access %s[0]
            : !aie.objectfifosubview<memref<8xbf16>> -> memref<8xbf16>
        aie.objectfifo.release @chan (Consume, 1)
      }
      aie.end
    }

    // Empty runtime_sequence — no rtp_write at all.
    aie.runtime_sequence() {
    }
  }
}
