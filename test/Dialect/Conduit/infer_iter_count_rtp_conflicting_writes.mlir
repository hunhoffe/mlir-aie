// RUN: aie-opt --objectfifo-to-conduit %s -verify-diagnostics 2>&1 | FileCheck %s
//
// Pattern D (Task #16) — negative case: two `aiex.npu.rtp_write` ops in
// the same runtime_sequence target the SAME (buffer_sym, index) key with
// DIFFERENT constant values (8 vs 16).
//
// `collectRtpConstants` flips the entry's `ambiguous` flag; the fold
// helper then refuses the lookup, the consumer scf.for upper bound stays
// non-static, and the inference path falls through to the existing
// "dynamic loop bounds in producer or consumer core" skip.
//
// Pre-Pattern-D: same outcome (no fold ever attempted, Dynamic skip).
// Post-Pattern-D: ambiguous-write guard preserves the conservative skip
// rather than silently picking the last-written value.

// CHECK-LABEL: module @infer_rtp_conflicting_writes
// CHECK: conduit.create @chan
// CHECK-NOT: dma_repeat
// CHECK-NEXT: aie.core

module @infer_rtp_conflicting_writes {
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

    // Consumer core: RTP-load upper bound; fold MUST miss because the
    // runtime_sequence writes (@my_rtp, 0) twice with conflicting values.
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

    // Two writes to (@my_rtp, 0) with different values → ambiguous fold.
    aie.runtime_sequence() {
      aiex.npu.rtp_write(@my_rtp, 0, 8)
      aiex.npu.rtp_write(@my_rtp, 0, 16)
    }
  }
}
