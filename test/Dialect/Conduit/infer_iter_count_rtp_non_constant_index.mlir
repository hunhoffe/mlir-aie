// RUN: aie-opt --objectfifo-to-conduit %s -verify-diagnostics 2>&1 | FileCheck %s
//
// Pattern D (Task #16) — negative case: the RTP `memref.load` index is a
// runtime-computed value (`arith.muli %c1, %c1`), not a constant.
//
// `tryFoldRtpBound` requires `getConstIndexOrInt(loadOp.getIndices()[0])`
// to succeed; on failure it returns nullopt and the inference falls
// through to the existing dynamic-loop skip path.
//
// This guards against accidentally folding bounds whose RTP slot is
// computed at core-runtime (e.g. selecting a slot per-tile via worker
// id) — the host write would not have set THAT specific slot in general.

// CHECK-LABEL: module @infer_rtp_non_constant_index
// CHECK: conduit.create @chan
// CHECK-NOT: dma_repeat
// CHECK-NEXT: aie.core

module @infer_rtp_non_constant_index {
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

    // Consumer: RTP load index is %dyn_idx = muli(c1, c1) — non-constant
    // SSA value as far as `getConstIndexOrInt` can tell.  Fold MUST miss.
    aie.core(%tile_0_3) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %dyn_idx = arith.muli %c1, %c1 : index
      %v = memref.load %my_rtp[%dyn_idx] : memref<2xi32>
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

    // Both slots written so the negative result is purely about the
    // load-index shape, not absence of host writes.
    aie.runtime_sequence() {
      aiex.npu.rtp_write(@my_rtp, 0, 8)
      aiex.npu.rtp_write(@my_rtp, 1, 8)
    }
  }
}
