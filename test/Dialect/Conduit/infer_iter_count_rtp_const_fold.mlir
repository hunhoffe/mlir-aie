// RUN: aie-opt --objectfifo-to-conduit %s | FileCheck %s
//
// Pattern D (Task #16) — positive RTP-constant fold for Pass A
// dma_repeat inference.
//
// IR shape:
//   * Compute-to-compute @chan (no shim BD) — emit.count = 0 so the
//     three-factor formula degrades to dma_repeat = trip.
//   * Producer core trip = 8 (literal scf.for upper bound).
//   * Consumer core trip = 8 (RTP-folded: scf.for upper bound is
//     `arith.index_cast (memref.load %my_rtp[%c0])`, and the
//     runtime_sequence writes `aiex.npu.rtp_write(@my_rtp, 0, 8)`).
//   * Both sides agree on trip = 8 → Pass A stamps dma_repeat = 8.
//
// Pre-Pattern-D Pass A returned TripStatus::Dynamic for the consumer side
// (memref.load is not a constant) and skipped with the "dynamic loop
// bounds in producer or consumer core" remark.  Post-fix, the lookup
// succeeds and the formula proceeds.

// CHECK-LABEL: module @infer_rtp_const_fold
// CHECK: conduit.create @chan
// CHECK-SAME: dma_repeat = 8

module @infer_rtp_const_fold {
  aie.device(npu1) {
    %tile_0_2 = aie.tile(0, 2)
    %tile_0_3 = aie.tile(0, 3)

    %my_rtp = aie.buffer(%tile_0_3) {sym_name = "my_rtp", use_write_rtp = true} : memref<2xi32>
    %my_barrier = aie.lock(%tile_0_3) {sym_name = "my_barrier"}

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

    // Consumer core: RTP-folded trip = 8.
    aie.core(%tile_0_3) {
      aie.use_lock(%my_barrier, Acquire, 1)
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

    aie.runtime_sequence() {
      aiex.npu.rtp_write(@my_rtp, 0, 8)
      aiex.set_lock(%my_barrier, 1)
    }
  }
}
