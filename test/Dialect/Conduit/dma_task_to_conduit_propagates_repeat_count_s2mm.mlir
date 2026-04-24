// RUN: aie-opt --objectfifo-to-conduit --dma-task-to-conduit %s | FileCheck %s
//
// =============================================================================
// REGRESSION (S2MM symmetric): --dma-task-to-conduit propagates IRON's
// `aiex.dma_configure_task_for {repeat_count = N : i32}` attribute onto
// the matching `conduit.create`'s `dma_repeat = N` attribute (verbatim)
// for the S2MM (consume / get_memref) direction as well as MM2S.
// =============================================================================
//
// Companion to `dma_task_to_conduit_propagates_repeat_count.mlir` (MM2S
// direction).  The canonical home for the channel-level replay count is the
// `conduit.create` op, which is direction-agnostic — so the same surfacing
// logic must fire for S2MM channels (where the lowering produces
// `conduit.get_memref` instead of `conduit.put_memref`).
//
// IRON's GEMM output channel for `issue_token=true` paths emits
// `repeat_count = 1` on the S2MM configure_task (= 2 total fires).  The
// fix surfaces this onto the conduit.create's `dma_repeat = 1` (verbatim).

// CHECK-LABEL: module @dma_task_to_conduit_propagates_repeat_count_s2mm
module @dma_task_to_conduit_propagates_repeat_count_s2mm {
  aie.device(npu2) {
    %shim_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)

    // Compute → shim output channel: tile produces, shim consumes (S2MM).
    aie.objectfifo @C_L2L3_0(%tile_0_2, {%shim_0}, 2 : i32)
        : !aie.objectfifo<memref<4096xbf16>>

    func.func private @gemm_kernel(memref<4096xbf16>)

    %core = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %cmax = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %i = %c0 to %cmax step %c1 {
        %out = aie.objectfifo.acquire @C_L2L3_0(Produce, 1)
            : !aie.objectfifosubview<memref<4096xbf16>>
        %out_buf = aie.objectfifo.subview.access %out[0]
            : !aie.objectfifosubview<memref<4096xbf16>> -> memref<4096xbf16>
        func.call @gemm_kernel(%out_buf) : (memref<4096xbf16>) -> ()
        aie.objectfifo.release @C_L2L3_0(Produce, 1)
      }
      aie.end
    } {link_with = "gemm.c"}

    // Runtime sequence: S2MM configure_task with IRON-emitted
    // {repeat_count = 1 : i32} (= 2 total fires).
    aie.runtime_sequence(%arg0: memref<8192xbf16>) {
      %t0 = aiex.dma_configure_task_for @C_L2L3_0 {
        aie.dma_bd(%arg0 : memref<8192xbf16>, 0, 4096,
          [<size = 1, stride = 0>,
           <size = 1, stride = 0>,
           <size = 64, stride = 64>,
           <size = 64, stride = 1>])
            {burst_length = 0 : i32}
        aie.end
      } {issue_token = true, repeat_count = 1 : i32}
      aiex.dma_start_task(%t0)
      aiex.dma_await_task(%t0)
      aiex.dma_free_task(%t0)
    }
  }
}

// -----------------------------------------------------------------------------
// EXPECTED behavior (post-fix): conduit.create stamped with dma_repeat = 1.
// The lowered conduit.get_memref carries no repeat-related field (the
// canonical home is the create).
// -----------------------------------------------------------------------------

// CHECK:       conduit.create @C_L2L3_0
// CHECK-SAME:  dma_repeat = 1

// CHECK:       conduit.get_memref
// CHECK-SAME:  name = @C_L2L3_0
// CHECK-NOT:   dma_repeat
// CHECK-NOT:   repeat_count
