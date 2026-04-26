// RUN: aie-opt --objectfifo-to-conduit --dma-task-to-conduit %s | FileCheck %s
//
// =============================================================================
// REGRESSION: --dma-task-to-conduit propagates IRON's
// `aiex.dma_configure_task_for {repeat_count = N : i32}` attribute onto
// the matching `conduit.create`'s `dma_repeat = N` attribute (verbatim).
// =============================================================================
//
// Convention (locked end-to-end by this fix):
//   * IRON emits `repeat_count = N` meaning "N additional firings"
//     (firmware fires the BD N+1 total times — see AIEDmaToNpu.cpp:180-183
//     where repeat_cnt is packed verbatim into the NPU command word).
//   * `--dma-task-to-conduit` copies this VERBATIM onto
//     `conduit.create.dma_repeat = N` (no +1 / -1 transformation).
//   * Pass C (`--conduit-to-dma`) re-emits `dma_repeat` VERBATIM as
//     `aiex.dma_configure_task_for.repeat_count = N` (see
//     ConduitToDMALower.cpp:1245-1248).
//   * End-to-end, the same N flows from IRON source through Pass C output
//     unchanged, and the firmware reads it as N+1 fires.
//
// History: this test originated as `dma_task_to_conduit_drops_repeat_count_BUG.mlir`
//   (committed 2b949033e9), pinning the WRONG behavior where
//   `--dma-task-to-conduit` silently dropped the `repeat_count` attribute.
//   The IRON GEMM operator's per-col N-tile broadcast pattern fired the BD
//   ONCE instead of N times — kernels stalled after 1/N of the work and
//   TIMEd OUT (#79: Llama 3.2 1B prefill `attn_query` GEMM).  This file was
//   renamed and the CHECK pin flipped when the fix landed.
//
//   Source:  lib/Dialect/Conduit/Transforms/ConduitDmaTaskToConduit.cpp
//            (the IRON repeat_count → conduit.create.dma_repeat surfacing
//            block, after PutMemref/GetMemref creation).
//
// Trigger geometry: IRON's GEMM operator emits a per-col N-tile broadcast as
//   aie.dma_bd(... [<size = R, stride = 0>, ...]) PLUS a sibling
//   {repeat_count = R-1 : i32} on the configure_task.  The shim DMA must fire
//   the BD R times per dispatch (once per N-tile).
//
// Pass A's emit.count==1 skip (commit dc792ebbd5) means this channel's
// `dma_repeat` is correctly absent from the create on the Pass A side; the
// IRON value is therefore the ONLY source for the stamp here.

// CHECK-LABEL: module @dma_task_to_conduit_propagates_repeat_count
module @dma_task_to_conduit_propagates_repeat_count {
  aie.device(npu2) {
    %shim_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)

    // External A-input from LPDDR5: per-col N-tile broadcast pattern
    // (4 BD firings per dispatch).  Mirrors the A_L3L2 shim channel of
    // IRON GEMM at the failing shape.
    aie.objectfifo @A_L3L2_0(%shim_0, {%tile_0_2}, 2 : i32)
        : !aie.objectfifo<memref<4096xbf16>>

    func.func private @gemm_kernel(memref<4096xbf16>)

    %core = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %cmax = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %i = %c0 to %cmax step %c1 {
        %in = aie.objectfifo.acquire @A_L3L2_0(Consume, 1)
            : !aie.objectfifosubview<memref<4096xbf16>>
        %in_buf = aie.objectfifo.subview.access %in[0]
            : !aie.objectfifosubview<memref<4096xbf16>> -> memref<4096xbf16>
        func.call @gemm_kernel(%in_buf) : (memref<4096xbf16>) -> ()
        aie.objectfifo.release @A_L3L2_0(Consume, 1)
      }
      aie.end
    } {link_with = "gemm.a"}

    // Runtime sequence: IRON-shaped MM2S configure_task with the per-col
    // N-tile broadcast.
    //
    //   {repeat_count = 3 : i32}                    ← IRON encodes "N+1 firings"
    //   aie.dma_bd(... [<size = 4, stride = 0>, ...])   ← matches: 4 firings
    //
    // Together: shim fires the BD 4× per dispatch (once per N-tile).
    aie.runtime_sequence(%arg0: memref<16384xbf16>) {
      %t0 = aiex.dma_configure_task_for @A_L3L2_0 {
        aie.dma_bd(%arg0 : memref<16384xbf16>, 0, 4096,
          [<size = 4, stride = 0>,
           <size = 1, stride = 0>,
           <size = 64, stride = 64>,
           <size = 64, stride = 1>])
            {burst_length = 0 : i32}
        aie.end
      } {repeat_count = 3 : i32}
      aiex.dma_start_task(%t0)
      aiex.dma_free_task(%t0)
    }
  }
}

// -----------------------------------------------------------------------------
// EXPECTED behavior (post-fix): the conduit.create carries `dma_repeat = 3`
// (verbatim copy of the IRON-emitted `repeat_count = 3`).  The lowered
// conduit.put_memref still does NOT carry any repeat-related field — the
// canonical home for the channel-level replay count is the create.
// -----------------------------------------------------------------------------

// CHECK:       conduit.create @A_L3L2_0
// CHECK-SAME:  dma_repeat = 3

// Async because the input has IRON dma_free_task — see conditional emission
// in --dma-task-to-conduit (ConduitDmaTaskToConduit.cpp file header).
// CHECK:       conduit.put_memref_async
// CHECK-SAME:  name = @A_L3L2_0
// CHECK-NOT:   dma_repeat
// CHECK-NOT:   repeat_count
