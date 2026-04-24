// RUN: aie-opt --objectfifo-to-conduit --conduit-fuse-channels %s | FileCheck %s
//
// Foundation Phase 2 (Task #19), gap #4 — Pass A inference followed by
// fuse-channels with channels that should land in the same
// dma_channel_group_s2mm but have DIFFERENT inferred dma_repeat values.
//
// Geometry:
//   * Shim @ tile(0,0) feeds two channels, both consumed by tile(0,2).
//   * Consumer body: `scf.for 0..32 { acquire chan_a; release;
//                                     acquire chan_b; release; }`
//     → per-channel consumer trip = 32; producer side is shim (no core
//       acquires) so the consumer trip dictates inference.
//   * Per-channel BD geometry differs:
//       - chan_a fifo elem = memref<2xbf16>, BD len = 32  →
//         acquires_per_BD = 16 → dma_repeat = 32 / 16 = 2.
//       - chan_b fifo elem = memref<2xbf16>, BD len = 4   →
//         acquires_per_BD = 2  → dma_repeat = 32 / 2  = 16.
//   * The two consumer acquires are sequentially interleaved in the
//     same scf.for body block, so their live intervals are disjoint and
//     fuse-channels will assign them to the same S2MM group.
//
// CURRENT BEHAVIOR (the gap this test pins):
//   fuse-channels does NOT inspect per-channel dma_repeat values when
//   forming an S2MM group; it only looks at producer/consumer-tile
//   shape and live-interval non-overlap.  Channels with mismatched
//   dma_repeat (2 vs 16) are silently grouped together.  Pass C
//   currently does NOT reject this either.  This is a known LOW-risk
//   gap — see PLAN.md backlog "Pass C should reject mismatched
//   dma_repeat within a dma_channel_group".
//
// If a future fix adds an error/remark in either pass, this test
// should be flipped from pinning the silent group to pinning the
// error/remark.  Until then, the regression net is "we noticed this
// happens silently, on purpose, for now".

// CHECK-LABEL: module @infer_then_fuse_channels_mismatched_repeat

// MLIR sorts attrs alphabetically: dma_channel_group_s2mm < dma_repeat.
// CHECK:       conduit.create @chan_a
// CHECK-SAME:  dma_channel_group_s2mm = "group0"
// CHECK-SAME:  dma_repeat = 2

// CHECK:       conduit.create @chan_b
// CHECK-SAME:  dma_channel_group_s2mm = "group0"
// CHECK-SAME:  dma_repeat = 16

module @infer_then_fuse_channels_mismatched_repeat {
  aie.device(npu1_1col) {
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)

    aie.objectfifo @chan_a(%tile_0_0, {%tile_0_2}, 2 : i32)
        : !aie.objectfifo<memref<2xbf16>>
    aie.objectfifo @chan_b(%tile_0_0, {%tile_0_2}, 2 : i32)
        : !aie.objectfifo<memref<2xbf16>>

    // Consumer: 32 acquires per channel, sequentially interleaved within
    // the scf.for body block.
    aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c32 = arith.constant 32 : index
      scf.for %i = %c0 to %c32 step %c1 {
        %wa = aie.objectfifo.acquire @chan_a(Consume, 1)
            : !aie.objectfifosubview<memref<2xbf16>>
        %ba = aie.objectfifo.subview.access %wa[0]
            : !aie.objectfifosubview<memref<2xbf16>> -> memref<2xbf16>
        aie.objectfifo.release @chan_a(Consume, 1)
        %wb = aie.objectfifo.acquire @chan_b(Consume, 1)
            : !aie.objectfifosubview<memref<2xbf16>>
        %bb = aie.objectfifo.subview.access %wb[0]
            : !aie.objectfifosubview<memref<2xbf16>> -> memref<2xbf16>
        aie.objectfifo.release @chan_b(Consume, 1)
      }
      aie.end
    }

    // Host side: distinct BD geometries per channel.
    aie.runtime_sequence(%a0: memref<64xbf16>, %b0: memref<64xbf16>) {
      // chan_a: BD covers 32 elems; fifo elem = 2 → acquires_per_BD = 16.
      %tA = aiex.dma_configure_task_for @chan_a {
        aie.dma_bd(%a0 : memref<64xbf16>, 0, 32,
          [<size = 1, stride = 0>, <size = 1, stride = 0>,
           <size = 1, stride = 0>, <size = 32, stride = 1>])
            {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%tA)
      aiex.dma_await_task(%tA)
      aiex.dma_free_task(%tA)
      // chan_b: BD covers 4 elems; fifo elem = 2 → acquires_per_BD = 2.
      %tB = aiex.dma_configure_task_for @chan_b {
        aie.dma_bd(%b0 : memref<64xbf16>, 0, 4,
          [<size = 1, stride = 0>, <size = 1, stride = 0>,
           <size = 1, stride = 0>, <size = 4, stride = 1>])
            {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%tB)
      aiex.dma_await_task(%tB)
      aiex.dma_free_task(%tB)
    }
  }
}
