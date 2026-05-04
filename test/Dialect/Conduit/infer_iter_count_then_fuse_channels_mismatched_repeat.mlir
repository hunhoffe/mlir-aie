// RUN: aie-opt --objectfifo-to-conduit --dma-task-to-conduit --conduit-fuse-channels --verify-diagnostics %s
//
// Foundation Phase 2 (Task #19), gap #4 — channel grouping rejection when
// two channels that fuse-channels would otherwise place in the same
// dma_channel_group_s2mm carry DIFFERENT dma_repeat values.
//
// Geometry (post Task #42, 2026-04-28):
//   * Shim @ tile(0,0) feeds two channels, both consumed by tile(0,2).
//   * Consumer body: `scf.for 0..64 { acquire chan_a; release;
//                                     acquire chan_b; release; }`
//     → per-channel consumer trip = 64.
//   * Two BD emissions per channel, identical per-channel BD shape.
//   * dma_repeat SOURCING: post Task #42 (#40 root cause) Pass A no
//     longer infers dma_repeat for any shim-bearing channel (the
//     host-side num_invocations is invisible to the IR and the
//     inference over-fires the shim BD).  This fixture now sources
//     dma_repeat from IRON-EXPLICIT `repeat_count` attributes on the
//     `aiex.dma_configure_task_for` ops, which `--dma-task-to-conduit`
//     surfaces verbatim onto the conduit.create's `dma_repeat`:
//       - chan_a IRON repeat_count = 2  → conduit.create dma_repeat = 2
//       - chan_b IRON repeat_count = 16 → conduit.create dma_repeat = 16
//   * The two consumer acquires are sequentially interleaved in the
//     same scf.for body block, so their live intervals are disjoint and
//     fuse-channels would otherwise assign them to the same S2MM group.
//
// EXPECTED BEHAVIOR (Task #45, unchanged):
//   --conduit-fuse-channels MUST reject grouping channels with mismatched
//   dma_repeat values.  Channels with different dma_repeat fundamentally
//   cannot share a hardware S2MM channel slot — they fire BDs at
//   different rates per dispatch.  The pass emits an error on the
//   offending (second-encountered) conduit.create and signalPassFailure.

module @infer_then_fuse_channels_mismatched_repeat {
  aie.device(npu1_1col) {
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)

    // expected-remark@+1 {{conduit-objectfifo: dma_repeat inference skipped: host-side num_invocations not observable in IR (shim-bearing channel); deferring dma_repeat to runtime}}
    aie.objectfifo @chan_a(%tile_0_0, {%tile_0_2}, 2 : i32)
        : !aie.objectfifo<memref<2xbf16>>
    // expected-error@+2 {{fuse-channels: cannot group channels with mismatched dma_repeat values 2 vs 16 (S2MM group)}}
    // expected-remark@+1 {{conduit-objectfifo: dma_repeat inference skipped: host-side num_invocations not observable in IR (shim-bearing channel); deferring dma_repeat to runtime}}
    aie.objectfifo @chan_b(%tile_0_0, {%tile_0_2}, 2 : i32)
        : !aie.objectfifo<memref<2xbf16>>

    // Consumer: 64 acquires per channel, sequentially interleaved within
    // the scf.for body block.
    aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c64 = arith.constant 64 : index
      scf.for %i = %c0 to %c64 step %c1 {
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

    // Host side: distinct BD geometries per channel.  TWO configure_task
    // emissions per channel (identical shape per channel) so Pass A's
    // emit.count = 2 and dma_repeat inference is sound (post-#74).
    aie.runtime_sequence(%a0: memref<64xbf16>, %b0: memref<64xbf16>) {
      // chan_a: BD covers 32 elems; fifo elem = 2 → acquires_per_BD = 16.
      %tA0 = aiex.dma_configure_task_for @chan_a {
        aie.dma_bd(%a0 : memref<64xbf16>, 0, 32,
          [<size = 1, stride = 0>, <size = 1, stride = 0>,
           <size = 1, stride = 0>, <size = 32, stride = 1>])
            {burst_length = 0 : i32}
        aie.end
      } {repeat_count = 2 : i32}
      aiex.dma_start_task(%tA0)
      aiex.dma_await_task(%tA0)
      aiex.dma_free_task(%tA0)
      %tA1 = aiex.dma_configure_task_for @chan_a {
        aie.dma_bd(%a0 : memref<64xbf16>, 0, 32,
          [<size = 1, stride = 0>, <size = 1, stride = 0>,
           <size = 1, stride = 0>, <size = 32, stride = 1>])
            {burst_length = 0 : i32}
        aie.end
      } {repeat_count = 2 : i32}
      aiex.dma_start_task(%tA1)
      aiex.dma_await_task(%tA1)
      aiex.dma_free_task(%tA1)
      // chan_b: BD covers 4 elems; fifo elem = 2 → acquires_per_BD = 2.
      %tB0 = aiex.dma_configure_task_for @chan_b {
        aie.dma_bd(%b0 : memref<64xbf16>, 0, 4,
          [<size = 1, stride = 0>, <size = 1, stride = 0>,
           <size = 1, stride = 0>, <size = 4, stride = 1>])
            {burst_length = 0 : i32}
        aie.end
      } {repeat_count = 16 : i32}
      aiex.dma_start_task(%tB0)
      aiex.dma_await_task(%tB0)
      aiex.dma_free_task(%tB0)
      %tB1 = aiex.dma_configure_task_for @chan_b {
        aie.dma_bd(%b0 : memref<64xbf16>, 0, 4,
          [<size = 1, stride = 0>, <size = 1, stride = 0>,
           <size = 1, stride = 0>, <size = 4, stride = 1>])
            {burst_length = 0 : i32}
        aie.end
      } {repeat_count = 16 : i32}
      aiex.dma_start_task(%tB1)
      aiex.dma_await_task(%tB1)
      aiex.dma_free_task(%tB1)
    }
  }
}
