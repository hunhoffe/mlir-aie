// RUN: aie-opt --objectfifo-to-conduit --dma-task-to-conduit --conduit-fuse-channels --verify-diagnostics %s
//
// Pattern D (Task #16) × fuse-channels (`1ac3119912`) cross-pass guard.
//
// RTP-folded trip variant of
// `infer_iter_count_then_fuse_channels_mismatched_repeat.mlir`: the
// consumer outer `scf.for` upper bound is RTP-folded
// (`arith.index_cast (memref.load %my_rtp[%c0])` → 64) instead of an
// `arith.constant 64 : index`.  Pins that RTP-folded trips flow through
// the dma_repeat compatibility check identically to literal-folded trips.
//
// Geometry (post Task #42, 2026-04-28):
//   * Both consumer acquires share ONE outer `scf.for` body block —
//     `ConduitFuseChannels.cpp::assignGroups` walks per-block, so the two
//     channels MUST live in the same body block to be compared and the
//     mismatch detected.
//   * Outer trip = `index_cast (load %my_rtp[%c0])` = 64.
//   * dma_repeat SOURCING: post Task #42 (#40 root cause) Pass A no longer
//     infers dma_repeat for any shim-bearing channel (host-side
//     num_invocations is invisible to the IR).  This fixture now sources
//     dma_repeat from IRON-EXPLICIT `repeat_count` attributes on the
//     `aiex.dma_configure_task_for` ops, which `--dma-task-to-conduit`
//     surfaces verbatim onto the conduit.create's `dma_repeat`:
//       - chan_a IRON repeat_count = 2  → conduit.create dma_repeat = 2
//       - chan_b IRON repeat_count = 16 → conduit.create dma_repeat = 16
//
// EXPECTED BEHAVIOR (Task #45, unchanged): --conduit-fuse-channels rejects
// the mismatched group with an error on the second (offending)
// conduit.create and signalPassFailure.

module @infer_rtp_then_fuse_channels_mismatch {
  aie.device(npu1_1col) {
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)

    %my_rtp = aie.buffer(%tile_0_2) {sym_name = "my_rtp", use_write_rtp = true} : memref<2xi32>

    // expected-remark@+1 {{conduit-objectfifo: dma_repeat inference skipped: host-side num_invocations not observable in IR (shim-bearing channel); deferring dma_repeat to runtime}}
    aie.objectfifo @chan_a(%tile_0_0, {%tile_0_2}, 2 : i32)
        : !aie.objectfifo<memref<2xbf16>>
    // expected-error@+2 {{fuse-channels: cannot group channels with mismatched dma_repeat values 2 vs 16 (S2MM group)}}
    // expected-remark@+1 {{conduit-objectfifo: dma_repeat inference skipped: host-side num_invocations not observable in IR (shim-bearing channel); deferring dma_repeat to runtime}}
    aie.objectfifo @chan_b(%tile_0_0, {%tile_0_2}, 2 : i32)
        : !aie.objectfifo<memref<2xbf16>>

    // Single shared scf.for body block; outer trip = RTP-folded 64.  Both
    // chan_a and chan_b acquires sit in the SAME block so assignGroups
    // sees them together.
    aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %i0 = arith.constant 0 : index
      %v = memref.load %my_rtp[%i0] : memref<2xi32>
      %ub = arith.index_cast %v : i32 to index
      scf.for %i = %c0 to %ub step %c1 {
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

    // Host: distinct BD geometry per channel; emit.count = 2 each.
    aie.runtime_sequence(%a0: memref<64xbf16>, %b0: memref<64xbf16>) {
      aiex.npu.rtp_write(@my_rtp, 0, 64)
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
