// RUN: aie-opt --conduit-to-dma --verify-diagnostics %s
//
// C7: fuse_mode = "runtime" must be rejected by Pass C with a hard error.
//
// When --conduit-fuse-channels classifies a group as fuse_mode = "runtime"
// (ops inside scf.if branches), Pass C cannot safely lower to static BD chains:
// the DMA engine would run both members unconditionally even when the branch is
// not taken, causing silent data corruption.
//
// The control-packet BD reprogramming path (Phase 3) is not yet implemented.
// Pass C must reject the program at Phase 1 attribute read-time.
//
// Expected behaviour:
//   - emitError fires on the conduit.create with fuse_mode = "runtime"
//   - signalPassFailure() + return halt compilation immediately
//   - --verify-diagnostics exits 0 when the expected-error annotation is matched
//
// Input: pre-annotated module (annotations normally set by --conduit-fuse-channels).
// Topology: npu1_1col, producer tile [0,2], consumer [0,4].

module @runtime_fuse_error {
  aie.device(npu1_1col) {
    %tile_0_2 = aie.tile(0, 2)
    %tile_0_4 = aie.tile(0, 4)

    // expected-error @+1 {{conduit-to-dma: fuse_mode="runtime" is not yet supported}}
    conduit.create {name = "chan_a", capacity = 8 : i64,
                    producer_tile = array<i64: 0, 2>,
                    consumer_tiles = array<i64: 0, 4>,
                    element_type = memref<8xi32>,
                    depth = 1 : i64,
                    fuse_mode = "runtime",
                    fused_dma_channel_group = "group0"}
  }
}
