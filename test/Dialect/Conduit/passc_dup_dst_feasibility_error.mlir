// RUN: aie-opt --verify-diagnostics --objectfifo-to-conduit --conduit-fuse-channels --conduit-to-dma %s
//
// Pin for the duplicate-dst circuit-flow feasibility check in Pass C
// (`emitFlow` in ConduitToDMACommon.cpp).  Bug #99 closure.
//
// `--conduit-fuse-channels` annotates two shim-produced channels with the
// same `dma_channel_group_s2mm` group key (legal per the path-c restriction
// in commit 431f853815: both producers are the same shim tile).  Pass C
// then folds the consumer-side S2MM port to a single channel via the group,
// but the producer-side MM2S channels are necessarily distinct (each shim
// conduit gets its own MM2S 0/1/...).  routePhase Sub-case 4a (lines 463-
// 531 in ConduitToDMARoute.cpp) emits ONE flow per (consumer tile,
// conduit) pair:
//   * (shim,DMA:0) → (cons,DMA:0)  for chan_a
//   * (shim,DMA:1) → (cons,DMA:0)  for chan_b
// Two distinct sources circuit-routing to the same dst port — `aie-routing`
// rejects this as a duplicate-dst circuit connect.  Rather than punt that
// failure downstream with an opaque routing error, Pass C now detects it
// at emit time and reports a clean diagnostic that names both source ports
// and the shared dst port so the user can correlate to conduit names by
// reading the IR.
//
// SILENT-DEDUP companion (the legal kept branch where two grouped channels
// resolve to the SAME physical (src,dst) port pair, exercising emitFlow's
// silent-dedup return) is covered by
//   fuse_channels_s2mm_same_producer.mlir
// — its second metafix-RUN line lowers two same-producer-tile fuse-grouped
// channels through Pass C and emits exactly ONE aie.flow on the shared
// producer→consumer leg (same producer compute tile means MM2S and S2MM
// both fold, so both conduits resolve to the same physical port pair).
// No need to duplicate that coverage here; this pin is exclusively the
// error path.
//
// Topology: shim(0,0) → tile(0,2) for both @ext_in_a and @ext_in_b.
// Consumer core acquires/releases each in turn (producer-side windows
// overlap on the shim, so this is the consumer-side S2MM fuse case
// specifically — annotation comes through the shim/compute-shared-
// consumer detection inside `--conduit-fuse-channels`).  expected-error
// pin attaches to the consumer TileOp's loc — that is the over-subscribed
// port and the diagnostic location chosen by emitFlow.

module @passc_dup_dst_error_test {
  aie.device(npu2_1col) {
    %shim   = aie.tile(0, 0)
    // expected-error @below {{conduit-to-dma: cannot circuit-route distinct sources to (0,2)}}
    %tile_0_2 = aie.tile(0, 2)

    // Two shim-producer FIFOs, same consumer tile.  Same producer tile
    // (shim) so the path-c restriction does NOT suppress the S2MM fuse
    // group annotation — it is the kept branch.  Different MM2S channels
    // on the shim, shared S2MM channel on the consumer → 2 flows, same
    // dst port.
    aie.objectfifo @ext_in_a (%shim, {%tile_0_2}, 1 : i32)
        : !aie.objectfifo<memref<8xi32>>
    aie.objectfifo @ext_in_b (%shim, {%tile_0_2}, 1 : i32)
        : !aie.objectfifo<memref<8xi32>>

    // Consumer core: acquire/release each FIFO sequentially (non-overlapping
    // consumer-side windows in the same parent block — the shape that
    // fuse-channels groups on the S2MM/consumer side).
    %core = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c4 = arith.constant 4 : index
      scf.for %i = %c0 to %c4 step %c1 {
        %sv_a = aie.objectfifo.acquire @ext_in_a (Consume, 1)
            : !aie.objectfifosubview<memref<8xi32>>
        aie.objectfifo.release @ext_in_a (Consume, 1)

        %sv_b = aie.objectfifo.acquire @ext_in_b (Consume, 1)
            : !aie.objectfifosubview<memref<8xi32>>
        aie.objectfifo.release @ext_in_b (Consume, 1)
      }
      aie.end
    }

    aie.runtime_sequence(%a: memref<8xi32>, %b: memref<8xi32>) {
      aiex.npu.dma_memcpy_nd (%a[0, 0, 0, 0][1, 1, 1, 8][0, 0, 0, 1])
          {metadata = @ext_in_a, id = 0 : i64} : memref<8xi32>
      aiex.npu.dma_memcpy_nd (%b[0, 0, 0, 0][1, 1, 1, 8][0, 0, 0, 1])
          {metadata = @ext_in_b, id = 1 : i64} : memref<8xi32>
      aiex.npu.dma_wait {symbol = @ext_in_a}
      aiex.npu.dma_wait {symbol = @ext_in_b}
    }
  }
}
