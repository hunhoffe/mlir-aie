// RUN: aie-opt --conduit-fuse-channels %s | FileCheck %s
//
// Regression test: S2MM consumer-side fusion in --conduit-fuse-channels —
// CROSS-PRODUCER suppression case (Path c, bug #99).
//
// Before the #99 fix, --conduit-fuse-channels would annotate
// dma_channel_group_s2mm on any S2MM group sharing a consumer tile,
// regardless of how many distinct producer tiles the group spanned.
// Pass C routePhase Sub-case 4a then emitted per-conduit aie.flow ops
// without honoring the grouping, producing two flows targeting the same
// consumer-tile DMA destination port — which aie-routing rejects.
//
// Path c restricts the dma_channel_group_s2mm annotation to groups whose
// members all share the same producer tile.  Cross-producer groups are
// punted to a future packet-routing path (Sprint N+4) and are NOT
// annotated by this pass.
//
// Topology (cross-producer, suppression case):
//   tile(0,2) = producer of chan_a
//   chan_b    = no producer core (producerTile null → treated as unknown,
//               distinct from tile(0,2))
//   tile(0,4) = shared consumer tile for BOTH chan_a and chan_b
//
// The consumer core on tile(0,4) acquires/releases chan_a, then
// acquires/releases chan_b — strictly sequential, so the live-interval
// pass would by itself coalesce them.  The Path c predicate suppresses
// the annotation because the producer tiles differ.
//
// The same-producer POSITIVE case is pinned in
// fuse_channels_s2mm_same_producer.mlir.

// CHECK-LABEL: module @fuse_channels_s2mm_test

// Path c suppression: NEITHER channel may carry the S2MM fuse annotation.
// CHECK-NOT:   dma_channel_group_s2mm
// CHECK-NOT:   fuse_mode_s2mm

module @fuse_channels_s2mm_test {
  aie.device(npu1_1col) {
    func.func @process(%buf: memref<8xi32>) -> () {
      return
    }

    %tile_0_2 = aie.tile(0, 2)
    %tile_0_4 = aie.tile(0, 4)

    // Two conduits with different producers but SAME consumer tile (0,4).
    conduit.create @chan_a {element_type = memref<8xi32>, depth = 1 : i64}
    conduit.create @chan_b {element_type = memref<8xi32>, depth = 1 : i64}

    // Producer core for chan_a on tile(0,2).
    aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c4 = arith.constant 4 : index
      scf.for %i = %c0 to %c4 step %c1 {
        %w = conduit.acquire {name = @chan_a, count = 1 : i64,
                              port = #conduit.port<Produce>}
                : !conduit.window<memref<8xi32>>
        %buf = conduit.subview_access %w {index = 0 : i64}
                  : !conduit.window<memref<8xi32>> -> memref<8xi32>
        func.call @process(%buf) : (memref<8xi32>) -> ()
        conduit.release %w {count = 1 : i64, port = #conduit.port<Produce>}
            : !conduit.window<memref<8xi32>>
      }
      aie.end
    } {dynamic_objfifo_lowering = true}

    // Consumer core on tile(0,4) — consumes BOTH conduits sequentially.
    aie.core(%tile_0_4) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c4 = arith.constant 4 : index
      scf.for %i = %c0 to %c4 step %c1 {
        // chan_a: acquire, use, release.
        %wa = conduit.acquire {name = @chan_a, count = 1 : i64,
                               port = #conduit.port<Consume>}
                 : !conduit.window<memref<8xi32>>
        %buf_a = conduit.subview_access %wa {index = 0 : i64}
                    : !conduit.window<memref<8xi32>> -> memref<8xi32>
        func.call @process(%buf_a) : (memref<8xi32>) -> ()
        conduit.release %wa {count = 1 : i64, port = #conduit.port<Consume>}
            : !conduit.window<memref<8xi32>>

        // chan_b: acquire, use, release — starts AFTER chan_a is released.
        %wb = conduit.acquire {name = @chan_b, count = 1 : i64,
                               port = #conduit.port<Consume>}
                 : !conduit.window<memref<8xi32>>
        %buf_b = conduit.subview_access %wb {index = 0 : i64}
                    : !conduit.window<memref<8xi32>> -> memref<8xi32>
        func.call @process(%buf_b) : (memref<8xi32>) -> ()
        conduit.release %wb {count = 1 : i64, port = #conduit.port<Consume>}
            : !conduit.window<memref<8xi32>>
      }
      aie.end
    } {dynamic_objfifo_lowering = true}
  }
}
