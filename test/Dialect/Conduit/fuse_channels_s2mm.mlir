// RUN: aie-opt --conduit-fuse-channels %s | FileCheck %s
//
// Regression test: S2MM consumer-side fusion in --conduit-fuse-channels.
//
// Before this sprint's fix, --conduit-fuse-channels only performed MM2S
// (producer-side) fusion.  S2MM fusion was added to handle the symmetric
// case: two conduits consumed on the same tile with non-overlapping
// acquire/release intervals can share one S2MM DMA channel.
//
// Topology:
//   tile(0,2) = producer of chan_a
//   tile(1,2) = producer of chan_b
//   tile(0,4) = shared consumer tile for BOTH chan_a and chan_b
//
// The consumer core on tile(0,4) acquires/releases chan_a, then
// acquires/releases chan_b — strictly sequential.  The pass should
// annotate both conduit.create ops with dma_channel_group_s2mm = "group0"
// and fuse_mode_s2mm = "static".

// CHECK-LABEL: module @fuse_channels_s2mm_test

// chan_a gets S2MM fusion annotation:
// CHECK:       conduit.create @chan_a
// CHECK-SAME:  dma_channel_group_s2mm = "group0"
// CHECK-SAME:  fuse_mode_s2mm = "static"

// chan_b gets S2MM fusion annotation (same group):
// CHECK:       conduit.create @chan_b
// CHECK-SAME:  dma_channel_group_s2mm = "group0"
// CHECK-SAME:  fuse_mode_s2mm = "static"

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
