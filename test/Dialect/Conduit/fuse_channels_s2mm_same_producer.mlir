// RUN: aie-opt --conduit-fuse-channels %s | FileCheck %s
// Metafix-style second RUN line: smoke through Pass C to confirm the
// annotated single-producer S2MM group lowers without aie-routing
// duplicate-dst-port collision (#99 verify-downstream evidence).
// RUN: aie-opt --conduit-fuse-channels --conduit-to-dma %s
//
// Companion to fuse_channels_s2mm.mlir: SAME-PRODUCER S2MM fusion is the
// POSITIVE case for the Path c predicate added in #99.
//
// Path c (#99) restricts dma_channel_group_s2mm to groups whose members
// all share the same producer tile.  This fixture exercises the kept
// branch: BOTH chan_a and chan_b are produced on tile(0,2) and consumed
// on tile(0,4) with non-overlapping consumer-side intervals.  The Path c
// predicate is satisfied (single producer tile in the group), so the
// dma_channel_group_s2mm + fuse_mode_s2mm annotation IS emitted.
//
// Contrast: in fuse_channels_s2mm.mlir, chan_b has no producer core
// (producer tile unknown, distinct from chan_a's tile(0,2)) → the
// predicate suppresses the annotation.
//
// Topology (same-producer, positive case):
//   tile(0,2) = producer of BOTH chan_a and chan_b
//   tile(0,4) = consumer of BOTH chan_a and chan_b
//   Producer-side and consumer-side ops are sequential (chan_a fully
//   precedes chan_b in both cores), so live-interval coalescing applies.

// CHECK-LABEL: module @fuse_channels_s2mm_same_producer_test

// chan_a gets S2MM fusion annotation (Path c kept branch):
// CHECK:       conduit.create @chan_a
// CHECK-SAME:  dma_channel_group_s2mm = "group{{[0-9]+}}"
// CHECK-SAME:  fuse_mode_s2mm = "static"

// chan_b gets the matching S2MM fusion annotation (same group):
// CHECK:       conduit.create @chan_b
// CHECK-SAME:  dma_channel_group_s2mm = "group{{[0-9]+}}"
// CHECK-SAME:  fuse_mode_s2mm = "static"

module @fuse_channels_s2mm_same_producer_test {
  aie.device(npu1_1col) {
    func.func @process(%buf: memref<8xi32>) -> () {
      return
    }

    %tile_0_2 = aie.tile(0, 2)
    %tile_0_4 = aie.tile(0, 4)

    // Both conduits with SAME producer tile (0,2) and SAME consumer tile (0,4).
    conduit.create @chan_a {element_type = memref<8xi32>, depth = 1 : i64}
    conduit.create @chan_b {element_type = memref<8xi32>, depth = 1 : i64}

    // Single producer core on tile(0,2) produces BOTH conduits sequentially.
    aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c4 = arith.constant 4 : index
      scf.for %i = %c0 to %c4 step %c1 {
        // chan_a: produce, use, release.
        %wa = conduit.acquire {name = @chan_a, count = 1 : i64,
                               port = #conduit.port<Produce>}
                 : !conduit.window<memref<8xi32>>
        %buf_a = conduit.subview_access %wa {index = 0 : i64}
                    : !conduit.window<memref<8xi32>> -> memref<8xi32>
        func.call @process(%buf_a) : (memref<8xi32>) -> ()
        conduit.release %wa {count = 1 : i64, port = #conduit.port<Produce>}
            : !conduit.window<memref<8xi32>>

        // chan_b: starts AFTER chan_a is released — non-overlapping.
        %wb = conduit.acquire {name = @chan_b, count = 1 : i64,
                               port = #conduit.port<Produce>}
                 : !conduit.window<memref<8xi32>>
        %buf_b = conduit.subview_access %wb {index = 0 : i64}
                    : !conduit.window<memref<8xi32>> -> memref<8xi32>
        func.call @process(%buf_b) : (memref<8xi32>) -> ()
        conduit.release %wb {count = 1 : i64, port = #conduit.port<Produce>}
            : !conduit.window<memref<8xi32>>
      }
      aie.end
    } {dynamic_objfifo_lowering = true}

    // Consumer core on tile(0,4) consumes BOTH conduits sequentially.
    aie.core(%tile_0_4) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c4 = arith.constant 4 : index
      scf.for %i = %c0 to %c4 step %c1 {
        %wa = conduit.acquire {name = @chan_a, count = 1 : i64,
                               port = #conduit.port<Consume>}
                 : !conduit.window<memref<8xi32>>
        %buf_a = conduit.subview_access %wa {index = 0 : i64}
                    : !conduit.window<memref<8xi32>> -> memref<8xi32>
        func.call @process(%buf_a) : (memref<8xi32>) -> ()
        conduit.release %wa {count = 1 : i64, port = #conduit.port<Consume>}
            : !conduit.window<memref<8xi32>>

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
