// RUN: not aie-opt --conduit-to-dma %s 2>&1 | FileCheck %s
//
// Pass C regression test: hard error on depth = 0.
//
// depth = 0 is the sentinel emitted by Pass A/B for unresolved channels.
// --conduit-depth-promote must run before --conduit-to-dma to replace it.
// If it doesn't, Pass C must hard-error rather than silently lowering with
// depth=0 (which would produce zero-BD BD chains and hardware deadlock).
//
// This test intentionally skips --conduit-depth-promote to trigger the error.

// CHECK: error: conduit-to-dma: channel @unresolved has depth = 0

module @depth_zero_error {
  aie.device(npu1_1col) {
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)

    // depth = 0: sentinel value emitted by Pass A/B when depth is unresolved.
    // --conduit-depth-promote is required before --conduit-to-dma to fill this in.
    conduit.create @unresolved {slot_elems = 8 : i64,
                    producer_tile = array<i64: 0, 0>,
                    consumer_tiles = array<i64: 0, 2>,
                    element_type = memref<8xi32>,
                    depth = 0 : i64}

    %core_0_2 = aie.core(%tile_0_2) {
      %w = conduit.acquire {name = @unresolved, count = 1 : i64,
                            port = #conduit.port<Consume>}
               : !conduit.window<memref<8xi32>>
      conduit.release %w {count = 1 : i64, port = #conduit.port<Consume>}
          : !conduit.window<memref<8xi32>>
      aie.end
    }
  }
}
