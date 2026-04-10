// RUN: aie-opt --conduit-to-dma %s | FileCheck %s
//
// Shared memory conduit — adjacent tiles.
//
// Topology:
//   producer  = tile(0, 2)  — compute tile
//   consumer  = tile(0, 3)  — adjacent to producer → shared-memory eligible
//
// alloc_tile was removed from conduit.create in Sprint 4 cleanup.
// This test now exercises normal shared-mem lowering (adjacent tiles).
//
// CHECK-LABEL: aie.device(npu1)
// CHECK-NOT: conduit.create

module {
  aie.device(npu1) {
    %tile_0_2 = aie.tile(0, 2)
    %tile_0_3 = aie.tile(0, 3)

    conduit.create @shm_chan {slot_elems = 1 : i64,
                    producer_tile = array<i64: 0, 2>,
                    consumer_tiles = array<i64: 0, 3>,
                    element_type = memref<16xi32>,
                    depth = 1 : i64}

    aie.core(%tile_0_2) {
      %win = conduit.acquire {name = @shm_chan, count = 1 : i64,
                              port = #conduit.port<Produce>}
                 : !conduit.window<memref<16xi32>>
      %buf = conduit.subview_access %win {index = 0 : i64}
                 : !conduit.window<memref<16xi32>> -> memref<16xi32>
      conduit.release %win {count = 1 : i64, port = #conduit.port<Produce>}
          : !conduit.window<memref<16xi32>>
      aie.end
    }

    aie.core(%tile_0_3) {
      %win = conduit.acquire {name = @shm_chan, count = 1 : i64,
                              port = #conduit.port<Consume>}
                 : !conduit.window<memref<16xi32>>
      %buf = conduit.subview_access %win {index = 0 : i64}
                 : !conduit.window<memref<16xi32>> -> memref<16xi32>
      conduit.release %win {count = 1 : i64, port = #conduit.port<Consume>}
          : !conduit.window<memref<16xi32>>
      aie.end
    }
  }
}
