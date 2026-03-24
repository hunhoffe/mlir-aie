// RUN: aie-opt --conduit-to-dma --verify-diagnostics %s
//
// P1-D: Shared memory adjacency verification — alloc_tile not adjacent.
//
// When a shared-memory conduit has an alloc_tile that is NOT adjacent to
// both the producer and consumer tiles, Pass C must reject it with a hard
// error. The buffer would not be reachable from both cores.
//
// Topology:
//   producer  = tile(0, 2)  — compute tile
//   consumer  = tile(0, 3)  — adjacent to producer → shared-memory eligible
//   alloc_tile= tile(3, 3)  — different column; NOT adjacent to tile(0,2)
//
// Expected: emitError on the conduit.create.

module {
  aie.device(npu1) {
    %tile_0_2 = aie.tile(0, 2)
    %tile_0_3 = aie.tile(0, 3)
    %tile_3_3 = aie.tile(3, 3)

    // expected-error @+1 {{shared-memory conduit requires adjacent tiles}}
    conduit.create @shm_chan {capacity = 1 : i64,
                    producer_tile = array<i64: 0, 2>,
                    consumer_tiles = array<i64: 0, 3>,
                    element_type = memref<16xi32>,
                    depth = 1 : i64,
                    alloc_tile = array<i64: 3, 3>}

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
