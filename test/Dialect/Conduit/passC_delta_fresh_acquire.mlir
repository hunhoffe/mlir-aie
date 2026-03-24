// RUN: aie-opt --conduit-check-channels --conduit-to-dma %s | FileCheck %s
//
// Fresh acquire: no prior held elements for the channel.
// conduit.acquire{count=2} → AcquireGreaterEqual(2).
//
// This is the simplest case: a single acquire with no prior state.  The delta
// equals the full count because heldCount starts at 0.
//
// Topology: shim(0,0) → compute(0,2), depth=4, element=memref<128xi32>.
//
// CHECK-LABEL: module @passC_delta_fresh_acquire
// CHECK: aie.core(%tile_0_2)
// CHECK: use_lock(%{{.*}}_cons_lock_0, AcquireGreaterEqual, 2)

module @passC_delta_fresh_acquire {
  aie.device(npu1_1col) {
    %shim = aie.tile(0, 0)
    %tile = aie.tile(0, 2)

    conduit.create @fifo {capacity = 512 : i64, depth = 4 : i64,
                    element_type = memref<128xi32>,
                    producer_tile = array<i64: 0, 0>,
                    consumer_tiles = array<i64: 0, 2>}

    aie.shim_dma_allocation @fifo_shim_alloc(%shim, MM2S, 0)

    %core = aie.core(%tile) {
      %c0 = arith.constant 0 : index
      %val = arith.constant 42 : i32

      %win = conduit.acquire {name = "fifo", count = 2 : i64,
                               port = #conduit.port<Consume>}
                 : !conduit.window<memref<128xi32>>
      %e0 = conduit.subview_access %win {index = 0 : i64}
                : !conduit.window<memref<128xi32>> -> memref<128xi32>
      %e1 = conduit.subview_access %win {index = 1 : i64}
                : !conduit.window<memref<128xi32>> -> memref<128xi32>
      memref.store %val, %e0[%c0] : memref<128xi32>

      conduit.release %win {count = 2 : i64, port = #conduit.port<Consume>}
          : !conduit.window<memref<128xi32>>

      aie.end
    }

    aie.runtime_sequence(%in: memref<512xi32>) {
      aiex.npu.dma_memcpy_nd (%in[0,0,0,0][1,1,1,512][0,0,0,1])
          {metadata = @fifo_shim_alloc, id = 0 : i64} : memref<512xi32>
    }
  }
}
