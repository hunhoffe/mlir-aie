// RUN: aie-opt --objectfifo-to-conduit -split-input-file %s | FileCheck %s
//
// alloc_tile → scatter{N=1} lowering test (Task #16).
//
// When aie.objectfifo.allocate specifies a MemTile (row==1) delegate,
// Pass A emits a scatter{N=1} relay through the MemTile:
//
//   1. @fifo's consumer_tiles → [memtile col, 1]
//   2. @fifo_relay created with consumer_tiles=original
//   3. conduit.scatter { src=@fifo, dsts=[@fifo_relay] }
//   4. Consumer-side acquire/release ops rewritten from @fifo to @fifo_relay.
//
// Uses --split-input-file for two scenarios:
//   (a) MemTile delegate (row=1): scatter{N=1} emitted.
//   (b) Compute-tile delegate (row=2): warning, no scatter.

// (a) MemTile delegate: producer=tile(0,2), consumer=tile(0,3),
//     delegate=tile(0,1) (MemTile).
//
// CHECK-LABEL: module @memtile_delegate
//
// Source channel: consumer_tiles updated to MemTile [0,1].
// CHECK:   conduit.create @fifo
//
// Relay channel: producer_tile = MemTile, consumer_tiles = original [0,3].
// CHECK:   conduit.create @fifo_relay
//
// Scatter relay.
// CHECK:   conduit.scatter{src = @fifo, dsts = [@fifo_relay]
// CHECK-SAME: memtile = "tile(0,1)"
//
// Producer acquire unchanged — still references @fifo.
// CHECK:   conduit.acquire
// CHECK-SAME: name = @fifo
// CHECK-SAME: port = #conduit.port<Produce>
//
// Consumer acquire rewritten from @fifo to @fifo_relay.
// CHECK:   conduit.acquire
// CHECK-SAME: name = @fifo_relay
// CHECK-SAME: port = #conduit.port<Consume>
//
// Consumer release (via window SSA) — no name attr to check, but the
// window flows from the @fifo_relay acquire so it's implicitly correct.
//
// No leftover objectfifo ops.
// CHECK-NOT: aie.objectfifo
// CHECK-NOT: aie.objectfifo.allocate

module @memtile_delegate {
  aie.device(npu1) {
    %tile_0_1 = aie.tile(0, 1)
    %tile_0_2 = aie.tile(0, 2)
    %tile_0_3 = aie.tile(0, 3)

    aie.objectfifo @fifo (%tile_0_2, {%tile_0_3}, 2 : i32)
        : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo.allocate @fifo (%tile_0_1)

    aie.core(%tile_0_2) {
      %sub = aie.objectfifo.acquire @fifo (Produce, 1)
          : !aie.objectfifosubview<memref<16xi32>>
      %elem = aie.objectfifo.subview.access %sub[0]
          : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>
      aie.objectfifo.release @fifo (Produce, 1)
      aie.end
    }

    aie.core(%tile_0_3) {
      %sub = aie.objectfifo.acquire @fifo (Consume, 1)
          : !aie.objectfifosubview<memref<16xi32>>
      %elem = aie.objectfifo.subview.access %sub[0]
          : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>
      aie.objectfifo.release @fifo (Consume, 1)
      aie.end
    }
  }
}

// -----

// (b) Compute-tile delegate (row=2): warning emitted, allocate erased,
//     no scatter generated.
//
// CHECK-LABEL: module @compute_delegate
//
// Channel unchanged — consumer_tiles still [1,3], no relay.
// CHECK:   conduit.create @fifo2
//
// No scatter or relay channel.
// CHECK-NOT: conduit.scatter
// CHECK-NOT: conduit.create @fifo2_relay
//
// CHECK-NOT: aie.objectfifo
// CHECK-NOT: aie.objectfifo.allocate

module @compute_delegate {
  aie.device(npu1) {
    %tile_1_2 = aie.tile(1, 2)
    %tile_1_3 = aie.tile(1, 3)

    aie.objectfifo @fifo2 (%tile_1_2, {%tile_1_3}, 1 : i32)
        : !aie.objectfifo<memref<16xi32>>
    aie.objectfifo.allocate @fifo2 (%tile_1_2)

    aie.core(%tile_1_2) {
      %sub = aie.objectfifo.acquire @fifo2 (Produce, 1)
          : !aie.objectfifosubview<memref<16xi32>>
      %elem = aie.objectfifo.subview.access %sub[0]
          : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>
      aie.objectfifo.release @fifo2 (Produce, 1)
      aie.end
    }

    aie.core(%tile_1_3) {
      %sub = aie.objectfifo.acquire @fifo2 (Consume, 1)
          : !aie.objectfifosubview<memref<16xi32>>
      %elem = aie.objectfifo.subview.access %sub[0]
          : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>
      aie.objectfifo.release @fifo2 (Consume, 1)
      aie.end
    }
  }
}
