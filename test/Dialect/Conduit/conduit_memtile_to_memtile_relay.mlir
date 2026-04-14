// RUN: aie-opt --conduit-to-dma --aie-assign-buffer-addresses %s | FileCheck %s
//
// Regression test: MemTile-to-MemTile relay via chained conduit.scatter ops.
//
// Validates that Pass C handles a topology where data flows through two
// MemTiles before reaching the consumer compute tile:
//
//   compute(0,2) → MemTile(0,1) → MemTile(1,1) → compute(1,2)
//
// The MemTile-to-MemTile hop is formed by chaining two scatter ops:
//   scatter #1 at MemTile(0,1): @src → @mid
//   scatter #2 at MemTile(1,1): @mid → @dst
//
// This is valid hardware topology (MemTiles can DMA to each other via the
// NoC), but was previously untested. Pass C must not assume relay endpoints
// are always compute tiles.
//
// Expected:
//   3 aie.flow: compute→MemTile(0,1), MemTile(0,1)→MemTile(1,1),
//               MemTile(1,1)→compute
//   2 aie.memtile_dma blocks (one per MemTile), each with S2MM + MM2S
//   No residual Conduit ops

// CHECK-LABEL: module @memtile_to_memtile_relay
// CHECK:   aie.device(npu2) {

// --- Scatter #1: flows from MemTile(0,1) ---
// CHECK: aie.flow(%mem_tile_0_1, DMA : 0, %mem_tile_1_1, DMA : 0)
// CHECK: aie.flow(%tile_0_2, DMA : 0, %mem_tile_0_1, DMA : 0)

// --- MemTile(0,1) DMA: 1 S2MM (ingest from compute) + 1 MM2S (relay to MemTile(1,1)) ---
// CHECK:     aie.memtile_dma(%mem_tile_0_1) {
// CHECK:       aie.dma_start(S2MM, 0,
// CHECK:       aie.dma_start(MM2S, 0,
// CHECK:       aie.end
// CHECK:     }

// --- Scatter #2: flow from MemTile(1,1) → compute(1,2) ---
// CHECK: aie.flow(%mem_tile_1_1, DMA : 0, %tile_1_2, DMA : 0)

// --- MemTile(1,1) DMA: 1 S2MM (ingest from MemTile(0,1)) + 1 MM2S (relay to compute) ---
// CHECK:     aie.memtile_dma(%mem_tile_1_1) {
// CHECK:       aie.dma_start(S2MM, 0,
// CHECK:       aie.dma_start(MM2S, 0,
// CHECK:       aie.end
// CHECK:     }

// --- No residual Conduit ops ---
// CHECK-NOT: conduit.create
// CHECK-NOT: conduit.scatter

module @memtile_to_memtile_relay {
  aie.device(npu2) {
    %tile_0_2 = aie.tile(0, 2)
    %mem_tile_0_1 = aie.tile(0, 1)
    %mem_tile_1_1 = aie.tile(1, 1)
    %tile_1_2 = aie.tile(1, 2)

    // Stage 1: compute(0,2) → MemTile(0,1), depth=2
    conduit.create @src {
      slot_elems = 4 : i64,
      element_type = memref<4xi32>,
      depth = 2 : i64
    }

    // Stage 2: MemTile(0,1) → MemTile(1,1), depth=2
    conduit.create @mid {
      slot_elems = 4 : i64,
      element_type = memref<4xi32>,
      depth = 2 : i64
    }

    // Stage 3: MemTile(1,1) → compute(1,2), depth=2
    conduit.create @dst {
      slot_elems = 4 : i64,
      element_type = memref<4xi32>,
      depth = 2 : i64
    }

    // Relay #1: forward @src → @mid at MemTile(0,1)
    conduit.scatter{src = @src, dsts = [@mid] {memtile = "tile(0,1)"}}

    // Relay #2: forward @mid → @dst at MemTile(1,1)
    conduit.scatter{src = @mid, dsts = [@dst] {memtile = "tile(1,1)"}}

    %core_prod = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c4 = arith.constant 4 : index

      scf.for %i = %c0 to %c4 step %c1 {
        %win = conduit.acquire {name = @src, count = 1 : i64,
                                port = #conduit.port<Produce>}
                 : !conduit.window<memref<4xi32>>
        conduit.release %win {count = 1 : i64, port = #conduit.port<Produce>}
          : !conduit.window<memref<4xi32>>
      }

      aie.end
    } {dynamic_objfifo_lowering = true}

    %core_cons = aie.core(%tile_1_2) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c4 = arith.constant 4 : index

      scf.for %i = %c0 to %c4 step %c1 {
        %win = conduit.acquire {name = @dst, count = 1 : i64,
                                port = #conduit.port<Consume>}
                 : !conduit.window<memref<4xi32>>
        conduit.release %win {count = 1 : i64, port = #conduit.port<Consume>}
          : !conduit.window<memref<4xi32>>
      }

      aie.end
    } {dynamic_objfifo_lowering = true}
  }
}
