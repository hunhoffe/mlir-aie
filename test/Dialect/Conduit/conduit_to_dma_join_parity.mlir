// RUN: aie-opt --objectfifo-to-conduit --conduit-to-dma %s | FileCheck %s
//
// Resource parity test: join lowering must produce exactly the same
// lock/buffer/flow counts as --aie-objectFifo-stateful-transform.
//
// Before the join-L2 fix, Conduit over-allocated:
//   +2 extra buffers (join destination re-allocated by Phase 5)
//   +2 extra locks   (Phase 3b allocated a lock pair before Phase 5 replaced it)
//   +1 extra flow    (Phase 5 duplicated the memtile→shim flow from Phase 4b)
//
// After the fix:
//   Phase 3b skips allocation for join destination conduits (linkDstNames check).
//   Phase 5 allocates buffers+locks for the join destination directly.
//   Phase 5 no longer emits the shim consumer flow (Phase 4b owns it).
//
// Expected resource counts (matching --aie-objectFifo-stateful-transform):
//   aie.buffer:   8  (link1: 2 on tile_2_2; link2: 2 on tile_2_3;
//                     link3: 2 on tile_3_3; link4: 2 on mem_tile_2_1)
//   aie.lock:    14  (link4: 2 on shim + 6 on memtile; link1/2/3: 2 each)
//   aie.flow:     4  (tile_2_2→memtile, tile_2_3→memtile, tile_3_3→memtile,
//                     mem_tile_2_1→shim_2_0)
//   aie.dma_bd:  18  (3 S2MM rings × 2 BDs + 1 MM2S ring × 6 BDs +
//                     3 compute tile MM2S rings × 2 BDs = 6+6+6 = 18)
//   aie.use_lock: 36 (2 per BD × 18 BDs = 36)

// CHECK-LABEL: module @link_join_parity
// CHECK: aie.device(xcve2302)
//
// Output ordering (reflects phase execution order):
//   Phase 4b: shim alloc + shim locks + memtile→shim flow
//   Phase 5 (join alloc): memtile buffers + 6 per-source locks (3 pairs)
//   Phase 5 (join flows): 3 per-source flows (tile→memtile)
//   Phase 5 (memtile_dma): 3 S2MM channels + 1 MM2S channel
//   Phase 5.5: 3 compute-tile aie.mem blocks
//
// Verify shim alloc and memtile→shim flow (Phase 4b, first).
// CHECK: aie.shim_dma_allocation @link4_shim_alloc
// CHECK: aie.flow(%mem_tile_2_1, DMA : 0, %shim_noc_tile_2_0, DMA : 0)
//
// Verify exactly 6 locks on memtile (3 pairs for 3 sources — no extra pair).
// CHECK: aie.lock(%mem_tile_2_1, 0) {init = 2
// CHECK: aie.lock(%mem_tile_2_1, 1) {init = 0
// CHECK: aie.lock(%mem_tile_2_1, 2) {init = 2
// CHECK: aie.lock(%mem_tile_2_1, 3) {init = 0
// CHECK: aie.lock(%mem_tile_2_1, 4) {init = 2
// CHECK: aie.lock(%mem_tile_2_1, 5) {init = 0
//
// Verify the 3 per-source flows (Phase 5, after lock allocation).
// CHECK: aie.flow(%tile_2_2, DMA : 0, %mem_tile_2_1, DMA : 0)
// CHECK: aie.flow(%tile_2_3, DMA : 0, %mem_tile_2_1, DMA : 1)
// CHECK: aie.flow(%tile_3_3, DMA : 0, %mem_tile_2_1, DMA : 2)
//
// MemTile DMA: 3 S2MM + 1 MM2S.
// CHECK: aie.memtile_dma(%mem_tile_2_1)
// CHECK:   aie.dma_start(S2MM, 0,
// CHECK:   aie.dma_start(S2MM, 1,
// CHECK:   aie.dma_start(S2MM, 2,
// CHECK:   aie.dma_start(MM2S, 0,
// CHECK:   aie.end

// No leftover conduit ops.
// CHECK-NOT: conduit.create
// CHECK-NOT: conduit.link

module @link_join_parity {
  aie.device(xcve2302) {
    %tile20 = aie.tile(2, 0)
    %tile21 = aie.tile(2, 1)
    %tile22 = aie.tile(2, 2)
    %tile23 = aie.tile(2, 3)
    %tile33 = aie.tile(3, 3)

    aie.objectfifo @link1 (%tile22, {%tile21}, 2 : i32) : !aie.objectfifo<memref<4x4xi32>>
    aie.objectfifo @link2 (%tile23, {%tile21}, 2 : i32) : !aie.objectfifo<memref<20xi32>>
    aie.objectfifo @link3 (%tile33, {%tile21}, 2 : i32) : !aie.objectfifo<memref<12xi32>>
    aie.objectfifo @link4 (%tile21, {%tile20}, 2 : i32) : !aie.objectfifo<memref<48xi32>>

    aie.objectfifo.link [@link1, @link2, @link3] -> [@link4] ([0, 16, 36][])
  }
}
