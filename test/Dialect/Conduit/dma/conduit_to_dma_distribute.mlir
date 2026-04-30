// RUN: aie-opt --conduit-to-dma %s | FileCheck %s
//
// Pass C test: 1→3 distribute link with byte offsets.
//
// After --conduit-to-dma the module should contain:
//   - aie.buffer ops for each conduit (depth-many per consumer)
//   - Independent lock pairs per destination slice on the MemTile (2*N=6 locks)
//   - aie.memtile_dma for the distribute MemTile DMA BD chain
//   - Per-destination aie.flow ops: MemTile MM2S ch i → compute_tile_i DMA 0
//
// Resource comparison target (from --aie-objectFifo-stateful-transform):
//   aie.buffer:  ≥8  (depth-many per conduit)
//   aie.lock:   ≥6   on MemTile (2 per destination × 3 destinations)
//   aie.flow:    4   (shim→memtile + memtile→tile22, 23, 33)
//   aie.dma_bd:  ≥6  (6 BD blocks in S2MM ingest chain for depth=2 × 3 slices)

// CHECK-LABEL: module @link_distribute_offsets
// CHECK:   aie.device(xcve2302) {

// -- per-destination independent lock pairs on MemTile (Defects 1 & 3) --
// CHECK:     aie.lock({{.*}}) {init = 2{{.*}}sym_name = "link1_link_prod_lock_0"
// CHECK:     aie.lock({{.*}}) {init = 0{{.*}}sym_name = "link1_link_cons_lock_0"
// CHECK:     aie.lock({{.*}}) {init = 2{{.*}}sym_name = "link1_link_prod_lock_1"
// CHECK:     aie.lock({{.*}}) {init = 0{{.*}}sym_name = "link1_link_cons_lock_1"
// CHECK:     aie.lock({{.*}}) {init = 2{{.*}}sym_name = "link1_link_prod_lock_2"
// CHECK:     aie.lock({{.*}}) {init = 0{{.*}}sym_name = "link1_link_cons_lock_2"

// -- per-destination flows from MemTile MM2S channels (Defect 2) --
// CHECK:     aie.flow(%mem_tile_2_1, DMA : 0, %tile_2_2, DMA : 0)
// CHECK:     aie.flow(%mem_tile_2_1, DMA : 1, %tile_2_3, DMA : 0)
// CHECK:     aie.flow(%mem_tile_2_1, DMA : 2, %tile_3_3, DMA : 0)

// -- MemTile DMA BD chain --
// CHECK:     aie.memtile_dma({{.*}}) {
// CHECK:       aie.dma_start(S2MM, 0

// S2MM ingest: slice 0 of buff_0 (offset 0, len 16)
// CHECK:       aie.use_lock({{.*link1_link_prod_lock_0.*}}, AcquireGreaterEqual, 1)
// CHECK:       aie.dma_bd({{.*}}, 0, 16)
// CHECK:       aie.use_lock({{.*link1_link_cons_lock_0.*}}, Release, 1)
// CHECK:       aie.next_bd

// S2MM ingest: slice 1 of buff_0 (offset 16, len 20)
// CHECK:       aie.use_lock({{.*link1_link_prod_lock_1.*}}, AcquireGreaterEqual, 1)
// CHECK:       aie.dma_bd({{.*}}, 16, 20)
// CHECK:       aie.use_lock({{.*link1_link_cons_lock_1.*}}, Release, 1)
// CHECK:       aie.next_bd

// S2MM ingest: slice 2 of buff_0 (offset 36, len 12)
// CHECK:       aie.use_lock({{.*link1_link_prod_lock_2.*}}, AcquireGreaterEqual, 1)
// CHECK:       aie.dma_bd({{.*}}, 36, 12)
// CHECK:       aie.use_lock({{.*link1_link_cons_lock_2.*}}, Release, 1)
// CHECK:       aie.next_bd

// MM2S ch 0: independent lock pair for slice 0
// CHECK:       aie.dma_start(MM2S, 0
// CHECK:       aie.use_lock({{.*link1_link_cons_lock_0.*}}, AcquireGreaterEqual, 1)
// CHECK:       aie.dma_bd({{.*}}, 0, 16)
// CHECK:       aie.use_lock({{.*link1_link_prod_lock_0.*}}, Release, 1)

// MM2S ch 1: independent lock pair for slice 1
// CHECK:       aie.dma_start(MM2S, 1
// CHECK:       aie.use_lock({{.*link1_link_cons_lock_1.*}}, AcquireGreaterEqual, 1)
// CHECK:       aie.dma_bd({{.*}}, 16, 20)
// CHECK:       aie.use_lock({{.*link1_link_prod_lock_1.*}}, Release, 1)

// MM2S ch 2: independent lock pair for slice 2
// CHECK:       aie.dma_start(MM2S, 2
// CHECK:       aie.use_lock({{.*link1_link_cons_lock_2.*}}, AcquireGreaterEqual, 1)
// CHECK:       aie.dma_bd({{.*}}, 36, 12)
// CHECK:       aie.use_lock({{.*link1_link_prod_lock_2.*}}, Release, 1)

// CHECK:       aie.end
// CHECK:     }
// CHECK-NOT: conduit.create
// CHECK-NOT: conduit.scatter

module @link_distribute_offsets {
  aie.device(xcve2302) {
    %tile20 = aie.tile(2, 0)
    %tile21 = aie.tile(2, 1)
    %tile22 = aie.tile(2, 2)
    %tile23 = aie.tile(2, 3)
    %tile33 = aie.tile(3, 3)

    // Ingress: shim → MemTile
    conduit.create @link1 {element_type = memref<48xi32>, depth = 2 : i64}
    // Egress: MemTile → compute tiles (3 destinations with byte offsets)
    conduit.create @link2 {element_type = memref<4x4xi32>, depth = 2 : i64}
    conduit.create @link3 {element_type = memref<20xi32>, depth = 2 : i64}
    conduit.create @link4 {element_type = memref<12xi32>, depth = 2 : i64}

    // Distribute link: MemTile(2,1) splits link1 into link2/link3/link4
    conduit.scatter{src = @link1, dsts = [@link2, @link3, @link4], memtile = %tile21, offsets = [0, 16, 36]}

    // Shim producer allocation for ingress channel.
    aie.shim_dma_allocation @link1_shim_alloc(%tile20, MM2S, 0) {conduit_channel = @link1}

    // Consumer cores — structural info for tile inference.
    %core_2_2 = aie.core(%tile22) {
      %0 = conduit.acquire {count = 1 : i64, name = @link2,
                            port = #conduit.port<Consume>} : <memref<4x4xi32>>
      conduit.release %0 {count = 1 : i64,
                          port = #conduit.port<Consume>} : <memref<4x4xi32>>
      aie.end
    }
    %core_2_3 = aie.core(%tile23) {
      %0 = conduit.acquire {count = 1 : i64, name = @link3,
                            port = #conduit.port<Consume>} : <memref<20xi32>>
      conduit.release %0 {count = 1 : i64,
                          port = #conduit.port<Consume>} : <memref<20xi32>>
      aie.end
    }
    %core_3_3 = aie.core(%tile33) {
      %0 = conduit.acquire {count = 1 : i64, name = @link4,
                            port = #conduit.port<Consume>} : <memref<12xi32>>
      conduit.release %0 {count = 1 : i64,
                          port = #conduit.port<Consume>} : <memref<12xi32>>
      aie.end
    }
  }
}
