// RUN: aie-opt --objectfifo-to-conduit --conduit-to-dma %s | FileCheck %s
//
// Pass A + Pass C end-to-end test: 1→3 distribute link with byte offsets.
// Input: same as objectfifo_to_conduit_distribute.mlir
//
// After --objectfifo-to-conduit --conduit-to-dma the module should contain:
//   - aie.buffer ops for each conduit (depth-many per consumer)
//   - aie.lock pairs for each conduit
//   - aie.memtile_dma for the distribute MemTile DMA BD chain
//   - aie.flow ops connecting tiles
//
// Resource comparison target (from --aie-objectFifo-stateful-transform):
//   aie.buffer:  8  (2 per conduit × 4 conduits)
//   aie.lock:   10  (link1: 2+2+2 for 3 dst-slices; link2,3,4: 1 pair each)
//   aie.flow:    4  (shim→memtile + memtile→tile22, 23, 33)
//   aie.dma_bd:  ≥4 (at least one per BD chain)

// CHECK-LABEL: module @link_distribute_offsets
// CHECK:   aie.device(xcve2302) {
// CHECK:     aie.buffer({{.*}})
// CHECK:     aie.lock({{.*}})
// CHECK:     aie.memtile_dma({{.*}}) {
// CHECK:       aie.dma_start(S2MM
// CHECK:       aie.dma_bd(
// CHECK:       aie.next_bd
// CHECK:       aie.end
// CHECK:     }
// CHECK-NOT: conduit.create
// CHECK-NOT: conduit.objectfifo_link

module @link_distribute_offsets {
  aie.device(xcve2302) {
    %tile20 = aie.tile(2, 0)
    %tile21 = aie.tile(2, 1)
    %tile22 = aie.tile(2, 2)
    %tile23 = aie.tile(2, 3)
    %tile33 = aie.tile(3, 3)

    aie.objectfifo @link1 (%tile20, {%tile21}, 2 : i32) : !aie.objectfifo<memref<48xi32>>
    aie.objectfifo @link2 (%tile21, {%tile22}, 2 : i32) : !aie.objectfifo<memref<4x4xi32>>
    aie.objectfifo @link3 (%tile21, {%tile23}, 2 : i32) : !aie.objectfifo<memref<20xi32>>
    aie.objectfifo @link4 (%tile21, {%tile33}, 2 : i32) : !aie.objectfifo<memref<12xi32>>

    aie.objectfifo.link [@link1] -> [@link2, @link3, @link4] ([][0, 16, 36])
  }
}
