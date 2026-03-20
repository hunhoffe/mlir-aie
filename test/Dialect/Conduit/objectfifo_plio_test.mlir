// RUN: aie-opt --objectfifo-to-conduit --conduit-to-dma %s | FileCheck %s

// Test PLIO objectfifo support: 3 objectfifos with plio=true on xcve2302.
// - of_0: shim→compute (PLIO producer)
// - of_1: compute→shim (PLIO consumer, single consumer)
// - of_2: compute→{shim, compute} (PLIO consumer + broadcast to compute tile)
//
// Conduit must:
// 1. Emit aie.shim_dma_allocation with {plio = true}
// 2. Use WireBundle::PLIO in flows for shim-side endpoints
// 3. Allocate distinct S2MM channels per shim consumer (of_1 → S2MM:0, of_2 → S2MM:1)
// 4. Use indexed lock names for multi-consumer conduits (_cons_0, _cons_1)
// 5. Share MM2S channel for broadcast (of_2 → both DMA:1 flows from tile_2_2)
// 6. Allocate producer-side buffers on tile_2_2 for of_2 (non-shared-mem path)

// CHECK: aie.shim_dma_allocation @of_0_shim_alloc(%{{.*}}, MM2S, 0) {plio = true}
// CHECK: aie.shim_dma_allocation @of_1_shim_alloc(%{{.*}}, S2MM, 0) {plio = true}
// CHECK: aie.shim_dma_allocation @of_2_shim_alloc(%{{.*}}, S2MM, 1) {plio = true}

// CHECK: aie.flow(%{{.*}}, PLIO : 0, %{{.*}}, DMA : 0)
// CHECK: aie.flow(%{{.*}}, DMA : 0, %{{.*}}, PLIO : 0)
// CHECK: aie.flow(%{{.*}}, DMA : 1, %{{.*}}, PLIO : 1)
// CHECK: aie.flow(%{{.*}}, DMA : 1, %{{.*}}, DMA : 0)

// Verify producer-side DMA on tile_2_2 for of_2 (MM2S channel for broadcast)
// CHECK: aie.mem(%{{.*}})
// CHECK:   aie.dma_start(MM2S, 1

module @plio {
    aie.device(xcve2302) {
        %tile20 = aie.tile(2, 0)
        %tile22 = aie.tile(2, 2)
        %tile23 = aie.tile(2, 3)

        aie.objectfifo @of_0 (%tile20, {%tile22}, 2 : i32) {plio = true} : !aie.objectfifo<memref<64xi16>>
        aie.objectfifo @of_1 (%tile22, {%tile20}, 2 : i32) {plio = true} : !aie.objectfifo<memref<64xi16>>
        aie.objectfifo @of_2 (%tile22, {%tile20, %tile23}, 2 : i32) {plio = true} : !aie.objectfifo<memref<64xi16>>
    }
}
