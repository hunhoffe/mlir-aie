// RUN: aie-opt --conduit-materialize-buffers %s \
// RUN:   | FileCheck %s --check-prefix=MAT
// RUN: aie-opt --conduit-materialize-buffers --conduit-place-buffers %s \
// RUN:   | FileCheck %s --check-prefix=PLACE
// RUN: aie-opt --conduit-to-dma %s \
// RUN:   | FileCheck %s --check-prefix=DMA
//
// Phase 5b: --conduit-materialize-buffers emits aie.buffer ops for channels
// whose consumer tile is inferred from conduit.acquire{port=Consume} inside
// aie.core.
//
// --conduit-place-buffers then assigns mem_bank = i % 4.
//
// The full pipeline (--conduit-depth-promote → --conduit-materialize-buffers
// → --conduit-to-dma) must lower cleanly with no residual conduit ops.
//
// Setup: depth=1 (explicit); consumer inferred from acquire{Consume} in core(0,2).

// --- After --conduit-materialize-buffers ---
// MAT-LABEL: module @materialize_buffers
// MAT:   aie.device(npu1_1col) {
// MAT:     %[[BUFF0:.*]] = aie.buffer(%{{.*}}tile_0_2)
// MAT-SAME:   sym_name = "fifo_shim_alloc_cons_buff_0"
// MAT-NOT:   mem_bank
// MAT-NOT: conduit.register_buffers

// --- After --conduit-place-buffers (adds mem_bank = 0 for slot 0) ---
// PLACE-LABEL: module @materialize_buffers
// PLACE:   %[[BUFF0:.*]] = aie.buffer(%{{.*}}tile_0_2)
// PLACE-SAME:   mem_bank = 0 : i32

// --- After full pipeline: no residual conduit ops; BD chain uses pre-materialized buffer ---
// DMA-LABEL: module @materialize_buffers
// DMA-NOT: conduit.create
// DMA-NOT: conduit.acquire
// DMA-NOT: conduit.release
// DMA-NOT: conduit.register_buffers
// DMA:   %[[BUFF:.*]] = aie.buffer(%{{.*}}tile_0_2) {sym_name = "fifo_shim_alloc_cons_buff_0"}
// DMA:   aie.mem(%{{.*}}tile_0_2) {
// DMA:     aie.dma_start(S2MM, 0
// DMA:     aie.dma_bd(%[[BUFF]] : memref<8xi32>, 0, 8)

module @materialize_buffers {
  aie.device(npu1_1col) {
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)

    // depth = 1: explicit single-buffer depth.
    // No consumer_tiles attr: consumer tile inferred from acquire{Consume} walk.
    conduit.create @fifo_shim_alloc {                    element_type = memref<8xi32>,
                    depth = 1 : i64}

    %core_0_2 = aie.core(%tile_0_2) {
      // acquire{Consume} → tile(0,2) identified as consumer by materialize pass.
      %w = conduit.acquire {name = @fifo_shim_alloc, count = 1 : i64,
                            port = #conduit.port<Consume>}
               : !conduit.window<memref<8xi32>>
      conduit.release %w {count = 1 : i64, port = #conduit.port<Consume>}
          : !conduit.window<memref<8xi32>>
      aie.end
    }
  }
}
