// RUN: aie-opt --conduit-to-dma %s | FileCheck %s
//
// Pass C Phase 5a: tile coordinate inference from IR structure (aie.core walk).
//
// Tests that Pass C correctly infers compute tile coordinates by walking
// conduit.acquire ops inside aie.core bodies — WITHOUT consumer_tiles attr.
//
// Channel @fifo_shim_alloc: the post-Pass-A name (Pass A renames @fifo →
// @fifo_shim_alloc and emits aie.shim_dma_allocation @fifo_shim_alloc).
// In this hand-written test we use the post-rewrite name directly, which
// means the conduit.create sym and the shim_dma_allocation sym are different:
//   conduit.create @fifo_shim_alloc  (the conduit channel)
//   aie.shim_dma_allocation @fifo_shim_alloc_dma  (distinct sym, same channel)
// and the producer_tile attr is used for the shim side until DEFERRED-13
// (conduit_channel back-ref attr) is implemented.
//
// What this test specifically exercises:
//   The consumer tile (0,2) is inferred SOLELY from the acquire{Consume} walk.
//   No consumer_tiles attr is present on conduit.create.
//
// Expected: Pass C allocates buffer + locks on tile(0,2) from IR walk alone.

// CHECK-LABEL: module @tile_inference_ir_walk
// CHECK:   aie.device(npu1_1col) {
// --- Buffer and locks on tile(0,2) — consumer inferred from acquire walk ---
// CHECK:     %[[BUFF:.*]] = aie.buffer(%{{.*}}tile_0_2)
// CHECK-SAME:   sym_name = "fifo_shim_alloc_cons_buff_0"
// CHECK:     %[[PROD_LOCK:.*]] = aie.lock(%{{.*}}tile_0_2
// CHECK-SAME:   init = 1
// CHECK-SAME:   sym_name = "fifo_shim_alloc_cons_prod_lock_0"
// CHECK:     %[[CONS_LOCK:.*]] = aie.lock(%{{.*}}tile_0_2
// CHECK-SAME:   init = 0
// CHECK-SAME:   sym_name = "fifo_shim_alloc_cons_cons_lock_0"
// CHECK:     aie.core(%{{.*}}tile_0_2) {
// CHECK:       aie.use_lock(%[[CONS_LOCK]], AcquireGreaterEqual, 1)
// CHECK:       aie.use_lock(%[[PROD_LOCK]], Release, 1)
// CHECK:     aie.mem(%{{.*}}tile_0_2) {
// CHECK:       aie.dma_start(S2MM, 0
// CHECK:       aie.dma_bd(%[[BUFF]]
// CHECK-NOT: conduit.create
// CHECK-NOT: conduit.acquire
// CHECK-NOT: conduit.release

module @tile_inference_ir_walk {
  aie.device(npu1_1col) {
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)

    // No consumer_tiles attr: consumer tile(0,2) inferred from acquire walk.
    // producer_tile attr required until DEFERRED-13 provides a structural
    // shim back-reference (conduit_channel attr on aie.shim_dma_allocation
    // or a new conduit.shim_endpoint op).
    conduit.create @fifo_shim_alloc {slot_elems = 8 : i64,
                    element_type = memref<8xi32>,
                    depth = 1 : i64}

    // acquire{port=Consume} in core(0,2) → Pass C infers tile(0,2) as consumer.
    %core_0_2 = aie.core(%tile_0_2) {
      %w = conduit.acquire {name = @fifo_shim_alloc, count = 1 : i64,
                            port = #conduit.port<Consume>}
               : !conduit.window<memref<8xi32>>
      conduit.release %w {count = 1 : i64, port = #conduit.port<Consume>}
          : !conduit.window<memref<8xi32>>
      aie.end
    }
  }
}
