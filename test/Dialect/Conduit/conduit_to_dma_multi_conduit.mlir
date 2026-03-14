// RUN: aie-opt --objectfifo-to-conduit --conduit-to-dma %s | FileCheck %s
//
// Pass A + Pass C end-to-end test: two conduits on the same compute tile.
//
// Verifies that the per-tile lock ID counter (lockIdCounter, keyed by tile
// SSA value) correctly assigns non-colliding lock IDs when multiple conduits
// land on the same tile.
//
// Two independent fifos both landing on tile_0_2:
//   fifo_a: shim(0,0) → tile(0,2), depth=1, memref<8xi32>
//   fifo_b: shim(0,0) → tile(0,2), depth=1, memref<16xi32>
//
// Expected resources:
//   aie.buffer on tile_0_2: 2  (fifo_a_cons_buff_0, fifo_b_cons_buff_0)
//   aie.lock on tile_0_2:   4  (fifo_?_cons_prod_lock_0 + fifo_?_cons_cons_lock_0 × 2,
//                                with lock IDs 0,1,2,3 — no collision)
//   aie.lock on shim tile:  4  (prod+cons lock per fifo for DMA host-side sync)
//   aie.flow: 2  (one per fifo, shim → tile_0_2)
//   aie.shim_dma_allocation: 2
//   aie.mem: 1  (single per tile with S2MM BD ring per fifo)
//
// The critical invariant: lock IDs on tile_0_2 are 0,1,2,3 (no two locks share
// the same ID on the same tile).

// CHECK-LABEL: module @two_conduits_same_tile
// CHECK:   aie.device(npu1_1col) {
// CHECK:     %{{.*}}tile_0_0 = aie.tile(0, 0)
// CHECK:     %{{.*}}tile_0_2 = aie.tile(0, 2)

// --- Two buffers on tile_0_2 (one per conduit, distinct element types) ---
// CHECK:     aie.buffer(%{{.*}}tile_0_2)
// CHECK:     aie.lock(%{{.*}}tile_0_2, {{[0-9]+}}) {init = 1
// CHECK:     aie.lock(%{{.*}}tile_0_2, {{[0-9]+}}) {init = 0
// CHECK:     aie.buffer(%{{.*}}tile_0_2)
// CHECK:     aie.lock(%{{.*}}tile_0_2, {{[0-9]+}}) {init = 1
// CHECK:     aie.lock(%{{.*}}tile_0_2, {{[0-9]+}}) {init = 0

// --- Core: acquire+release for fifo_a, then fifo_b ---
// CHECK:     aie.core(%{{.*}}tile_0_2) {
// CHECK:       scf.for
// CHECK:         aie.use_lock({{.*}}, AcquireGreaterEqual, 1)
// CHECK:         func.call @process_a
// CHECK:         aie.use_lock({{.*}}, Release, 1)
// CHECK:         aie.use_lock({{.*}}, AcquireGreaterEqual, 1)
// CHECK:         func.call @process_b
// CHECK:         aie.use_lock({{.*}}, Release, 1)
// CHECK:     }

// --- Shim allocations (Pass A batches all shim_dma_allocation before locks) ---
// CHECK:     aie.shim_dma_allocation
// CHECK:     aie.shim_dma_allocation

// --- Shim locks + flow for fifo_a ---
// CHECK:     aie.lock(%{{.*}}tile_0_0,
// CHECK:     aie.lock(%{{.*}}tile_0_0,
// CHECK:     aie.flow(%{{.*}}tile_0_0, DMA :

// --- Shim locks + flow for fifo_b ---
// CHECK:     aie.lock(%{{.*}}tile_0_0,
// CHECK:     aie.lock(%{{.*}}tile_0_0,
// CHECK:     aie.flow(%{{.*}}tile_0_0, DMA :

// --- Single aie.mem per tile with two S2MM channels (one per conduit) ---
// CHECK:     aie.mem(%{{.*}}tile_0_2) {
// CHECK:       aie.dma_start(S2MM, 0,
// CHECK:       aie.dma_bd(
// CHECK:       aie.dma_start(S2MM, 1,
// CHECK:       aie.dma_bd(

// --- No residual Conduit ops ---
// CHECK-NOT: conduit.create
// CHECK-NOT: conduit.acquire
// CHECK-NOT: conduit.release

module @two_conduits_same_tile {
  aie.device(npu1_1col) {
    func.func @process_a(%a: memref<8xi32>) -> () {
      return
    }
    func.func @process_b(%b: memref<16xi32>) -> () {
      return
    }

    %tile_0_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)

    // Two independent fifos landing on the same compute tile
    aie.objectfifo @fifo_a(%tile_0_0, {%tile_0_2}, 1 : i32) : !aie.objectfifo<memref<8xi32>>
    aie.objectfifo @fifo_b(%tile_0_0, {%tile_0_2}, 1 : i32) : !aie.objectfifo<memref<16xi32>>

    %core_0_2 = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c4 = arith.constant 4 : index

      scf.for %arg0 = %c0 to %c4 step %c1 {
        // Acquire both fifos independently — lock IDs must not collide
        %a0 = aie.objectfifo.acquire @fifo_a(Consume, 1) : !aie.objectfifosubview<memref<8xi32>>
        %a1 = aie.objectfifo.subview.access %a0[0] : !aie.objectfifosubview<memref<8xi32>> -> memref<8xi32>
        func.call @process_a(%a1) : (memref<8xi32>) -> ()
        aie.objectfifo.release @fifo_a(Consume, 1)

        %b0 = aie.objectfifo.acquire @fifo_b(Consume, 1) : !aie.objectfifosubview<memref<16xi32>>
        %b1 = aie.objectfifo.subview.access %b0[0] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>
        func.call @process_b(%b1) : (memref<16xi32>) -> ()
        aie.objectfifo.release @fifo_b(Consume, 1)
      }

      aie.end
    } {dynamic_objfifo_lowering = true}
  }
}
