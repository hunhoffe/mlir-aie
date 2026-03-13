// RUN: aie-opt --objectfifo-to-conduit --conduit-to-dma %s | FileCheck %s
//
// Pass A + Pass C end-to-end test: shared memory path for adjacent compute tiles.
//
// tile(2,2) [producer] and tile(2,3) [consumer] are N/S adjacent compute tiles
// on npu1.  For AIE2, isLegalMemAffinity(2,2,2,3) returns true (IsMemNorth =
// isNorth(2,2,2,3) = true), so the producer and consumer share a memory bank
// without any DMA.
//
// The stateful transform (--aie-objectFifo-stateful-transform) detects this
// adjacency via requiresDMAs() → isSharedMemory() and:
//   - allocates buffers and locks on the PRODUCER tile (share_direction != 1)
//   - emits NO aie.flow, NO aie.dma_bd, NO aie.mem
//
// Pass C must reproduce the same shared memory lowering:
//   aie.buffer  on tile(2,2)  [producer tile]
//   aie.lock    on tile(2,2)  [producer tile]
//   NO aie.flow
//   NO aie.dma_bd
//   NO aie.mem
//
// Ground truth (from --aie-objectFifo-stateful-transform):
//
//   %tile_2_2 = aie.tile(2, 2)
//   %tile_2_3 = aie.tile(2, 3)
//   %buf = aie.buffer(%tile_2_2) : memref<16xi32>   -- producer tile
//   %prod_lock = aie.lock(%tile_2_2, 0) {init = 1}  -- producer tile
//   %cons_lock = aie.lock(%tile_2_2, 1) {init = 0}  -- producer tile
//   (no aie.flow, no aie.dma_bd, no aie.mem)
//
// Core body: use_lock on the PRODUCER-tile locks (accessible via shared memory).
//
// CHECK-LABEL: module @shared_memory_adjacent_tiles
// CHECK:   aie.device(npu1) {

// --- Tile declarations ---
// CHECK:     %[[TILE_2_2:.*]] = aie.tile(2, 2)
// CHECK:     %[[TILE_2_3:.*]] = aie.tile(2, 3)

// --- Buffer MUST be on the PRODUCER tile (tile_2_2), not tile_2_3 ---
// CHECK:     aie.buffer(%[[TILE_2_2]])
// CHECK-SAME:   sym_name = "shared_fifo_buff_0"

// --- Locks MUST be on the PRODUCER tile (tile_2_2) ---
// CHECK:     %[[PROD_LOCK:.*]] = aie.lock(%[[TILE_2_2]], {{[0-9]+}}) {init = 1
// CHECK-SAME:   sym_name = "shared_fifo_prod_lock_0"
// CHECK:     %[[CONS_LOCK:.*]] = aie.lock(%[[TILE_2_2]], {{[0-9]+}}) {init = 0
// CHECK-SAME:   sym_name = "shared_fifo_cons_lock_0"

// --- Consumer core: acquire/release still lowers to aie.use_lock ---
// CHECK:     aie.core(%[[TILE_2_3]]) {
// CHECK:       aie.use_lock(%[[CONS_LOCK]], AcquireGreaterEqual, 1)
// CHECK:       aie.use_lock(%[[PROD_LOCK]], Release, 1)
// CHECK:     }

// --- No DMA infrastructure (shared memory bypasses the DMA engine) ---
// CHECK-NOT: aie.dma_bd
// CHECK-NOT: aie.flow
// CHECK-NOT: aie.mem
// CHECK-NOT: aie.shim_dma_allocation

// --- No residual Conduit ops ---
// CHECK-NOT: conduit.create
// CHECK-NOT: conduit.acquire
// CHECK-NOT: conduit.release

module @shared_memory_adjacent_tiles {
  aie.device(npu1) {
    func.func @process(%buf: memref<16xi32>) -> () {
      return
    }

    // tile(2,2) and tile(2,3): same column, rows 2 and 3 → adjacent compute
    // tiles on npu1.  isLegalMemAffinity(2,2,2,3) = true → shared memory.
    %tile_2_2 = aie.tile(2, 2)
    %tile_2_3 = aie.tile(2, 3)

    // depth=1, single consumer, no DMA dimensions.
    // The stateful transform detects shared memory and skips DMA setup.
    aie.objectfifo @shared_fifo(%tile_2_2, {%tile_2_3}, 1 : i32) : !aie.objectfifo<memref<16xi32>>

    %core_2_3 = aie.core(%tile_2_3) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c4 = arith.constant 4 : index

      scf.for %arg0 = %c0 to %c4 step %c1 {
        %0 = aie.objectfifo.acquire @shared_fifo(Consume, 1) : !aie.objectfifosubview<memref<16xi32>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>
        func.call @process(%1) : (memref<16xi32>) -> ()
        aie.objectfifo.release @shared_fifo(Consume, 1)
      }

      aie.end
    } {dynamic_objfifo_lowering = true}
  }
}
