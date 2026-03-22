// RUN: aie-opt --objectfifo-to-conduit --conduit-to-dma %s | FileCheck %s
//
// Full 512×512 row element type test: memref<4096xi8>.
//
// Verifies that large element types (a full image row at 512 bytes × 8 channels
// = 4096 bytes) lower correctly through Pass A + Pass C.
//
// Topology:
//   shim(0,0) → MemTile(0,1) → tile(0,2) → MemTile(0,1) → shim(0,0)
//
// The input path (shim → memtile → compute) and output path (compute → memtile → shim)
// exercise the full relay chain with a large element type.
//
// Expected:
//   aie.buffer with memref<4096xi8> on both MemTile and compute tile
//   aie.flow for each hop in the relay chain

// CHECK-LABEL: module @full_res_elements
// CHECK:   aie.device(npu1_1col) {

// --- Large buffer allocations ---
// CHECK:     aie.buffer({{.*}}) {{.*}} memref<4096xi8>

// --- MemTile DMA for relay ---
// CHECK:     aie.memtile_dma(%{{.*}}mem_tile_0_1) {
// CHECK:       aie.dma_start(S2MM
// CHECK:       aie.dma_bd
// CHECK:     }

// --- Compute tile DMA ---
// CHECK:     aie.mem(%{{.*}}tile_0_2) {
// CHECK:       aie.dma_start(S2MM
// CHECK:     }

// --- No residual Conduit ops ---
// CHECK-NOT: conduit.create
// CHECK-NOT: conduit.acquire
// CHECK-NOT: conduit.release

module @full_res_elements {
  aie.device(npu1_1col) {
    func.func @process_row(%in: memref<4096xi8>, %out: memref<4096xi8>) -> () {
      return
    }

    %tile_0_0 = aie.tile(0, 0)
    %mem_tile_0_1 = aie.tile(0, 1)
    %tile_0_2 = aie.tile(0, 2)

    // Input: shim → MemTile (relay) → compute tile
    aie.objectfifo @input_row (%tile_0_0, {%mem_tile_0_1}, 2 : i32) : !aie.objectfifo<memref<4096xi8>>
    aie.objectfifo @input_relay (%mem_tile_0_1, {%tile_0_2}, 2 : i32) : !aie.objectfifo<memref<4096xi8>>
    aie.objectfifo.link [@input_row] -> [@input_relay] ([][])

    // Output: compute tile → MemTile (relay) → shim
    aie.objectfifo @output_relay (%tile_0_2, {%mem_tile_0_1}, 2 : i32) : !aie.objectfifo<memref<4096xi8>>
    aie.objectfifo @output_row (%mem_tile_0_1, {%tile_0_0}, 2 : i32) : !aie.objectfifo<memref<4096xi8>>
    aie.objectfifo.link [@output_relay] -> [@output_row] ([][])

    %core_0_2 = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c8 = arith.constant 8 : index

      scf.for %arg0 = %c0 to %c8 step %c1 {
        %in_view = aie.objectfifo.acquire @input_relay(Consume, 1) : !aie.objectfifosubview<memref<4096xi8>>
        %in_buf = aie.objectfifo.subview.access %in_view[0] : !aie.objectfifosubview<memref<4096xi8>> -> memref<4096xi8>

        %out_view = aie.objectfifo.acquire @output_relay(Produce, 1) : !aie.objectfifosubview<memref<4096xi8>>
        %out_buf = aie.objectfifo.subview.access %out_view[0] : !aie.objectfifosubview<memref<4096xi8>> -> memref<4096xi8>

        func.call @process_row(%in_buf, %out_buf) : (memref<4096xi8>, memref<4096xi8>) -> ()

        aie.objectfifo.release @input_relay(Consume, 1)
        aie.objectfifo.release @output_relay(Produce, 1)
      }

      aie.end
    } {dynamic_objfifo_lowering = true}
  }
}
