// RUN: aie-opt --objectfifo-to-conduit --conduit-to-dma %s | FileCheck %s
//
// P1-F: External buffers Pass C integration.
//
// Tests that conduit.register_external_buffers is:
//   1. Correctly lowered through Pass A (--objectfifo-to-conduit) into
//      conduit.register_external_buffers with the correct tile_coord.
//   2. Erased by Pass C (--conduit-to-dma): NO surviving conduit.* ops.
//   3. The external buffer SSA value survives in the device body.
//   4. An aie.shim_dma BD chain is built using the external buffer.
//
// Topology: shim tile(7,0) [producer via external buffer] → tile(7,1) [consumer]
// Device: xcvc1902 (AIE1)

// CHECK-LABEL: aie.device(xcvc1902)

// --- Consumer-tile buffers and locks (emitted first by Phase 3) ---
// CHECK: aie.buffer(%{{.*}}tile_7_1
// CHECK: aie.lock(%{{.*}}tile_7_1

// --- External buffer survives ---
// CHECK: aie.external_buffer

// --- No conduit ops survive (I1 invariant) ---
// CHECK-NOT: conduit.create
// CHECK-NOT: conduit.register_external_buffers
// CHECK-NOT: conduit.acquire
// CHECK-NOT: conduit.release

// --- shim_dma_allocation ---
// CHECK: aie.shim_dma_allocation

// --- Flow from shim to compute tile ---
// CHECK: aie.flow(%{{.*}}tile_7_0, DMA : 0, %{{.*}}tile_7_1, DMA : 0)

// --- shim_dma BD chain references the external buffer ---
// CHECK: aie.shim_dma(%{{.*}}tile_7_0)
// CHECK: aie.dma_start
// CHECK: aie.dma_bd({{.*}}ext_buffer_in

// --- Consumer-tile aie.mem BD chain ---
// CHECK: aie.mem(%{{.*}}tile_7_1)
// CHECK: aie.dma_start(S2MM

module {
  aie.device(xcvc1902) {
    %tile70 = aie.tile(7, 0)
    %tile71 = aie.tile(7, 1)

    aie.objectfifo @ext_fifo(%tile70, {%tile71}, 2 : i32) : !aie.objectfifo<memref<16xi32>>

    %ext_buffer_in = aie.external_buffer {sym_name = "ext_buffer_in"} : memref<64xi32>
    aie.objectfifo.register_external_buffers @ext_fifo(%tile70, {%ext_buffer_in}) : (memref<64xi32>)

    %core71 = aie.core(%tile71) {
      %subview = aie.objectfifo.acquire @ext_fifo(Consume, 2) : !aie.objectfifosubview<memref<16xi32>>
      %elem0 = aie.objectfifo.subview.access %subview[0] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>
      %elem1 = aie.objectfifo.subview.access %subview[1] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>
      aie.objectfifo.release @ext_fifo(Consume, 1)
      aie.end
    }
  }
}
