// RUN: aie-opt --conduit-to-dma %s | FileCheck %s
//
// Pass C regression test: BD transfer length for Tier 3 channels.
//
// Without the numElems fix in ConduitToDMALink.cpp, the BD transfer length
// is computed as capacity/depth = 1/1 = 1. The fix checks info.numElems
// first (populated from put/get_memref_async num_elems attributes) and uses
// that as the BD length instead.
//
// Setup:
//   conduit.create with capacity=1, depth=1, element_type=memref<128xi32>
//   put_memref_async/get_memref_async with num_elems=128
//
// Expected: aie.dma_bd emits length 128 (from num_elems), not 1.
//
// This test exercises Pass C directly on hand-written Conduit IR.

// CHECK-LABEL: module @tier3_bd_length
// CHECK:   aie.device(npu1_1col) {
// CHECK:     aie.tile(0, 0)
// CHECK:     aie.tile(0, 2)
// CHECK:     %[[BUFF0:.*]] = aie.buffer(%{{.*}}tile_0_2)
// CHECK-SAME:   sym_name = "t3_bd_cons_buff_0"

// --- BD chain on consumer tile: length must be 128 (not 1) ---
// CHECK:     aie.mem(%{{.*}}tile_0_2) {
// CHECK:       aie.dma_start(S2MM
// CHECK:       aie.dma_bd(%[[BUFF0]] : memref<128xi32>, 0, 128)
// CHECK-NOT:   aie.dma_bd({{.*}}, 0, 1)

// --- No residual Conduit ops ---
// CHECK-NOT: conduit.create
// CHECK-NOT: conduit.put_memref_async
// CHECK-NOT: conduit.get_memref_async

module @tier3_bd_length {
  aie.device(npu1_1col) {
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)

    // capacity=1 (slot count from air.channel), depth=1 (single-buffered).
    // element_type=memref<128xi32> determines the buffer allocation size.
    // Without the numElems fix, perBufLen = capacity/depth = 1/1 = 1.
    conduit.create @t3_bd {capacity = 1 : i64,
                    producer_tile = array<i64: 0, 0>,
                    consumer_tiles = array<i64: 0, 2>,
                    element_type = memref<128xi32>,
                    depth = 1 : i64}

    %core_0_2 = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c4 = arith.constant 4 : index

      scf.for %arg0 = %c0 to %c4 step %c1 {
        // Tier 3 DMA send: num_elems=128 is the actual transfer size.
        %put_tok = conduit.put_memref_async {name = @t3_bd,
                       num_elems = 128 : i64,
                       offsets = array<i64: 0>,
                       sizes = array<i64: 128>,
                       strides = array<i64: 1>}
                       : !conduit.dma.token
        conduit.wait %put_tok : !conduit.dma.token

        // Tier 3 DMA receive: num_elems=128 is the actual transfer size.
        %get_tok = conduit.get_memref_async {name = @t3_bd,
                       num_elems = 128 : i64,
                       offsets = array<i64: 0>,
                       sizes = array<i64: 128>,
                       strides = array<i64: 1>}
                       : !conduit.dma.token
        conduit.wait %get_tok : !conduit.dma.token
      }
      aie.end
    } {dynamic_objfifo_lowering = true}
  }
}
