// RUN: aie-opt --conduit-to-dma %s | FileCheck %s
//
// Pass C (--conduit-to-dma) cascade mode test.
//
// Verifies that a conduit.create with routing_mode = #conduit.routing_mode<cascade> is lowered to:
//   - aie.cascade_flow(src_tile, dst_tile) in the device body
//   - aie.put_cascade(value : type) in the producer core body
//   - aie.get_cascade() : type in the consumer core body
//
// No aie.buffer, aie.lock, aie.dma_bd, or aie.flow should be emitted for
// the cascade conduit.
//
// NOTE: aie.cascade_flow requires --aie-lower-cascade-flows for full lowering.
// This test checks only the Pass C output.

// CHECK-LABEL: aie.device
// CASCADE VALUE: aie.put_cascade / aie.get_cascade in core bodies (emitted first).
// CHECK: aie.put_cascade
// CHECK: aie.get_cascade
// CASCADE FLOW: aie.cascade_flow appears after cores in device body.
// CHECK: aie.cascade_flow
// No DMA infrastructure for cascade conduit.
// CHECK-NOT: aie.flow
// CHECK-NOT: aie.lock
// CHECK-NOT: aie.buffer
// CHECK-NOT: aie.dma_start

module {
  aie.device(npu1) {
    %tile03 = aie.tile(0, 3)
    %tile13 = aie.tile(1, 3)

    // Cascade conduit: producer is tile(0,3), consumer is tile(1,3).
    // No buffers, no locks, no DMA — just a cascade_flow connection.
    conduit.create @cas {capacity = 1 : i64,
                    producer_tile = array<i64: 0, 3>,
                    consumer_tiles = array<i64: 1, 3>,
                    depth = 1 : i64,
                    routing_mode = #conduit.routing_mode<cascade>}

    // Producer core: computes a vector and puts it on the cascade stream.
    aie.core(%tile03) {
      %v = arith.constant dense<42> : vector<16xi32>
      aie.put_cascade(%v : vector<16xi32>)
      aie.end
    }

    // Consumer core: reads the cascade value.
    aie.core(%tile13) {
      %r = aie.get_cascade() : vector<16xi32>
      aie.end
    }
  }
}
