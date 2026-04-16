// RUN: aie-opt -split-input-file -verify-diagnostics %s
//
// Tests for new routing_mode values added in Sprint 6:
//   shared_memory — force shared memory path; error if tiles not adjacent
//   dma           — force DMA path; --conduit-infer-modes picks circuit or packet
//
// Note: plio is no longer an attribute on conduit.create. It is set on
// aie.shim_dma_allocation by Pass A and read by Pass C from there.

// -----

// routing_mode = shared_memory is a valid enum value — parses and round-trips.

// CHECK-LABEL: aie.device(npu1)
// CHECK: conduit.create @shared_mem_chan
// CHECK-SAME: #conduit.routing_mode<shared_memory>
aie.device(npu1) {
conduit.create @shared_mem_chan {element_type = memref<16xi32>, depth = 2 : i64,
                                  routing_mode = #conduit.routing_mode<shared_memory>}
}

// -----

// routing_mode = dma is a valid enum value — parses and round-trips.

// CHECK-LABEL: aie.device(npu1)
// CHECK: conduit.create @dma_chan
// CHECK-SAME: #conduit.routing_mode<dma>
aie.device(npu1) {
conduit.create @dma_chan {element_type = memref<16xi32>, depth = 2 : i64,
                           routing_mode = #conduit.routing_mode<dma>}
}

// -----

// routing_mode = circuit forces DMA even for adjacent tiles (replaces viaDMA).

// CHECK-LABEL: aie.device(npu1)
// CHECK: conduit.create @circuit_chan
// CHECK-SAME: #conduit.routing_mode<circuit>
aie.device(npu1) {
conduit.create @circuit_chan {element_type = memref<16xi32>, depth = 1 : i64,
                               routing_mode = #conduit.routing_mode<circuit>}
}

// -----

// sync_mode = none suppresses lock emission (replaces disable_synchronization).

// CHECK-LABEL: aie.device(npu1)
// CHECK: conduit.create @no_sync_chan
// CHECK-SAME: #conduit.sync_mode<none>
aie.device(npu1) {
conduit.create @no_sync_chan {element_type = memref<8xi32>, depth = 1 : i64,
                               sync_mode = #conduit.sync_mode<none>}
}
