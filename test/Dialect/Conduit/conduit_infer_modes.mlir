// RUN: aie-opt --conduit-infer-modes --split-input-file %s | FileCheck %s
//
// Tests for --conduit-infer-modes (P2-C).
//
// The pass walks conduit.create ops with routing_mode="any" and resolves them
// to "circuit" or "packet" using the R3 + Step 3.5 decision procedure:
//
//   R3a. Adjacent tiles (isLegalMemAffinity), no via_DMA → "circuit"
//        (shared-memory path; Pass C Phase 3c handles it, no DMA channel used).
//   R3b. Circuit DMA channel available on producer tile → "circuit".
//   Step 3.5. Circuit channels exhausted → "packet".
//   Already-resolved conduits (routing_mode != "any") and cascade conduits
//   are left untouched.

// -----

// Test 1: R3a — adjacent tiles → "circuit" (shared-memory path).
//
// tile(0,2) and tile(0,3) are vertically adjacent on npu1.
// isLegalMemAffinity(0,2,0,3) is true.
// Even though no DMA channel assignment happens here, the mode is "circuit"
// because Pass C Phase 3c will use shared memory.

// CHECK-LABEL: aie.device(npu1)
// CHECK: conduit.create @adj
// CHECK-SAME: #conduit.routing_mode<circuit>
module @test_adjacent_shared_mem {
  aie.device(npu1) {
    %t02 = aie.tile(0, 2)
    %t03 = aie.tile(0, 3)
    conduit.create @adj {slot_elems = 4 : i64,
                    producer_tile = array<i64: 0, 2>,
                    consumer_tiles = array<i64: 0, 3>,
                    element_type = memref<4xi32>,
                    depth = 1 : i64
                    }
  }
}

// -----

// Test 2: R3b — non-adjacent tiles, circuit DMA channel available → "circuit".
//
// tile(0,2) and tile(1,4) are not adjacent.
// One MM2S channel is available on tile(0,2) (AIE2 compute tile has 2).
// The conduit gets the first available circuit-mode channel.

// CHECK-LABEL: aie.device(npu1)
// CHECK: conduit.create @non_adj
// CHECK-SAME: #conduit.routing_mode<circuit>
module @test_non_adjacent_circuit {
  aie.device(npu1) {
    %t02 = aie.tile(0, 2)
    %t14 = aie.tile(1, 4)
    conduit.create @non_adj {slot_elems = 4 : i64,
                    producer_tile = array<i64: 0, 2>,
                    consumer_tiles = array<i64: 1, 4>,
                    element_type = memref<4xi32>,
                    depth = 1 : i64
                    }
  }
}

// -----

// Test 3: Already-resolved conduits are left unchanged.
//
// routing_mode="circuit" and routing_mode="packet" conduits are skipped.
// Only the "any" conduit is resolved.

// CHECK-LABEL: aie.device(npu1)
// CHECK: conduit.create @already_circuit
// CHECK-SAME: #conduit.routing_mode<circuit>
// CHECK: conduit.create @already_packet
// CHECK-SAME: routing_mode = #conduit.routing_mode<packet>
// CHECK: conduit.create @to_infer
// CHECK-SAME: #conduit.routing_mode<circuit>
module @test_already_resolved {
  aie.device(npu1) {
    %t02 = aie.tile(0, 2)
    %t03 = aie.tile(0, 3)
    %t04 = aie.tile(0, 4)
    %t14 = aie.tile(1, 4)
    conduit.create @already_circuit {slot_elems = 4 : i64,
                    producer_tile = array<i64: 0, 2>,
                    consumer_tiles = array<i64: 0, 3>,
                    element_type = memref<4xi32>,
                    depth = 1 : i64,
                    routing_mode = #conduit.routing_mode<circuit>}
    conduit.create @already_packet {slot_elems = 4 : i64,
                    producer_tile = array<i64: 0, 4>,
                    consumer_tiles = array<i64: 1, 4>,
                    element_type = memref<4xi32>,
                    depth = 1 : i64,
                    routing_mode = #conduit.routing_mode<packet>}
    conduit.create @to_infer {slot_elems = 4 : i64,
                    producer_tile = array<i64: 0, 2>,
                    consumer_tiles = array<i64: 1, 4>,
                    element_type = memref<4xi32>,
                    depth = 1 : i64
                    }
  }
}

// -----

// Test 4: Cascade conduits are never touched.
//
// routing_mode="cascade" conduits are skipped even if they exist alongside
// "any" conduits.  The cascade conduit retains its routing_mode.

// CHECK-LABEL: aie.device(npu1)
// CHECK: conduit.create @cas
// CHECK-SAME: routing_mode = #conduit.routing_mode<cascade>
// CHECK: conduit.create @dma_any
// CHECK-SAME: #conduit.routing_mode<circuit>
module @test_cascade_unchanged {
  aie.device(npu1) {
    %t02 = aie.tile(0, 2)
    %t03 = aie.tile(0, 3)
    %t14 = aie.tile(1, 4)
    conduit.create @cas {slot_elems = 1 : i64,
                    producer_tile = array<i64: 0, 2>,
                    consumer_tiles = array<i64: 0, 3>,
                    depth = 1 : i64,
                    routing_mode = #conduit.routing_mode<cascade>}
    conduit.create @dma_any {slot_elems = 4 : i64,
                    producer_tile = array<i64: 0, 2>,
                    consumer_tiles = array<i64: 1, 4>,
                    element_type = memref<4xi32>,
                    depth = 1 : i64
                    }
  }
}

// -----

// Test 5: Step 3.5 — circuit DMA exhausted → packet fallback.
//
// tile(0,2) has 2 MM2S channels (AIE2 npu1).
// Two circuit-mode conduits already occupy both channels.
// A third conduit with routing_mode="any" finds no circuit channel and falls
// back to "packet".

// CHECK-LABEL: aie.device(npu1)
// CHECK: conduit.create @c1
// CHECK-SAME: #conduit.routing_mode<circuit>
// CHECK: conduit.create @c2
// CHECK-SAME: #conduit.routing_mode<circuit>
// CHECK: conduit.create @c3_any
// CHECK-SAME: routing_mode = #conduit.routing_mode<packet>
module @test_packet_fallback {
  aie.device(npu1) {
    %t02 = aie.tile(0, 2)
    %t13 = aie.tile(1, 3)
    %t14 = aie.tile(1, 4)
    %t15 = aie.tile(1, 5)
    // Two circuit conduits consuming both MM2S channels on tile(0,2).
    conduit.create @c1 {slot_elems = 4 : i64,
                    producer_tile = array<i64: 0, 2>,
                    consumer_tiles = array<i64: 1, 3>,
                    element_type = memref<4xi32>,
                    depth = 1 : i64,
                    routing_mode = #conduit.routing_mode<circuit>}
    conduit.create @c2 {slot_elems = 4 : i64,
                    producer_tile = array<i64: 0, 2>,
                    consumer_tiles = array<i64: 1, 4>,
                    element_type = memref<4xi32>,
                    depth = 1 : i64,
                    routing_mode = #conduit.routing_mode<circuit>}
    // Third conduit: circuit exhausted → packet fallback.
    conduit.create @c3_any {slot_elems = 4 : i64,
                    producer_tile = array<i64: 0, 2>,
                    consumer_tiles = array<i64: 1, 5>,
                    element_type = memref<4xi32>,
                    depth = 1 : i64
                    }
  }
}

// -----

// Test 6: Two "any" conduits on the same tile, both fit in circuit mode.
//
// tile(0,2) has 2 MM2S channels.  Two "any" conduits consume them both
// and both get resolved to "circuit".

// CHECK-LABEL: aie.device(npu1)
// CHECK: conduit.create @a1
// CHECK-SAME: #conduit.routing_mode<circuit>
// CHECK: conduit.create @a2
// CHECK-SAME: #conduit.routing_mode<circuit>
module @test_two_any_both_circuit {
  aie.device(npu1) {
    %t02 = aie.tile(0, 2)
    %t13 = aie.tile(1, 3)
    %t14 = aie.tile(1, 4)
    conduit.create @a1 {slot_elems = 4 : i64,
                    producer_tile = array<i64: 0, 2>,
                    consumer_tiles = array<i64: 1, 3>,
                    element_type = memref<4xi32>,
                    depth = 1 : i64
                    }
    conduit.create @a2 {slot_elems = 4 : i64,
                    producer_tile = array<i64: 0, 2>,
                    consumer_tiles = array<i64: 1, 4>,
                    element_type = memref<4xi32>,
                    depth = 1 : i64
                    }
  }
}

// -----

// Test 7: via_DMA=true overrides R3a — adjacent tiles still use DMA channel.
//
// tile(0,2) and tile(0,3) are adjacent, but via_DMA=true forces DMA path.
// R3a is skipped; R3b assigns a circuit DMA channel.

// CHECK-LABEL: aie.device(npu1)
// CHECK: conduit.create @forced_dma
// CHECK-SAME: #conduit.routing_mode<circuit>
module @test_via_dma_override {
  aie.device(npu1) {
    %t02 = aie.tile(0, 2)
    %t03 = aie.tile(0, 3)
    conduit.create @forced_dma {slot_elems = 4 : i64,
                    producer_tile = array<i64: 0, 2>,
                    consumer_tiles = array<i64: 0, 3>,
                    element_type = memref<4xi32>,
                    depth = 1 : i64,
                    viaDMA = true
                    }
  }
}

// -----

// Test 8: R3a.5 — multi-consumer uniform conduit → packet (multicast).
//
// tile(0,2) produces to tile(1,3), tile(1,4), tile(1,5).
// consumer_dimensions absent → uniform.  Expect routing_mode=packet.
// Multicast costs 1 packet ID regardless of N consumers.

// CHECK-LABEL: aie.device(npu1)
// CHECK: conduit.create @bcast
// CHECK-SAME: routing_mode = #conduit.routing_mode<packet>
module @test_multicast_uniform {
  aie.device(npu1) {
    %t02 = aie.tile(0, 2)
    %t13 = aie.tile(1, 3)
    %t14 = aie.tile(1, 4)
    %t15 = aie.tile(1, 5)
    conduit.create @bcast {slot_elems = 4 : i64,
                    producer_tile = array<i64: 0, 2>,
                    consumer_tiles = array<i64: 1, 3, 1, 4, 1, 5>,
                    element_type = memref<4xi32>,
                    depth = 1 : i64
                    }
  }
}

// -----

// Test 9: R3a.5 — multi-consumer, consumer_dimensions present and all
// identical → packet (multicast).
//
// Same topology as Test 8 but with explicit consumer_dimensions.
// All three sub-arrays are identical → uniform = true → packet.

// CHECK-LABEL: aie.device(npu1)
// CHECK: conduit.create @bcast_dims_uniform
// CHECK-SAME: routing_mode = #conduit.routing_mode<packet>
module @test_multicast_uniform_dims {
  aie.device(npu1) {
    %t02 = aie.tile(0, 2)
    %t13 = aie.tile(1, 3)
    %t14 = aie.tile(1, 4)
    %t15 = aie.tile(1, 5)
    conduit.create @bcast_dims_uniform {slot_elems = 4 : i64,
                    producer_tile = array<i64: 0, 2>,
                    consumer_tiles = array<i64: 1, 3, 1, 4, 1, 5>,
                    element_type = memref<4xi32>,
                    depth = 1 : i64,
                    consumer_dimensions = #aie<bd_dim_layout_array_array[[<size = 1, stride = 2>], [<size = 1, stride = 2>], [<size = 1, stride = 2>]]>
                    }
  }
}

// -----

// Test 10: R3a.5 — multi-consumer, consumer_dimensions present and
// NON-uniform → falls through to R3b (circuit DMA available).
//
// Same topology as Test 8 but consumer_dimensions sub-arrays differ.
// uniform = false → R3a.5 does NOT fire → R3b assigns circuit
// (tile(0,2) has 2 MM2S channels, none consumed).

// CHECK-LABEL: aie.device(npu1)
// CHECK: conduit.create @bcast_dims_nonuniform
// CHECK-SAME: routing_mode = #conduit.routing_mode<circuit>
module @test_multicast_nonuniform_dims {
  aie.device(npu1) {
    %t02 = aie.tile(0, 2)
    %t13 = aie.tile(1, 3)
    %t14 = aie.tile(1, 4)
    %t15 = aie.tile(1, 5)
    conduit.create @bcast_dims_nonuniform {slot_elems = 4 : i64,
                    producer_tile = array<i64: 0, 2>,
                    consumer_tiles = array<i64: 1, 3, 1, 4, 1, 5>,
                    element_type = memref<4xi32>,
                    depth = 1 : i64,
                    consumer_dimensions = #aie<bd_dim_layout_array_array[[<size = 1, stride = 2>], [<size = 3, stride = 4>], [<size = 1, stride = 2>]]>
                    }
  }
}
