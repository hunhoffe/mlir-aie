// RUN: aie-opt --conduit-fuse-operators --split-input-file %s | FileCheck %s
//
// Routing-axis Step 3 regression: --conduit-fuse-operators must propagate
// the `routing_mode` of the two fused endpoints to the new
// `@fused_intermediate_N` conduit.create it emits, applying this conflict
// policy:
//
//   * Both absent           → fused absent (downstream resolves)
//   * Both equal             → that value
//   * Exactly one set        → the one that's set
//   * Otherwise (no error)   → producer (devA's output channel) wins
//
// Each split-input-file case below is a minimal two-device fixture: each
// device exposes one external shim-side channel and one fusion-marked
// intermediate channel.  No aie.core / aie.runtime_sequence is needed —
// tile inference uses the aie.shim_dma_allocation pairs to identify the
// output / input channels and the fusion_group attr drives the pair
// match.
//
// Behavior under test is the routing_mode resolution only; the rest of
// the pass behavior is exercised by fuse_operators_basic.mlir et al.

// -----

// Case A: both endpoints absent → fused absent.  None of the surviving
// conduit.create ops in this module ever carried a routing_mode in the
// input, so a CHECK-NOT over the post-fusion body proves the fused
// intermediate did not invent one.
//
// CHECK-LABEL: module @case_a_both_absent
// CHECK:       conduit.create @fused_intermediate_0
// CHECK-NOT:   routing_mode
module @case_a_both_absent {
  aie.device(npu2) @devA {
    %shim_a = aie.tile(0, 0)
    %tile_a = aie.tile(0, 2)
    conduit.create @ext_in_a {element_type = memref<128xbf16>, depth = 2 : i64}
    conduit.create @inter_a {element_type = memref<128xbf16>, depth = 2 : i64,
                              fusion_group = "fg0"}
    aie.shim_dma_allocation @ext_in_a_shim_alloc (%shim_a, MM2S, 0)
    aie.shim_dma_allocation @inter_a_shim_alloc (%shim_a, S2MM, 0)
  }
  aie.device(npu2) @devB {
    %shim_b = aie.tile(0, 0)
    %tile_b = aie.tile(0, 2)
    conduit.create @inter_b {element_type = memref<128xbf16>, depth = 2 : i64,
                              fusion_group = "fg0"}
    conduit.create @ext_out_b {element_type = memref<128xbf16>, depth = 2 : i64}
    aie.shim_dma_allocation @inter_b_shim_alloc (%shim_b, MM2S, 0)
    aie.shim_dma_allocation @ext_out_b_shim_alloc (%shim_b, S2MM, 0)
  }
}

// -----

// Case B: producer=circuit, consumer=circuit → fused=circuit.
//
// CHECK-LABEL: module @case_b_both_circuit
// CHECK:       conduit.create @fused_intermediate_0
// CHECK-SAME:  routing_mode = #conduit.routing_mode<circuit>
module @case_b_both_circuit {
  aie.device(npu2) @devA {
    %shim_a = aie.tile(0, 0)
    %tile_a = aie.tile(0, 2)
    conduit.create @ext_in_a {element_type = memref<128xbf16>, depth = 2 : i64}
    conduit.create @inter_a {element_type = memref<128xbf16>, depth = 2 : i64,
                              fusion_group = "fg0",
                              routing_mode = #conduit.routing_mode<circuit>}
    aie.shim_dma_allocation @ext_in_a_shim_alloc (%shim_a, MM2S, 0)
    aie.shim_dma_allocation @inter_a_shim_alloc (%shim_a, S2MM, 0)
  }
  aie.device(npu2) @devB {
    %shim_b = aie.tile(0, 0)
    %tile_b = aie.tile(0, 2)
    conduit.create @inter_b {element_type = memref<128xbf16>, depth = 2 : i64,
                              fusion_group = "fg0",
                              routing_mode = #conduit.routing_mode<circuit>}
    conduit.create @ext_out_b {element_type = memref<128xbf16>, depth = 2 : i64}
    aie.shim_dma_allocation @inter_b_shim_alloc (%shim_b, MM2S, 0)
    aie.shim_dma_allocation @ext_out_b_shim_alloc (%shim_b, S2MM, 0)
  }
}

// -----

// Case C: producer=circuit, consumer absent → fused=circuit (one-set takes).
//
// CHECK-LABEL: module @case_c_producer_circuit_consumer_absent
// CHECK:       conduit.create @fused_intermediate_0
// CHECK-SAME:  routing_mode = #conduit.routing_mode<circuit>
module @case_c_producer_circuit_consumer_absent {
  aie.device(npu2) @devA {
    %shim_a = aie.tile(0, 0)
    %tile_a = aie.tile(0, 2)
    conduit.create @ext_in_a {element_type = memref<128xbf16>, depth = 2 : i64}
    conduit.create @inter_a {element_type = memref<128xbf16>, depth = 2 : i64,
                              fusion_group = "fg0",
                              routing_mode = #conduit.routing_mode<circuit>}
    aie.shim_dma_allocation @ext_in_a_shim_alloc (%shim_a, MM2S, 0)
    aie.shim_dma_allocation @inter_a_shim_alloc (%shim_a, S2MM, 0)
  }
  aie.device(npu2) @devB {
    %shim_b = aie.tile(0, 0)
    %tile_b = aie.tile(0, 2)
    conduit.create @inter_b {element_type = memref<128xbf16>, depth = 2 : i64,
                              fusion_group = "fg0"}
    conduit.create @ext_out_b {element_type = memref<128xbf16>, depth = 2 : i64}
    aie.shim_dma_allocation @inter_b_shim_alloc (%shim_b, MM2S, 0)
    aie.shim_dma_allocation @ext_out_b_shim_alloc (%shim_b, S2MM, 0)
  }
}

// -----

// Case D: producer absent, consumer=dma → fused=dma (one-set takes).
//
// CHECK-LABEL: module @case_d_producer_absent_consumer_dma
// CHECK:       conduit.create @fused_intermediate_0
// CHECK-SAME:  routing_mode = #conduit.routing_mode<dma>
module @case_d_producer_absent_consumer_dma {
  aie.device(npu2) @devA {
    %shim_a = aie.tile(0, 0)
    %tile_a = aie.tile(0, 2)
    conduit.create @ext_in_a {element_type = memref<128xbf16>, depth = 2 : i64}
    conduit.create @inter_a {element_type = memref<128xbf16>, depth = 2 : i64,
                              fusion_group = "fg0"}
    aie.shim_dma_allocation @ext_in_a_shim_alloc (%shim_a, MM2S, 0)
    aie.shim_dma_allocation @inter_a_shim_alloc (%shim_a, S2MM, 0)
  }
  aie.device(npu2) @devB {
    %shim_b = aie.tile(0, 0)
    %tile_b = aie.tile(0, 2)
    conduit.create @inter_b {element_type = memref<128xbf16>, depth = 2 : i64,
                              fusion_group = "fg0",
                              routing_mode = #conduit.routing_mode<dma>}
    conduit.create @ext_out_b {element_type = memref<128xbf16>, depth = 2 : i64}
    aie.shim_dma_allocation @inter_b_shim_alloc (%shim_b, MM2S, 0)
    aie.shim_dma_allocation @ext_out_b_shim_alloc (%shim_b, S2MM, 0)
  }
}

// -----

// Case E: producer=dma, consumer=circuit → fused=dma (producer-wins fallthrough
// for the otherwise-non-conflicting pair).
//
// CHECK-LABEL: module @case_e_producer_wins
// CHECK:       conduit.create @fused_intermediate_0
// CHECK-SAME:  routing_mode = #conduit.routing_mode<dma>
module @case_e_producer_wins {
  aie.device(npu2) @devA {
    %shim_a = aie.tile(0, 0)
    %tile_a = aie.tile(0, 2)
    conduit.create @ext_in_a {element_type = memref<128xbf16>, depth = 2 : i64}
    conduit.create @inter_a {element_type = memref<128xbf16>, depth = 2 : i64,
                              fusion_group = "fg0",
                              routing_mode = #conduit.routing_mode<dma>}
    aie.shim_dma_allocation @ext_in_a_shim_alloc (%shim_a, MM2S, 0)
    aie.shim_dma_allocation @inter_a_shim_alloc (%shim_a, S2MM, 0)
  }
  aie.device(npu2) @devB {
    %shim_b = aie.tile(0, 0)
    %tile_b = aie.tile(0, 2)
    conduit.create @inter_b {element_type = memref<128xbf16>, depth = 2 : i64,
                              fusion_group = "fg0",
                              routing_mode = #conduit.routing_mode<circuit>}
    conduit.create @ext_out_b {element_type = memref<128xbf16>, depth = 2 : i64}
    aie.shim_dma_allocation @inter_b_shim_alloc (%shim_b, MM2S, 0)
    aie.shim_dma_allocation @ext_out_b_shim_alloc (%shim_b, S2MM, 0)
  }
}
