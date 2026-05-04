// RUN: aie-opt --conduit-fuse-operators --split-input-file --verify-diagnostics %s
//
// Routing-axis Step 3 conflict-policy regression: when the two endpoints
// of a fused intermediate channel carry incompatible routing_mode values,
// --conduit-fuse-operators MUST emit a diagnostic naming both modes and
// signal pass failure rather than silently picking one.
//
// Hard-conflict pairs:
//   * Cascade vs anything not-cascade  (cascade is the only single-register
//     pass-through path; mixing it with a buffered DMA / shared-memory
//     consumer is a deadlock-or-corruption hazard with no legal lowering)
//   * SharedMemory vs Circuit          (these reach different lowering
//     phases in Pass C — shared_memory skips DMA, circuit forces DMA — so
//     reconciling them at the fusion boundary is undefined)

// -----

// Conflict 1: Cascade vs SharedMemory.  The diagnostic must name both
// modes so an operator author can see what they pinned.
module @conflict_cascade_vs_shared_memory {
  aie.device(npu2) @devA {
    %shim_a = aie.tile(0, 0)
    %tile_a = aie.tile(0, 2)
    conduit.create @ext_in_a {element_type = memref<128xbf16>, depth = 2 : i64}
    // expected-error@+1 {{conduit-fuse-operators: incompatible routing_mode for fused intermediate channel: producer=cascade, consumer=shared_memory}}
    conduit.create @inter_a {element_type = memref<128xbf16>, depth = 1 : i64,
                              fusion_group = "fg0",
                              routing_mode = #conduit.routing_mode<cascade>}
    aie.shim_dma_allocation @ext_in_a_shim_alloc (%shim_a, MM2S, 0)
    aie.shim_dma_allocation @inter_a_shim_alloc (%shim_a, S2MM, 0)
  }
  aie.device(npu2) @devB {
    %shim_b = aie.tile(0, 0)
    %tile_b = aie.tile(0, 2)
    conduit.create @inter_b {element_type = memref<128xbf16>, depth = 1 : i64,
                              fusion_group = "fg0",
                              routing_mode = #conduit.routing_mode<shared_memory>}
    conduit.create @ext_out_b {element_type = memref<128xbf16>, depth = 2 : i64}
    aie.shim_dma_allocation @inter_b_shim_alloc (%shim_b, MM2S, 0)
    aie.shim_dma_allocation @ext_out_b_shim_alloc (%shim_b, S2MM, 0)
  }
}

// -----

// Conflict 2: SharedMemory vs Circuit.  The diagnostic must name both
// modes (in producer→consumer order) so the conflict is unambiguous.
module @conflict_shared_memory_vs_circuit {
  aie.device(npu2) @devA {
    %shim_a = aie.tile(0, 0)
    %tile_a = aie.tile(0, 2)
    conduit.create @ext_in_a {element_type = memref<128xbf16>, depth = 2 : i64}
    // expected-error@+1 {{conduit-fuse-operators: incompatible routing_mode for fused intermediate channel: producer=shared_memory, consumer=circuit}}
    conduit.create @inter_a {element_type = memref<128xbf16>, depth = 2 : i64,
                              fusion_group = "fg0",
                              routing_mode = #conduit.routing_mode<shared_memory>}
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
