// RUN: aie-opt -split-input-file -verify-diagnostics %s
//
// Negative tests for cascade-related Conduit verifiers.
//
// After cascade migration (#27), conduit.put_cascade and conduit.get_cascade
// no longer exist in the Conduit dialect.  Pass A/B emit aie.put_cascade /
// aie.get_cascade directly, so the per-op routing_mode and type verifiers
// that lived in PutCascade::verify() / GetCascade::verify() are gone.
//
// NOTE: The cascade channel check for scatter/gather is enforced by Pass C
// (conduit-to-dma linkPhase), NOT by the op-level verifier. ScatterOp
// verifiers only check DMA budget and memtile format.
//
// This file validates that conduit.scatter with a cascade-mode src parses
// correctly — the cascade rejection fires at lowering time (--conduit-to-dma).

// -----

// (e) conduit.scatter with a cascade-mode src — parses without verifier error.
// Cascade incompatibility is detected by Pass C, not by ScatterOp::verify().

aie.device(npu1) {
conduit.create @src {depth = 1 : i64,
                element_type = memref<64xi32>,
                routing_mode = #conduit.routing_mode<cascade>
                }
conduit.create @dst {depth = 1 : i64,
                element_type = memref<64xi32>
                }
func.func @scatter_cascade_src_parse_ok() {
  conduit.scatter{src = @src, dsts = [@dst] {memtile = "tile(0,1)"}}
  return
}
}
