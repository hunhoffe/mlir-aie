// RUN: aie-opt -split-input-file -verify-diagnostics %s
//
// Negative tests for cascade-related Conduit verifiers.
//
// After cascade migration (#27), conduit.put_cascade and conduit.get_cascade
// no longer exist in the Conduit dialect.  Pass A/B emit aie.put_cascade /
// aie.get_cascade directly, so the per-op routing_mode and type verifiers
// that lived in PutCascade::verify() / GetCascade::verify() are gone.
//
// This file retains test (e): conduit.distribute with a cascade-mode src
// is still rejected by the M5 distribute verifier.

// -----

// (e) conduit.distribute with a cascade-mode src is rejected by the verifier.
// Cascade is incompatible with distribute fan-out.

func.func @bad_distribute_cascade_src() {
  conduit.create @src {capacity = 1 : i64, depth = 1 : i64,
                  routing_mode = #conduit.routing_mode<cascade>,
                  producer_tile = array<i64: 0, 2>,
                  consumer_tiles = array<i64: 0, 1>}
  conduit.create @dst {capacity = 1 : i64, depth = 1 : i64,
                  producer_tile = array<i64: 0, 1>,
                  consumer_tiles = array<i64: 1, 2>}
  // expected-error @+1 {{'conduit.distribute' op cascade channel 'src' cannot be used in a distribute src}}
  conduit.distribute {srcs = [@src], dsts = [@dst], memtile = "tile(0,1)"}
  return
}
