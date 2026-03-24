// RUN: aie-opt -split-input-file -verify-diagnostics %s
//
// Tests for plio as a first-class ODS attribute on conduit.create.
//
// plio=true requires at least one shim-row (row == 0) endpoint: either
// producer_tile at row 0, or shim_consumer_tiles non-empty.
// Create::verify() in ConduitOps.cpp enforces this.
//
// Cases:
//   (a) plio=true with shim producer (producer_tile row=0) — must PASS
//   (b) plio=true with shim consumer (shim_consumer_tiles set) — must PASS
//   (c) plio=true with no shim endpoint at all — must emit error

// -----

// (a) plio=true with producer_tile=[0,0] (shim row) — valid, no error expected.

conduit.create {name = "plio_shim_producer", capacity = 4 : i64,
               producer_tile = array<i64: 0, 0>,
               plio = true}

// -----

// (b) plio=true with compute producer and shim consumer — valid, no error expected.
// Mirrors of_1/of_2 in objectfifo_plio_test.mlir (compute→shim direction).

conduit.create {name = "plio_shim_consumer", capacity = 4 : i64,
               producer_tile = array<i64: 0, 2>,
               shim_consumer_tiles = array<i64: 0, 0>,
               plio = true}

// -----

// (c) plio=true with no shim endpoint — must be rejected.

// expected-error @+1 {{'conduit.create' op plio=true requires a shim tile (row 0) as either producer_tile or in shim_consumer_tiles}}
conduit.create {name = "plio_no_shim", capacity = 4 : i64,
               producer_tile = array<i64: 0, 2>,
               consumer_tiles = array<i64: 0, 3>,
               plio = true}
