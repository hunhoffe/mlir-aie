// RUN: aie-opt -split-input-file -verify-diagnostics %s
//
// Tests for plio as a first-class ODS attribute on conduit.create.
//
// plio=true requires at least one shim-row (row == 0) endpoint: either
// producer_tile at row 0, or a consumer_tiles entry at row 0.
// Create::verify() in ConduitOps.cpp enforces this.
//
// Cases:
//   (a) plio=true with shim producer (producer_tile row=0) — must PASS
//   (b) plio=true with compute producer and shim consumer — must PASS
//   (c) plio=true with no shim endpoint at all — must emit error

// -----

// (a) plio=true with producer_tile=[0,0] (shim row) — valid, no error expected.

conduit.create @plio_shim_producer {slot_elems = 4 : i64, depth = 0 : i64,
               producer_tile = array<i64: 0, 0>,
               plio = true}

// -----

// (b) plio=true with compute producer and shim consumer — valid, no error expected.
// Mirrors of_1/of_2 in objectfifo_plio_test.mlir (compute→shim direction).

conduit.create @plio_shim_consumer {slot_elems = 4 : i64, depth = 0 : i64,
               producer_tile = array<i64: 0, 2>,
               consumer_tiles = array<i64: 0, 0>,
               plio = true}

// -----

// (c) plio=true with no shim endpoint — must be rejected.

// expected-error @+1 {{'conduit.create' op plio=true requires a shim tile (row 0) as producer_tile or consumer_tiles}}
conduit.create @plio_no_shim {slot_elems = 4 : i64, depth = 0 : i64,
               producer_tile = array<i64: 0, 2>,
               consumer_tiles = array<i64: 0, 3>,
               plio = true}
