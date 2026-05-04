// RUN: aie-opt -split-input-file -verify-diagnostics %s | FileCheck %s
// TODO Sprint 2 Phase 5b: re-add test sections for sync_mode=barrier/independent
// and routing_mode=packet once those attrs are added to ScatterOp/GatherOp in Conduit.td.
//
// Lit tests for conduit.gather (N source tiles → 1 destination via MemTile relay).
//
// NOTE: Pass C lowering for conduit.gather is not yet implemented (Sprint 2
// Phase 5c — ConduitToDMALink.cpp relay op migration). These tests verify
// that the ops parse, round-trip, and pass verifier checks.
//
// Design intent (what Pass C WILL emit after Sprint 2 Phase 5c):
//   For a 4:1 gather through memtile(0,1):
//     - 4 MemTile S2MM channels — one per source tile feeding its slice
//       into the assembled destination buffer (with per-src BD offset)
//     - 1 MemTile MM2S channel — outputs the assembled destination buffer
//     - 4 aie.flow ops: tile(0,2..5) MM2S → memtile(0,1) S2MM ch 0..3
//     - 1 aie.flow op: memtile(0,1) MM2S ch 0 → dest (shim or next tile)
//     - Per-source lock pairs on the MemTile (init=depth) for flow control
//
// Resource comparison vs. separate channels (no gather):
//   Without gather: 4 separate channels, each wiring its own MM2S → S2MM pair.
//   With gather: S2MM channels are fused at the MemTile; output is assembled.
//
// See DIALECT_REDESIGN.md §3 "conduit.gather (replaces conduit.join)".

// -----

// Valid conduit.gather: 4 srcs → 1 dst (full 4-tile gather).
// This is the canonical C output gather from a 4-tile GEMV computation.
// CHECK-LABEL: func.func @gather_4tile_parses
func.func @gather_4tile_parses() {
  // CHECK: %[[MT:.*]] = aie.tile(0, 1)
  // CHECK: conduit.gather
  // CHECK-SAME: srcs = [@C_tile0, @C_tile1, @C_tile2, @C_tile3]
  // CHECK-SAME: dst = @C_dst
  // CHECK-SAME: memtile = %[[MT]]
  %mt = aie.tile(0, 1)
  conduit.gather{srcs = [@C_tile0, @C_tile1, @C_tile2, @C_tile3], dst = @C_dst, memtile = %mt}
  return
}

// -----

// Valid conduit.gather: 2 srcs → 1 dst (minimal gather).
// CHECK-LABEL: func.func @gather_2tile_parses
func.func @gather_2tile_parses() {
  // CHECK: %[[MT:.*]] = aie.tile(0, 1)
  // CHECK: conduit.gather
  // CHECK-SAME: srcs = [@out0, @out1]
  // CHECK-SAME: dst = @result
  // CHECK-SAME: memtile = %[[MT]]
  %mt = aie.tile(0, 1)
  conduit.gather{srcs = [@out0, @out1], dst = @result, memtile = %mt}
  return
}

// -----

// Valid conduit.gather: 1 src → 1 dst (N=1 relay / passthrough case).
// N=1 gather is semantically a relay: the single source feeds the single
// destination through the MemTile with no fan-in.
// CHECK-LABEL: func.func @gather_single_src_parses
func.func @gather_single_src_parses() {
  // CHECK: %[[MT:.*]] = aie.tile(0, 1)
  // CHECK: conduit.gather
  // CHECK-SAME: srcs = [@single_src]
  // CHECK-SAME: dst = @single_dst
  // CHECK-SAME: memtile = %[[MT]]
  %mt = aie.tile(0, 1)
  conduit.gather{srcs = [@single_src], dst = @single_dst, memtile = %mt}
  return
}

