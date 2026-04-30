// RUN: aie-opt -split-input-file -verify-diagnostics %s | FileCheck %s
// TODO Sprint 2 Phase 5b: re-add test sections for sync_mode=barrier/independent
// and routing_mode=packet once those attrs are added to ScatterOp/GatherOp in Conduit.td.
//
// Lit tests for conduit.scatter multicast mode (1 src → N dsts, uniform data).
//
// NOTE: Pass C lowering for conduit.scatter is not yet implemented (Sprint 2
// Phase 5c — ConduitToDMALink.cpp relay op migration). These tests verify
// that the ops parse, round-trip, and pass verifier checks.
//
// Multicast vs. time-multiplexed unicast:
//   Multicast (offsets absent or all equal, routing_mode=packet inferred):
//     Pass C WILL emit: 1 MM2S channel, 1 BD, 1 aie.packet_flow with
//     N aie.packet_dest entries (simultaneous delivery, same data per tile).
//     Packet ID cost: 1 (multicast, not time-multiplexed unicast).
//
//   Time-multiplexed unicast (non-uniform offsets + routing_mode=packet):
//     Pass C WILL emit: 1 MM2S channel, N BDs with N distinct pkt_ids,
//     N aie.packet_flow ops each with a single aie.packet_dest.
//     Packet ID cost: N (one per destination).
//
// This file tests the multicast case (no offsets → uniform data → single BD).
// For the time-multiplexed unicast case see conduit_scatter_packet_4tile.mlir.
//
// See DIALECT_REDESIGN.md §3 "Multicast inference".

// -----

// Valid conduit.scatter: 1 src → 4 dsts, no offsets (multicast).
// --conduit-infer-modes will set routing_mode=packet automatically for absent offsets.
// Pass C emits: 1 MM2S, 1 BD, aie.packet_flow with 4 aie.packet_dest entries.
// CHECK-LABEL: func.func @scatter_multicast_4tile_parses
func.func @scatter_multicast_4tile_parses() {
  // CHECK: %[[MT:.*]] = aie.tile(0, 1)
  // CHECK: conduit.scatter
  // CHECK-SAME: src = @B_src
  // CHECK-SAME: dsts = [@B_tile0, @B_tile1, @B_tile2, @B_tile3]
  // CHECK-SAME: memtile = %[[MT]]
  %mt = aie.tile(0, 1)
  conduit.scatter{src = @B_src, dsts = [@B_tile0, @B_tile1, @B_tile2, @B_tile3], memtile = %mt}
  return
}

// -----

// Valid conduit.scatter: 1 src → 2 dsts, no offsets (minimal multicast).
// CHECK-LABEL: func.func @scatter_multicast_2tile_parses
func.func @scatter_multicast_2tile_parses() {
  // CHECK: %[[MT:.*]] = aie.tile(0, 1)
  // CHECK: conduit.scatter
  // CHECK-SAME: src = @bcast
  // CHECK-SAME: dsts = [@c0, @c1]
  // CHECK-SAME: memtile = %[[MT]]
  %mt = aie.tile(0, 1)
  conduit.scatter{src = @bcast, dsts = [@c0, @c1], memtile = %mt}
  return
}

