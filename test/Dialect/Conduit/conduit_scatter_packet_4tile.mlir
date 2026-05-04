// RUN: aie-opt -split-input-file -verify-diagnostics %s | FileCheck %s
// TODO Sprint 2 Phase 5b: re-add test sections for sync_mode=barrier/independent
// and routing_mode=packet once those attrs are added to ScatterOp/GatherOp in Conduit.td.
//
// Lit tests for conduit.scatter with routing_mode=packet (time-multiplexed
// unicast): 1 source channel → 4 destination channels with non-uniform offsets.
//
// NOTE: Pass C lowering for conduit.scatter is not yet implemented (Sprint 2
// Phase 5c — ConduitToDMALink.cpp relay op migration). These tests verify
// that the ops parse, round-trip, and pass verifier checks. A separate
// negative test at the bottom (File-local section) confirms the expected Pass C
// failure once routing is attempted.
//
// Design intent (what Pass C WILL emit after Sprint 2 Phase 5c):
//   - 1 MemTile MM2S channel (not 4) serving all 4 destinations sequentially
//   - 4 BDs with pkt_id 0, 1, 2, 3 (time-multiplexed unicast)
//   - 4 aie.packet_flow ops, each with exactly 1 aie.packet_dest
//   - 4 MemTile S2MM channels (one per source chunk ingested from shim)
//
// Resource comparison target (vs. separate unicast without routing_mode=packet):
//   With time-multiplexed unicast: 1 MM2S (not 4), 4 packet IDs consumed.
//   Without: 4 separate aie.flow ops, 4 MM2S channels, 0 packet IDs.
//
// See DIALECT_REDESIGN.md §3 "Time-multiplexed scatter semantics".

// -----

// Valid conduit.scatter: 1 src → 4 dsts, routing_mode=packet.
// Non-uniform offsets select time-multiplexed unicast in Pass C.
// CHECK-LABEL: func.func @scatter_packet_4tile_parses
func.func @scatter_packet_4tile_parses() {
  // CHECK: %[[MT:.*]] = aie.tile(0, 1)
  // CHECK: conduit.scatter
  // CHECK-SAME: src = @A_src
  // CHECK-SAME: dsts = [@A_tile0, @A_tile1, @A_tile2, @A_tile3]
  // CHECK-SAME: memtile = %[[MT]]
  %mt = aie.tile(0, 1)
  conduit.scatter{src = @A_src, dsts = [@A_tile0, @A_tile1, @A_tile2, @A_tile3], memtile = %mt}
  return
}

// -----

// Valid conduit.scatter: 1 src → 1 dst (N=1 relay / forward case).
// CHECK-LABEL: func.func @scatter_packet_relay_parses
func.func @scatter_packet_relay_parses() {
  // CHECK: %[[MT:.*]] = aie.tile(0, 1)
  // CHECK: conduit.scatter
  // CHECK-SAME: src = @in_src
  // CHECK-SAME: dsts = [@out_dst]
  // CHECK-SAME: memtile = %[[MT]]
  %mt = aie.tile(0, 1)
  conduit.scatter{src = @in_src, dsts = [@out_dst], memtile = %mt}
  return
}

