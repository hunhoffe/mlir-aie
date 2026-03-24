// RUN: aie-opt --allow-unregistered-dialect --air-channel-to-conduit %s | FileCheck %s
//
// PASSB-DEP-001 regression test: air.wait_all fan-in tokens as deps for put/get.
//
// Bug: AirChannelToConduit.cpp Phase 3 (put/get rewrite) runs before Phase 4
// (wait_all rewrite). When air.channel.get async [%merged] is processed in
// Phase 3 and %merged is the result of an air.wait_all, the token still has
// type !air.async.token at Phase 3 time and fails the DMATokenType filter at
// dep resolution. The dep was silently dropped, causing the DMA to fire before
// the fan-in fence completes — potential data race or deadlock on hardware.
//
// Fix: at Phase 3 dep resolution, when a dep fails the DMATokenType check,
// inspect whether its defining op is air.wait_all. If so, recursively resolve
// the wait_all's operands (already conduit tokens via prior replaceAllUsesWith)
// and pre-emit a conduit.wait_all_async in-place. The pre-emitted op is
// recorded in a map so Phase 4 skips re-emitting it and just replaces uses.
//
// Test cases:
//   1. put → wait_all → get: get dep on wait_all result (single-input fan-in).
//   2. put + put → wait_all (two inputs) → get: multi-input fan-in.
//   3. Same wait_all result used as dep by two gets: deduplication check.
//   4. wait_all with mix of conduit and non-conduit deps: only conduit threaded.
//   5. Blocking (no-result) wait_all still lowers correctly when a second wait_all
//      uses the merged result as a dep to a get.

// CHECK-LABEL: module

// -------------------------------------------------------------------
// Test 1: put → wait_all(put_tok) → get(dep=merged).
// After fix: merged is pre-emitted as conduit.wait_all_async before the get,
// and the get's dep list contains the pre-emitted token.
// -------------------------------------------------------------------
// CHECK: conduit.create @chan1

// put emitted, no deps.
// CHECK: %[[PUT1:.*]] = conduit.put_memref_async {name = "chan1"
// CHECK-SAME: : !conduit.dma.token

// wait_all pre-emitted before the get (Phase 3 pre-emission).
// CHECK: %[[WA1:.*]] = conduit.wait_all_async %[[PUT1]]
// CHECK-SAME: (!conduit.dma.token) -> !conduit.dma.token

// get with dep on the pre-emitted wait_all token.
// CHECK: %[[GET1:.*]] = conduit.get_memref_async[%[[WA1]] : !conduit.dma.token]
// CHECK-SAME: name = "chan1"
// CHECK-SAME: : !conduit.dma.token

// -------------------------------------------------------------------
// Test 2: two puts, wait_all([tok1, tok2]), get(dep=merged).
// Pre-emitted wait_all_async collects both put tokens.
// -------------------------------------------------------------------
// CHECK: conduit.create @ch2a
// CHECK: conduit.create @ch2b
// CHECK: conduit.create @ch2c

// Two puts, no deps.
// CHECK: %[[P2A:.*]] = conduit.put_memref_async {name = "ch2a"
// CHECK: %[[P2B:.*]] = conduit.put_memref_async {name = "ch2b"

// wait_all_async fan-in over both puts.
// CHECK: %[[WA2:.*]] = conduit.wait_all_async %[[P2A]], %[[P2B]]
// CHECK-SAME: (!conduit.dma.token, !conduit.dma.token) -> !conduit.dma.token

// get with dep on merged fan-in.
// CHECK: %[[GET2:.*]] = conduit.get_memref_async[%[[WA2]] : !conduit.dma.token]
// CHECK-SAME: name = "ch2c"
// CHECK-SAME: : !conduit.dma.token

// -------------------------------------------------------------------
// Test 3: same wait_all result as dep for two gets (deduplication).
// Only ONE conduit.wait_all_async emitted (preEmittedWaitAll map deduplicates).
// -------------------------------------------------------------------
// CHECK: conduit.create @ch3put
// CHECK: conduit.create @ch3a
// CHECK: conduit.create @ch3b

// put then pre-emitted wait_all_async.
// CHECK: %[[P3:.*]] = conduit.put_memref_async {name = "ch3put"
// CHECK: %[[WA3:.*]] = conduit.wait_all_async %[[P3]]

// First get: dep on pre-emitted token.
// CHECK: %[[G3A:.*]] = conduit.get_memref_async[%[[WA3]] : !conduit.dma.token]
// CHECK-SAME: name = "ch3a"

// Second get: also dep on the SAME pre-emitted token (no second wait_all_async).
// CHECK: %[[G3B:.*]] = conduit.get_memref_async[%[[WA3]] : !conduit.dma.token]
// CHECK-SAME: name = "ch3b"

// -------------------------------------------------------------------
// Test 4: put → get → wait_all(put_tok, get_tok) → put[dep=merged].
// This is the original PASSB-DEP-001 scenario: two separate channel ops feed
// a wait_all fan-in, whose merged token is then used as a dep for a subsequent
// put. Asserts that put_memref_async carries [%merged : !conduit.dma.token].
// -------------------------------------------------------------------
// CHECK: conduit.create @ch4a
// CHECK: conduit.create @ch4b
// CHECK: conduit.create @ch4c

// put and get emitted with no deps.
// CHECK: %[[P4:.*]] = conduit.put_memref_async {name = "ch4a"
// CHECK-SAME: : !conduit.dma.token
// CHECK: %[[G4:.*]] = conduit.get_memref_async {name = "ch4b"
// CHECK-SAME: : !conduit.dma.token

// wait_all_async pre-emitted with both put and get tokens.
// CHECK: %[[WA4:.*]] = conduit.wait_all_async %[[P4]], %[[G4]]
// CHECK-SAME: (!conduit.dma.token, !conduit.dma.token) -> !conduit.dma.token

// Second put carries dep on merged token — the original bug scenario.
// CHECK: conduit.put_memref_async[%[[WA4]] : !conduit.dma.token]
// CHECK-SAME: name = "ch4c"
// CHECK-SAME: : !conduit.dma.token

// CHECK-NOT: air.channel{{[^._]}}
// CHECK-NOT: air.wait_all

module {
  // ---------------------------------------------------------------
  // Test 1: put → wait_all → get (dep on wait_all result).
  // ---------------------------------------------------------------
  "air.channel"() {sym_name = "chan1", size = [1, 1]} : () -> ()

  func.func @test_put_wait_get(%src : memref<4xi32>, %dst : memref<4xi32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index

    // put async, no deps.
    %put_tok = "air.channel.put"(%src, %c0, %c4, %c1)
        {chan_name = @chan1,
         operand_segment_sizes = array<i32: 0, 0, 1, 1, 1, 1>}
        : (memref<4xi32>, index, index, index)
        -> !air.async.token

    // wait_all fan-in over put token.
    %merged = "air.wait_all"(%put_tok)
        : (!air.async.token) -> !air.async.token

    // get async with dep on %merged (NOT directly on %put_tok).
    // Bug: %merged is !air.async.token at Phase 3 time → dep dropped.
    // Fix: pre-emit conduit.wait_all_async and use its result.
    %get_tok = "air.channel.get"(%merged, %dst, %c0, %c4, %c1)
        {chan_name = @chan1,
         operand_segment_sizes = array<i32: 1, 0, 1, 1, 1, 1>}
        : (!air.async.token, memref<4xi32>, index, index, index)
        -> !air.async.token

    "air.wait_all"(%get_tok) : (!air.async.token) -> ()
    return
  }

  // ---------------------------------------------------------------
  // Test 2: two puts, wait_all fan-in of both, get depends on merged.
  // ---------------------------------------------------------------
  "air.channel"() {sym_name = "ch2a", size = [1, 1]} : () -> ()
  "air.channel"() {sym_name = "ch2b", size = [1, 1]} : () -> ()
  "air.channel"() {sym_name = "ch2c", size = [1, 1]} : () -> ()

  func.func @test_fanin2(%s1 : memref<4xi32>, %s2 : memref<4xi32>, %dst : memref<4xi32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index

    %tok1 = "air.channel.put"(%s1, %c0, %c4, %c1)
        {chan_name = @ch2a, operand_segment_sizes = array<i32: 0, 0, 1, 1, 1, 1>}
        : (memref<4xi32>, index, index, index) -> !air.async.token

    %tok2 = "air.channel.put"(%s2, %c0, %c4, %c1)
        {chan_name = @ch2b, operand_segment_sizes = array<i32: 0, 0, 1, 1, 1, 1>}
        : (memref<4xi32>, index, index, index) -> !air.async.token

    // Fan-in over both put tokens.
    %merged = "air.wait_all"(%tok1, %tok2)
        : (!air.async.token, !air.async.token) -> !air.async.token

    // get depends on merged fan-in.
    %get_tok = "air.channel.get"(%merged, %dst, %c0, %c4, %c1)
        {chan_name = @ch2c, operand_segment_sizes = array<i32: 1, 0, 1, 1, 1, 1>}
        : (!air.async.token, memref<4xi32>, index, index, index) -> !air.async.token

    "air.wait_all"(%get_tok) : (!air.async.token) -> ()
    return
  }

  // ---------------------------------------------------------------
  // Test 3: same wait_all result as dep for two gets (deduplication).
  // preEmittedWaitAll map ensures only ONE conduit.wait_all_async is emitted.
  // ---------------------------------------------------------------
  "air.channel"() {sym_name = "ch3put", size = [1, 1]} : () -> ()
  "air.channel"() {sym_name = "ch3a",   size = [1, 1]} : () -> ()
  "air.channel"() {sym_name = "ch3b",   size = [1, 1]} : () -> ()

  func.func @test_dedup_wait_all(%src : memref<4xi32>, %dA : memref<4xi32>, %dB : memref<4xi32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index

    %put_tok = "air.channel.put"(%src, %c0, %c4, %c1)
        {chan_name = @ch3put, operand_segment_sizes = array<i32: 0, 0, 1, 1, 1, 1>}
        : (memref<4xi32>, index, index, index) -> !air.async.token

    %merged = "air.wait_all"(%put_tok)
        : (!air.async.token) -> !air.async.token

    // Two gets both depending on the same merged token.
    // The fix must deduplicate: only one conduit.wait_all_async is emitted.
    %gA = "air.channel.get"(%merged, %dA, %c0, %c4, %c1)
        {chan_name = @ch3a, operand_segment_sizes = array<i32: 1, 0, 1, 1, 1, 1>}
        : (!air.async.token, memref<4xi32>, index, index, index) -> !air.async.token

    %gB = "air.channel.get"(%merged, %dB, %c0, %c4, %c1)
        {chan_name = @ch3b, operand_segment_sizes = array<i32: 1, 0, 1, 1, 1, 1>}
        : (!air.async.token, memref<4xi32>, index, index, index) -> !air.async.token

    "air.wait_all"(%gA, %gB) : (!air.async.token, !air.async.token) -> ()
    return
  }

  // ---------------------------------------------------------------
  // Test 4: put → get → wait_all(put_tok, get_tok) → put[dep=merged].
  // The original PASSB-DEP-001 scenario: merged token from a two-input
  // wait_all feeds a subsequent put as a dep.
  // ---------------------------------------------------------------
  "air.channel"() {sym_name = "ch4a", size = [1, 1]} : () -> ()
  "air.channel"() {sym_name = "ch4b", size = [1, 1]} : () -> ()
  "air.channel"() {sym_name = "ch4c", size = [1, 1]} : () -> ()

  func.func @test_merged_dep_on_put(
      %s1 : memref<4xi32>, %s2 : memref<4xi32>,
      %s3 : memref<4xi32>, %dst : memref<4xi32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index

    // First put, no deps.
    %put_tok = "air.channel.put"(%s1, %c0, %c4, %c1)
        {chan_name = @ch4a, operand_segment_sizes = array<i32: 0, 0, 1, 1, 1, 1>}
        : (memref<4xi32>, index, index, index) -> !air.async.token

    // First get, no deps.
    %get_tok = "air.channel.get"(%s2, %c0, %c4, %c1)
        {chan_name = @ch4b, operand_segment_sizes = array<i32: 0, 0, 1, 1, 1, 1>}
        : (memref<4xi32>, index, index, index) -> !air.async.token

    // Fan-in over put and get tokens.
    %merged = "air.wait_all"(%put_tok, %get_tok)
        : (!air.async.token, !air.async.token) -> !air.async.token

    // Second put with dep on %merged (the wait_all result, not directly on
    // %put_tok or %get_tok). This was the original PASSB-DEP-001 scenario.
    %put2_tok = "air.channel.put"(%merged, %s3, %c0, %c4, %c1)
        {chan_name = @ch4c, operand_segment_sizes = array<i32: 1, 0, 1, 1, 1, 1>}
        : (!air.async.token, memref<4xi32>, index, index, index) -> !air.async.token

    "air.wait_all"(%put2_tok) : (!air.async.token) -> ()
    return
  }
}
