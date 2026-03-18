// RUN: aie-opt --allow-unregistered-dialect --air-channel-to-conduit %s | FileCheck %s
//
// Pass B dep-list threading test.
//
// Verifies that async dependency tokens from prior air.channel.put/get results
// are correctly threaded through to the emitted conduit put/get_memref_async ops.
//
// The key mechanism: Pass B calls replaceAllUsesWith() on the old air async token
// result after emitting the conduit op. This redirects all uses (including dep-list
// operands of subsequent ops) to the new !conduit.dma.token. When a later put/get
// is processed, its dep operand already has DMATokenType and passes the isa<> filter.
//
// Test cases:
//   1. put-then-get: get dep on put token (put→get chain, common in producer-consumer)
//   2. get-then-put: put dep on get token (get→put chain, common in read-modify-write)
//   3. wait_all fan-in: wait_all with two conduit deps (both from put/get)
//   4. wait_all dep propagation: put → wait_all(put_tok) → get(dep=merged_wait_all_result)
//      Tests that the merged token from wait_all propagates as a dep to a subsequent get.
//      This is the exact PASSB-DEP-001 regression scenario (fixed in Task #24).
//
// NOT tested here (known limitation, documented in AirChannelToConduit.cpp):
//   - air.execute tokens as deps: these remain !air.async.token and are dropped
//     because there is no conduit equivalent for non-DMA ordering tokens.

// CHECK-LABEL: module

// -------------------------------------------------------------------
// Test 1: put-then-get dep chain.
// get has ndeps=1 pointing to the put result.
// After rewrite: put → conduit.put_memref_async (result %0 : !conduit.dma.token)
//                get → conduit.get_memref_async[%0 : !conduit.dma.token]
// -------------------------------------------------------------------
// CHECK: conduit.create
// CHECK-SAME: name = "putGet"

// put emitted first, no deps: no bracket dep list before the attr-dict.
// CHECK: %[[PUT:.*]] = conduit.put_memref_async {name = "putGet"
// CHECK-SAME: : !conduit.dma.token

// get emitted second, dep on put token.
// CHECK: %[[GET:.*]] = conduit.get_memref_async
// CHECK-SAME: [%[[PUT]] : !conduit.dma.token]
// CHECK-SAME: name = "putGet"
// CHECK-SAME: : !conduit.dma.token

// -------------------------------------------------------------------
// Test 2: get-then-put dep chain (separate channels).
// put has ndeps=1 pointing to the get result.
// After rewrite: get → conduit.get_memref_async (result %0)
//                put → conduit.put_memref_async[%0 : !conduit.dma.token]
// -------------------------------------------------------------------
// CHECK: conduit.create
// CHECK-SAME: name = "inChan"
// CHECK: conduit.create
// CHECK-SAME: name = "outChan"

// get emitted first, no deps.
// CHECK: %[[GTOK:.*]] = conduit.get_memref_async
// CHECK-SAME: name = "inChan"
// CHECK-SAME: : !conduit.dma.token

// put emitted second, dep on get token.
// CHECK: %[[PTOK:.*]] = conduit.put_memref_async
// CHECK-SAME: [%[[GTOK]] : !conduit.dma.token]
// CHECK-SAME: name = "outChan"
// CHECK-SAME: : !conduit.dma.token

// -------------------------------------------------------------------
// Test 3: wait_all fan-in with two conduit deps.
// wait_all async over [put_tok, get_tok] → conduit.wait_all_async
// -------------------------------------------------------------------
// CHECK: conduit.create
// CHECK-SAME: name = "waChan"

// CHECK: %[[W0:.*]] = conduit.put_memref_async
// CHECK-SAME: name = "waChan"
// CHECK: %[[W1:.*]] = conduit.get_memref_async
// CHECK-SAME: name = "waChan"
// CHECK: conduit.wait_all_async %[[W0]], %[[W1]]
// CHECK-SAME: (!conduit.dma.token, !conduit.dma.token) -> !conduit.dma.token

// -------------------------------------------------------------------
// Test 4: put → wait_all(put_tok) → get(dep=merged_wait_all_result).
// The wait_all result token (still !air.async.token at Phase 3 time) must
// propagate as a dep into the subsequent get. This is the PASSB-DEP-001
// regression case: without the fix, the dep is silently dropped.
// With the fix (preEmittedWaitAll), the wait_all is pre-emitted as a
// conduit.wait_all_async and the get carries [%merged : !conduit.dma.token].
// -------------------------------------------------------------------
// CHECK: conduit.create
// CHECK-SAME: name = "waDep"

// put emitted, no deps.
// CHECK: %[[WD_PUT:.*]] = conduit.put_memref_async {name = "waDep"
// CHECK-SAME: : !conduit.dma.token

// wait_all_async pre-emitted before the get (Phase 3 pre-emission for PASSB-DEP-001).
// CHECK: %[[WD_MERGED:.*]] = conduit.wait_all_async %[[WD_PUT]]
// CHECK-SAME: (!conduit.dma.token) -> !conduit.dma.token

// get carries dep on the pre-emitted wait_all token — not directly on put.
// CHECK: conduit.get_memref_async[%[[WD_MERGED]] : !conduit.dma.token]
// CHECK-SAME: name = "waDep"
// CHECK-SAME: : !conduit.dma.token

// No residual air ops.
// CHECK-NOT: air.channel{{[^._]}}
// CHECK-NOT: air.wait_all

module {
  // ---------------------------------------------------------------
  // Test 1: put-then-get dep chain.
  // ---------------------------------------------------------------
  "air.channel"() {sym_name = "putGet", size = [1, 1]} : () -> ()

  func.func @test_put_then_get_dep(%src : memref<4xi32>, %dst : memref<4xi32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index

    // put async, no deps.
    %put_tok = "air.channel.put"(%src, %c0, %c4, %c1)
        {chan_name = @putGet,
         operand_segment_sizes = array<i32: 0, 0, 1, 1, 1, 1>}
        : (memref<4xi32>, index, index, index)
        -> !air.async.token

    // get async, dep on %put_tok (ndeps=1).
    // After Pass B rewrites put first, %put_tok's uses are redirected to
    // the conduit put token. This dep is a !conduit.dma.token by the time
    // the get is processed.
    %get_tok = "air.channel.get"(%put_tok, %dst, %c0, %c4, %c1)
        {chan_name = @putGet,
         operand_segment_sizes = array<i32: 1, 0, 1, 1, 1, 1>}
        : (!air.async.token, memref<4xi32>, index, index, index)
        -> !air.async.token

    "air.wait_all"(%get_tok) : (!air.async.token) -> ()
    return
  }

  // ---------------------------------------------------------------
  // Test 2: get-then-put dep chain (separate channels).
  // ---------------------------------------------------------------
  "air.channel"() {sym_name = "inChan", size = [1, 1]} : () -> ()
  "air.channel"() {sym_name = "outChan", size = [1, 1]} : () -> ()

  func.func @test_get_then_put_dep(%src : memref<4xi32>, %dst : memref<4xi32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index

    // get async, no deps.
    %get_tok = "air.channel.get"(%dst, %c0, %c4, %c1)
        {chan_name = @inChan,
         operand_segment_sizes = array<i32: 0, 0, 1, 1, 1, 1>}
        : (memref<4xi32>, index, index, index)
        -> !air.async.token

    // put async, dep on %get_tok.
    // After Pass B rewrites get, %get_tok's uses redirect to the conduit token.
    %put_tok = "air.channel.put"(%get_tok, %src, %c0, %c4, %c1)
        {chan_name = @outChan,
         operand_segment_sizes = array<i32: 1, 0, 1, 1, 1, 1>}
        : (!air.async.token, memref<4xi32>, index, index, index)
        -> !air.async.token

    "air.wait_all"(%put_tok) : (!air.async.token) -> ()
    return
  }

  // ---------------------------------------------------------------
  // Test 3: wait_all fan-in with two conduit dep tokens.
  // ---------------------------------------------------------------
  "air.channel"() {sym_name = "waChan", size = [1, 1]} : () -> ()

  func.func @test_wait_all_fanin(%src : memref<4xi32>, %dst : memref<4xi32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index

    // put async, no deps.
    %put_tok = "air.channel.put"(%src, %c0, %c4, %c1)
        {chan_name = @waChan,
         operand_segment_sizes = array<i32: 0, 0, 1, 1, 1, 1>}
        : (memref<4xi32>, index, index, index)
        -> !air.async.token

    // get async, no deps.
    %get_tok = "air.channel.get"(%dst, %c0, %c4, %c1)
        {chan_name = @waChan,
         operand_segment_sizes = array<i32: 0, 0, 1, 1, 1, 1>}
        : (memref<4xi32>, index, index, index)
        -> !air.async.token

    // wait_all async: fan-in over both conduit tokens.
    %merged = "air.wait_all"(%put_tok, %get_tok)
        : (!air.async.token, !air.async.token) -> !air.async.token

    "air.wait_all"(%merged) : (!air.async.token) -> ()
    return
  }

  // ---------------------------------------------------------------
  // Test 4: put → wait_all(put_tok) → get(dep=merged_wait_all_result).
  // Regression test for PASSB-DEP-001: the wait_all result is still
  // !air.async.token at Phase 3 time, so without the fix it fails the
  // DMATokenType filter and the get's dep is silently dropped.
  // With the fix, a conduit.wait_all_async is pre-emitted and the get
  // correctly carries [%merged : !conduit.dma.token].
  // ---------------------------------------------------------------
  "air.channel"() {sym_name = "waDep", size = [1, 1]} : () -> ()

  func.func @test_wait_all_dep_propagation(
      %src : memref<4xi32>, %dst : memref<4xi32>) {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index

    // put async, no deps.
    %put_tok = "air.channel.put"(%src, %c0, %c4, %c1)
        {chan_name = @waDep,
         operand_segment_sizes = array<i32: 0, 0, 1, 1, 1, 1>}
        : (memref<4xi32>, index, index, index)
        -> !air.async.token

    // wait_all fan-in over put token.
    // At Phase 3 time this result is still !air.async.token — the exact
    // type that PASSB-DEP-001 failed to thread through.
    %merged = "air.wait_all"(%put_tok)
        : (!air.async.token) -> !air.async.token

    // get async with dep on %merged (NOT directly on %put_tok).
    %get_tok = "air.channel.get"(%merged, %dst, %c0, %c4, %c1)
        {chan_name = @waDep,
         operand_segment_sizes = array<i32: 1, 0, 1, 1, 1, 1>}
        : (!air.async.token, memref<4xi32>, index, index, index)
        -> !air.async.token

    "air.wait_all"(%get_tok) : (!air.async.token) -> ()
    return
  }
}
