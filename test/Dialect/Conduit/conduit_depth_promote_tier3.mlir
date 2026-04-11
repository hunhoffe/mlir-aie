// RUN: aie-opt --conduit-depth-promote %s | FileCheck %s
// RUN: aie-opt --conduit-depth-promote --verify-diagnostics %s | FileCheck %s
//
// Test for --conduit-depth-promote on Tier 3 (air.channel-originated) channels.
//
// Tier 3 channels use conduit.put_memref_async / conduit.get_memref_async
// instead of conduit.acquire / conduit.release. The pass must detect these
// ops for loop context and passthrough checks, and promote eligible channels.
//
// This test verifies:
//   (a) "tier3_loop" — depth-1 with put/get_memref_async inside scf.for + compute:
//       PROMOTED from depth=1,slot_elems =128 to depth=2,slot_elems =256
//   (b) "tier3_no_loop" — depth-1 but put/get_memref_async outside any loop:
//       NOT promoted (no loop context)
//   (c) "tier2_loop" — standard Tier 2 acquire/release inside loop (regression):
//       PROMOTED from depth=1,slot_elems =32 to depth=2,slot_elems =64

// (a) tier3_loop: promoted — capacity doubles 128→256, depth 1→2
// CHECK-DAG: conduit.create @tier3_loop {{{.*}}depth = 2 : i64, {{.*}}slot_elems = 256 : i64
// (b) tier3_no_loop: NOT promoted — capacity stays 128, depth stays 1
// CHECK-DAG: conduit.create @tier3_no_loop {{{.*}}depth = 1 : i64, {{.*}}slot_elems = 128 : i64
// (c) tier2_loop: promoted (regression check) — capacity doubles 32→64, depth 1→2
// CHECK-DAG: conduit.create @tier2_loop {{{.*}}depth = 2 : i64, {{.*}}slot_elems = 64 : i64
// expected-remark @+1 {{conduit-depth-promote: promoted 2 conduit(s)}}
module {
aie.device(npu1) {

// (a) Tier 3 eligible: depth-1 with put/get_memref_async inside loop + compute.
// expected-remark @+1 {{conduit-depth-promote: promoted 'tier3_loop' from depth-1 to depth-2}}
conduit.create @tier3_loop {slot_elems = 128 : i64,
                producer_tile = array<i64: 0, 2>,
                consumer_tiles = array<i64: 0, 3>,
                element_type = memref<32xi32>,
                depth = 1 : i64}

// (b) Tier 3 no loop: depth-1 but not inside a loop — should NOT be promoted.
// expected-remark @+1 {{conduit-depth-promote: skipping 'tier3_no_loop' -- no loop context}}
conduit.create @tier3_no_loop {slot_elems = 128 : i64,
                producer_tile = array<i64: 0, 4>,
                consumer_tiles = array<i64: 0, 5>,
                element_type = memref<32xi32>,
                depth = 1 : i64}

// (c) Tier 2 regression: standard acquire/release inside loop — must still promote.
// expected-remark @+1 {{conduit-depth-promote: promoted 'tier2_loop' from depth-1 to depth-2}}
conduit.create @tier2_loop {slot_elems = 32 : i64,
                producer_tile = array<i64: 0, 0>,
                consumer_tiles = array<i64: 0, 2>,
                element_type = memref<8xi32>,
                depth = 1 : i64}

func.func @tier3_eligible(%buf: memref<32xi32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  scf.for %i = %c0 to %c8 step %c1 {
    %tok = conduit.put_memref_async {name = @tier3_loop, num_elems = 32 : i64,
               offsets = array<i64: 0>, sizes = array<i64: 32>,
               strides = array<i64: 1>} : !conduit.dma.token
    conduit.wait_all %tok : !conduit.dma.token
    // Real compute between put and get.
    %c42 = arith.constant 42 : i32
    memref.store %c42, %buf[%c0] : memref<32xi32>
    %tok2 = conduit.get_memref_async {name = @tier3_loop, num_elems = 32 : i64,
               offsets = array<i64: 0>, sizes = array<i64: 32>,
               strides = array<i64: 1>} : !conduit.dma.token
    conduit.wait_all %tok2 : !conduit.dma.token
  }
  return
}

func.func @tier3_no_loop_kernel(%buf: memref<32xi32>) {
  %tok = conduit.put_memref_async {name = @tier3_no_loop, num_elems = 32 : i64,
             offsets = array<i64: 0>, sizes = array<i64: 32>,
             strides = array<i64: 1>} : !conduit.dma.token
  conduit.wait_all %tok : !conduit.dma.token
  return
}

func.func @tier2_regression(%result: memref<8xi32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  scf.for %i = %c0 to %c8 step %c1 {
    %win = conduit.acquire {name = @tier2_loop, count = 1 : i64, port = #conduit.port<Consume>}
               : !conduit.window<memref<8xi32>>
    %elem = conduit.subview_access %win {index = 0 : i64}
               : !conduit.window<memref<8xi32>> -> memref<8xi32>
    memref.copy %elem, %result : memref<8xi32> to memref<8xi32>
    conduit.release %win {count = 1 : i64, port = #conduit.port<Consume>}
        : !conduit.window<memref<8xi32>>
  }
  return
}

} // aie.device
} // module
