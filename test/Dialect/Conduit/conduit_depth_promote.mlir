// RUN: aie-opt --conduit-depth-promote %s | FileCheck %s
// RUN: aie-opt --conduit-depth-promote --verify-diagnostics %s | FileCheck %s
//
// Test for --conduit-depth-promote pass.
//
// The pass promotes eligible depth-1 conduits to depth-2 (double-buffering).
// Exclusion criteria (any one prevents promotion):
//   1. CSDF/cyclostatic access pattern present
//   2. Conduit appears in conduit.link srcs or dsts (linked conduit)
//   3. No surrounding loop context for its acquire ops
//   4. Passthrough-only (no compute between acquire and release)
//   5. Non-uniform acquire/release counts
//   6. Memory budget exceeded
//
// This test verifies:
//   (a) "loop_fifo" — depth-1 with acquire inside scf.for + real compute:
//       PROMOTED from depth=1,capacity=8 to depth=2,capacity=16
//   (b) "linked_fifo" — depth-1 but referenced in conduit.link:
//       NOT promoted (stays at depth=1,capacity=8)
//   (c) "passthrough_fifo" — depth-1 but acquire→release with no compute:
//       NOT promoted (passthrough-only)

// Use CHECK-DAG to match conduit.create ops regardless of output order.
// Each conduit.create is on one line so CHECK-DAG matching works.

// (a) loop_fifo: promoted — capacity doubles 8→16, depth 1→2
// CHECK-DAG: conduit.create {capacity = 16 : i64, {{.*}} depth = 2 : i64, {{.*}} name = "loop_fifo"

// (b) linked_fifo: NOT promoted — capacity stays 8, depth stays 1
// CHECK-DAG: conduit.create {capacity = 8 : i64, {{.*}} depth = 1 : i64, {{.*}} name = "linked_fifo"

// (c) passthrough_fifo: NOT promoted — capacity stays 4, depth stays 1
// CHECK-DAG: conduit.create {capacity = 4 : i64, {{.*}} depth = 1 : i64, {{.*}} name = "passthrough_fifo"

// conduit.link must survive unchanged (also CHECK-DAG to allow any order)
// CHECK-DAG: conduit.link

// expected-remark @+1 {{conduit-depth-promote: promoted 1 conduit(s) to depth-2}}
module {

// (a) Eligible: depth-1 with loop-enclosed acquire and compute.
// Pass must promote to depth=2, capacity=16.
func.func @eligible_loop_fifo(%result: memref<8xi32>) {
  // expected-remark @+1 {{conduit-depth-promote: promoted 'loop_fifo' from depth-1 to depth-2}}
  conduit.create {name = "loop_fifo", capacity = 8 : i64,
                  producer_tile = array<i64: 0, 0>,
                  consumer_tiles = array<i64: 0, 2>,
                  element_type = memref<8xi32>,
                  depth = 1 : i64}
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  scf.for %i = %c0 to %c8 step %c1 {
    %win = conduit.acquire {name = "loop_fifo", count = 1 : i64, port = #conduit.port<Consume>}
               : !conduit.window<memref<8xi32>>
    %elem = conduit.subview_access %win {index = 0 : i64}
               : !conduit.window<memref<8xi32>> -> memref<8xi32>
    // Real compute: copy element to result (not a passthrough).
    memref.copy %elem, %result : memref<8xi32> to memref<8xi32>
    conduit.release %win {count = 1 : i64, port = #conduit.port<Consume>}
        : !conduit.window<memref<8xi32>>
  }
  return
}

// (b) Linked: depth-1 but conduit.link references it.
// Pass must skip it (exclusion criterion #2).
func.func @linked_conduit_not_promoted() {
  // expected-remark @+1 {{conduit-depth-promote: skipping 'linked_fifo' -- linked conduit}}
  conduit.create {name = "linked_fifo", capacity = 8 : i64,
                  producer_tile = array<i64: 0, 0>,
                  consumer_tiles = array<i64: 0, 1>,
                  element_type = memref<8xi32>,
                  depth = 1 : i64}
  // expected-remark @+1 {{conduit-depth-promote: skipping 'linked_out' -- linked conduit}}
  conduit.create {name = "linked_out", capacity = 8 : i64,
                  producer_tile = array<i64: 0, 1>,
                  consumer_tiles = array<i64: 0, 2>,
                  element_type = memref<8xi32>,
                  depth = 1 : i64}
  // This link causes both "linked_fifo" and "linked_out" to be excluded.
  conduit.link {srcs = ["linked_fifo"], dsts = ["linked_out"],
                           mode = "forward", memtile = "tile(0,1)"}
  return
}

// (c) Passthrough: depth-1 with acquire immediately followed by release, no compute.
// Pass must skip it (exclusion criterion #4).
func.func @passthrough_not_promoted() {
  // expected-remark @+1 {{conduit-depth-promote: skipping 'passthrough_fifo' -- passthrough-only (no compute)}}
  conduit.create {name = "passthrough_fifo", capacity = 4 : i64,
                  producer_tile = array<i64: 0, 0>,
                  consumer_tiles = array<i64: 0, 3>,
                  element_type = memref<4xi32>,
                  depth = 1 : i64}
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  scf.for %i = %c0 to %c4 step %c1 {
    %win = conduit.acquire {name = "passthrough_fifo", count = 1 : i64, port = #conduit.port<Consume>}
               : !conduit.window<memref<4xi32>>
    // No compute between acquire and release — pure passthrough.
    conduit.release %win {count = 1 : i64, port = #conduit.port<Consume>}
        : !conduit.window<memref<4xi32>>
  }
  return
}

} // module
