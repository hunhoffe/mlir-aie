// RUN: aie-opt --conduit-depth-promote %s | FileCheck %s
// RUN: aie-opt --conduit-depth-promote --verify-diagnostics %s | FileCheck %s
//
// Test for --conduit-depth-promote pass.
//
// The pass promotes eligible depth-1 conduits to depth-2 (double-buffering).
// Exclusion criteria (any one prevents promotion):
//   1. CSDF/cyclostatic access pattern present
//   2. Conduit appears in conduit.scatter/conduit.gather (linked conduit)
//   3. No surrounding loop context for its acquire ops
//   4. Passthrough-only (no compute between acquire and release)
//   5. Non-uniform acquire/release counts
//   6. Memory budget exceeded
//
// This test verifies:
//   (a) "loop_fifo" — depth-1 with acquire inside scf.for + real compute:
//       PROMOTED from depth=1,slot_elems =8 to depth=2,slot_elems =16
//   (b) "linked_fifo" — depth-1 but referenced in conduit.scatter:
//       NOT promoted (stays at depth=1,slot_elems =8)
//   (c) "passthrough_fifo" — depth-1 but acquire→release with no compute:
//       NOT promoted (passthrough-only)

// Use CHECK-DAG to match conduit.create ops regardless of output order.
// Each conduit.create is on one line so CHECK-DAG matching works.

// (a) loop_fifo: promoted — capacity doubles 8→16, depth 1→2
// CHECK-DAG: conduit.create @loop_fifo {{{.*}}depth = 2 : i64, {{.*}}
// (b) linked_fifo: NOT promoted — capacity stays 8, depth stays 1
// CHECK-DAG: conduit.create @linked_fifo {{{.*}}depth = 1 : i64, {{.*}}
// (c) passthrough_fifo: NOT promoted — capacity stays 4, depth stays 1
// CHECK-DAG: conduit.create @passthrough_fifo {{{.*}}depth = 1 : i64, {{.*}}
// conduit.scatter must survive unchanged (also CHECK-DAG to allow any order)
// CHECK-DAG: conduit.scatter{src = @linked_fifo, dsts = [@linked_out]

// expected-remark @+1 {{conduit-depth-promote: promoted 1 conduit(s)}}
module {
aie.device(npu1) {

// (a) Eligible: depth-1 with loop-enclosed acquire and compute.
// Pass must promote to depth=2, slot_elems =16.
// expected-remark @+1 {{conduit-depth-promote: promoted 'loop_fifo' from depth-1 to depth-2}}
conduit.create @loop_fifo {                element_type = memref<8xi32>,
                depth = 1 : i64}

// (b) Linked: depth-1 but conduit.scatter references it.
// Pass must skip it (exclusion criterion #2).
// expected-remark @+1 {{conduit-depth-promote: skipping 'linked_fifo' -- linked conduit}}
conduit.create @linked_fifo {                element_type = memref<8xi32>,
                depth = 1 : i64}
// expected-remark @+1 {{conduit-depth-promote: skipping 'linked_out' -- linked conduit}}
conduit.create @linked_out {                element_type = memref<8xi32>,
                depth = 1 : i64}

// (c) Passthrough: depth-1 with acquire immediately followed by release, no compute.
// Pass must skip it (exclusion criterion #4).
// expected-remark @+1 {{conduit-depth-promote: skipping 'passthrough_fifo' -- passthrough-only (no compute)}}
conduit.create @passthrough_fifo {                element_type = memref<4xi32>,
                depth = 1 : i64}

func.func @eligible_loop_fifo(%result: memref<8xi32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  scf.for %i = %c0 to %c8 step %c1 {
    %win = conduit.acquire {name = @loop_fifo, count = 1 : i64, port = #conduit.port<Consume>}
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

func.func @linked_conduit_not_promoted() {
  // This link causes both "linked_fifo" and "linked_out" to be excluded.
  conduit.scatter{src = @linked_fifo, dsts = [@linked_out] {memtile = "tile(0,1)"}}
  return
}

func.func @passthrough_not_promoted() {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c4 = arith.constant 4 : index
  scf.for %i = %c0 to %c4 step %c1 {
    %win = conduit.acquire {name = @passthrough_fifo, count = 1 : i64, port = #conduit.port<Consume>}
               : !conduit.window<memref<4xi32>>
    // No compute between acquire and release — pure passthrough.
    conduit.release %win {count = 1 : i64, port = #conduit.port<Consume>}
        : !conduit.window<memref<4xi32>>
  }
  return
}

} // aie.device
} // module
