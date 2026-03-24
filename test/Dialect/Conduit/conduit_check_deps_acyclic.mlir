// RUN: aie-opt %s | FileCheck %s
// NOTE: --conduit-check-deps is not yet implemented (Task #26 identified it as
// future work). This test validates the IR parses and round-trips correctly with
// the proper $deps assembly syntax: [%tok : type] not (%tok) : (type) -> type.
//
// Verify that --conduit-check-deps accepts acyclic dep-token DAGs without
// error.  The pass should produce no diagnostics and the IR should pass
// through unchanged.
//
// Dep graph tested here (all acyclic):
//
//   Case 1 — linear chain:
//     put_A → get_B → wait_all_async → (consumed by conduit.wait)
//
//   Case 2 — fan-in (diamond):
//     put_X ──┐
//             ├──► wait_all_async → (consumed)
//     get_Y ──┘
//
//   Case 3 — independent (no deps):
//     put_P   (no $deps)
//     get_Q   (no $deps)
//
// None of these form a cycle; the pass must accept all three.

// CHECK: module

module {
  conduit.create @chA {capacity = 64 : i64}
  conduit.create @chB {capacity = 64 : i64}
  conduit.create @chX {capacity = 32 : i64}
  conduit.create @chY {capacity = 32 : i64}
  conduit.create @chP {capacity = 16 : i64}
  conduit.create @chQ {capacity = 16 : i64}

  // Case 1: linear chain A → B → wait_all_async
  func.func @linear_chain(%bufA : memref<64xi32>, %bufB : memref<64xi32>) {
    // put_A produces tok_a (no deps)
    %tok_a = conduit.put_memref_async {name = "chA", num_elems = 64 : i64,
                 offsets = array<i64: 0>, sizes = array<i64: 64>,
                 strides = array<i64: 1>} : !conduit.dma.token
    // get_B depends on tok_a (tok_a → tok_b)
    %tok_b = conduit.get_memref_async[%tok_a : !conduit.dma.token]
                 {name = "chB", num_elems = 64 : i64,
                 offsets = array<i64: 0>, sizes = array<i64: 64>,
                 strides = array<i64: 1>} : !conduit.dma.token
    // wait_all_async depends on tok_b (tok_b → tok_merged)
    %tok_merged = conduit.wait_all_async %tok_b : (!conduit.dma.token) -> !conduit.dma.token
    conduit.wait %tok_merged : !conduit.dma.token
    return
  }

  // Case 2: fan-in diamond — two independent ops merge into wait_all_async
  func.func @fan_in(%bufX : memref<32xi32>, %bufY : memref<32xi32>) {
    // put_X and get_Y are independent (no deps between them)
    %tok_x = conduit.put_memref_async {name = "chX", num_elems = 32 : i64,
                 offsets = array<i64: 0>, sizes = array<i64: 32>,
                 strides = array<i64: 1>} : !conduit.dma.token
    %tok_y = conduit.get_memref_async {name = "chY", num_elems = 32 : i64,
                 offsets = array<i64: 0>, sizes = array<i64: 32>,
                 strides = array<i64: 1>} : !conduit.dma.token
    // Both fan into wait_all_async — tok_x and tok_y are both predecessors
    %tok_merged = conduit.wait_all_async %tok_x, %tok_y
                      : (!conduit.dma.token, !conduit.dma.token) -> !conduit.dma.token
    conduit.wait %tok_merged : !conduit.dma.token
    return
  }

  // Case 3: no deps at all — isolated nodes, trivially acyclic
  func.func @no_deps(%bufP : memref<16xi32>, %bufQ : memref<16xi32>) {
    %tok_p = conduit.put_memref_async {name = "chP", num_elems = 16 : i64,
                 offsets = array<i64: 0>, sizes = array<i64: 16>,
                 strides = array<i64: 1>} : !conduit.dma.token
    %tok_q = conduit.get_memref_async {name = "chQ", num_elems = 16 : i64,
                 offsets = array<i64: 0>, sizes = array<i64: 16>,
                 strides = array<i64: 1>} : !conduit.dma.token
    conduit.wait_all %tok_p, %tok_q : !conduit.dma.token, !conduit.dma.token
    return
  }
}
