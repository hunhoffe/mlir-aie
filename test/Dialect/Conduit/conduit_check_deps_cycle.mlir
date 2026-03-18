// RUN: not aie-opt --conduit-check-deps %s 2>&1 | FileCheck %s
//
// Verify that --conduit-check-deps detects dep-token cycles and emits a hard
// error.  The pass walks the $deps operand DAG over !conduit.dma.token SSA
// values; a cycle means the program is statically deadlocked.
//
// SSA dominance prevents direct value cycles (a token value cannot appear in
// its own def chain).  A realistic dep cycle arises from a loop-carried token
// passed as a block argument: each iteration's put_memref_async depends on the
// previous iteration's result, and the result feeds back into the next
// iteration's dep.  When the dep chain from iteration N wraps back to depend on
// the result of a wait_all_async that itself depends on the put from iteration
// N, the cycle is closed.
//
// Test structure:
//
//   %init_tok = conduit.put_memref_async {...}          : !conduit.dma.token
//   scf.while (%iter_tok = %init_tok) {
//     // %iter_tok is a block argument — loop-carried dep token.
//     // put_B depends on %iter_tok (the previous iteration's token).
//     %tok_b = conduit.put_memref_async [%iter_tok : !conduit.dma.token] {...}
//     // wait_all_async merges tok_b back → tok_merged, which becomes the
//     // next iteration's %iter_tok via scf.while yield.
//     %tok_merged = conduit.wait_all_async %tok_b : (...) -> !conduit.dma.token
//     // Cycle: tok_b depends on iter_tok; tok_merged depends on tok_b;
//     // iter_tok in the next iteration IS tok_merged — so the dep chain
//     // tok_merged → tok_b → iter_tok → tok_merged is a cycle in the DAG
//     // the pass builds (block args are pre-registered as nodes; the yield
//     // back-edge closes the cycle).
//     scf.condition(%true) %tok_merged : !conduit.dma.token
//   } do {
//   ^bb(%iter_tok2 : !conduit.dma.token):
//     scf.yield %iter_tok2 : !conduit.dma.token
//   }
//
// The pass must detect the cycle through the block argument and emit M12.

module {
  conduit.create {name = "chA", capacity = 64 : i64}
  conduit.create {name = "chB", capacity = 64 : i64}

  func.func @dep_cycle() {
    %true = arith.constant true

    // Seed token: no deps.
    %init_tok = conduit.put_memref_async {name = "chA", num_elems = 64 : i64,
                     offsets = array<i64: 0>, sizes = array<i64: 64>,
                     strides = array<i64: 1>} : !conduit.dma.token

    // scf.while carries %iter_tok across iterations.
    // The block argument %iter_tok represents the loop-carried dep token.
    // Each iteration: put_B depends on the previous iteration's merged token.
    // The merged token feeds back as the next %iter_tok — closing the cycle.
    //
    // CHECK: M12: dep token cycle detected: program is statically deadlocked
    // CHECK: circular completion dependency through loop-carried DMA token
    %_ = scf.while (%iter_tok = %init_tok) : (!conduit.dma.token) -> !conduit.dma.token {
      // put_B deps on %iter_tok: creates edge iter_tok → tok_b in the DAG.
      %tok_b = conduit.put_memref_async [%iter_tok : !conduit.dma.token]
                   {name = "chB", num_elems = 64 : i64,
                    offsets = array<i64: 0>, sizes = array<i64: 64>,
                    strides = array<i64: 1>} : !conduit.dma.token
      // wait_all_async deps on tok_b: creates edge tok_b → tok_merged.
      %tok_merged = conduit.wait_all_async %tok_b
                        : (!conduit.dma.token) -> !conduit.dma.token
      // scf.condition yields tok_merged as the next iteration's iter_tok.
      // This closes the cycle: tok_merged is yielded back as iter_tok,
      // which is the dep for the next iteration's tok_b.
      // DAG cycle: iter_tok → tok_b → tok_merged → (iter_tok in next iter)
      scf.condition(%true) %tok_merged : !conduit.dma.token
    } do {
    ^bb0(%iter_tok2 : !conduit.dma.token):
      scf.yield %iter_tok2 : !conduit.dma.token
    }

    conduit.wait %_ : !conduit.dma.token
    return
  }
}
