// RUN: aie-opt %s -split-input-file -verify-diagnostics

// distribute mode: offsets count must equal dsts count
func.func @bad_distribute_offsets() {
  // expected-error@+1 {{'conduit.objectfifo_link' op distribute mode: offsets count (1) must equal dsts count (2)}}
  conduit.objectfifo_link {srcs = ["in"], dsts = ["out0", "out1"],
                           mode = "distribute", memtile = "tile(0,1)",
                           offsets = array<i64: 0>}
  return
}

// -----

// join mode: offsets count must equal srcs count
func.func @bad_join_offsets() {
  // expected-error@+1 {{'conduit.objectfifo_link' op join mode: offsets count (1) must equal srcs count (2)}}
  conduit.objectfifo_link {srcs = ["in0", "in1"], dsts = ["out"],
                           mode = "join", memtile = "tile(0,1)",
                           offsets = array<i64: 0>}
  return
}

// -----

// M2: subview_access index out of bounds for conduit depth
func.func @bad_subview_index() {
  conduit.create {name = "fifo", capacity = 8 : i64,
                  producer_tile = array<i64: 0, 2>,
                  consumer_tiles = array<i64: 0, 3>,
                  element_type = memref<8xi32>,
                  depth = 2 : i64}
  %win = conduit.acquire {name = "fifo", count = 1 : i64, port = "Consume"}
             : !conduit.window<memref<8xi32>>
  // expected-error@+1 {{'conduit.subview_access' op index 2 out of bounds for conduit of depth 2}}
  %elem = conduit.subview_access %win {index = 2 : i64}
             : !conduit.window<memref<8xi32>> -> memref<8xi32>
  conduit.release %win {count = 1 : i64, port = "Consume"}
      : !conduit.window<memref<8xi32>>
  return
}

// -----

// M3: distribute mode requires exactly 1 src
func.func @bad_distribute_multiple_srcs() {
  // expected-error@+1 {{'conduit.objectfifo_link' op distribute mode requires exactly 1 src, got 2}}
  conduit.objectfifo_link {srcs = ["in0", "in1"], dsts = ["out0", "out1"],
                           mode = "distribute", memtile = "tile(0,1)"}
  return
}

// -----

// M3: join mode requires exactly 1 dst
func.func @bad_join_multiple_dsts() {
  // expected-error@+1 {{'conduit.objectfifo_link' op join mode requires exactly 1 dst, got 2}}
  conduit.objectfifo_link {srcs = ["in0", "in1"], dsts = ["out0", "out1"],
                           mode = "join", memtile = "tile(0,1)"}
  return
}

// -----

// M3: unknown mode
func.func @bad_unknown_mode() {
  // expected-error@+1 {{'conduit.objectfifo_link' op unknown mode 'relay'; expected distribute, join, or forward}}
  conduit.objectfifo_link {srcs = ["in"], dsts = ["out"],
                           mode = "relay", memtile = "tile(0,1)"}
  return
}

// -----

// M6: CSDF — unbalanced producer and consumer rates (different sums, same period)
// P=[1,2] sum=3 period=2, C=[2,3] sum=5 period=2
// CSDF check: 3*2 != 5*2 → 6 != 10
func.func @bad_csdf_imbalanced_rates() {
  // expected-error@+1 {{'conduit.create' op CSDF rate imbalance: sum(producer_rates)*len(consumer_rates)=6 != sum(consumer_rates)*len(producer_rates)=10}}
  conduit.create {name = "csdf_bad", capacity = 8 : i64,
                  producer_tile = array<i64: 0, 2>,
                  consumer_tiles = array<i64: 0, 3>,
                  element_type = memref<i32>,
                  depth = 8 : i64,
                  producer_rates = array<i64: 1, 2>,
                  consumer_rates = array<i64: 2, 3>}
  return
}

// -----

// M6: CSDF — sum-equal but period-imbalanced (catches the wrong sum==sum check)
// P=[3] sum=3 period=1, C=[1,2] sum=3 period=2
// Wrong check (sum==sum): 3==3 → would PASS (incorrectly)
// Correct check: sum(P)*len(C) == sum(C)*len(P) → 3*2 != 3*1 → 6 != 3 → FAIL (correct)
// No integer firing vector exists: producer fires 1x/period, consumer fires 2x/period,
// but producer delivers 3 tokens and consumer expects 1+2=3 tokens per consumer cycle —
// except producer period=1 and consumer period=2 means in 2 producer firings (6 tokens)
// vs 1 consumer firing (3 tokens): 6 != 3 → imbalanced.
func.func @bad_csdf_sum_equal_period_imbalanced() {
  // expected-error@+1 {{'conduit.create' op CSDF rate imbalance: sum(producer_rates)*len(consumer_rates)=6 != sum(consumer_rates)*len(producer_rates)=3}}
  conduit.create {name = "csdf_sum_equal_bad", capacity = 6 : i64,
                  producer_tile = array<i64: 0, 2>,
                  consumer_tiles = array<i64: 0, 3>,
                  element_type = memref<i32>,
                  depth = 6 : i64,
                  producer_rates = array<i64: 3>,
                  consumer_rates = array<i64: 1, 2>}
  return
}

// -----

// M6: CSDF — only producer_rates provided (missing consumer_rates)
func.func @bad_csdf_missing_consumer_rates() {
  // expected-error@+1 {{'conduit.create' op CSDF requires both producer_rates and consumer_rates; only one was provided}}
  conduit.create {name = "csdf_incomplete", capacity = 4 : i64,
                  producer_tile = array<i64: 0, 2>,
                  consumer_tiles = array<i64: 0, 3>,
                  element_type = memref<i32>,
                  depth = 4 : i64,
                  producer_rates = array<i64: 1, 2>}
  return
}

// -----

// M6: CSDF — only consumer_rates provided (missing producer_rates)
func.func @bad_csdf_missing_producer_rates() {
  // expected-error@+1 {{'conduit.create' op CSDF requires both producer_rates and consumer_rates; only one was provided}}
  conduit.create {name = "csdf_incomplete2", capacity = 4 : i64,
                  producer_tile = array<i64: 0, 2>,
                  consumer_tiles = array<i64: 0, 3>,
                  element_type = memref<i32>,
                  depth = 4 : i64,
                  consumer_rates = array<i64: 1, 2>}
  return
}

// -----

// M5: bad routing_mode value
func.func @bad_routing_mode() {
  // expected-error@+1 {{'conduit.create' op routing_mode must be "circuit" or "packet", got "broadcast"}}
  conduit.create {name = "bad_mode_ch", capacity = 4 : i64,
                  routing_mode = "broadcast"}
  return
}

// -----

// Token type mismatch: conduit.wait requires !conduit.dma.token;
// passing !conduit.window.token must fail type checking.
// (Type enforcement is TableGen-generated; this tests that the type system rejects it.)
func.func @bad_wait_window_token() {
  conduit.create {name = "w", capacity = 1 : i64}
  %tok = conduit.acquire_async {name = "w", count = 1 : i64}
             : !conduit.window.token
  // expected-error@+1 {{operand #0 must be}}
  conduit.wait %tok : !conduit.window.token
  return
}

// -----

// Token type mismatch: conduit.wait_window requires !conduit.window.token;
// passing !conduit.dma.token must fail type checking.
func.func @bad_wait_with_dma_token() {
  conduit.create {name = "ch", capacity = 64 : i64}
  %tok = conduit.put_memref_async {name = "ch", num_elems = 64 : i64,
             offsets = array<i64: 0>, sizes = array<i64: 64>,
             strides = array<i64: 1>} : !conduit.dma.token
  // expected-error@+2 {{invalid kind of type specified: expected}}
  %win = conduit.wait_window %tok for "ch"
             : !conduit.dma.token -> !conduit.window<memref<64xi32>>
  return
}

// -----

// forward mode: requires exactly 1 src and 1 dst; 2 srcs must fail.
func.func @bad_forward_two_srcs() {
  // expected-error@+1 {{'conduit.objectfifo_link' op forward mode requires exactly 1 src and 1 dst}}
  conduit.objectfifo_link {srcs = ["in0", "in1"], dsts = ["out"],
                           mode = "forward", memtile = "tile(0,1)"}
  return
}

// -----

// forward mode: requires exactly 1 src and 1 dst; 2 dsts must fail.
func.func @bad_forward_two_dsts() {
  // expected-error@+1 {{'conduit.objectfifo_link' op forward mode requires exactly 1 src and 1 dst}}
  conduit.objectfifo_link {srcs = ["in"], dsts = ["out0", "out1"],
                           mode = "forward", memtile = "tile(0,1)"}
  return
}

// -----

// M7: CSDF buffer capacity insufficient.
// producer_rates = [3, 1] (sum=4, period=2); consumer_rates = [2] (sum=2, period=1)
// M6 balance check: sum(P)*len(C) = 4*1 = 4 == sum(C)*len(P) = 2*2 = 4  ✓  (passes M6)
// M7 hyper-period simulation (H=lcm(2,1)=2 steps, produce-then-consume each step):
//   t=0: produce 3 (occ=3), consume 2 (occ=1) — peak=3
//   t=1: produce 1 (occ=2), consume 2 (occ=0) — peak still 3
// Peak occupancy = 3 > capacity = 2 → M7 error.
func.func @bad_csdf_capacity_insufficient() {
  // expected-error@+1 {{M7: CSDF buffer capacity insufficient: peak token occupancy over one hyper-period=3 exceeds capacity=2}}
  conduit.create {name = "csdf_cap_bad", capacity = 2 : i64,
                  producer_tile = array<i64: 0, 2>,
                  consumer_tiles = array<i64: 0, 3>,
                  element_type = memref<i32>,
                  depth = 2 : i64,
                  producer_rates = array<i64: 3, 1>,
                  consumer_rates = array<i64: 2>}
  return
}

// -----

// M7: CSDF hyper-period simulation underflow warning (not an error).
// producer_rates = [1, 3] (sum=4, period=2); consumer_rates = [3, 1] (sum=4, period=2)
// M6 balance check: sum(P)*len(C) = 4*2 = 8 == sum(C)*len(P) = 4*2 = 8  ✓  (passes M6)
// M7 hyper-period simulation (H=lcm(2,2)=2 steps, produce-before-consume each step):
//   t=0: produce 1 (occ=1), consume 3 → occ=-2 → underflow warning at step 0
//   t=1: produce 3 (occ=1, after reset), consume 1 → occ=0
// Peak occupancy = 3 (at t=1 after produce), capacity=4 → no capacity error.
// This tests that underflow emits emitWarning (not emitOpError): the program is
// accepted (M6 guarantees feasibility) but the user is warned that the
// produce-before-consume simulation interleaving hits underflow.
func.func @warn_csdf_underflow() {
  // expected-warning@+1 {{M7: CSDF hyper-period simulation: momentary underflow at step 0}}
  conduit.create {name = "csdf_underflow", capacity = 4 : i64,
                  producer_tile = array<i64: 0, 2>,
                  consumer_tiles = array<i64: 0, 3>,
                  element_type = memref<i32>,
                  depth = 4 : i64,
                  producer_rates = array<i64: 1, 3>,
                  consumer_rates = array<i64: 3, 1>}
  return
}

// -----

// M8a: double release — acquire(count=1) + release(count=1) + release(count=1)
// = cumulative 2 > acquired 1 → hardware lock-counter overflow.
func.func @m8a_double_release() {
  conduit.create {name = "dbl", capacity = 1 : i64,
                  producer_tile = array<i64: 0, 2>,
                  consumer_tiles = array<i64: 0, 3>,
                  element_type = memref<1xi32>,
                  depth = 1 : i64}
  // expected-error@+1 {{'conduit.acquire' op M8: cumulative release count (2) exceeds acquired count (1) -- double-release causes hardware lock-counter overflow}}
  %win = conduit.acquire {name = "dbl", count = 1 : i64, port = "Consume"}
             : !conduit.window<memref<1xi32>>
  conduit.release %win {count = 1 : i64, port = "Consume"}
      : !conduit.window<memref<1xi32>>
  conduit.release %win {count = 1 : i64, port = "Consume"}
      : !conduit.window<memref<1xi32>>
  return
}

// -----

// M8c: !conduit.window<T> is not a token type — rejected by wait_all.
func.func @m8c_wait_all_window_value() {
  conduit.create {name = "unx", capacity = 1 : i64,
                  producer_tile = array<i64: 0, 2>,
                  consumer_tiles = array<i64: 0, 3>,
                  element_type = memref<1xi32>,
                  depth = 1 : i64}
  %win = conduit.acquire {name = "unx", count = 1 : i64, port = "Consume"}
             : !conduit.window<memref<1xi32>>
  // expected-error@+1 {{'conduit.wait_all' op operand #0 must be variadic of conduit token type, but got '!conduit.window<memref<1xi32>>'}}
  conduit.wait_all %win : !conduit.window<memref<1xi32>>
  return
}

// -----

// M8b: two wait_window on same token → double-materialization, deadlock.
func.func @m8b_double_wait_window() {
  conduit.create {name = "dbl_tok", capacity = 1 : i64,
                  producer_tile = array<i64: 0, 2>,
                  consumer_tiles = array<i64: 0, 3>,
                  element_type = memref<1xi32>,
                  depth = 1 : i64}
  // expected-error@+1 {{'conduit.acquire_async' op M8: window.token has 2 conduit.wait_window uses}}
  %tok = conduit.acquire_async {name = "dbl_tok", count = 1 : i64}
             : !conduit.window.token
  %win1 = conduit.wait_window %tok for "dbl_tok"
              : !conduit.window.token -> !conduit.window<memref<1xi32>>
  %win2 = conduit.wait_window %tok for "dbl_tok"
              : !conduit.window.token -> !conduit.window<memref<1xi32>>
  conduit.release %win1 {count = 1 : i64, port = "Consume"}
      : !conduit.window<memref<1xi32>>
  conduit.release %win2 {count = 1 : i64, port = "Consume"}
      : !conduit.window<memref<1xi32>>
  return
}

// -----

// M8c: i32 operand in wait_all is not a token type.
func.func @m8c_wait_all_non_token(%bad : i32) {
  conduit.create {name = "ntok", capacity = 1 : i64}
  %tok = conduit.put_memref_async {name = "ntok", num_elems = 1 : i64,
             offsets = array<i64: 0>, sizes = array<i64: 1>,
             strides = array<i64: 1>} : !conduit.dma.token
  // expected-error@+1 {{'conduit.wait_all' op operand #1 must be variadic of conduit token type, but got 'i32'}}
  conduit.wait_all %tok, %bad : !conduit.dma.token, i32
  return
}

// -----

// M10: window.token escapes via return — hardware state is not portable.
func.func @m10_window_token_escape_return() -> !conduit.window.token {
  conduit.create {name = "esc", capacity = 1 : i64}
  // expected-error@+1 {{'conduit.acquire_async' op M10: token escapes function scope via return}}
  %tok = conduit.acquire_async {name = "esc", count = 1 : i64}
             : !conduit.window.token
  return %tok : !conduit.window.token
}

// -----

// M10: dma.token escapes via return — hardware state is not portable.
func.func @m10_dma_token_escape_return() -> !conduit.dma.token {
  conduit.create {name = "esc_dma", capacity = 64 : i64}
  // expected-error@+1 {{'conduit.put_memref_async' op M10: token escapes function scope via return}}
  %tok = conduit.put_memref_async {name = "esc_dma", num_elems = 64 : i64,
             offsets = array<i64: 0>, sizes = array<i64: 64>,
             strides = array<i64: 1>} : !conduit.dma.token
  return %tok : !conduit.dma.token
}

// -----

// M10: window.token escapes via call argument.
func.func private @callee(%tok : !conduit.window.token)
func.func @m10_token_escape_call() {
  conduit.create {name = "esc_call", capacity = 1 : i64}
  // expected-error@+1 {{'conduit.acquire_async' op M10: token escapes function scope via call argument}}
  %tok = conduit.acquire_async {name = "esc_call", count = 1 : i64}
             : !conduit.window.token
  func.call @callee(%tok) : (!conduit.window.token) -> ()
  return
}

// -----

// M10: wait_all_async result (dma.token) escapes via return.
// wait_all_async merges completion tokens into a single dma.token result;
// that result must not escape function scope.
func.func @m10_wait_all_async_token_escape() -> !conduit.dma.token {
  conduit.create {name = "wa_esc", capacity = 64 : i64}
  %tok = conduit.put_memref_async {name = "wa_esc", num_elems = 64 : i64,
             offsets = array<i64: 0>, sizes = array<i64: 64>,
             strides = array<i64: 1>} : !conduit.dma.token
  // expected-error@+1 {{'conduit.wait_all_async' op M10: token escapes function scope via return}}
  %merged = conduit.wait_all_async %tok : (!conduit.dma.token) -> !conduit.dma.token
  return %merged : !conduit.dma.token
}
