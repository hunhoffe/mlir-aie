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
