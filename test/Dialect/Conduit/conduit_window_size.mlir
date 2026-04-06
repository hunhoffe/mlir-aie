// RUN: aie-opt --split-input-file --verify-diagnostics %s

// MVE-3: window_size attribute on conduit.acquire / conduit.acquire_async.
// M7-window check: channel depth must be >= window_size.

// ---- Valid: depth=4 >= window_size=3 → no error ----

aie.device(npu2) {
  conduit.create @sliding_window {
    capacity = 64 : i64,
    depth = 4 : i64,
    element_type = memref<32xi32>,
    producer_tile = array<i64: 0, 0>,
    consumer_tiles = array<i64: 0, 2>
  }
  %tile = aie.tile(0, 2)
  %core = aie.core(%tile) {
    %w = conduit.acquire {name = @sliding_window, count = 3 : i64, port = "Consume", window_size = 3 : i64} : !conduit.window<memref<32xi32>>
    conduit.release {name = @sliding_window, count = 1 : i64, port = "Consume"}
    aie.end
  }
}

// -----

// ---- Error: depth=2 < window_size=3 → M7-window error ----

aie.device(npu2) {
  conduit.create @bad_window {
    capacity = 64 : i64,
    depth = 2 : i64,
    element_type = memref<32xi32>,
    producer_tile = array<i64: 0, 0>,
    consumer_tiles = array<i64: 0, 2>
  }
  %tile = aie.tile(0, 2)
  %core = aie.core(%tile) {
    // expected-error @+1 {{depth must be >= window_size}}
    %w = conduit.acquire {name = @bad_window, count = 3 : i64, port = "Consume", window_size = 3 : i64} : !conduit.window<memref<32xi32>>
    conduit.release {name = @bad_window, count = 1 : i64, port = "Consume"}
    aie.end
  }
}
