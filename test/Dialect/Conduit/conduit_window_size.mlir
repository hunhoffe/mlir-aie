// RUN: aie-opt --split-input-file --verify-diagnostics %s

// MVE-3: sliding-window acquire/release pattern validity tests.
// M7-window check (window_size attr on conduit.create, Sprint 1): placeholder.
// For now verifies that sliding-window acquire(count=3)/release(count=1) is
// valid IR when depth >= count.

// ---- Valid: depth=4, acquire count=3, partial release count=1 → no error ----

aie.device(npu2) {
  conduit.create @sliding_window {
    slot_elems = 64 : i64,
    depth = 4 : i64,
    element_type = memref<32xi32>
  }
  %tile = aie.tile(0, 2)
  %core = aie.core(%tile) {
    %w = conduit.acquire {name = @sliding_window, count = 3 : i64,
                          port = #conduit.port<Consume>}
                         : !conduit.window<memref<32xi32>>
    conduit.release %w {count = 1 : i64, port = #conduit.port<Consume>}
        : !conduit.window<memref<32xi32>>
    aie.end
  }
}
