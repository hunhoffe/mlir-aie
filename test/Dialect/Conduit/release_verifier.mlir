// RUN: aie-opt --split-input-file --verify-diagnostics %s
//
// M2: conduit.release verifier regression tests.
//
// Section 1: valid release — no errors expected.
// Section 2: count=0 — release count must be > 0.
// Section 3: port mismatch — release port != acquire port.
// Section 4: count exceeds acquire count.
// Section 5: valid async path (wait_window → acquire_async).
// Section 6: port mismatch via async path.
// Section 7: count exceeds acquire_async count via async path.

// -----

// Section 1: PASS — valid acquire + release with matching port and count.

aie.device(npu2) {
  conduit.create @fifo {
    depth = 2 : i64,
    element_type = memref<32xi32>
  }
  %tile = aie.tile(0, 2)
  %core = aie.core(%tile) {
    %w = conduit.acquire {name = @fifo, count = 2 : i64,
                          port = #conduit.port<Consume>}
                         : !conduit.window<memref<32xi32>>
    conduit.release %w {count = 1 : i64, port = #conduit.port<Consume>}
        : !conduit.window<memref<32xi32>>
    aie.end
  }
}

// -----

// Section 2: count=0 — release count must be > 0.

aie.device(npu2) {
  conduit.create @fifo {
    depth = 2 : i64,
    element_type = memref<32xi32>
  }
  %tile = aie.tile(0, 2)
  %core = aie.core(%tile) {
    %w = conduit.acquire {name = @fifo, count = 1 : i64,
                          port = #conduit.port<Consume>}
                         : !conduit.window<memref<32xi32>>
    // expected-error @+1 {{'conduit.release' op release count must be > 0}}
    conduit.release %w {count = 0 : i64, port = #conduit.port<Consume>}
        : !conduit.window<memref<32xi32>>
    aie.end
  }
}

// -----

// Section 3: port mismatch — acquired Consume, releasing Produce.

aie.device(npu2) {
  conduit.create @fifo {
    depth = 2 : i64,
    element_type = memref<32xi32>
  }
  %tile = aie.tile(0, 2)
  %core = aie.core(%tile) {
    %w = conduit.acquire {name = @fifo, count = 1 : i64,
                          port = #conduit.port<Consume>}
                         : !conduit.window<memref<32xi32>>
    // expected-error @+1 {{'conduit.release' op release port (Produce) does not match acquire port (Consume)}}
    conduit.release %w {count = 1 : i64, port = #conduit.port<Produce>}
        : !conduit.window<memref<32xi32>>
    aie.end
  }
}

// -----

// Section 4: release count exceeds acquire count.

aie.device(npu2) {
  conduit.create @fifo {
    depth = 4 : i64,
    element_type = memref<32xi32>
  }
  %tile = aie.tile(0, 2)
  %core = aie.core(%tile) {
    // expected-error @+1 {{'conduit.acquire' op M8: cumulative release count (3) exceeds acquired count (2) -- double-release causes hardware lock-counter overflow}}
    %w = conduit.acquire {name = @fifo, count = 2 : i64,
                          port = #conduit.port<Consume>}
                         : !conduit.window<memref<32xi32>>
    conduit.release %w {count = 3 : i64, port = #conduit.port<Consume>}
        : !conduit.window<memref<32xi32>>
    aie.end
  }
}

// -----

// Section 5: PASS — valid async path: acquire_async → wait_window → release.

aie.device(npu2) {
  conduit.create @fifo {
    depth = 2 : i64,
    element_type = memref<32xi32>
  }
  %tile = aie.tile(0, 2)
  %core = aie.core(%tile) {
    %tok = conduit.acquire_async {name = @fifo, count = 2 : i64,
                                  port = #conduit.port<Consume>}
                                 : !conduit.window.token
    %w = conduit.wait_window %tok for @fifo
             : !conduit.window.token -> !conduit.window<memref<32xi32>>
    conduit.release %w {count = 1 : i64, port = #conduit.port<Consume>}
        : !conduit.window<memref<32xi32>>
    aie.end
  }
}

// -----

// Section 6: port mismatch via async path — acquire_async Consume, release Produce.

aie.device(npu2) {
  conduit.create @fifo {
    depth = 2 : i64,
    element_type = memref<32xi32>
  }
  %tile = aie.tile(0, 2)
  %core = aie.core(%tile) {
    %tok = conduit.acquire_async {name = @fifo, count = 1 : i64,
                                  port = #conduit.port<Consume>}
                                 : !conduit.window.token
    %w = conduit.wait_window %tok for @fifo
             : !conduit.window.token -> !conduit.window<memref<32xi32>>
    // expected-error @+1 {{'conduit.release' op release port (Produce) does not match acquire_async port (Consume)}}
    conduit.release %w {count = 1 : i64, port = #conduit.port<Produce>}
        : !conduit.window<memref<32xi32>>
    aie.end
  }
}

// -----

// Section 7: count exceeds acquire_async count via async path.

aie.device(npu2) {
  conduit.create @fifo {
    depth = 4 : i64,
    element_type = memref<32xi32>
  }
  %tile = aie.tile(0, 2)
  %core = aie.core(%tile) {
    %tok = conduit.acquire_async {name = @fifo, count = 1 : i64,
                                  port = #conduit.port<Consume>}
                                 : !conduit.window.token
    // expected-error @+1 {{'conduit.wait_window' op M8: cumulative release count (3) exceeds acquired count (1) -- double-release causes hardware lock-counter overflow}}
    %w = conduit.wait_window %tok for @fifo
             : !conduit.window.token -> !conduit.window<memref<32xi32>>
    conduit.release %w {count = 3 : i64, port = #conduit.port<Consume>}
        : !conduit.window<memref<32xi32>>
    aie.end
  }
}
