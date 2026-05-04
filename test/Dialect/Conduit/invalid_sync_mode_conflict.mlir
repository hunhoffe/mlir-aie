// RUN: aie-opt -split-input-file -verify-diagnostics %s
//
// Negative tests: sync_mode enum validation on conduit.create.
// sync_mode = none is the canonical way to suppress lock emission.
// sync_mode values barrier, independent, none are all valid.
// Invalid (non-enum) sync_mode values are rejected at parse time.

// -----

// Valid: sync_mode = none suppresses lock allocation.
// No error expected.
aie.device(npu1) {
conduit.create @no_locks {element_type = memref<4xi32>, depth = 1 : i64,
                           sync_mode = #conduit.sync_mode<none>}
}

// -----

// Valid: sync_mode = barrier (default protocol).
aie.device(npu1) {
conduit.create @with_barrier {element_type = memref<4xi32>, depth = 1 : i64,
                               sync_mode = #conduit.sync_mode<barrier>}
}

// -----

// Valid: sync_mode = independent (per-endpoint lock pair).
aie.device(npu1) {
conduit.create @with_independent {element_type = memref<4xi32>, depth = 1 : i64,
                                   sync_mode = #conduit.sync_mode<independent>}
}
