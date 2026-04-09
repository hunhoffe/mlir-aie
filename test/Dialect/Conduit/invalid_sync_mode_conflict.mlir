// RUN: aie-opt -split-input-file -verify-diagnostics %s
//
// Negative tests: sync_mode and disable_synchronization=true are mutually
// exclusive on conduit.create.
//
// sync_mode specifies an active synchronization protocol (barrier or
// independent N-way ack locks).  disable_synchronization=true suppresses ALL
// lock allocation and use_lock emission.  Setting both is contradictory:
// which protocol should Pass C implement?  The verifier rejects this early
// so the programmer receives a clear error rather than a silent miscompile.

// -----

// sync_mode=barrier + disable_synchronization=true is rejected.
// expected-error @+1 {{'conduit.create' op sync_mode and disable_synchronization=true are mutually exclusive}}
conduit.create @conflict_barrier {slot_elems = 8 : i64, depth = 0 : i64,
                                  sync_mode = #conduit.sync_mode<barrier>,
                                  disable_synchronization = true}

// -----

// sync_mode=independent + disable_synchronization=true is also rejected.
// expected-error @+1 {{'conduit.create' op sync_mode and disable_synchronization=true are mutually exclusive}}
conduit.create @conflict_independent {slot_elems = 8 : i64, depth = 0 : i64,
                                      sync_mode = #conduit.sync_mode<independent>,
                                      disable_synchronization = true}
