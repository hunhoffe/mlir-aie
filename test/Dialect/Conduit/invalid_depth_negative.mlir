// RUN: aie-opt -verify-diagnostics %s
//
// Negative test: depth < 0 is always invalid on conduit.create.
//
// depth=0 is the sentinel for "unresolved" (emitted by Pass A/B, resolved by
// --conduit-depth-promote).  depth>0 is an explicit hardware ring depth.
// Negative depth has no valid hardware interpretation and must be rejected.

// expected-error@+1 {{'conduit.create' op depth must be >= 0 (0 = unresolved sentinel, >0 = explicit depth); got -1}}
conduit.create @bad_depth {slot_elems = 10 : i64, depth = -1 : i64}
