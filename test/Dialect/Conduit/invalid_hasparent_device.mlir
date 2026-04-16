// RUN: aie-opt -verify-diagnostics %s
//
// Phase 9: conduit.create requires HasParent<DeviceOp>.
// A conduit.create outside any aie.device block must be rejected.

// expected-error @+1 {{'conduit.create' op expects parent op 'aie.device'}}
conduit.create @ch {depth = 0 : i64}
