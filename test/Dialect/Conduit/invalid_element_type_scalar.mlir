// RUN: aie-opt -verify-diagnostics %s
//
// Negative test: element_type must be a MemRefType when present.
//
// conduit.create's element_type is used by Pass C to allocate aie.buffer ops.
// A scalar type (i32, f32, etc.) cannot be used as an aie.buffer allocation
// target and would cause a silent crash in Pass C.  The verifier rejects it
// with a clear error message so the user gets actionable feedback early.

aie.device(npu1) {
// expected-error@+1 {{'conduit.create' op element_type must be a MemRefType when present, got 'i32'}}
conduit.create @bad_elem_type {slot_elems = 10 : i64, depth = 0 : i64, element_type = i32}
}
