// RUN: aie-opt -verify-diagnostics %s
//
// M4: conduit.create with a dynamic-dimension element_type must emit a warning.
//
// ConduitOps.cpp Create::verify() emits emitWarning() when element_type is a
// ShapedType with at least one dynamic dimension (ShapedType::isDynamic).
// This confirms the warning fires and its exact text, which is important for
// users who provide memref<?xi32> instead of a statically-sized type.
//
// Note: this is a WARNING, not an error — the program is accepted.
// Use expected-warning annotation with -verify-diagnostics.

aie.device(npu1) {
// expected-warning @+1 {{conduit.create: element_type has dynamic dimensions; capacity is approximate}}
conduit.create @dyn_fifo {slot_elems = 16 : i64,
                producer_tile = array<i64: 0, 2>,
                consumer_tiles = array<i64: 0, 3>,
                element_type = memref<?xi32>,
                depth = 1 : i64}
}
