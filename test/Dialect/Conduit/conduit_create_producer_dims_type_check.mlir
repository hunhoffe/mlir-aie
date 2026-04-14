// RUN: not aie-opt %s 2>&1 | FileCheck %s
//
// Regression test: B-5 — conduit.create with producer_dimensions set to a
// scalar attribute (StringAttr) triggers a verifier error.
//
// The producer_dimensions attribute is constrained to AIE::BDDimLayoutArrayAttr
// via an ODS CPred constraint. Invalid attribute types are rejected by the
// ODS-generated verifier.
//
// CHECK: failed to satisfy constraint: AIE::BDDimLayoutArrayAttr

module @conduit_create_producer_dims_type_check {
  aie.device(npu1_1col) {
    %shim = aie.tile(0, 0)
    %tile = aie.tile(0, 2)

    // producer_dimensions set to a string attribute — obviously wrong, never a
    // BDDimLayoutArrayAttr. The ODS constraint should catch and reject this.
    conduit.create @badDims {slot_elems = 32 : i64, depth = 1 : i64,
                    element_type = memref<32xi32>,
                    producer_dimensions = "not_a_bd_dim_layout_array"}
  }
}
