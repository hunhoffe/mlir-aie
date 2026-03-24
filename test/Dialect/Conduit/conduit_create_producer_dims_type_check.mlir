// RUN: not aie-opt %s 2>&1 | FileCheck %s
//
// Regression test: B-5 — conduit.create with producer_dimensions set to a
// scalar attribute (StringAttr) triggers a verifier error instead of a silent
// crash in Pass C.
//
// The producer_dimensions attribute is stored as AnyAttr. At runtime it must
// be an AIE::BDDimLayoutArrayAttr. Create::verify() now checks for obviously
// wrong scalar types (StringAttr, IntegerAttr) and emits a clear error.
//
// Note: AIE::BDDimLayoutArrayAttr is NOT an mlir::ArrayAttr subclass, so we
// cannot use mlir::isa<mlir::ArrayAttr>() to validate it from the Conduit
// dialect (which does not include the AIE dialect). Only scalar attributes
// are rejected here; valid BDDimLayoutArrayAttr values always pass.
//
// CHECK: producer_dimensions must be an AIE::BDDimLayoutArrayAttr
// CHECK: scalar attribute

module @conduit_create_producer_dims_type_check {
  aie.device(npu1_1col) {
    %shim = aie.tile(0, 0)
    %tile = aie.tile(0, 2)

    // producer_dimensions set to a string attribute — obviously wrong, never a
    // BDDimLayoutArrayAttr. The verifier should catch and reject this.
    conduit.create {name = "badDims", capacity = 32 : i64, depth = 1 : i64,
                    element_type = memref<32xi32>,
                    producer_tile = array<i64: 0, 0>,
                    consumer_tiles = array<i64: 0, 2>,
                    shim_consumer_tiles = array<i64>,
                    producer_dimensions = "not_a_bd_dim_layout_array"}
  }
}
