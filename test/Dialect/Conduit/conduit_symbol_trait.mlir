// RUN: aie-opt %s | FileCheck %s
//
// C-2 Phase 1: verify conduit.create uses @symbol prefix syntax and
// sym_name attribute for SymbolTable participation.
//
// After renaming $name → $sym_name, conduit.create must:
//   1. Parse using the @symbol prefix syntax (matching aie.objectfifo).
//   2. Round-trip cleanly through aie-opt (print then re-parse).
//   3. NOT print sym_name in the attr-dict (it is consumed by the @prefix).
//
// CHECK-LABEL: aie.device(npu1)
// CHECK: conduit.create @my_channel
// CHECK-NOT: sym_name = "my_channel"
// CHECK-NOT: name = @my_channel
// CHECK: conduit.create @another_chan

module {
  aie.device(npu1) {
    conduit.create @my_channel {
      slot_elems = 128 : i64,
      depth = 2 : i64,
      element_type = memref<32xi32>,
      producer_tile = array<i64: 0, 0>,
      consumer_tiles = array<i64: 0, 2>
    }
    conduit.create @another_chan {
      slot_elems = 64 : i64,
      depth = 1 : i64,
      element_type = memref<16xi32>,
      producer_tile = array<i64: 1, 0>,
      consumer_tiles = array<i64: 1, 2>
    }
  }
}
