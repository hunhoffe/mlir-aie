// RUN: aie-opt -pass-pipeline='builtin.module(conduit-depth-promote{csdf=true eta=1.5})' --verify-diagnostics %s | FileCheck %s
//
// DEFERRED-7: CSDFa-grounded depth selection with uneven eta.
//
// CSDFa minimum buffer depth formula (Denolf 2007, Koek 2016 §4):
//   min_depth = ceil(max(P, C) * eta / min(P, C))
//
// Test case: P = sum([2]) = 2, C = sum([2]) = 2, eta = 1.5.
//   min_depth = ceil(max(2,2) * 1.5 / min(2,2)) = ceil(2*1.5/2) = ceil(1.5) = 2.
//
// The conduit should be promoted from depth-1 to depth-2.
// (Rates are balanced per M6; eta > 1.0 forces over-provisioned depth.)

// CHECK: conduit.create @csdf_uneven {{{.*}}depth = 2 : i64, {{.*}}slot_elems = 12 : i64
// expected-remark @+1 {{conduit-depth-promote: promoted 1 conduit(s)}}
module {

func.func @csdf_uneven_rates(%result: memref<6xi32>) {
  // expected-remark @+1 {{conduit-depth-promote: promoted 'csdf_uneven' from depth-1 to depth-2}}
  conduit.create @csdf_uneven {slot_elems = 6 : i64,
                  producer_tile = array<i64: 0, 0>,
                  consumer_tiles = array<i64: 0, 2>,
                  element_type = memref<6xi32>,
                  depth = 1 : i64,
                  producer_rates = array<i64: 2>,
                  consumer_rates = array<i64: 2>}
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c6 = arith.constant 6 : index
  scf.for %i = %c0 to %c6 step %c1 {
    %win = conduit.acquire {name = @csdf_uneven, count = 1 : i64, port = #conduit.port<Consume>}
               : !conduit.window<memref<6xi32>>
    %elem = conduit.subview_access %win {index = 0 : i64}
               : !conduit.window<memref<6xi32>> -> memref<6xi32>
    memref.copy %elem, %result : memref<6xi32> to memref<6xi32>
    conduit.release %win {count = 1 : i64, port = #conduit.port<Consume>}
        : !conduit.window<memref<6xi32>>
  }
  return
}

} // module
