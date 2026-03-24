// RUN: aie-opt -pass-pipeline='builtin.module(conduit-depth-promote{csdf=true eta=2.0})' --verify-diagnostics %s | FileCheck %s
//
// DEFERRED-7: CSDFa-grounded depth selection in --conduit-depth-promote.
//
// When csdf=true and producer_rates/consumer_rates are present, the pass
// uses the CSDFa minimum buffer depth formula (Denolf 2007, Koek 2016 §4):
//
//   min_depth = ceil(max(P, C) * eta / min(P, C))
//
// Test case: P = sum([1]) = 1, C = sum([1]) = 1, eta = 2.0.
//   min_depth = ceil(max(1,1) * 2.0 / min(1,1)) = ceil(2.0) = 2.
//
// The conduit should be promoted from depth-1 to depth-2.
// (Rates are balanced per M6; eta > 1.0 forces over-provisioned depth.)

// CHECK: conduit.create @csdf_ch {capacity = 16 : i64, {{.*}}depth = 2 : i64, 
// expected-remark @+1 {{conduit-depth-promote: promoted 1 conduit(s)}}
module {

func.func @csdf_uniform_rates(%result: memref<8xi32>) {
  // expected-remark @+1 {{conduit-depth-promote: promoted 'csdf_ch' from depth-1 to depth-2}}
  conduit.create @csdf_ch {capacity = 8 : i64,
                  producer_tile = array<i64: 0, 0>,
                  consumer_tiles = array<i64: 0, 2>,
                  element_type = memref<8xi32>,
                  depth = 1 : i64,
                  producer_rates = array<i64: 1>,
                  consumer_rates = array<i64: 1>}
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  scf.for %i = %c0 to %c8 step %c1 {
    %win = conduit.acquire {name = "csdf_ch", count = 1 : i64, port = #conduit.port<Consume>}
               : !conduit.window<memref<8xi32>>
    %elem = conduit.subview_access %win {index = 0 : i64}
               : !conduit.window<memref<8xi32>> -> memref<8xi32>
    memref.copy %elem, %result : memref<8xi32> to memref<8xi32>
    conduit.release %win {count = 1 : i64, port = #conduit.port<Consume>}
        : !conduit.window<memref<8xi32>>
  }
  return
}

} // module
