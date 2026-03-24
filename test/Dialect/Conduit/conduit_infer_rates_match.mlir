// RUN: aie-opt --conduit-infer-rates --verify-diagnostics %s | FileCheck %s
//
// --conduit-infer-rates: uniform SDF case (put num_elems == get num_elems).
//
// A conduit.create with no explicit rates, one put_memref_async with
// num_elems=64 and one get_memref_async with num_elems=64.
//
// Expected behaviour:
//   - Pass infers producer_rates = [64], consumer_rates = [64].
//   - M6 balance check: sum(P)*len(C) = 64*1 == sum(C)*len(P) = 64*1 → passes.
//   - M7 hyper-period simulation: H=lcm(1,1)=1 step.
//       t=0: produce 64 (occ=64), consume 64 (occ=0). Peak=64 <= capacity=64. Passes.
//   - A remark is emitted by the pass on the conduit.create op.
//   - Output IR has producer_rates = array<i64: 64> and consumer_rates = array<i64: 64>.
//
// Note: --verify-diagnostics intercepts expected-remark annotations.
// Attributes are printed in alphabetical order; use CHECK-DAG for unordered matching.

// CHECK-LABEL: module @infer_rates_match
// CHECK:       conduit.create
// CHECK-DAG:   consumer_rates = array<i64: 64>
// CHECK-DAG:   producer_rates = array<i64: 64>
// CHECK-DAG:   name = @chan

module @infer_rates_match {
  // expected-remark@+1 {{conduit-infer-rates: attached producer_rates=[64] consumer_rates=[64] to conduit 'chan'}}
  conduit.create @chan {capacity = 64 : i64,
                  depth = 1 : i64,
                  element_type = memref<64xi32>}

  func.func @producer(%buf : memref<64xi32>) {
    %tok = conduit.put_memref_async {name = @chan, num_elems = 64 : i64,
                                     offsets = array<i64: 0>,
                                     sizes   = array<i64: 64>,
                                     strides = array<i64: 1>}
                                    : !conduit.dma.token
    conduit.wait %tok : !conduit.dma.token
    return
  }

  func.func @consumer(%buf : memref<64xi32>) {
    %tok = conduit.get_memref_async {name = @chan, num_elems = 64 : i64,
                                     offsets = array<i64: 0>,
                                     sizes   = array<i64: 64>,
                                     strides = array<i64: 1>}
                                    : !conduit.dma.token
    conduit.wait %tok : !conduit.dma.token
    return
  }
}
