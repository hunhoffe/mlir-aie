// RUN: aie-opt --conduit-infer-rates -split-input-file -verify-diagnostics %s
//
// --conduit-infer-rates: rate mismatch triggers M6 CSDF balance error.
//
// A conduit.create with no explicit rates.
// put_memref_async num_elems=64 → producer_rates = [64] (sum=64, period=1)
// get_memref_async num_elems=128 → consumer_rates = [128] (sum=128, period=1)
//
// M6 balance check:
//   sum(P)*len(C) = 64*1 = 64
//   sum(C)*len(P) = 128*1 = 128
//   64 != 128 → CSDF rate imbalance error.
//
// The pass attaches the rates (emitting a remark) and M6 fires immediately
// inside Create::verify() as the op is re-verified after the attribute update.

// -----

module @infer_rates_mismatch {
  // expected-remark@+2 {{conduit-infer-rates: attached producer_rates=[64] consumer_rates=[128] to conduit 'chan'}}
  // expected-error@+1 {{'conduit.create' op CSDF rate imbalance: sum(producer_rates)*len(consumer_rates)=64 != sum(consumer_rates)*len(producer_rates)=128}}
  conduit.create @chan {capacity = 128 : i64,
                  depth = 1 : i64,
                  element_type = memref<128xi32>}

  func.func @producer(%buf : memref<64xi32>) {
    %tok = conduit.put_memref_async {name = @chan, num_elems = 64 : i64,
                                     offsets = array<i64: 0>,
                                     sizes   = array<i64: 64>,
                                     strides = array<i64: 1>}
                                    : !conduit.dma.token
    conduit.wait %tok : !conduit.dma.token
    return
  }

  func.func @consumer(%buf : memref<128xi32>) {
    %tok = conduit.get_memref_async {name = @chan, num_elems = 128 : i64,
                                     offsets = array<i64: 0>,
                                     sizes   = array<i64: 128>,
                                     strides = array<i64: 1>}
                                    : !conduit.dma.token
    conduit.wait %tok : !conduit.dma.token
    return
  }
}
