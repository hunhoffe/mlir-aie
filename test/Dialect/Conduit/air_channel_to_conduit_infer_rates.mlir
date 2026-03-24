// RUN: aie-opt --allow-unregistered-dialect "--air-channel-to-conduit=infer-rates=true" %s 2>&1 | FileCheck %s
// RUN: aie-opt --allow-unregistered-dialect --air-channel-to-conduit %s 2>&1 | FileCheck %s --check-prefix=NOINFER
//
// --air-channel-to-conduit infer-rates pipeline integration test.
//
// When inferRates=true, the pass attaches producer_rates and
// consumer_rates to conduit.create based on the num_elems of put/get ops.
//
// When inferRates=false (the default), rates are NOT attached.
//
// This test uses scalar transfers (no sizes → num_elems=1) so that the
// inferred rates [1] are compatible with capacity=1 (M7 requires
// peak_occupancy ≤ capacity).
//
// Topology: scalar channel @chan with one scalar put and one scalar get.

// Explicit inferRates=true: rates should be attached.
// MLIR prints attributes alphabetically: consumer_rates, name, producer_rates.
// CHECK-LABEL: module
// CHECK: conduit.create @chan
// CHECK-SAME: consumer_rates = array<i64: 1>
// CHECK-SAME: producer_rates = array<i64: 1>

// Default (inferRates=false): rates must NOT be attached.
// NOINFER-LABEL: module
// NOINFER: conduit.create @chan
// NOINFER-NOT: producer_rates
// NOINFER-NOT: consumer_rates

module {
  "air.channel"() {sym_name = "chan", size = [1, 1]} : () -> ()

  func.func @producer(%buf : memref<64xi32>) {
    // Scalar transfer: no offsets/sizes/strides → num_elems = 1.
    "air.channel.put"(%buf)
        {chan_name = @chan,
         operand_segment_sizes = array<i32: 0, 0, 1, 0, 0, 0>}
        : (memref<64xi32>) -> ()
    return
  }

  func.func @consumer(%buf : memref<64xi32>) {
    // Scalar transfer: no offsets/sizes/strides → num_elems = 1.
    "air.channel.get"(%buf)
        {chan_name = @chan,
         operand_segment_sizes = array<i32: 0, 0, 1, 0, 0, 0>}
        : (memref<64xi32>) -> ()
    return
  }
}
