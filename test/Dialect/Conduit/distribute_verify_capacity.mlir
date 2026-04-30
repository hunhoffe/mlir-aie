// RUN: aie-opt %s -split-input-file -verify-diagnostics
//
// M7-dist composed-consume: Denolf Eq. 48 buffer capacity check for 1:N distribute.
//
// NOTE: The cross-conduit composed consume check was implemented in the
// removed Distribute::verify(). These tests now verify that the individual
// conduit.create ops pass their own M6/M7 checks and the conduit.scatter
// relay op is structurally valid (no per-edge CSDF checks on scatter).
//
// Section 1: 1→2 scatter, slow consumer (period 3 vs 1) — all creates valid.
// Section 2: 1→3 scatter, one very slow consumer — all creates valid.

// -----

// Section 1: PASS — all conduit.create ops individually balanced.
//
// src: P=[3], C=[3], cap=3 → per-edge OK, M7 peak=3 = cap ✓
// dst1: P=[3], C=[3], cap=6 → per-edge OK ✓
// dst2: P=[1,1,1], C=[1,1,1], cap=6 → per-edge OK ✓ (sum=3, period=3: 3*3=9=3*3)

aie.device(npu1) {
conduit.create @ov_src {                element_type = memref<3xi32>,
                depth = 3 : i64,
                producer_rates = array<i64: 3>,
                consumer_rates = array<i64: 3>}
conduit.create @ov_d1 {                element_type = memref<6xi32>,
                depth = 2 : i64,
                producer_rates = array<i64: 3>,
                consumer_rates = array<i64: 3>}
conduit.create @ov_d2 {                element_type = memref<6xi32>,
                depth = 2 : i64,
                producer_rates = array<i64: 1, 1, 1>,
                consumer_rates = array<i64: 1, 1, 1>}
func.func @distribute_slow_consumer_overflow() {
  %mt = aie.tile(0, 1)
  conduit.scatter{src = @ov_src, dsts = [@ov_d1, @ov_d2], memtile = %mt}
  return
}
}

// -----

// Section 2: PASS — all conduit.create ops individually balanced.
//
// src: P=[4], C=[4], cap=4 → per-edge OK, M7 peak=4 = cap ✓
// dst1: P=[4], C=[4], cap=8 → per-edge OK ✓
// dst2: P=[2,2], C=[2,2], cap=8 → per-edge OK ✓ (sum=4, period=2: 4*2=8=4*2)
// dst3: P=[1,1,1,1], C=[1,1,1,1], cap=8 → per-edge OK ✓ (sum=4, period=4: 4*4=16=4*4)

aie.device(npu1) {
conduit.create @t3_src {                element_type = memref<4xi32>,
                depth = 4 : i64,
                producer_rates = array<i64: 4>,
                consumer_rates = array<i64: 4>}
conduit.create @t3_d1 {                element_type = memref<8xi32>,
                depth = 2 : i64,
                producer_rates = array<i64: 4>,
                consumer_rates = array<i64: 4>}
conduit.create @t3_d2 {                element_type = memref<8xi32>,
                depth = 2 : i64,
                producer_rates = array<i64: 2, 2>,
                consumer_rates = array<i64: 2, 2>}
conduit.create @t3_d3 {                element_type = memref<8xi32>,
                depth = 2 : i64,
                producer_rates = array<i64: 1, 1, 1, 1>,
                consumer_rates = array<i64: 1, 1, 1, 1>}
func.func @distribute_three_consumer_overflow() {
  %mt = aie.tile(0, 1)
  conduit.scatter{src = @t3_src, dsts = [@t3_d1, @t3_d2, @t3_d3], memtile = %mt}
  return
}
}
