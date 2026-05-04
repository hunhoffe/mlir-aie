// RUN: aie-opt --conduit-canonicalize-channel-puts %s | FileCheck %s
//
// HISTORICAL: this fixture originally pinned cap+1 = 17 (just over) for
// ArithProgressionPattern — canon REFUSED to collapse via the BD-cap
// emitWarning ("refusing to collapse 17 puts on @chan — exceeds tile BD
// cap of 16"); pinned with --verify-diagnostics + expected-warning.
//
// FLIPPED 2026-05-03 (canon refuse-to-collapse-on-await predicate, this
// commit): the 17 puts each carry `wait_all{token=true}` +
// `wait_all{token=false}` (chain shape `[true, false]`).  Per the new
// `chainHasAwait` predicate, ArithProgressionPattern now refuses
// EARLIER — before the cap-refuse fires — so the cap-refuse warning no
// longer reaches the user for this fixture's chain shape.  The
// --verify-diagnostics + expected-warning directive is removed (the
// emitRemark from chainHasAwait fires from the greedy driver and would
// not match a single expected-warning anyway).
//
// Same root-cause class as the canon link-refusal landed in commit
// 375b0e5233; per CLAUDE.md USER-LOCKED 2026-04-28 "wrong is right"
// anti-pattern, the prior pinned shape (warning text + producer_dimensions
// absent) was correct in OUTCOME (canon refused) but the REASON was the
// wrong gate — canon should have refused on chain-await first, not on
// cap.  Empirical HW backing for the new gate:
// `test/npu-xrt/conduit_canon_no_collapse_on_puts_with_await/`.
//
// Boundary-stress role (cap+1) is preserved for the LEGITIMATE
// chain-without-await shape on the homogeneous-repeat sibling cap pins.

// CHECK-LABEL: aie.device(npu1)

// Channel must NOT carry the canon-introduced outer wrap dim
// (canon refused — chain has token=true).
// CHECK: conduit.create @chan
// CHECK-NOT: producer_dimensions

// All 17 puts survive on @chan at original offsets (canon left the IR alone).
// CHECK-COUNT-17: conduit.put_memref_async {{.*}}name = @chan

module @boundary_stress_putcount_17 {
  aie.device(npu1) {
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)
    conduit.create @chan { element_type = memref<8xi32>, depth = 2 : i64 }
    aie.shim_dma_allocation @chan_shim_alloc(%tile_0_0, MM2S, 0) {conduit_channel = @chan}
    %core_0_2 = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c17 = arith.constant 17 : index
      %c1 = arith.constant 1 : index
      scf.for %i = %c0 to %c17 step %c1 {
        %g = conduit.get_memref_async {name = @chan, num_elems = 8 : i64,
                  offsets = array<i64: 0>, sizes = array<i64: 8>, strides = array<i64: 1>} : !conduit.dma.token
        conduit.wait_all %g : !conduit.dma.token
      }
      aie.end
    } {dynamic_objfifo_lowering = true}
    func.func @sequence(%arg0: memref<136xi32>) {
      %t0 = conduit.put_memref_async {name = @chan, num_elems = 8 : i64, offsets = array<i64: 0>, sizes = array<i64: 8>, strides = array<i64: 1>} : !conduit.dma.token
      conduit.wait_all %t0 {token = true} : !conduit.dma.token
      conduit.wait_all %t0 {token = false} : !conduit.dma.token
      %t1 = conduit.put_memref_async {name = @chan, num_elems = 8 : i64, offsets = array<i64: 8>, sizes = array<i64: 8>, strides = array<i64: 1>} : !conduit.dma.token
      conduit.wait_all %t1 {token = true} : !conduit.dma.token
      conduit.wait_all %t1 {token = false} : !conduit.dma.token
      %t2 = conduit.put_memref_async {name = @chan, num_elems = 8 : i64, offsets = array<i64: 16>, sizes = array<i64: 8>, strides = array<i64: 1>} : !conduit.dma.token
      conduit.wait_all %t2 {token = true} : !conduit.dma.token
      conduit.wait_all %t2 {token = false} : !conduit.dma.token
      %t3 = conduit.put_memref_async {name = @chan, num_elems = 8 : i64, offsets = array<i64: 24>, sizes = array<i64: 8>, strides = array<i64: 1>} : !conduit.dma.token
      conduit.wait_all %t3 {token = true} : !conduit.dma.token
      conduit.wait_all %t3 {token = false} : !conduit.dma.token
      %t4 = conduit.put_memref_async {name = @chan, num_elems = 8 : i64, offsets = array<i64: 32>, sizes = array<i64: 8>, strides = array<i64: 1>} : !conduit.dma.token
      conduit.wait_all %t4 {token = true} : !conduit.dma.token
      conduit.wait_all %t4 {token = false} : !conduit.dma.token
      %t5 = conduit.put_memref_async {name = @chan, num_elems = 8 : i64, offsets = array<i64: 40>, sizes = array<i64: 8>, strides = array<i64: 1>} : !conduit.dma.token
      conduit.wait_all %t5 {token = true} : !conduit.dma.token
      conduit.wait_all %t5 {token = false} : !conduit.dma.token
      %t6 = conduit.put_memref_async {name = @chan, num_elems = 8 : i64, offsets = array<i64: 48>, sizes = array<i64: 8>, strides = array<i64: 1>} : !conduit.dma.token
      conduit.wait_all %t6 {token = true} : !conduit.dma.token
      conduit.wait_all %t6 {token = false} : !conduit.dma.token
      %t7 = conduit.put_memref_async {name = @chan, num_elems = 8 : i64, offsets = array<i64: 56>, sizes = array<i64: 8>, strides = array<i64: 1>} : !conduit.dma.token
      conduit.wait_all %t7 {token = true} : !conduit.dma.token
      conduit.wait_all %t7 {token = false} : !conduit.dma.token
      %t8 = conduit.put_memref_async {name = @chan, num_elems = 8 : i64, offsets = array<i64: 64>, sizes = array<i64: 8>, strides = array<i64: 1>} : !conduit.dma.token
      conduit.wait_all %t8 {token = true} : !conduit.dma.token
      conduit.wait_all %t8 {token = false} : !conduit.dma.token
      %t9 = conduit.put_memref_async {name = @chan, num_elems = 8 : i64, offsets = array<i64: 72>, sizes = array<i64: 8>, strides = array<i64: 1>} : !conduit.dma.token
      conduit.wait_all %t9 {token = true} : !conduit.dma.token
      conduit.wait_all %t9 {token = false} : !conduit.dma.token
      %t10 = conduit.put_memref_async {name = @chan, num_elems = 8 : i64, offsets = array<i64: 80>, sizes = array<i64: 8>, strides = array<i64: 1>} : !conduit.dma.token
      conduit.wait_all %t10 {token = true} : !conduit.dma.token
      conduit.wait_all %t10 {token = false} : !conduit.dma.token
      %t11 = conduit.put_memref_async {name = @chan, num_elems = 8 : i64, offsets = array<i64: 88>, sizes = array<i64: 8>, strides = array<i64: 1>} : !conduit.dma.token
      conduit.wait_all %t11 {token = true} : !conduit.dma.token
      conduit.wait_all %t11 {token = false} : !conduit.dma.token
      %t12 = conduit.put_memref_async {name = @chan, num_elems = 8 : i64, offsets = array<i64: 96>, sizes = array<i64: 8>, strides = array<i64: 1>} : !conduit.dma.token
      conduit.wait_all %t12 {token = true} : !conduit.dma.token
      conduit.wait_all %t12 {token = false} : !conduit.dma.token
      %t13 = conduit.put_memref_async {name = @chan, num_elems = 8 : i64, offsets = array<i64: 104>, sizes = array<i64: 8>, strides = array<i64: 1>} : !conduit.dma.token
      conduit.wait_all %t13 {token = true} : !conduit.dma.token
      conduit.wait_all %t13 {token = false} : !conduit.dma.token
      %t14 = conduit.put_memref_async {name = @chan, num_elems = 8 : i64, offsets = array<i64: 112>, sizes = array<i64: 8>, strides = array<i64: 1>} : !conduit.dma.token
      conduit.wait_all %t14 {token = true} : !conduit.dma.token
      conduit.wait_all %t14 {token = false} : !conduit.dma.token
      %t15 = conduit.put_memref_async {name = @chan, num_elems = 8 : i64, offsets = array<i64: 120>, sizes = array<i64: 8>, strides = array<i64: 1>} : !conduit.dma.token
      conduit.wait_all %t15 {token = true} : !conduit.dma.token
      conduit.wait_all %t15 {token = false} : !conduit.dma.token
      %t16 = conduit.put_memref_async {name = @chan, num_elems = 8 : i64, offsets = array<i64: 128>, sizes = array<i64: 8>, strides = array<i64: 1>} : !conduit.dma.token
      conduit.wait_all %t16 {token = true} : !conduit.dma.token
      conduit.wait_all %t16 {token = false} : !conduit.dma.token
      return
    }
  }
}
