// RUN: aie-opt --conduit-canonicalize-channel-puts %s | FileCheck %s
// RUN: aie-opt --conduit-canonicalize-channel-puts --conduit-depth-promote --conduit-to-dma --aie-substitute-shim-dma-allocations --aie-assign-runtime-sequence-bd-ids %s
//
// HISTORICAL: this fixture originally pinned cap = 16 (exact) for
// ArithProgressionPattern — 16 sliding-offset puts → 1 surviving put +
// producer_dimensions outer wrap <size = 16, stride = 8> as an
// off-by-one guard in the cap check.
//
// FLIPPED 2026-05-03 (canon refuse-to-collapse-on-await predicate, this
// commit): the 16 puts each carry `wait_all{token=true}` +
// `wait_all{token=false}` (chain shape `[true, false]`).  Per the new
// `chainHasAwait` predicate, ArithProgressionPattern now REFUSES to
// collapse such chains; the cap-exact case is moot for this chain shape.
// Same root-cause class as the canon link-refusal landed in commit
// 375b0e5233; per CLAUDE.md USER-LOCKED 2026-04-28 "wrong is right"
// anti-pattern, the prior collapse-asserting CHECKs encoded HW-broken
// behavior.  Empirical HW backing:
// `test/npu-xrt/conduit_canon_no_collapse_on_puts_with_await/`.
//
// Boundary-stress role (cap-exact) is preserved for the LEGITIMATE
// chain-without-await shape on the homogeneous-repeat sibling cap pins
// (those exercise chain `[false]` only).

// CHECK-LABEL: aie.device(npu1)

// Channel must NOT carry the canon-introduced outer wrap dim
// (canon refused — chain has token=true).
// CHECK: conduit.create @chan
// CHECK-NOT: producer_dimensions

// All 16 puts survive on @chan at original offsets (canon left the IR alone).
// CHECK-COUNT-16: conduit.put_memref_async {{.*}}name = @chan

module @boundary_stress_putcount_16 {
  aie.device(npu1) {
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)
    conduit.create @chan { element_type = memref<8xi32>, depth = 2 : i64 }
    aie.shim_dma_allocation @chan_shim_alloc(%tile_0_0, MM2S, 0) {conduit_channel = @chan}
    %core_0_2 = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c16 = arith.constant 16 : index
      %c1 = arith.constant 1 : index
      scf.for %i = %c0 to %c16 step %c1 {
        %g = conduit.get_memref_async {name = @chan, num_elems = 8 : i64,
                  offsets = array<i64: 0>, sizes = array<i64: 8>, strides = array<i64: 1>} : !conduit.dma.token
        conduit.wait_all %g : !conduit.dma.token
      }
      aie.end
    } {dynamic_objfifo_lowering = true}
    func.func @sequence(%arg0: memref<128xi32>) {
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
      return
    }
  }
}
