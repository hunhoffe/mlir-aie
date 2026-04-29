// RUN: aie-opt --verify-diagnostics --conduit-canonicalize-channel-puts %s | FileCheck %s
//
// Boundary-stress pin (cap+1 = 17, just over) for ArithProgressionPattern.
// AIE2 shim/compute BD cap = 16; N = 17 sits ONE over, so canon must
// REFUSE to collapse (cannot encode <size = 17, ...> into a single BD's
// dim layout — would crash the downstream HasValidBDs verifier).  Canon
// emits the structured warning and leaves the IR un-collapsed.  This is
// the FAIL-SAFE tail of the boundary triplet (15 / 16 / 17).

// CHECK-LABEL: aie.device(npu1)
// CHECK: conduit.create @chan
// CHECK-NOT: producer_dimensions
// All 17 puts survive (canon refused).
// CHECK: conduit.put_memref_async
// CHECK-SAME: name = @chan
// CHECK: conduit.put_memref_async
// CHECK-SAME: name = @chan

module @boundary_stress_putcount_17 {
  aie.device(npu1) {
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)
    // expected-warning@+1 {{canonicalize-loop-unroll-puts: refusing to collapse 17 puts on @chan}}
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
