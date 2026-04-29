// RUN: aie-opt --conduit-canonicalize-channel-puts %s | FileCheck %s
// RUN: aie-opt --conduit-canonicalize-channel-puts --conduit-depth-promote --conduit-to-dma --aie-substitute-shim-dma-allocations --aie-assign-runtime-sequence-bd-ids %s
//
// Pin the basic arith-progression collapse behavior of
// --conduit-canonicalize-channel-puts (ArithProgressionPattern, Sprint N
// sibling of HomogeneousRepeatPattern):
//   * 4 conduit.put_memref_async ops on @chan with SAME shape/sizes/strides
//     but offsets sliding in arithmetic progression [0, 8, 16, 24] (delta = 8)
//     and matching wait_all{token=true} await + wait_all{token=false} free
//     chains → collapsed to 1 put + 1 await + 1 free with producer_dimensions
//     gaining an outer wrap dimension <size = 4, stride = 8> on the
//     conduit.create channel.
//
// IR shape models the IRON `for batch in range(4)` where IRON's per-batch
// kernel emits structurally-similar dma_configure ops that ONLY differ in
// the configure offset (e.g. op11_GEMV @op11_A_L3L1_0 in the captured
// reproducer at /tmp/npu_run_conduit_20260428_104918_1277633/build/
// fused_op_fused.mlir lines 5467/5563/5659/... with offsets 0, 131072,
// 262144, ... — the actual Llama bug).  HomogeneousRepeatPattern doesn't
// fire because offsets differ; the arith-progression pattern is the
// next-strictest collapse — it encodes the per-fire offset increment as a
// new outer wrap+stride on the channel's producer_dimensions.
//
// Geometry: shim(0,0) producer → compute(0,2) consumer, depth=2,
//           memref<8xi32> per put, 4 host dispatches at offsets [0,8,16,24].

// CHECK-LABEL: aie.device(npu1)

// Channel gains an outer wrap dimension matching the per-put delta.
// CHECK: conduit.create @chan
// CHECK-SAME: producer_dimensions = #aie<bd_dim_layout_array[<size = 4, stride = 8>]>

// Exactly one surviving put_memref_async on @chan, at the FIRST offset (0).
// CHECK: conduit.put_memref_async
// CHECK-SAME: name = @chan
// CHECK-SAME: offsets = array<i64: 0>
// CHECK-NOT: conduit.put_memref_async{{.*}}name = @chan

module @arith_progression_1d_collapse {
  aie.device(npu1) {
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)

    conduit.create @chan {
      element_type = memref<8xi32>,
      depth = 2 : i64
    }

    aie.shim_dma_allocation @chan_shim_alloc(%tile_0_0, MM2S, 0) {conduit_channel = @chan}

    %core_0_2 = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c4 = arith.constant 4 : index
      %c1 = arith.constant 1 : index
      scf.for %i = %c0 to %c4 step %c1 {
        %g = conduit.get_memref_async {name = @chan,
                  num_elems = 8 : i64,
                  offsets = array<i64: 0>,
                  sizes = array<i64: 8>,
                  strides = array<i64: 1>} : !conduit.dma.token
        conduit.wait_all %g : !conduit.dma.token
      }
      aie.end
    } {dynamic_objfifo_lowering = true}

    func.func @sequence(%arg0: memref<32xi32>) {
      // 4 sliding-offset puts, each with await + free chain — IRON's
      // host-Python-unroll pattern with per-batch base-offset increment.
      %t0 = conduit.put_memref_async {name = @chan, num_elems = 8 : i64,
            offsets = array<i64: 0>, sizes = array<i64: 8>,
            strides = array<i64: 1>} : !conduit.dma.token
      conduit.wait_all %t0 {token = true} : !conduit.dma.token
      conduit.wait_all %t0 {token = false} : !conduit.dma.token

      %t1 = conduit.put_memref_async {name = @chan, num_elems = 8 : i64,
            offsets = array<i64: 8>, sizes = array<i64: 8>,
            strides = array<i64: 1>} : !conduit.dma.token
      conduit.wait_all %t1 {token = true} : !conduit.dma.token
      conduit.wait_all %t1 {token = false} : !conduit.dma.token

      %t2 = conduit.put_memref_async {name = @chan, num_elems = 8 : i64,
            offsets = array<i64: 16>, sizes = array<i64: 8>,
            strides = array<i64: 1>} : !conduit.dma.token
      conduit.wait_all %t2 {token = true} : !conduit.dma.token
      conduit.wait_all %t2 {token = false} : !conduit.dma.token

      %t3 = conduit.put_memref_async {name = @chan, num_elems = 8 : i64,
            offsets = array<i64: 24>, sizes = array<i64: 8>,
            strides = array<i64: 1>} : !conduit.dma.token
      conduit.wait_all %t3 {token = true} : !conduit.dma.token
      conduit.wait_all %t3 {token = false} : !conduit.dma.token
      return
    }
  }
}
