// RUN: aie-opt --conduit-canonicalize-channel-puts --conduit-expand-channel-puts %s | FileCheck %s
//
// Round-trip pin for ArithProgressionPattern: collapse → expand returns to
// the original IR shape (modulo SSA naming).  After collapse, the channel
// carries producer_dimensions <size = 3, stride = 8> + 1 surviving put at
// offset 0; expand replicates back to N=3 sliding-offset puts at offsets
// [0, 8, 16] and clears the wrap dimension from producer_dimensions.
//
// This is the symmetric counterpart of homogeneous_repeat_roundtrip.mlir
// for the arith-progression case.  Reversibility is required by fusion
// authors that need per-batch IR-level mutation post-canon (e.g.
// per-batch fusion-group annotation).

// CHECK-LABEL: aie.device(npu1)

// Channel must NOT carry the canon-introduced outer wrap after expand.
// (We use CHECK-NOT for the size-3 stride-8 wrap specifically — a
// pre-existing producer_dimensions on the channel would be left alone,
// but this fixture has none.)
// CHECK: conduit.create @chan
// CHECK-NOT: producer_dimensions

// Exactly N=3 put_memref_async ops on @chan, each at the original offsets.
// CHECK: conduit.put_memref_async
// CHECK-SAME: name = @chan
// CHECK-SAME: offsets = array<i64: 0>
// CHECK: conduit.put_memref_async
// CHECK-SAME: name = @chan
// CHECK-SAME: offsets = array<i64: 8>
// CHECK: conduit.put_memref_async
// CHECK-SAME: name = @chan
// CHECK-SAME: offsets = array<i64: 16>
// CHECK-NOT: conduit.put_memref_async{{.*}}name = @chan

module @arith_progression_1d_roundtrip {
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
      %c3 = arith.constant 3 : index
      %c1 = arith.constant 1 : index
      scf.for %i = %c0 to %c3 step %c1 {
        %g = conduit.get_memref_async {name = @chan,
                  num_elems = 8 : i64,
                  offsets = array<i64: 0>,
                  sizes = array<i64: 8>,
                  strides = array<i64: 1>} : !conduit.dma.token
        conduit.wait_all %g : !conduit.dma.token
      }
      aie.end
    } {dynamic_objfifo_lowering = true}

    func.func @sequence(%arg0: memref<24xi32>) {
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
      return
    }
  }
}
