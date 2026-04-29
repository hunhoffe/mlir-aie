// RUN: aie-opt --conduit-canonicalize-channel-puts %s | FileCheck %s
// RUN: aie-opt --conduit-canonicalize-channel-puts --conduit-depth-promote --conduit-to-dma --aie-substitute-shim-dma-allocations --aie-assign-runtime-sequence-bd-ids %s
//
// Negative-fire pin for ArithProgressionPattern: 4 puts at offsets
// [0, 5, 12, 18] — NOT in arithmetic progression (deltas are 5, 7, 6).
// Canon must NOT fire; the IR is left fully intact (4 puts + 4 await/free
// chains, no producer_dimensions added to the channel).
//
// Guards against an arith-impl that would over-aggressively pattern-match
// any monotonic offset sequence.  Only EXACT arith progression is safe to
// encode as a single outer wrap+stride.

// CHECK-LABEL: aie.device(npu1)

// Channel must NOT carry the canon-introduced outer wrap.
// CHECK: conduit.create @chan
// CHECK-NOT: producer_dimensions

// All 4 puts survive at their original offsets.
// CHECK: conduit.put_memref_async
// CHECK-SAME: name = @chan
// CHECK-SAME: offsets = array<i64: 0>
// CHECK: conduit.put_memref_async
// CHECK-SAME: name = @chan
// CHECK-SAME: offsets = array<i64: 5>
// CHECK: conduit.put_memref_async
// CHECK-SAME: name = @chan
// CHECK-SAME: offsets = array<i64: 12>
// CHECK: conduit.put_memref_async
// CHECK-SAME: name = @chan
// CHECK-SAME: offsets = array<i64: 18>

module @arith_progression_1d_non_progression_no_fire {
  aie.device(npu1) {
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)

    conduit.create @chan {
      element_type = memref<4xi32>,
      depth = 2 : i64
    }

    aie.shim_dma_allocation @chan_shim_alloc(%tile_0_0, MM2S, 0) {conduit_channel = @chan}

    %core_0_2 = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c4 = arith.constant 4 : index
      %c1 = arith.constant 1 : index
      scf.for %i = %c0 to %c4 step %c1 {
        %g = conduit.get_memref_async {name = @chan,
                  num_elems = 4 : i64,
                  offsets = array<i64: 0>,
                  sizes = array<i64: 4>,
                  strides = array<i64: 1>} : !conduit.dma.token
        conduit.wait_all %g : !conduit.dma.token
      }
      aie.end
    } {dynamic_objfifo_lowering = true}

    func.func @sequence(%arg0: memref<32xi32>) {
      // Offsets [0, 5, 12, 18] — deltas [5, 7, 6] are NOT constant.
      %t0 = conduit.put_memref_async {name = @chan, num_elems = 4 : i64,
            offsets = array<i64: 0>, sizes = array<i64: 4>,
            strides = array<i64: 1>} : !conduit.dma.token
      conduit.wait_all %t0 {token = true} : !conduit.dma.token
      conduit.wait_all %t0 {token = false} : !conduit.dma.token

      %t1 = conduit.put_memref_async {name = @chan, num_elems = 4 : i64,
            offsets = array<i64: 5>, sizes = array<i64: 4>,
            strides = array<i64: 1>} : !conduit.dma.token
      conduit.wait_all %t1 {token = true} : !conduit.dma.token
      conduit.wait_all %t1 {token = false} : !conduit.dma.token

      %t2 = conduit.put_memref_async {name = @chan, num_elems = 4 : i64,
            offsets = array<i64: 12>, sizes = array<i64: 4>,
            strides = array<i64: 1>} : !conduit.dma.token
      conduit.wait_all %t2 {token = true} : !conduit.dma.token
      conduit.wait_all %t2 {token = false} : !conduit.dma.token

      %t3 = conduit.put_memref_async {name = @chan, num_elems = 4 : i64,
            offsets = array<i64: 18>, sizes = array<i64: 4>,
            strides = array<i64: 1>} : !conduit.dma.token
      conduit.wait_all %t3 {token = true} : !conduit.dma.token
      conduit.wait_all %t3 {token = false} : !conduit.dma.token
      return
    }
  }
}
