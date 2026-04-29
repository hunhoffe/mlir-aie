// RUN: aie-opt --conduit-canonicalize-channel-puts %s | FileCheck %s
//
// Negative-stride REFUSE-pin for ArithProgressionPattern: 4 puts at
// descending offsets [24, 16, 8, 0] (delta = -8) on @chan.
//
// HW limitation: AIE `BDDimLayoutAttr.stride` is `uint32_t` (per
// `mlir-aie/include/aie/Dialect/AIE/IR/AIEAttrs.td:153-166`) AND the
// `aie.dma_bd` verifier (`AIEDialect.cpp:2267-2279`) explicitly rejects
// non-positive strides with "must be a positive integer".
//
// Conduit's BDDimLayoutArrayAttr reuses the same BDDimLayoutAttr, so a
// negative-stride collapse here would emit IR that fails downstream Pass C
// verification.  Canon MUST refuse descending progressions.
//
// This pattern is rare in IRON's host-Python loops (e.g. reverse-traversal
// of a per-batch input window).  Pre-canon shape (4 separate puts) is
// preserved; Pass C lowers each to its own BD chain entry as today.

// CHECK-LABEL: aie.device(npu1)

// Canon does NOT collapse — channel still has no producer_dimensions.
// CHECK: conduit.create @chan
// CHECK-NOT: producer_dimensions

// All 4 puts SURVIVE at their original offsets (descending).
// CHECK: conduit.put_memref_async
// CHECK-SAME: offsets = array<i64: 24>
// CHECK: conduit.put_memref_async
// CHECK-SAME: offsets = array<i64: 16>
// CHECK: conduit.put_memref_async
// CHECK-SAME: offsets = array<i64: 8>
// CHECK: conduit.put_memref_async
// CHECK-SAME: offsets = array<i64: 0>

module @arith_progression_1d_negative_stride {
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
      %t0 = conduit.put_memref_async {name = @chan, num_elems = 8 : i64,
            offsets = array<i64: 24>, sizes = array<i64: 8>,
            strides = array<i64: 1>} : !conduit.dma.token
      conduit.wait_all %t0 {token = true} : !conduit.dma.token
      conduit.wait_all %t0 {token = false} : !conduit.dma.token

      %t1 = conduit.put_memref_async {name = @chan, num_elems = 8 : i64,
            offsets = array<i64: 16>, sizes = array<i64: 8>,
            strides = array<i64: 1>} : !conduit.dma.token
      conduit.wait_all %t1 {token = true} : !conduit.dma.token
      conduit.wait_all %t1 {token = false} : !conduit.dma.token

      %t2 = conduit.put_memref_async {name = @chan, num_elems = 8 : i64,
            offsets = array<i64: 8>, sizes = array<i64: 8>,
            strides = array<i64: 1>} : !conduit.dma.token
      conduit.wait_all %t2 {token = true} : !conduit.dma.token
      conduit.wait_all %t2 {token = false} : !conduit.dma.token

      %t3 = conduit.put_memref_async {name = @chan, num_elems = 8 : i64,
            offsets = array<i64: 0>, sizes = array<i64: 8>,
            strides = array<i64: 1>} : !conduit.dma.token
      conduit.wait_all %t3 {token = true} : !conduit.dma.token
      conduit.wait_all %t3 {token = false} : !conduit.dma.token
      return
    }
  }
}
