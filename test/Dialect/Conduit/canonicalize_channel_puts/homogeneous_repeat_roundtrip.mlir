// RUN: aie-opt --conduit-canonicalize-channel-puts --conduit-expand-channel-puts %s | FileCheck %s
//
// Round-trip pin: the canonicalize → expand pair is a no-op on IR shape.
// After collapse to (1 put + dma_repeat=N), expand replicates back to N
// structurally-identical puts + matching wait_all chains and clears
// dma_repeat on the channel.  This proves canon is reversible (which is
// required by fusion authors that need per-batch IR-level mutation).

// CHECK-LABEL: aie.device(npu1)

// Channel must NOT carry dma_repeat after expand.
// CHECK: conduit.create @chan
// CHECK-NOT: dma_repeat

// Exactly N=3 put_memref_async ops on @chan, each with one await and one
// free wait_all op.  We don't pin operand chains exactly (greedy driver
// + module walk leaves IR-order deterministic only by construction); we
// pin the count via three CHECK lines for put_memref_async.
// CHECK: conduit.put_memref_async
// CHECK-SAME: name = @chan
// CHECK: conduit.put_memref_async
// CHECK-SAME: name = @chan
// CHECK: conduit.put_memref_async
// CHECK-SAME: name = @chan
// CHECK-NOT: conduit.put_memref_async{{.*}}name = @chan

module @conduit_canonicalize_loop_unroll_puts_roundtrip {
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

    func.func @sequence(%arg0: memref<8xi32>) {
      %t0 = conduit.put_memref_async {name = @chan, num_elems = 8 : i64,
            offsets = array<i64: 0>, sizes = array<i64: 8>,
            strides = array<i64: 1>} : !conduit.dma.token
      conduit.wait_all %t0 {token = true} : !conduit.dma.token
      conduit.wait_all %t0 {token = false} : !conduit.dma.token

      %t1 = conduit.put_memref_async {name = @chan, num_elems = 8 : i64,
            offsets = array<i64: 0>, sizes = array<i64: 8>,
            strides = array<i64: 1>} : !conduit.dma.token
      conduit.wait_all %t1 {token = true} : !conduit.dma.token
      conduit.wait_all %t1 {token = false} : !conduit.dma.token

      %t2 = conduit.put_memref_async {name = @chan, num_elems = 8 : i64,
            offsets = array<i64: 0>, sizes = array<i64: 8>,
            strides = array<i64: 1>} : !conduit.dma.token
      conduit.wait_all %t2 {token = true} : !conduit.dma.token
      conduit.wait_all %t2 {token = false} : !conduit.dma.token
      return
    }
  }
}
