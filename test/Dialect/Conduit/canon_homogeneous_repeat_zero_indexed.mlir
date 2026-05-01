// RUN: aie-opt --conduit-canonicalize-channel-puts %s | FileCheck %s
//
// Bug #98 / Task #39 pin: `--conduit-canonicalize-channel-puts` stamps
// `dma_repeat` using the 0-INDEXED convention ("additional fires beyond
// the initial one" → total fires = dma_repeat + 1).  This matches IRON's
// `aiex.dma_configure_task_for.repeat_count` semantic (aiex.py:289-291,
// `repeat_count = sizes[0] - 1`) so Pass C's verbatim surface to
// `configure_task.repeat_count` (ConduitToDMALower.cpp:1356-1359) yields
// the correct firmware fire count (`value + 1` per
// AIEDmaToNpu.cpp:180-183).
//
// Before #98: canon stamped `dma_repeat = N` (1-indexed = "fire N
// times"), Pass C surfaced verbatim → firmware fired N+1 times = over-fire
// by 1.  Masked by canon NPU smokes' separate structural bug (wrap-in-BD
// vs N dispatches), but real Llama-scale risk.
//
// Geometry: shim(0,0) producer → compute(0,2) consumer, depth=2,
//           memref<16xi32>, 4 host dispatches → canon stamps
//           dma_repeat = 3 (= 4 total fires).
//
// Companion fixture `canonicalize_channel_puts/homogeneous_repeat_collapse.mlir`
// pins the surrounding collapse semantics (single surviving put + chain
// preservation) and is updated to pin `dma_repeat = 3` post-#98.  This
// fixture's role is to be the named-by-the-fix lit-pin so future
// convention drift is caught at this exact site.

// CHECK-LABEL: aie.device(npu1)

// CHECK: conduit.create @chan
// CHECK-SAME: dma_repeat = 3

// Exactly one surviving put_memref_async on @chan (collapse worked).
// CHECK: conduit.put_memref_async
// CHECK-SAME: name = @chan
// CHECK-NOT: conduit.put_memref_async{{.*}}name = @chan

module @canon_homogeneous_repeat_zero_indexed {
  aie.device(npu1) {
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)

    conduit.create @chan {
      element_type = memref<16xi32>,
      depth = 2 : i64
    }

    aie.shim_dma_allocation @chan_shim_alloc(%tile_0_0, MM2S, 0) {conduit_channel = @chan}

    %core_0_2 = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c4 = arith.constant 4 : index
      %c1 = arith.constant 1 : index
      scf.for %i = %c0 to %c4 step %c1 {
        %g = conduit.get_memref_async {name = @chan,
                  num_elems = 16 : i64,
                  offsets = array<i64: 0>,
                  sizes = array<i64: 16>,
                  strides = array<i64: 1>} : !conduit.dma.token
        conduit.wait_all %g : !conduit.dma.token
      }
      aie.end
    } {dynamic_objfifo_lowering = true}

    func.func @sequence(%arg0: memref<16xi32>) {
      // 4 IRON-pattern identical puts.  Canon collapses to:
      //   1 surviving put + dma_repeat = 3 (= 4 total fires).
      %t0 = conduit.put_memref_async {name = @chan, num_elems = 16 : i64,
            offsets = array<i64: 0>, sizes = array<i64: 16>,
            strides = array<i64: 1>} : !conduit.dma.token
      conduit.wait_all %t0 {token = true} : !conduit.dma.token
      conduit.wait_all %t0 {token = false} : !conduit.dma.token

      %t1 = conduit.put_memref_async {name = @chan, num_elems = 16 : i64,
            offsets = array<i64: 0>, sizes = array<i64: 16>,
            strides = array<i64: 1>} : !conduit.dma.token
      conduit.wait_all %t1 {token = true} : !conduit.dma.token
      conduit.wait_all %t1 {token = false} : !conduit.dma.token

      %t2 = conduit.put_memref_async {name = @chan, num_elems = 16 : i64,
            offsets = array<i64: 0>, sizes = array<i64: 16>,
            strides = array<i64: 1>} : !conduit.dma.token
      conduit.wait_all %t2 {token = true} : !conduit.dma.token
      conduit.wait_all %t2 {token = false} : !conduit.dma.token

      %t3 = conduit.put_memref_async {name = @chan, num_elems = 16 : i64,
            offsets = array<i64: 0>, sizes = array<i64: 16>,
            strides = array<i64: 1>} : !conduit.dma.token
      conduit.wait_all %t3 {token = true} : !conduit.dma.token
      conduit.wait_all %t3 {token = false} : !conduit.dma.token
      return
    }
  }
}
