// RUN: aie-opt --conduit-canonicalize-channel-puts %s | FileCheck %s
//
// Pin the basic collapse behavior of --conduit-canonicalize-channel-puts:
//   * 4 structurally-identical conduit.put_memref_async ops on @chan with
//     matching wait_all{token=true} await + wait_all{token=false} free chains
//     → collapsed to 1 put + 1 await + 1 free
//   * conduit.create @chan gains dma_repeat = 4
//
// This is the canonical IRON `for batch in range(4)` shape that the IRON
// `task_group` / `finish_task_group` lowering produces after the upstream
// --dma-task-to-conduit pass round-trips it into Conduit IR.
//
// Geometry: shim(0,0) producer → compute(0,2) consumer, depth=2,
//           memref<16xi32>, 4 host dispatches.

// CHECK-LABEL: aie.device(npu1)

// CHECK: conduit.create @chan
// CHECK-SAME: dma_repeat = 4

// Exactly one surviving put_memref_async on @chan.
// CHECK: conduit.put_memref_async
// CHECK-SAME: name = @chan
// CHECK-NOT: conduit.put_memref_async{{.*}}name = @chan

// One await + one free on the surviving token; canon kept the matched pair.
// The await keeps the put[0] release-marker semantic (token defaults to true,
// printer omits the attr when default per Conduit.td:994
// DefaultValuedOptionalAttr<BoolAttr, "true">).  The free is explicit
// {token = false}.
// CHECK: conduit.wait_all %{{[^ ]+}} : !conduit.dma.token
// CHECK-NEXT: conduit.wait_all %{{[^ ]+}} {token = false} : !conduit.dma.token

module @conduit_canonicalize_loop_unroll_puts_collapse {
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
      // 4 IRON-pattern identical puts, each with await + free.
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
