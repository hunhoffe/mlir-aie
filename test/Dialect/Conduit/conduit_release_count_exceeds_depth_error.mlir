// RUN: not aie-opt --conduit-to-dma %s 2>&1 | FileCheck %s
//
// Regression test (A-8): release count > depth must emit a hard error.
//
// emitFastModulo assumes counter + delta < 2*depth, which requires delta <= depth.
// If a conduit.release has count=3 on a depth=2 conduit, the modulo optimization
// would produce incorrect results — the branchless conditional subtract only
// removes at most `depth`, so (counter + 3) % 2 would wrap incorrectly.
//
// Before the fix, this was silently accepted, producing wrong modular arithmetic
// that could cause the rotation counter to jump to wrong buffer indices.
//
// CHECK: error:{{.*}}M8: cumulative release count (3) exceeds acquired count (2)

module @release_count_exceeds_depth {
  aie.device(npu1_1col) {
    func.func @consume(%buf: memref<8xi32>) -> () {
      return
    }

    %tile_0_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)

    conduit.create @fifo {slot_elems = 16 : i64,
                    producer_tile = array<i64: 0, 0>,
                    consumer_tiles = array<i64: 0, 2>,
                    element_type = memref<8xi32>,
                    depth = 2 : i64}

    aie.shim_dma_allocation @fifo_shim_alloc(%tile_0_0, MM2S, 0)

    %core = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c4 = arith.constant 4 : index
      scf.for %arg0 = %c0 to %c4 step %c1 {
        %win = conduit.acquire {name = @fifo, count = 2 : i64,
                               port = #conduit.port<Consume>}
                   : !conduit.window<memref<8xi32>>
        %elem = conduit.subview_access %win {index = 0 : i64}
                    : !conduit.window<memref<8xi32>> -> memref<8xi32>
        func.call @consume(%elem) : (memref<8xi32>) -> ()
        // count=3 exceeds depth=2 — must be a hard error.
        conduit.release %win {count = 3 : i64, port = #conduit.port<Consume>}
            : !conduit.window<memref<8xi32>>
      }
      aie.end
    } {dynamic_objfifo_lowering = true}
  }
}
