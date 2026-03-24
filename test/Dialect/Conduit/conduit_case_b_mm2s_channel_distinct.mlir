// RUN: aie-opt --objectfifo-to-conduit --conduit-to-dma %s | FileCheck %s
//
// Regression test (A-1): Case B MM2S channel must be dynamically allocated.
//
// When two objectfifos share the same compute tile as producer and both route
// to a shim consumer, the Case B path must allocate distinct MM2S channel
// indices (0 and 1) instead of hardcoding channel 0 for both.
//
// Before the fix, both `aie.dma_start(MM2S, ...)` in the producer tile's
// aie.mem block had channel index 0, causing a hardware conflict.
//
// Topology:
//   tile(0,2) [compute producer]
//     -> tile(0,0) [shim consumer] via fifo_out_a   (MM2S channel 0)
//     -> tile(0,0) [shim consumer] via fifo_out_b   (MM2S channel 1)
//
// Expected: the producer tile's aie.mem block has two aie.dma_start ops with
//   MM2S channel indices 0 and 1 (not both 0).
//
// CHECK-LABEL: module @case_b_mm2s_channel_distinct
// CHECK: aie.device(npu1_1col)
// Check that both MM2S chains appear in the same aie.mem block on tile(0,2).
// CHECK:     aie.mem(%{{.*}}tile_0_2) {
// CHECK:       aie.dma_start(MM2S, 0,
// CHECK:       aie.dma_start(MM2S, 1,
// CHECK:     }
// CHECK-NOT: conduit.create
// CHECK-NOT: conduit.acquire
// CHECK-NOT: conduit.release

module @case_b_mm2s_channel_distinct {
  aie.device(npu1_1col) {
    func.func @produce(%buf: memref<8xi32>) -> () {
      return
    }

    %shim = aie.tile(0, 0)
    %tile02 = aie.tile(0, 2)

    // First objectfifo: compute tile(0,2) produces to shim tile(0,0).
    aie.objectfifo @fifo_out_a(%tile02, {%shim}, 1 : i32) :
        !aie.objectfifo<memref<8xi32>>

    // Second objectfifo: same compute tile(0,2) also produces to shim.
    aie.objectfifo @fifo_out_b(%tile02, {%shim}, 1 : i32) :
        !aie.objectfifo<memref<8xi32>>

    // Producer core: acquires from both fifos and produces data.
    %core02 = aie.core(%tile02) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c4 = arith.constant 4 : index
      scf.for %arg0 = %c0 to %c4 step %c1 {
        %subview_a = aie.objectfifo.acquire @fifo_out_a(Produce, 1) :
            !aie.objectfifosubview<memref<8xi32>>
        %elem_a = aie.objectfifo.subview.access %subview_a[0] :
            !aie.objectfifosubview<memref<8xi32>> -> memref<8xi32>
        func.call @produce(%elem_a) : (memref<8xi32>) -> ()
        aie.objectfifo.release @fifo_out_a(Produce, 1)

        %subview_b = aie.objectfifo.acquire @fifo_out_b(Produce, 1) :
            !aie.objectfifosubview<memref<8xi32>>
        %elem_b = aie.objectfifo.subview.access %subview_b[0] :
            !aie.objectfifosubview<memref<8xi32>> -> memref<8xi32>
        func.call @produce(%elem_b) : (memref<8xi32>) -> ()
        aie.objectfifo.release @fifo_out_b(Produce, 1)
      }
      aie.end
    } {dynamic_objfifo_lowering = true}
  }
}
