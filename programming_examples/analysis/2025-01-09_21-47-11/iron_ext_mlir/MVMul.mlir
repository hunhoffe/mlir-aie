module {
  aie.device(npu1_4col) {
    %tile_0_2 = aie.tile(0, 2)
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_1 = aie.tile(0, 1)
    aie.objectfifo @inB(%tile_0_0, {%tile_0_2}, 2 : i32) : !aie.objectfifo<memref<32xi16>> 
    aie.objectfifo @inA0(%tile_0_1, {%tile_0_2}, 2 : i32) : !aie.objectfifo<memref<32x32xi16>> 
    aie.objectfifo @memA0(%tile_0_0, {%tile_0_1}, 2 : i32) : !aie.objectfifo<memref<32x32xi16>> 
    aie.objectfifo.link [@memA0] -> [@inA0]([] [0])
    aie.objectfifo @outC0(%tile_0_2, {%tile_0_0}, 2 : i32) : !aie.objectfifo<memref<32xi32>> 
    func.func private @zero_scalar_i32(memref<32xi32>)
    func.func private @matvec_scalar_i16_i32(memref<32x32xi16>, memref<32xi16>, memref<32xi32>)
    %core_0_2 = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %0 = aie.objectfifo.acquire @outC0(Produce, 1) : !aie.objectfifosubview<memref<32xi32>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<32xi32>> -> memref<32xi32>
        func.call @zero_scalar_i32(%1) : (memref<32xi32>) -> ()
        %c0_0 = arith.constant 0 : index
        %c9 = arith.constant 9 : index
        %c1_1 = arith.constant 1 : index
        scf.for %arg1 = %c0_0 to %c9 step %c1_1 {
          %2 = aie.objectfifo.acquire @inA0(Consume, 1) : !aie.objectfifosubview<memref<32x32xi16>>
          %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<32x32xi16>> -> memref<32x32xi16>
          %4 = aie.objectfifo.acquire @inB(Consume, 1) : !aie.objectfifosubview<memref<32xi16>>
          %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<32xi16>> -> memref<32xi16>
          func.call @matvec_scalar_i16_i32(%3, %5, %1) : (memref<32x32xi16>, memref<32xi16>, memref<32xi32>) -> ()
          aie.objectfifo.release @inA0(Consume, 1)
          aie.objectfifo.release @inB(Consume, 1)
        }
        aie.objectfifo.release @outC0(Produce, 1)
      }
      aie.end
    } {link_with = "mv_32x32.o"}
    aiex.runtime_sequence(%arg0: memref<288x288xi16>, %arg1: memref<288xi16>, %arg2: memref<288xi32>) {
      %0 = aiex.dma_configure_task_for @inB {
        aie.dma_bd(%arg1 : memref<288xi16>, 0, 288, [<size = 9, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 288, stride = 1>])
        aie.end
      } {repeat_count = 8 : i32}
      aiex.dma_start_task(%0)
      %1 = aiex.dma_configure_task_for @memA0 {
        aie.dma_bd(%arg0 : memref<288x288xi16>, 0, 9216, [<size = 9, stride = 9216>, <size = 9, stride = 32>, <size = 32, stride = 288>, <size = 32, stride = 1>])
        aie.end
      } {repeat_count = 8 : i32}
      aiex.dma_start_task(%1)
      %2 = aiex.dma_configure_task_for @outC0 {
        aie.dma_bd(%arg2 : memref<288xi32>, 0, 288, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 288, stride = 1>])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%2)
      aiex.dma_await_task(%2)
      aiex.dma_free_task(%0)
      aiex.dma_free_task(%1)
    }
  }
}

