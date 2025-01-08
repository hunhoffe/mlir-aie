module {
  aie.device(npu1_1col) {
    %tile_0_2 = aie.tile(0, 2)
    %tile_0_0 = aie.tile(0, 0)
    aie.objectfifo @bias_fifo(%tile_0_0, {%tile_0_2}, 2 : i32) : !aie.objectfifo<memref<32xf32>> 
    aie.objectfifo @out_fifo(%tile_0_2, {%tile_0_0}, 2 : i32) : !aie.objectfifo<memref<3072xf32>> 
    aie.objectfifo @in_fifo(%tile_0_0, {%tile_0_2}, 2 : i32) : !aie.objectfifo<memref<3072xf32>> 
    func.func private @row_wise_bias_add_f32_f32(memref<3072xf32>, memref<32xf32>, memref<3072xf32>)
    %core_0_2 = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %c0_0 = arith.constant 0 : index
        %c72 = arith.constant 72 : index
        %c1_1 = arith.constant 1 : index
        scf.for %arg1 = %c0_0 to %c72 step %c1_1 {
          %0 = aie.objectfifo.acquire @bias_fifo(Consume, 1) : !aie.objectfifosubview<memref<32xf32>>
          %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<32xf32>> -> memref<32xf32>
          %c0_2 = arith.constant 0 : index
          %c8 = arith.constant 8 : index
          %c1_3 = arith.constant 1 : index
          scf.for %arg2 = %c0_2 to %c8 step %c1_3 {
            %2 = aie.objectfifo.acquire @in_fifo(Consume, 1) : !aie.objectfifosubview<memref<3072xf32>>
            %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<3072xf32>> -> memref<3072xf32>
            %4 = aie.objectfifo.acquire @out_fifo(Produce, 1) : !aie.objectfifosubview<memref<3072xf32>>
            %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<3072xf32>> -> memref<3072xf32>
            func.call @row_wise_bias_add_f32_f32(%3, %1, %5) : (memref<3072xf32>, memref<32xf32>, memref<3072xf32>) -> ()
            aie.objectfifo.release @out_fifo(Produce, 1)
            aie.objectfifo.release @in_fifo(Consume, 1)
          }
          aie.objectfifo.release @bias_fifo(Consume, 1)
        }
      }
      aie.end
    } {link_with = "kernel.o"}
    aiex.runtime_sequence(%arg0: memref<3072xf32>, %arg1: memref<32xf32>, %arg2: memref<3072xf32>) {
      %0 = aiex.dma_configure_task_for @in_fifo {
        aie.dma_bd(%arg0 : memref<3072xf32>, 0, 1769472, [<size = 1, stride = 0>, <size = 72, stride = 32>, <size = 768, stride = 2304>, <size = 32, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%0)
      %1 = aiex.dma_configure_task_for @bias_fifo {
        aie.dma_bd(%arg1 : memref<32xf32>, 0, 2304, [<size = 1, stride = 0>, <size = 72, stride = 32>, <size = 1, stride = 2304>, <size = 32, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%1)
      %2 = aiex.dma_configure_task_for @out_fifo {
        aie.dma_bd(%arg2 : memref<3072xf32>, 0, 1769472, [<size = 1, stride = 0>, <size = 72, stride = 32>, <size = 768, stride = 2304>, <size = 32, stride = 1>])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%2)
      aiex.dma_await_task(%2)
      aiex.dma_free_task(%0)
      aiex.dma_free_task(%1)
    }
  }
}

