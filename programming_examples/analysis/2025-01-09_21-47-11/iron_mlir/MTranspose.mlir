module {
  aie.device(npu1_1col) {
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)
    aie.objectfifo @in(%tile_0_0, {%tile_0_2}, 2 : i32) : !aie.objectfifo<memref<64x32xi32>> 
    aie.objectfifo @out(%tile_0_2, {%tile_0_0}, 2 : i32) : !aie.objectfifo<memref<64x32xi32>> 
    aie.objectfifo.link [@in] -> [@out]([] [])
    aiex.runtime_sequence(%arg0: memref<64x32xi32>, %arg1: memref<64x32xi32>, %arg2: memref<64x32xi32>) {
      %0 = aiex.dma_configure_task_for @in {
        aie.dma_bd(%arg0 : memref<64x32xi32>, 0, 2048, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 32, stride = 1>, <size = 64, stride = 32>])
        aie.end
      }
      %1 = aiex.dma_configure_task_for @out {
        aie.dma_bd(%arg2 : memref<64x32xi32>, 0, 2048, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 2048, stride = 1>])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%0)
      aiex.dma_start_task(%1)
      aiex.dma_await_task(%1)
      aiex.dma_free_task(%0)
    }
  }
}

