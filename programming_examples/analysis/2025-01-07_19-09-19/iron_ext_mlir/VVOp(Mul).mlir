module {
  aie.device(npu1_1col) {
    %tile_0_2 = aie.tile(0, 2)
    %tile_0_0 = aie.tile(0, 0)
    aie.objectfifo @in1(%tile_0_0, {%tile_0_2}, 2 : i32) : !aie.objectfifo<memref<16xi32>> 
    aie.objectfifo @in2(%tile_0_0, {%tile_0_2}, 2 : i32) : !aie.objectfifo<memref<16xi32>> 
    aie.objectfifo @out(%tile_0_2, {%tile_0_0}, 2 : i32) : !aie.objectfifo<memref<16xi32>> 
    %core_0_2 = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %c0_0 = arith.constant 0 : index
        %c16 = arith.constant 16 : index
        %c1_1 = arith.constant 1 : index
        scf.for %arg1 = %c0_0 to %c16 step %c1_1 {
          %0 = aie.objectfifo.acquire @in1(Consume, 1) : !aie.objectfifosubview<memref<16xi32>>
          %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>
          %2 = aie.objectfifo.acquire @in2(Consume, 1) : !aie.objectfifosubview<memref<16xi32>>
          %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>
          %4 = aie.objectfifo.acquire @out(Produce, 1) : !aie.objectfifosubview<memref<16xi32>>
          %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>
          %c0_2 = arith.constant 0 : index
          %c16_3 = arith.constant 16 : index
          %c1_4 = arith.constant 1 : index
          scf.for %arg2 = %c0_2 to %c16_3 step %c1_4 {
            %6 = memref.load %1[%arg2] : memref<16xi32>
            %7 = memref.load %3[%arg2] : memref<16xi32>
            %8 = arith.muli %6, %7 : i32
            memref.store %8, %5[%arg2] : memref<16xi32>
          }
          aie.objectfifo.release @in1(Consume, 1)
          aie.objectfifo.release @in2(Consume, 1)
          aie.objectfifo.release @out(Produce, 1)
        }
      }
      aie.end
    }
    aiex.runtime_sequence(%arg0: memref<256xi32>, %arg1: memref<256xi32>, %arg2: memref<256xi32>) {
      %0 = aiex.dma_configure_task_for @in1 {
        aie.dma_bd(%arg0 : memref<256xi32>, 0, 256, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 256, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%0)
      %1 = aiex.dma_configure_task_for @in2 {
        aie.dma_bd(%arg1 : memref<256xi32>, 0, 256, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 256, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%1)
      %2 = aiex.dma_configure_task_for @out {
        aie.dma_bd(%arg2 : memref<256xi32>, 0, 256, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 256, stride = 1>])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%2)
      aiex.dma_await_task(%2)
      aiex.dma_free_task(%0)
      aiex.dma_free_task(%1)
    }
  }
}

