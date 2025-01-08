module {
  aie.device(npu1_1col) {
    %tile_0_2 = aie.tile(0, 2)
    %tile_0_0 = aie.tile(0, 0)
    aie.objectfifo @in(%tile_0_0, {%tile_0_2}, 2 : i32) : !aie.objectfifo<memref<1024xi16>> 
    aie.objectfifo @out(%tile_0_2, {%tile_0_0}, 2 : i32) : !aie.objectfifo<memref<1024xi16>> 
    aie.objectfifo @infactor(%tile_0_0, {%tile_0_2}, 2 : i32) : !aie.objectfifo<memref<1xi32>> 
    func.func private @vector_scalar_mul_int16_vector(memref<1024xi16>, memref<1024xi16>, memref<1xi32>, i32)
    %core_0_2 = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %0 = aie.objectfifo.acquire @infactor(Consume, 1) : !aie.objectfifosubview<memref<1xi32>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<1xi32>> -> memref<1xi32>
        %c0_0 = arith.constant 0 : index
        %c4 = arith.constant 4 : index
        %c1_1 = arith.constant 1 : index
        scf.for %arg1 = %c0_0 to %c4 step %c1_1 {
          %2 = aie.objectfifo.acquire @out(Produce, 1) : !aie.objectfifosubview<memref<1024xi16>>
          %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<1024xi16>> -> memref<1024xi16>
          %4 = aie.objectfifo.acquire @in(Consume, 1) : !aie.objectfifosubview<memref<1024xi16>>
          %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<1024xi16>> -> memref<1024xi16>
          %c1024_i32 = arith.constant 1024 : i32
          func.call @vector_scalar_mul_int16_vector(%5, %3, %1, %c1024_i32) : (memref<1024xi16>, memref<1024xi16>, memref<1xi32>, i32) -> ()
          aie.objectfifo.release @in(Consume, 1)
          aie.objectfifo.release @out(Produce, 1)
        }
        aie.objectfifo.release @infactor(Consume, 1)
      }
      aie.end
    } {link_with = "scale.o"}
    aiex.runtime_sequence(%arg0: memref<4096xi16>, %arg1: memref<1xi32>, %arg2: memref<4096xi16>) {
      %0 = aiex.dma_configure_task_for @in {
        aie.dma_bd(%arg0 : memref<4096xi16>, 0, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 4096, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%0)
      %1 = aiex.dma_configure_task_for @infactor {
        aie.dma_bd(%arg1 : memref<1xi32>, 0, 1, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%1)
      %2 = aiex.dma_configure_task_for @out {
        aie.dma_bd(%arg2 : memref<4096xi16>, 0, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 4096, stride = 1>])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%2)
      aiex.dma_await_task(%2)
      aiex.dma_free_task(%0)
      aiex.dma_free_task(%1)
    }
  }
}

