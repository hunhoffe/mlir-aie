module {
  aie.device(npu1_1col) {
    %tile_0_2 = aie.tile(0, 2)
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_1 = aie.tile(0, 1)
    aie.objectfifo @inOF_wts_0_L3L2(%tile_0_0, {%tile_0_2}, 1 : i32) : !aie.objectfifo<memref<4096xi8>> 
    aie.objectfifo @out_02_L2(%tile_0_2, {%tile_0_1}, 2 : i32) : !aie.objectfifo<memref<2048xi8>> 
    aie.objectfifo @outOFL2L3(%tile_0_1, {%tile_0_0}, 2 : i32) : !aie.objectfifo<memref<4096xi8>> 
    aie.objectfifo.link [@out_02_L2] -> [@outOFL2L3]([] [0])
    aie.objectfifo @act_L2_02(%tile_0_1, {%tile_0_2}, 2 : i32) : !aie.objectfifo<memref<2048xi8>> 
    aie.objectfifo @inOF_act_L3L2(%tile_0_0, {%tile_0_1}, 2 : i32) : !aie.objectfifo<memref<4096xi8>> 
    aie.objectfifo.link [@inOF_act_L3L2] -> [@act_L2_02]([] [0])
    %rtp2 = aie.buffer(%tile_0_2) {sym_name = "rtp2"} : memref<16xi32> 
    func.func private @conv2dk1_i8(memref<2048xi8>, memref<4096xi8>, memref<2048xi8>, i32, i32, i32, i32)
    %core_0_2 = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %0 = aie.objectfifo.acquire @inOF_wts_0_L3L2(Consume, 1) : !aie.objectfifosubview<memref<4096xi8>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<4096xi8>> -> memref<4096xi8>
        %c0_0 = arith.constant 0 : index
        %2 = memref.load %rtp2[%c0_0] : memref<16xi32>
        %c0_1 = arith.constant 0 : index
        %c32 = arith.constant 32 : index
        %c1_2 = arith.constant 1 : index
        scf.for %arg1 = %c0_1 to %c32 step %c1_2 {
          %3 = aie.objectfifo.acquire @act_L2_02(Consume, 1) : !aie.objectfifosubview<memref<2048xi8>>
          %4 = aie.objectfifo.subview.access %3[0] : !aie.objectfifosubview<memref<2048xi8>> -> memref<2048xi8>
          %5 = aie.objectfifo.acquire @out_02_L2(Produce, 1) : !aie.objectfifosubview<memref<2048xi8>>
          %6 = aie.objectfifo.subview.access %5[0] : !aie.objectfifosubview<memref<2048xi8>> -> memref<2048xi8>
          %c32_i32 = arith.constant 32 : i32
          %c64_i32 = arith.constant 64 : i32
          %c64_i32_3 = arith.constant 64 : i32
          func.call @conv2dk1_i8(%4, %1, %6, %c32_i32, %c64_i32, %c64_i32_3, %2) : (memref<2048xi8>, memref<4096xi8>, memref<2048xi8>, i32, i32, i32, i32) -> ()
          aie.objectfifo.release @act_L2_02(Consume, 1)
          aie.objectfifo.release @out_02_L2(Produce, 1)
        }
        aie.objectfifo.release @inOF_wts_0_L3L2(Consume, 1)
      }
      aie.end
    } {link_with = "conv2dk1.o"}
    aiex.runtime_sequence(%arg0: memref<65536xi8>, %arg1: memref<4096xi8>, %arg2: memref<65536xi8>) {
      aiex.npu.rtp_write(@rtp2, 0, 1)
      %0 = aiex.dma_configure_task_for @inOF_act_L3L2 {
        aie.dma_bd(%arg0 : memref<65536xi8>, 0, 65536, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 65536, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%0)
      %1 = aiex.dma_configure_task_for @inOF_wts_0_L3L2 {
        aie.dma_bd(%arg1 : memref<4096xi8>, 0, 4096, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 4096, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%1)
      %2 = aiex.dma_configure_task_for @outOFL2L3 {
        aie.dma_bd(%arg2 : memref<65536xi8>, 0, 65536, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 65536, stride = 1>])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%2)
      aiex.dma_await_task(%2)
      aiex.dma_free_task(%0)
      aiex.dma_free_task(%1)
    }
  }
}

