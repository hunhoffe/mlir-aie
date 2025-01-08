module {
  aie.device(npu1_1col) {
    func.func private @rgba2grayLine(memref<7680xui8>, memref<1920xui8>, i32)
    func.func private @filter2dLine(memref<1920xui8>, memref<1920xui8>, memref<1920xui8>, memref<1920xui8>, i32, memref<3x3xi16>)
    func.func private @thresholdLine(memref<1920xui8>, memref<1920xui8>, i32, i16, i16, i8)
    func.func private @gray2rgbaLine(memref<1920xui8>, memref<7680xui8>, i32)
    func.func private @addWeightedLine(memref<7680xui8>, memref<7680xui8>, memref<7680xui8>, i32, i16, i16, i8)
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_1 = aie.tile(0, 1)
    %tile_0_2 = aie.tile(0, 2)
    %tile_0_3 = aie.tile(0, 3)
    %tile_0_4 = aie.tile(0, 4)
    %tile_0_5 = aie.tile(0, 5)
    aie.objectfifo @inOF_L3L2(%tile_0_0, {%tile_0_2, %tile_0_1}, [2 : i32, 2 : i32, 7 : i32]) : !aie.objectfifo<memref<7680xui8>> 
    aie.objectfifo @inOF_L2L1(%tile_0_1, {%tile_0_5}, 7 : i32) : !aie.objectfifo<memref<7680xui8>> 
    aie.objectfifo.link [@inOF_L3L2] -> [@inOF_L2L1]([] [])
    aie.objectfifo @outOF_L2L3(%tile_0_1, {%tile_0_0}, 2 : i32) : !aie.objectfifo<memref<7680xui8>> 
    aie.objectfifo @outOF_L1L2(%tile_0_5, {%tile_0_1}, 2 : i32) : !aie.objectfifo<memref<7680xui8>> 
    aie.objectfifo.link [@outOF_L1L2] -> [@outOF_L2L3]([] [])
    aie.objectfifo @OF_2to3(%tile_0_2, {%tile_0_3}, 4 : i32) : !aie.objectfifo<memref<1920xui8>> 
    aie.objectfifo @OF_3to4(%tile_0_3, {%tile_0_4}, 2 : i32) : !aie.objectfifo<memref<1920xui8>> 
    aie.objectfifo @OF_4to5(%tile_0_4, {%tile_0_5}, 2 : i32) : !aie.objectfifo<memref<1920xui8>> 
    aie.objectfifo @OF_5to5(%tile_0_5, {%tile_0_5}, 1 : i32) : !aie.objectfifo<memref<7680xui8>> 
    %core_0_2 = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %0 = aie.objectfifo.acquire @inOF_L3L2(Consume, 1) : !aie.objectfifosubview<memref<7680xui8>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<7680xui8>> -> memref<7680xui8>
        %2 = aie.objectfifo.acquire @OF_2to3(Produce, 1) : !aie.objectfifosubview<memref<1920xui8>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<1920xui8>> -> memref<1920xui8>
        %c1920_i32 = arith.constant 1920 : i32
        func.call @rgba2grayLine(%1, %3, %c1920_i32) : (memref<7680xui8>, memref<1920xui8>, i32) -> ()
        aie.objectfifo.release @inOF_L3L2(Consume, 1)
        aie.objectfifo.release @OF_2to3(Produce, 1)
      }
      aie.end
    } {link_with = "rgba2gray.cc.o"}
    %core_0_3 = aie.core(%tile_0_3) {
      %kernel = aie.buffer(%tile_0_3) {sym_name = "kernel"} : memref<3x3xi16> = dense<[[0, 4096, 0], [4096, -16384, 4096], [0, 4096, 0]]>
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %0 = aie.objectfifo.acquire @OF_2to3(Consume, 2) : !aie.objectfifosubview<memref<1920xui8>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<1920xui8>> -> memref<1920xui8>
        %2 = aie.objectfifo.subview.access %0[1] : !aie.objectfifosubview<memref<1920xui8>> -> memref<1920xui8>
        %3 = aie.objectfifo.acquire @OF_3to4(Produce, 1) : !aie.objectfifosubview<memref<1920xui8>>
        %4 = aie.objectfifo.subview.access %3[0] : !aie.objectfifosubview<memref<1920xui8>> -> memref<1920xui8>
        %c1920_i32 = arith.constant 1920 : i32
        func.call @filter2dLine(%1, %1, %2, %4, %c1920_i32, %kernel) : (memref<1920xui8>, memref<1920xui8>, memref<1920xui8>, memref<1920xui8>, i32, memref<3x3xi16>) -> ()
        aie.objectfifo.release @OF_3to4(Produce, 1)
        %c1_0 = arith.constant 1 : index
        %c1079 = arith.constant 1079 : index
        %c1_1 = arith.constant 1 : index
        scf.for %arg1 = %c1_0 to %c1079 step %c1_1 {
          %10 = aie.objectfifo.acquire @OF_2to3(Consume, 3) : !aie.objectfifosubview<memref<1920xui8>>
          %11 = aie.objectfifo.subview.access %10[0] : !aie.objectfifosubview<memref<1920xui8>> -> memref<1920xui8>
          %12 = aie.objectfifo.subview.access %10[1] : !aie.objectfifosubview<memref<1920xui8>> -> memref<1920xui8>
          %13 = aie.objectfifo.subview.access %10[2] : !aie.objectfifosubview<memref<1920xui8>> -> memref<1920xui8>
          %14 = aie.objectfifo.acquire @OF_3to4(Produce, 1) : !aie.objectfifosubview<memref<1920xui8>>
          %15 = aie.objectfifo.subview.access %14[0] : !aie.objectfifosubview<memref<1920xui8>> -> memref<1920xui8>
          %c1920_i32_3 = arith.constant 1920 : i32
          func.call @filter2dLine(%11, %12, %13, %15, %c1920_i32_3, %kernel) : (memref<1920xui8>, memref<1920xui8>, memref<1920xui8>, memref<1920xui8>, i32, memref<3x3xi16>) -> ()
          aie.objectfifo.release @OF_2to3(Consume, 1)
          aie.objectfifo.release @OF_3to4(Produce, 1)
        }
        %5 = aie.objectfifo.acquire @OF_2to3(Consume, 2) : !aie.objectfifosubview<memref<1920xui8>>
        %6 = aie.objectfifo.subview.access %5[0] : !aie.objectfifosubview<memref<1920xui8>> -> memref<1920xui8>
        %7 = aie.objectfifo.subview.access %5[1] : !aie.objectfifosubview<memref<1920xui8>> -> memref<1920xui8>
        %8 = aie.objectfifo.acquire @OF_3to4(Produce, 1) : !aie.objectfifosubview<memref<1920xui8>>
        %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<1920xui8>> -> memref<1920xui8>
        %c1920_i32_2 = arith.constant 1920 : i32
        func.call @filter2dLine(%6, %7, %7, %9, %c1920_i32_2, %kernel) : (memref<1920xui8>, memref<1920xui8>, memref<1920xui8>, memref<1920xui8>, i32, memref<3x3xi16>) -> ()
        aie.objectfifo.release @OF_2to3(Consume, 2)
        aie.objectfifo.release @OF_3to4(Produce, 1)
      }
      aie.end
    } {link_with = "filter2d.cc.o"}
    %core_0_4 = aie.core(%tile_0_4) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %0 = aie.objectfifo.acquire @OF_3to4(Consume, 1) : !aie.objectfifosubview<memref<1920xui8>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<1920xui8>> -> memref<1920xui8>
        %2 = aie.objectfifo.acquire @OF_4to5(Produce, 1) : !aie.objectfifosubview<memref<1920xui8>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<1920xui8>> -> memref<1920xui8>
        %c1920_i32 = arith.constant 1920 : i32
        %c10_i16 = arith.constant 10 : i16
        %c255_i16 = arith.constant 255 : i16
        %c0_i8 = arith.constant 0 : i8
        func.call @thresholdLine(%1, %3, %c1920_i32, %c10_i16, %c255_i16, %c0_i8) : (memref<1920xui8>, memref<1920xui8>, i32, i16, i16, i8) -> ()
        aie.objectfifo.release @OF_3to4(Consume, 1)
        aie.objectfifo.release @OF_4to5(Produce, 1)
      }
      aie.end
    } {link_with = "threshold.cc.o"}
    %core_0_5 = aie.core(%tile_0_5) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %0 = aie.objectfifo.acquire @OF_4to5(Consume, 1) : !aie.objectfifosubview<memref<1920xui8>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<1920xui8>> -> memref<1920xui8>
        %2 = aie.objectfifo.acquire @OF_5to5(Produce, 1) : !aie.objectfifosubview<memref<7680xui8>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<7680xui8>> -> memref<7680xui8>
        %c1920_i32 = arith.constant 1920 : i32
        func.call @gray2rgbaLine(%1, %3, %c1920_i32) : (memref<1920xui8>, memref<7680xui8>, i32) -> ()
        aie.objectfifo.release @OF_4to5(Consume, 1)
        aie.objectfifo.release @OF_5to5(Produce, 1)
        %4 = aie.objectfifo.acquire @OF_5to5(Consume, 1) : !aie.objectfifosubview<memref<7680xui8>>
        %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<7680xui8>> -> memref<7680xui8>
        %6 = aie.objectfifo.acquire @inOF_L2L1(Consume, 1) : !aie.objectfifosubview<memref<7680xui8>>
        %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<7680xui8>> -> memref<7680xui8>
        %8 = aie.objectfifo.acquire @outOF_L1L2(Produce, 1) : !aie.objectfifosubview<memref<7680xui8>>
        %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<7680xui8>> -> memref<7680xui8>
        %c7680_i32 = arith.constant 7680 : i32
        %c16384_i16 = arith.constant 16384 : i16
        %c16384_i16_0 = arith.constant 16384 : i16
        %c0_i8 = arith.constant 0 : i8
        func.call @addWeightedLine(%5, %7, %9, %c7680_i32, %c16384_i16, %c16384_i16_0, %c0_i8) : (memref<7680xui8>, memref<7680xui8>, memref<7680xui8>, i32, i16, i16, i8) -> ()
        aie.objectfifo.release @OF_5to5(Consume, 1)
        aie.objectfifo.release @inOF_L2L1(Consume, 1)
        aie.objectfifo.release @outOF_L1L2(Produce, 1)
      }
      aie.end
    } {link_with = "combined_gray2rgba_addWeighted.a"}
    aiex.runtime_sequence(%arg0: memref<8294400xi8>, %arg1: memref<16x16xi32>, %arg2: memref<8294400xi8>) {
      %0 = aiex.dma_configure_task_for @inOF_L3L2 {
        aie.dma_bd(%arg0 : memref<8294400xi8>, 0, 8294400, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 8294400, stride = 1>])
        aie.end
      }
      %1 = aiex.dma_configure_task_for @outOF_L2L3 {
        aie.dma_bd(%arg2 : memref<8294400xi8>, 0, 8294400, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 8294400, stride = 1>])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%0)
      aiex.dma_start_task(%1)
      aiex.dma_await_task(%1)
      aiex.dma_free_task(%0)
    }
  }
}

