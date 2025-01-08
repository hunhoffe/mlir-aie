module {
  aie.device(npu1_1col) {
    %tile_0_2 = aie.tile(0, 2)
    %tile_0_3 = aie.tile(0, 3)
    %tile_0_4 = aie.tile(0, 4)
    %tile_0_5 = aie.tile(0, 5)
    %tile_0_1 = aie.tile(0, 1)
    %tile_0_0 = aie.tile(0, 0)
    aie.objectfifo @OF_2to34(%tile_0_2, {%tile_0_3, %tile_0_4}, 2 : i32) : !aie.objectfifo<memref<1920xui8>> 
    aie.objectfifo @OF_3to3(%tile_0_3, {%tile_0_3}, 1 : i32) : !aie.objectfifo<memref<1920xui8>> 
    aie.objectfifo @OF_3to5(%tile_0_3, {%tile_0_5}, 2 : i32) : !aie.objectfifo<memref<1920xui8>> 
    aie.objectfifo @OF_4to5(%tile_0_4, {%tile_0_5}, 2 : i32) : !aie.objectfifo<memref<1920xui8>> 
    aie.objectfifo @OF_5to5a(%tile_0_5, {%tile_0_5}, 1 : i32) : !aie.objectfifo<memref<1920xui8>> 
    aie.objectfifo @OF_5to5b(%tile_0_5, {%tile_0_5}, 1 : i32) : !aie.objectfifo<memref<7680xui8>> 
    aie.objectfifo @OF_4to4(%tile_0_4, {%tile_0_4}, 1 : i32) : !aie.objectfifo<memref<1920xui8>> 
    aie.objectfifo @inOF_L2L1(%tile_0_1, {%tile_0_5}, 6 : i32) : !aie.objectfifo<memref<7680xui8>> 
    aie.objectfifo @inOF_L3L2(%tile_0_0, {%tile_0_1, %tile_0_2}, [2 : i32, 6 : i32, 2 : i32]) : !aie.objectfifo<memref<7680xui8>> 
    aie.objectfifo.link [@inOF_L3L2] -> [@inOF_L2L1]([] [0])
    aie.objectfifo @outOF_L1L2(%tile_0_5, {%tile_0_1}, 2 : i32) : !aie.objectfifo<memref<7680xui8>> 
    aie.objectfifo @outOF_L2L3(%tile_0_1, {%tile_0_0}, 2 : i32) : !aie.objectfifo<memref<7680xui8>> 
    aie.objectfifo.link [@outOF_L1L2] -> [@outOF_L2L3]([] [0])
    func.func private @rgba2hueLine(memref<7680xui8>, memref<1920xui8>, i32)
    func.func private @thresholdLine(memref<1920xui8>, memref<1920xui8>, i32, i16, i16, i8)
    func.func private @bitwiseORLine(memref<1920xui8>, memref<1920xui8>, memref<1920xui8>, i32)
    func.func private @gray2rgbaLine(memref<1920xui8>, memref<7680xui8>, i32)
    func.func private @bitwiseANDLine(memref<7680xui8>, memref<7680xui8>, memref<7680xui8>, i32)
    %core_0_2 = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %0 = aie.objectfifo.acquire @inOF_L3L2(Consume, 1) : !aie.objectfifosubview<memref<7680xui8>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<7680xui8>> -> memref<7680xui8>
        %2 = aie.objectfifo.acquire @OF_2to34(Produce, 1) : !aie.objectfifosubview<memref<1920xui8>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<1920xui8>> -> memref<1920xui8>
        %c1920_i32 = arith.constant 1920 : i32
        func.call @rgba2hueLine(%1, %3, %c1920_i32) : (memref<7680xui8>, memref<1920xui8>, i32) -> ()
        aie.objectfifo.release @inOF_L3L2(Consume, 1)
        aie.objectfifo.release @OF_2to34(Produce, 1)
      }
      aie.end
    } {link_with = "rgba2hue.cc.o"}
    %core_0_3 = aie.core(%tile_0_3) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %0 = aie.objectfifo.acquire @OF_2to34(Consume, 1) : !aie.objectfifosubview<memref<1920xui8>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<1920xui8>> -> memref<1920xui8>
        %2 = aie.objectfifo.acquire @OF_3to3(Produce, 1) : !aie.objectfifosubview<memref<1920xui8>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<1920xui8>> -> memref<1920xui8>
        %c1920_i32 = arith.constant 1920 : i32
        %c40_i16 = arith.constant 40 : i16
        %c255_i16 = arith.constant 255 : i16
        %c4_i8 = arith.constant 4 : i8
        func.call @thresholdLine(%1, %3, %c1920_i32, %c40_i16, %c255_i16, %c4_i8) : (memref<1920xui8>, memref<1920xui8>, i32, i16, i16, i8) -> ()
        aie.objectfifo.release @OF_2to34(Consume, 1)
        aie.objectfifo.release @OF_3to3(Produce, 1)
        %4 = aie.objectfifo.acquire @OF_3to3(Consume, 1) : !aie.objectfifosubview<memref<1920xui8>>
        %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<1920xui8>> -> memref<1920xui8>
        %6 = aie.objectfifo.acquire @OF_3to5(Produce, 1) : !aie.objectfifosubview<memref<1920xui8>>
        %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<1920xui8>> -> memref<1920xui8>
        %c1920_i32_0 = arith.constant 1920 : i32
        %c30_i16 = arith.constant 30 : i16
        %c255_i16_1 = arith.constant 255 : i16
        %c0_i8 = arith.constant 0 : i8
        func.call @thresholdLine(%5, %7, %c1920_i32_0, %c30_i16, %c255_i16_1, %c0_i8) : (memref<1920xui8>, memref<1920xui8>, i32, i16, i16, i8) -> ()
        aie.objectfifo.release @OF_3to3(Consume, 1)
        aie.objectfifo.release @OF_3to5(Produce, 1)
      }
      aie.end
    } {link_with = "threshold.cc.o"}
    %core_0_4 = aie.core(%tile_0_4) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %0 = aie.objectfifo.acquire @OF_2to34(Consume, 1) : !aie.objectfifosubview<memref<1920xui8>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<1920xui8>> -> memref<1920xui8>
        %2 = aie.objectfifo.acquire @OF_4to4(Produce, 1) : !aie.objectfifosubview<memref<1920xui8>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<1920xui8>> -> memref<1920xui8>
        %c1920_i32 = arith.constant 1920 : i32
        %c160_i16 = arith.constant 160 : i16
        %c255_i16 = arith.constant 255 : i16
        %c4_i8 = arith.constant 4 : i8
        func.call @thresholdLine(%1, %3, %c1920_i32, %c160_i16, %c255_i16, %c4_i8) : (memref<1920xui8>, memref<1920xui8>, i32, i16, i16, i8) -> ()
        aie.objectfifo.release @OF_2to34(Consume, 1)
        aie.objectfifo.release @OF_4to4(Produce, 1)
        %4 = aie.objectfifo.acquire @OF_4to4(Consume, 1) : !aie.objectfifosubview<memref<1920xui8>>
        %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<1920xui8>> -> memref<1920xui8>
        %6 = aie.objectfifo.acquire @OF_4to5(Produce, 1) : !aie.objectfifosubview<memref<1920xui8>>
        %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<1920xui8>> -> memref<1920xui8>
        %c1920_i32_0 = arith.constant 1920 : i32
        %c90_i16 = arith.constant 90 : i16
        %c255_i16_1 = arith.constant 255 : i16
        %c0_i8 = arith.constant 0 : i8
        func.call @thresholdLine(%5, %7, %c1920_i32_0, %c90_i16, %c255_i16_1, %c0_i8) : (memref<1920xui8>, memref<1920xui8>, i32, i16, i16, i8) -> ()
        aie.objectfifo.release @OF_4to4(Consume, 1)
        aie.objectfifo.release @OF_4to5(Produce, 1)
      }
      aie.end
    } {link_with = "threshold.cc.o"}
    %core_0_5 = aie.core(%tile_0_5) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %0 = aie.objectfifo.acquire @OF_3to5(Consume, 1) : !aie.objectfifosubview<memref<1920xui8>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<1920xui8>> -> memref<1920xui8>
        %2 = aie.objectfifo.acquire @OF_4to5(Consume, 1) : !aie.objectfifosubview<memref<1920xui8>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<1920xui8>> -> memref<1920xui8>
        %4 = aie.objectfifo.acquire @OF_5to5a(Produce, 1) : !aie.objectfifosubview<memref<1920xui8>>
        %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<1920xui8>> -> memref<1920xui8>
        %c1920_i32 = arith.constant 1920 : i32
        func.call @bitwiseORLine(%1, %3, %5, %c1920_i32) : (memref<1920xui8>, memref<1920xui8>, memref<1920xui8>, i32) -> ()
        aie.objectfifo.release @OF_3to5(Consume, 1)
        aie.objectfifo.release @OF_4to5(Consume, 1)
        aie.objectfifo.release @OF_5to5a(Produce, 1)
        %6 = aie.objectfifo.acquire @OF_5to5a(Consume, 1) : !aie.objectfifosubview<memref<1920xui8>>
        %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<1920xui8>> -> memref<1920xui8>
        %8 = aie.objectfifo.acquire @OF_5to5b(Produce, 1) : !aie.objectfifosubview<memref<7680xui8>>
        %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<7680xui8>> -> memref<7680xui8>
        %c1920_i32_0 = arith.constant 1920 : i32
        func.call @gray2rgbaLine(%7, %9, %c1920_i32_0) : (memref<1920xui8>, memref<7680xui8>, i32) -> ()
        aie.objectfifo.release @OF_5to5a(Consume, 1)
        aie.objectfifo.release @OF_5to5b(Produce, 1)
        %10 = aie.objectfifo.acquire @OF_5to5b(Consume, 1) : !aie.objectfifosubview<memref<7680xui8>>
        %11 = aie.objectfifo.subview.access %10[0] : !aie.objectfifosubview<memref<7680xui8>> -> memref<7680xui8>
        %12 = aie.objectfifo.acquire @inOF_L2L1(Consume, 1) : !aie.objectfifosubview<memref<7680xui8>>
        %13 = aie.objectfifo.subview.access %12[0] : !aie.objectfifosubview<memref<7680xui8>> -> memref<7680xui8>
        %14 = aie.objectfifo.acquire @outOF_L1L2(Produce, 1) : !aie.objectfifosubview<memref<7680xui8>>
        %15 = aie.objectfifo.subview.access %14[0] : !aie.objectfifosubview<memref<7680xui8>> -> memref<7680xui8>
        %c7680_i32 = arith.constant 7680 : i32
        func.call @bitwiseANDLine(%11, %13, %15, %c7680_i32) : (memref<7680xui8>, memref<7680xui8>, memref<7680xui8>, i32) -> ()
        aie.objectfifo.release @OF_5to5b(Consume, 1)
        aie.objectfifo.release @inOF_L2L1(Consume, 1)
        aie.objectfifo.release @outOF_L1L2(Produce, 1)
      }
      aie.end
    } {link_with = "combined_bitwiseOR_gray2rgba_bitwiseAND.a"}
    aiex.runtime_sequence(%arg0: memref<8294400xi8>, %arg1: memref<16x16xi32>, %arg2: memref<8294400xi8>) {
      %0 = aiex.dma_configure_task_for @inOF_L3L2 {
        aie.dma_bd(%arg0 : memref<8294400xi8>, 0, 8294400, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 8294400, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%0)
      %1 = aiex.dma_configure_task_for @outOF_L2L3 {
        aie.dma_bd(%arg2 : memref<8294400xi8>, 0, 8294400, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 8294400, stride = 1>])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1)
      aiex.dma_await_task(%1)
      aiex.dma_free_task(%0)
    }
  }
}

