module {
  aie.device(npu1_1col) {
    %tile_0_2 = aie.tile(0, 2)
    %tile_0_3 = aie.tile(0, 3)
    %tile_0_4 = aie.tile(0, 4)
    %tile_0_5 = aie.tile(0, 5)
    %tile_0_1 = aie.tile(0, 1)
    %tile_0_0 = aie.tile(0, 0)
    aie.objectfifo @inOOB_L2L1_0(%tile_0_1, {%tile_0_2}, 2 : i32) : !aie.objectfifo<memref<1920xui8>> 
    aie.objectfifo @inOOB_L3L2(%tile_0_0, {%tile_0_1}, 2 : i32) : !aie.objectfifo<memref<7680xui8>> 
    aie.objectfifo @inOOB_L2L1_1(%tile_0_1, {%tile_0_3}, 2 : i32) : !aie.objectfifo<memref<1920xui8>> 
    aie.objectfifo @inOOB_L2L1_2(%tile_0_1, {%tile_0_4}, 2 : i32) : !aie.objectfifo<memref<1920xui8>> 
    aie.objectfifo @inOOB_L2L1_3(%tile_0_1, {%tile_0_5}, 2 : i32) : !aie.objectfifo<memref<1920xui8>> 
    aie.objectfifo.link [@inOOB_L3L2] -> [@inOOB_L2L1_0, @inOOB_L2L1_1, @inOOB_L2L1_2, @inOOB_L2L1_3]([] [0, 1920, 3840, 5760])
    aie.objectfifo @outOOB_L1L2_2(%tile_0_4, {%tile_0_1}, 2 : i32) : !aie.objectfifo<memref<1920xui8>> 
    aie.objectfifo @outOOB_L1L2_0(%tile_0_2, {%tile_0_1}, 2 : i32) : !aie.objectfifo<memref<1920xui8>> 
    aie.objectfifo @outOOB_L1L2_1(%tile_0_3, {%tile_0_1}, 2 : i32) : !aie.objectfifo<memref<1920xui8>> 
    aie.objectfifo @outOOB_L1L2_3(%tile_0_5, {%tile_0_1}, 2 : i32) : !aie.objectfifo<memref<1920xui8>> 
    aie.objectfifo @outOOB_L2L3(%tile_0_1, {%tile_0_0}, 2 : i32) : !aie.objectfifo<memref<7680xui8>> 
    aie.objectfifo.link [@outOOB_L1L2_0, @outOOB_L1L2_1, @outOOB_L1L2_2, @outOOB_L1L2_3] -> [@outOOB_L2L3]([0, 1920, 3840, 5760] [])
    %rtpComputeTile2 = aie.buffer(%tile_0_2) {sym_name = "rtpComputeTile2"} : memref<16xi32> 
    func.func private @thresholdLine(memref<1920xui8>, memref<1920xui8>, i32, i16, i16, i8)
    %rtpComputeTile3 = aie.buffer(%tile_0_3) {sym_name = "rtpComputeTile3"} : memref<16xi32> 
    %rtpComputeTile4 = aie.buffer(%tile_0_4) {sym_name = "rtpComputeTile4"} : memref<16xi32> 
    %rtpComputeTile5 = aie.buffer(%tile_0_5) {sym_name = "rtpComputeTile5"} : memref<16xi32> 
    %core_0_2 = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %0 = aie.objectfifo.acquire @inOOB_L2L1_0(Consume, 1) : !aie.objectfifosubview<memref<1920xui8>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<1920xui8>> -> memref<1920xui8>
        %2 = aie.objectfifo.acquire @outOOB_L1L2_0(Produce, 1) : !aie.objectfifosubview<memref<1920xui8>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<1920xui8>> -> memref<1920xui8>
        %c0_0 = arith.constant 0 : index
        %4 = memref.load %rtpComputeTile2[%c0_0] : memref<16xi32>
        %5 = arith.trunci %4 : i32 to i16
        %c1_1 = arith.constant 1 : index
        %6 = memref.load %rtpComputeTile2[%c1_1] : memref<16xi32>
        %7 = arith.trunci %6 : i32 to i16
        %c2 = arith.constant 2 : index
        %8 = memref.load %rtpComputeTile2[%c2] : memref<16xi32>
        %9 = arith.trunci %8 : i32 to i8
        %c1920_i32 = arith.constant 1920 : i32
        func.call @thresholdLine(%1, %3, %c1920_i32, %5, %7, %9) : (memref<1920xui8>, memref<1920xui8>, i32, i16, i16, i8) -> ()
        aie.objectfifo.release @inOOB_L2L1_0(Consume, 1)
        aie.objectfifo.release @outOOB_L1L2_0(Produce, 1)
      }
      aie.end
    } {link_with = "threshold.cc.o"}
    %core_0_3 = aie.core(%tile_0_3) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %0 = aie.objectfifo.acquire @inOOB_L2L1_1(Consume, 1) : !aie.objectfifosubview<memref<1920xui8>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<1920xui8>> -> memref<1920xui8>
        %2 = aie.objectfifo.acquire @outOOB_L1L2_1(Produce, 1) : !aie.objectfifosubview<memref<1920xui8>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<1920xui8>> -> memref<1920xui8>
        %c0_0 = arith.constant 0 : index
        %4 = memref.load %rtpComputeTile3[%c0_0] : memref<16xi32>
        %5 = arith.trunci %4 : i32 to i16
        %c1_1 = arith.constant 1 : index
        %6 = memref.load %rtpComputeTile3[%c1_1] : memref<16xi32>
        %7 = arith.trunci %6 : i32 to i16
        %c2 = arith.constant 2 : index
        %8 = memref.load %rtpComputeTile3[%c2] : memref<16xi32>
        %9 = arith.trunci %8 : i32 to i8
        %c1920_i32 = arith.constant 1920 : i32
        func.call @thresholdLine(%1, %3, %c1920_i32, %5, %7, %9) : (memref<1920xui8>, memref<1920xui8>, i32, i16, i16, i8) -> ()
        aie.objectfifo.release @inOOB_L2L1_1(Consume, 1)
        aie.objectfifo.release @outOOB_L1L2_1(Produce, 1)
      }
      aie.end
    } {link_with = "threshold.cc.o"}
    %core_0_4 = aie.core(%tile_0_4) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %0 = aie.objectfifo.acquire @inOOB_L2L1_2(Consume, 1) : !aie.objectfifosubview<memref<1920xui8>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<1920xui8>> -> memref<1920xui8>
        %2 = aie.objectfifo.acquire @outOOB_L1L2_2(Produce, 1) : !aie.objectfifosubview<memref<1920xui8>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<1920xui8>> -> memref<1920xui8>
        %c0_0 = arith.constant 0 : index
        %4 = memref.load %rtpComputeTile4[%c0_0] : memref<16xi32>
        %5 = arith.trunci %4 : i32 to i16
        %c1_1 = arith.constant 1 : index
        %6 = memref.load %rtpComputeTile4[%c1_1] : memref<16xi32>
        %7 = arith.trunci %6 : i32 to i16
        %c2 = arith.constant 2 : index
        %8 = memref.load %rtpComputeTile4[%c2] : memref<16xi32>
        %9 = arith.trunci %8 : i32 to i8
        %c1920_i32 = arith.constant 1920 : i32
        func.call @thresholdLine(%1, %3, %c1920_i32, %5, %7, %9) : (memref<1920xui8>, memref<1920xui8>, i32, i16, i16, i8) -> ()
        aie.objectfifo.release @inOOB_L2L1_2(Consume, 1)
        aie.objectfifo.release @outOOB_L1L2_2(Produce, 1)
      }
      aie.end
    } {link_with = "threshold.cc.o"}
    %core_0_5 = aie.core(%tile_0_5) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %0 = aie.objectfifo.acquire @inOOB_L2L1_3(Consume, 1) : !aie.objectfifosubview<memref<1920xui8>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<1920xui8>> -> memref<1920xui8>
        %2 = aie.objectfifo.acquire @outOOB_L1L2_3(Produce, 1) : !aie.objectfifosubview<memref<1920xui8>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<1920xui8>> -> memref<1920xui8>
        %c0_0 = arith.constant 0 : index
        %4 = memref.load %rtpComputeTile5[%c0_0] : memref<16xi32>
        %5 = arith.trunci %4 : i32 to i16
        %c1_1 = arith.constant 1 : index
        %6 = memref.load %rtpComputeTile5[%c1_1] : memref<16xi32>
        %7 = arith.trunci %6 : i32 to i16
        %c2 = arith.constant 2 : index
        %8 = memref.load %rtpComputeTile5[%c2] : memref<16xi32>
        %9 = arith.trunci %8 : i32 to i8
        %c1920_i32 = arith.constant 1920 : i32
        func.call @thresholdLine(%1, %3, %c1920_i32, %5, %7, %9) : (memref<1920xui8>, memref<1920xui8>, i32, i16, i16, i8) -> ()
        aie.objectfifo.release @inOOB_L2L1_3(Consume, 1)
        aie.objectfifo.release @outOOB_L1L2_3(Produce, 1)
      }
      aie.end
    } {link_with = "threshold.cc.o"}
    aiex.runtime_sequence(%arg0: memref<2073600xi8>, %arg1: memref<32xi32>, %arg2: memref<2073600xi8>) {
      aiex.npu.rtp_write(@rtpComputeTile2, 0, 50)
      aiex.npu.rtp_write(@rtpComputeTile2, 1, 255)
      aiex.npu.rtp_write(@rtpComputeTile2, 2, 0)
      aiex.npu.rtp_write(@rtpComputeTile3, 0, 50)
      aiex.npu.rtp_write(@rtpComputeTile3, 1, 255)
      aiex.npu.rtp_write(@rtpComputeTile3, 2, 0)
      aiex.npu.rtp_write(@rtpComputeTile4, 0, 50)
      aiex.npu.rtp_write(@rtpComputeTile4, 1, 255)
      aiex.npu.rtp_write(@rtpComputeTile4, 2, 0)
      aiex.npu.rtp_write(@rtpComputeTile5, 0, 50)
      aiex.npu.rtp_write(@rtpComputeTile5, 1, 255)
      aiex.npu.rtp_write(@rtpComputeTile5, 2, 0)
      %0 = aiex.dma_configure_task_for @inOOB_L3L2 {
        aie.dma_bd(%arg0 : memref<2073600xi8>, 0, 2073600, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 2073600, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%0)
      %1 = aiex.dma_configure_task_for @outOOB_L2L3 {
        aie.dma_bd(%arg2 : memref<2073600xi8>, 0, 2073600, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 2073600, stride = 1>])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%1)
      aiex.dma_await_task(%1)
      aiex.dma_free_task(%0)
    }
  }
}

