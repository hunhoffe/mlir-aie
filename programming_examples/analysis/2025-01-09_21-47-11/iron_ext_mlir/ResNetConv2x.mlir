module {
  aie.device(npu1_3col) {
    %tile_0_2 = aie.tile(0, 2)
    %tile_0_3 = aie.tile(0, 3)
    %tile_0_4 = aie.tile(0, 4)
    %tile_0_5 = aie.tile(0, 5)
    %tile_1_5 = aie.tile(1, 5)
    %tile_1_4 = aie.tile(1, 4)
    %tile_1_3 = aie.tile(1, 3)
    %tile_1_2 = aie.tile(1, 2)
    %tile_2_2 = aie.tile(2, 2)
    %tile_2_3 = aie.tile(2, 3)
    %tile_2_4 = aie.tile(2, 4)
    %tile_2_5 = aie.tile(2, 5)
    %tile_2_1 = aie.tile(2, 1)
    %tile_0_1 = aie.tile(0, 1)
    %tile_1_1 = aie.tile(1, 1)
    %tile_1_0 = aie.tile(1, 0)
    %tile_0_0 = aie.tile(0, 0)
    %tile_2_0 = aie.tile(2, 0)
    aie.objectfifo @act1_13_22_21(%tile_1_3, {%tile_2_1, %tile_2_2}, [2 : i32, 4 : i32, 2 : i32]) : !aie.objectfifo<memref<32x1x256xui8>> 
    aie.objectfifo @skip_2(%tile_2_1, {%tile_2_4}, 2 : i32) : !aie.objectfifo<memref<32x1x256xui8>> 
    aie.objectfifo.link [@act1_13_22_21] -> [@skip_2]([] [0])
    aie.objectfifo @skip_1(%tile_0_1, {%tile_1_3}, 2 : i32) : !aie.objectfifo<memref<32x1x256xui8>> 
    aie.objectfifo @act1_04_15_11(%tile_0_4, {%tile_0_1, %tile_1_5}, [2 : i32, 4 : i32, 2 : i32]) : !aie.objectfifo<memref<32x1x256xui8>> 
    aie.objectfifo.link [@act1_04_15_11] -> [@skip_1]([] [0])
    aie.objectfifo @wts_buf_11(%tile_1_1, {%tile_1_4, %tile_1_2}, 1 : i32) : !aie.objectfifo<memref<36864xi8>> 
    aie.objectfifo @wts_1_L3L2(%tile_1_0, {%tile_1_1}, 1 : i32) : !aie.objectfifo<memref<69632xi8>> 
    aie.objectfifo @wts_buf_10(%tile_1_1, {%tile_1_5}, 1 : i32) : !aie.objectfifo<memref<16384xi8>> 
    aie.objectfifo @wts_buf_12(%tile_1_1, {%tile_1_3}, 1 : i32) : !aie.objectfifo<memref<16384xi8>> 
    aie.objectfifo.link [@wts_1_L3L2] -> [@wts_buf_10, @wts_buf_11, @wts_buf_12]([] [0, 16384, 53248])
    aie.objectfifo @act2_15_12_14(%tile_1_5, {%tile_1_4, %tile_1_2}, 4 : i32) : !aie.objectfifo<memref<32x1x64xui8>> 
    aie.objectfifo @act3_12_13(%tile_1_2, {%tile_1_3}, 2 : i32) : !aie.objectfifo<memref<32x1x32xui8>> 
    aie.objectfifo @wts_buf_20(%tile_2_1, {%tile_2_2}, 1 : i32) : !aie.objectfifo<memref<16384xi8>> 
    aie.objectfifo @wts_2_L3L2(%tile_2_0, {%tile_2_1}, 1 : i32) : !aie.objectfifo<memref<69632xi8>> 
    aie.objectfifo @wts_buf_21(%tile_2_1, {%tile_2_3, %tile_2_5}, 1 : i32) : !aie.objectfifo<memref<36864xi8>> 
    aie.objectfifo @wts_buf_22(%tile_2_1, {%tile_2_4}, 1 : i32) : !aie.objectfifo<memref<16384xi8>> 
    aie.objectfifo.link [@wts_2_L3L2] -> [@wts_buf_20, @wts_buf_21, @wts_buf_22]([] [0, 16384, 53248])
    aie.objectfifo @act2_22_23_25(%tile_2_2, {%tile_2_3, %tile_2_5}, 4 : i32) : !aie.objectfifo<memref<32x1x64xui8>> 
    aie.objectfifo @act3_23_24(%tile_2_3, {%tile_2_4}, 2 : i32) : !aie.objectfifo<memref<32x1x32xui8>> 
    aie.objectfifo @act3_25_24(%tile_2_5, {%tile_2_4}, 2 : i32) : !aie.objectfifo<memref<32x1x32xui8>> 
    aie.objectfifo @outOFL2L3(%tile_2_4, {%tile_1_0}, 2 : i32) : !aie.objectfifo<memref<32x1x256xui8>> 
    aie.objectfifo @act1_00_02_01(%tile_0_0, {%tile_0_1, %tile_0_2}, [2 : i32, 4 : i32, 2 : i32]) : !aie.objectfifo<memref<32x1x64xi8>> 
    aie.objectfifo @skip_0(%tile_0_1, {%tile_0_4}, 2 : i32) : !aie.objectfifo<memref<32x1x64xi8>> 
    aie.objectfifo.link [@act1_00_02_01] -> [@skip_0]([] [0])
    aie.objectfifo @wts_buf_00(%tile_0_1, {%tile_0_2}, 1 : i32) : !aie.objectfifo<memref<4096xi8>> 
    aie.objectfifo @wts_0_L3L2(%tile_0_0, {%tile_0_1}, 1 : i32) : !aie.objectfifo<memref<73728xi8>> 
    aie.objectfifo @wts_buf_01(%tile_0_1, {%tile_0_3, %tile_0_5}, 1 : i32) : !aie.objectfifo<memref<36864xi8>> 
    aie.objectfifo @wts_buf_02(%tile_0_1, {%tile_0_4}, 1 : i32) : !aie.objectfifo<memref<32768xi8>> 
    aie.objectfifo.link [@wts_0_L3L2] -> [@wts_buf_00, @wts_buf_01, @wts_buf_02]([] [0, 4096, 40960])
    aie.objectfifo @act2_02_03_05(%tile_0_2, {%tile_0_3, %tile_0_5}, 4 : i32) : !aie.objectfifo<memref<32x1x64xui8>> 
    aie.objectfifo @act3_03_04(%tile_0_3, {%tile_0_4}, 2 : i32) : !aie.objectfifo<memref<32x1x32xui8>> 
    aie.objectfifo @act3_05_04(%tile_0_5, {%tile_0_4}, 2 : i32) : !aie.objectfifo<memref<32x1x32xui8>> 
    aie.objectfifo @act3_14_13(%tile_1_4, {%tile_1_3}, 2 : i32) : !aie.objectfifo<memref<32x1x32xui8>> 
    func.func private @conv2dk1_i8(memref<32x1x64xi8>, memref<4096xi8>, memref<32x1x64xui8>, i32, i32, i32, i32)
    %rtpComputeTile02 = aie.buffer(%tile_0_2) {sym_name = "rtpComputeTile02"} : memref<16xi32> 
    func.func private @conv2dk3_ui8(memref<32x1x64xui8>, memref<32x1x64xui8>, memref<32x1x64xui8>, memref<36864xi8>, memref<32x1x32xui8>, i32, i32, i32, i32, i32, i32, i32, i32)
    func.func private @conv2dk1_skip_init_i8(memref<32x1x32xui8>, memref<32x1x32xui8>, memref<32768xi8>, memref<32x1x256xui8>, memref<32x1x64xi8>, i32, i32, i32, i32, i32, i32, i32)
    %rtpComputeTile05 = aie.buffer(%tile_0_4) {sym_name = "rtpComputeTile05"} : memref<16xi32> 
    func.func private @conv2dk1_ui8(memref<32x1x256xui8>, memref<16384xi8>, memref<32x1x64xui8>, i32, i32, i32, i32)
    %rtpComputeTile15 = aie.buffer(%tile_1_5) {sym_name = "rtpComputeTile15"} : memref<16xi32> 
    func.func private @conv2dk1_skip_ui8(memref<32x1x32xui8>, memref<32x1x32xui8>, memref<16384xi8>, memref<32x1x256xui8>, memref<32x1x256xui8>, i32, i32, i32, i32, i32)
    %rtpComputeTile13 = aie.buffer(%tile_1_3) {sym_name = "rtpComputeTile13"} : memref<16xi32> 
    %rtpComputeTile22 = aie.buffer(%tile_2_2) {sym_name = "rtpComputeTile22"} : memref<16xi32> 
    %rtpComputeTile24 = aie.buffer(%tile_2_4) {sym_name = "rtpComputeTile24"} : memref<16xi32> 
    %core_0_2 = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %0 = aie.objectfifo.acquire @wts_buf_00(Consume, 1) : !aie.objectfifosubview<memref<4096xi8>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<4096xi8>> -> memref<4096xi8>
        %c0_0 = arith.constant 0 : index
        %2 = memref.load %rtpComputeTile02[%c0_0] : memref<16xi32>
        %c0_1 = arith.constant 0 : index
        %c32 = arith.constant 32 : index
        %c1_2 = arith.constant 1 : index
        scf.for %arg1 = %c0_1 to %c32 step %c1_2 {
          %3 = aie.objectfifo.acquire @act1_00_02_01(Consume, 1) : !aie.objectfifosubview<memref<32x1x64xi8>>
          %4 = aie.objectfifo.subview.access %3[0] : !aie.objectfifosubview<memref<32x1x64xi8>> -> memref<32x1x64xi8>
          %5 = aie.objectfifo.acquire @act2_02_03_05(Produce, 1) : !aie.objectfifosubview<memref<32x1x64xui8>>
          %6 = aie.objectfifo.subview.access %5[0] : !aie.objectfifosubview<memref<32x1x64xui8>> -> memref<32x1x64xui8>
          %c32_i32 = arith.constant 32 : i32
          %c64_i32 = arith.constant 64 : i32
          %c64_i32_3 = arith.constant 64 : i32
          func.call @conv2dk1_i8(%4, %1, %6, %c32_i32, %c64_i32, %c64_i32_3, %2) : (memref<32x1x64xi8>, memref<4096xi8>, memref<32x1x64xui8>, i32, i32, i32, i32) -> ()
          aie.objectfifo.release @act1_00_02_01(Consume, 1)
          aie.objectfifo.release @act2_02_03_05(Produce, 1)
        }
        aie.objectfifo.release @wts_buf_00(Consume, 1)
      }
      aie.end
    } {link_with = "conv2dk1_i8.o"}
    %core_0_3 = aie.core(%tile_0_3) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %0 = aie.objectfifo.acquire @wts_buf_01(Consume, 1) : !aie.objectfifosubview<memref<36864xi8>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<36864xi8>> -> memref<36864xi8>
        %2 = aie.objectfifo.acquire @act2_02_03_05(Consume, 2) : !aie.objectfifosubview<memref<32x1x64xui8>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<32x1x64xui8>> -> memref<32x1x64xui8>
        %4 = aie.objectfifo.subview.access %2[1] : !aie.objectfifosubview<memref<32x1x64xui8>> -> memref<32x1x64xui8>
        %5 = aie.objectfifo.acquire @act3_03_04(Produce, 1) : !aie.objectfifosubview<memref<32x1x32xui8>>
        %6 = aie.objectfifo.subview.access %5[0] : !aie.objectfifosubview<memref<32x1x32xui8>> -> memref<32x1x32xui8>
        %c32_i32 = arith.constant 32 : i32
        %c64_i32 = arith.constant 64 : i32
        %c32_i32_0 = arith.constant 32 : i32
        %c3_i32 = arith.constant 3 : i32
        %c3_i32_1 = arith.constant 3 : i32
        %c0_i32 = arith.constant 0 : i32
        %c1_i32 = arith.constant 1 : i32
        %c32_i32_2 = arith.constant 32 : i32
        func.call @conv2dk3_ui8(%3, %3, %4, %1, %6, %c32_i32, %c64_i32, %c32_i32_0, %c3_i32, %c3_i32_1, %c0_i32, %c1_i32, %c32_i32_2) : (memref<32x1x64xui8>, memref<32x1x64xui8>, memref<32x1x64xui8>, memref<36864xi8>, memref<32x1x32xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
        aie.objectfifo.release @act3_03_04(Produce, 1)
        %c0_3 = arith.constant 0 : index
        %c30 = arith.constant 30 : index
        %c1_4 = arith.constant 1 : index
        scf.for %arg1 = %c0_3 to %c30 step %c1_4 {
          %12 = aie.objectfifo.acquire @act2_02_03_05(Consume, 3) : !aie.objectfifosubview<memref<32x1x64xui8>>
          %13 = aie.objectfifo.subview.access %12[0] : !aie.objectfifosubview<memref<32x1x64xui8>> -> memref<32x1x64xui8>
          %14 = aie.objectfifo.subview.access %12[1] : !aie.objectfifosubview<memref<32x1x64xui8>> -> memref<32x1x64xui8>
          %15 = aie.objectfifo.subview.access %12[2] : !aie.objectfifosubview<memref<32x1x64xui8>> -> memref<32x1x64xui8>
          %16 = aie.objectfifo.acquire @act3_03_04(Produce, 1) : !aie.objectfifosubview<memref<32x1x32xui8>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<32x1x32xui8>> -> memref<32x1x32xui8>
          %c32_i32_12 = arith.constant 32 : i32
          %c64_i32_13 = arith.constant 64 : i32
          %c32_i32_14 = arith.constant 32 : i32
          %c3_i32_15 = arith.constant 3 : i32
          %c3_i32_16 = arith.constant 3 : i32
          %c1_i32_17 = arith.constant 1 : i32
          %c1_i32_18 = arith.constant 1 : i32
          %c32_i32_19 = arith.constant 32 : i32
          func.call @conv2dk3_ui8(%13, %14, %15, %1, %17, %c32_i32_12, %c64_i32_13, %c32_i32_14, %c3_i32_15, %c3_i32_16, %c1_i32_17, %c1_i32_18, %c32_i32_19) : (memref<32x1x64xui8>, memref<32x1x64xui8>, memref<32x1x64xui8>, memref<36864xi8>, memref<32x1x32xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
          aie.objectfifo.release @act2_02_03_05(Consume, 1)
          aie.objectfifo.release @act3_03_04(Produce, 1)
        }
        %7 = aie.objectfifo.acquire @act2_02_03_05(Consume, 2) : !aie.objectfifosubview<memref<32x1x64xui8>>
        %8 = aie.objectfifo.subview.access %7[0] : !aie.objectfifosubview<memref<32x1x64xui8>> -> memref<32x1x64xui8>
        %9 = aie.objectfifo.subview.access %7[1] : !aie.objectfifosubview<memref<32x1x64xui8>> -> memref<32x1x64xui8>
        %10 = aie.objectfifo.acquire @act3_03_04(Produce, 1) : !aie.objectfifosubview<memref<32x1x32xui8>>
        %11 = aie.objectfifo.subview.access %10[0] : !aie.objectfifosubview<memref<32x1x32xui8>> -> memref<32x1x32xui8>
        %c32_i32_5 = arith.constant 32 : i32
        %c64_i32_6 = arith.constant 64 : i32
        %c32_i32_7 = arith.constant 32 : i32
        %c3_i32_8 = arith.constant 3 : i32
        %c3_i32_9 = arith.constant 3 : i32
        %c2_i32 = arith.constant 2 : i32
        %c1_i32_10 = arith.constant 1 : i32
        %c32_i32_11 = arith.constant 32 : i32
        func.call @conv2dk3_ui8(%8, %9, %9, %1, %11, %c32_i32_5, %c64_i32_6, %c32_i32_7, %c3_i32_8, %c3_i32_9, %c2_i32, %c1_i32_10, %c32_i32_11) : (memref<32x1x64xui8>, memref<32x1x64xui8>, memref<32x1x64xui8>, memref<36864xi8>, memref<32x1x32xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
        aie.objectfifo.release @act2_02_03_05(Consume, 2)
        aie.objectfifo.release @act3_03_04(Produce, 1)
        aie.objectfifo.release @wts_buf_01(Consume, 1)
      }
      aie.end
    } {link_with = "conv2dk3.o"}
    %core_0_4 = aie.core(%tile_0_4) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %0 = aie.objectfifo.acquire @wts_buf_02(Consume, 1) : !aie.objectfifosubview<memref<32768xi8>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<32768xi8>> -> memref<32768xi8>
        %c0_0 = arith.constant 0 : index
        %2 = memref.load %rtpComputeTile05[%c0_0] : memref<16xi32>
        %c1_1 = arith.constant 1 : index
        %3 = memref.load %rtpComputeTile05[%c1_1] : memref<16xi32>
        %c2 = arith.constant 2 : index
        %4 = memref.load %rtpComputeTile05[%c2] : memref<16xi32>
        %c0_2 = arith.constant 0 : index
        %c32 = arith.constant 32 : index
        %c1_3 = arith.constant 1 : index
        scf.for %arg1 = %c0_2 to %c32 step %c1_3 {
          %5 = aie.objectfifo.acquire @act3_03_04(Consume, 1) : !aie.objectfifosubview<memref<32x1x32xui8>>
          %6 = aie.objectfifo.subview.access %5[0] : !aie.objectfifosubview<memref<32x1x32xui8>> -> memref<32x1x32xui8>
          %7 = aie.objectfifo.acquire @act3_05_04(Consume, 1) : !aie.objectfifosubview<memref<32x1x32xui8>>
          %8 = aie.objectfifo.subview.access %7[0] : !aie.objectfifosubview<memref<32x1x32xui8>> -> memref<32x1x32xui8>
          %9 = aie.objectfifo.acquire @act1_04_15_11(Produce, 1) : !aie.objectfifosubview<memref<32x1x256xui8>>
          %10 = aie.objectfifo.subview.access %9[0] : !aie.objectfifosubview<memref<32x1x256xui8>> -> memref<32x1x256xui8>
          %11 = aie.objectfifo.acquire @skip_0(Consume, 1) : !aie.objectfifosubview<memref<32x1x64xi8>>
          %12 = aie.objectfifo.subview.access %11[0] : !aie.objectfifosubview<memref<32x1x64xi8>> -> memref<32x1x64xi8>
          %c32_i32 = arith.constant 32 : i32
          %c64_i32 = arith.constant 64 : i32
          %c256_i32 = arith.constant 256 : i32
          %c64_i32_4 = arith.constant 64 : i32
          func.call @conv2dk1_skip_init_i8(%6, %8, %1, %10, %12, %c32_i32, %c64_i32, %c256_i32, %c64_i32_4, %2, %3, %4) : (memref<32x1x32xui8>, memref<32x1x32xui8>, memref<32768xi8>, memref<32x1x256xui8>, memref<32x1x64xi8>, i32, i32, i32, i32, i32, i32, i32) -> ()
          aie.objectfifo.release @act3_03_04(Consume, 1)
          aie.objectfifo.release @act3_05_04(Consume, 1)
          aie.objectfifo.release @act1_04_15_11(Produce, 1)
          aie.objectfifo.release @skip_0(Consume, 1)
        }
        aie.objectfifo.release @wts_buf_02(Consume, 1)
      }
      aie.end
    } {link_with = "conv2dk1_skip_init.o"}
    %core_0_5 = aie.core(%tile_0_5) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %0 = aie.objectfifo.acquire @wts_buf_01(Consume, 1) : !aie.objectfifosubview<memref<36864xi8>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<36864xi8>> -> memref<36864xi8>
        %2 = aie.objectfifo.acquire @act2_02_03_05(Consume, 2) : !aie.objectfifosubview<memref<32x1x64xui8>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<32x1x64xui8>> -> memref<32x1x64xui8>
        %4 = aie.objectfifo.subview.access %2[1] : !aie.objectfifosubview<memref<32x1x64xui8>> -> memref<32x1x64xui8>
        %5 = aie.objectfifo.acquire @act3_05_04(Produce, 1) : !aie.objectfifosubview<memref<32x1x32xui8>>
        %6 = aie.objectfifo.subview.access %5[0] : !aie.objectfifosubview<memref<32x1x32xui8>> -> memref<32x1x32xui8>
        %c32_i32 = arith.constant 32 : i32
        %c64_i32 = arith.constant 64 : i32
        %c32_i32_0 = arith.constant 32 : i32
        %c3_i32 = arith.constant 3 : i32
        %c3_i32_1 = arith.constant 3 : i32
        %c0_i32 = arith.constant 0 : i32
        %c1_i32 = arith.constant 1 : i32
        %c0_i32_2 = arith.constant 0 : i32
        func.call @conv2dk3_ui8(%3, %3, %4, %1, %6, %c32_i32, %c64_i32, %c32_i32_0, %c3_i32, %c3_i32_1, %c0_i32, %c1_i32, %c0_i32_2) : (memref<32x1x64xui8>, memref<32x1x64xui8>, memref<32x1x64xui8>, memref<36864xi8>, memref<32x1x32xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
        aie.objectfifo.release @act3_05_04(Produce, 1)
        %c0_3 = arith.constant 0 : index
        %c30 = arith.constant 30 : index
        %c1_4 = arith.constant 1 : index
        scf.for %arg1 = %c0_3 to %c30 step %c1_4 {
          %12 = aie.objectfifo.acquire @act2_02_03_05(Consume, 3) : !aie.objectfifosubview<memref<32x1x64xui8>>
          %13 = aie.objectfifo.subview.access %12[0] : !aie.objectfifosubview<memref<32x1x64xui8>> -> memref<32x1x64xui8>
          %14 = aie.objectfifo.subview.access %12[1] : !aie.objectfifosubview<memref<32x1x64xui8>> -> memref<32x1x64xui8>
          %15 = aie.objectfifo.subview.access %12[2] : !aie.objectfifosubview<memref<32x1x64xui8>> -> memref<32x1x64xui8>
          %16 = aie.objectfifo.acquire @act3_05_04(Produce, 1) : !aie.objectfifosubview<memref<32x1x32xui8>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<32x1x32xui8>> -> memref<32x1x32xui8>
          %c32_i32_12 = arith.constant 32 : i32
          %c64_i32_13 = arith.constant 64 : i32
          %c32_i32_14 = arith.constant 32 : i32
          %c3_i32_15 = arith.constant 3 : i32
          %c3_i32_16 = arith.constant 3 : i32
          %c1_i32_17 = arith.constant 1 : i32
          %c1_i32_18 = arith.constant 1 : i32
          %c0_i32_19 = arith.constant 0 : i32
          func.call @conv2dk3_ui8(%13, %14, %15, %1, %17, %c32_i32_12, %c64_i32_13, %c32_i32_14, %c3_i32_15, %c3_i32_16, %c1_i32_17, %c1_i32_18, %c0_i32_19) : (memref<32x1x64xui8>, memref<32x1x64xui8>, memref<32x1x64xui8>, memref<36864xi8>, memref<32x1x32xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
          aie.objectfifo.release @act2_02_03_05(Consume, 1)
          aie.objectfifo.release @act3_05_04(Produce, 1)
        }
        %7 = aie.objectfifo.acquire @act2_02_03_05(Consume, 2) : !aie.objectfifosubview<memref<32x1x64xui8>>
        %8 = aie.objectfifo.subview.access %7[0] : !aie.objectfifosubview<memref<32x1x64xui8>> -> memref<32x1x64xui8>
        %9 = aie.objectfifo.subview.access %7[1] : !aie.objectfifosubview<memref<32x1x64xui8>> -> memref<32x1x64xui8>
        %10 = aie.objectfifo.acquire @act3_05_04(Produce, 1) : !aie.objectfifosubview<memref<32x1x32xui8>>
        %11 = aie.objectfifo.subview.access %10[0] : !aie.objectfifosubview<memref<32x1x32xui8>> -> memref<32x1x32xui8>
        %c32_i32_5 = arith.constant 32 : i32
        %c64_i32_6 = arith.constant 64 : i32
        %c32_i32_7 = arith.constant 32 : i32
        %c3_i32_8 = arith.constant 3 : i32
        %c3_i32_9 = arith.constant 3 : i32
        %c2_i32 = arith.constant 2 : i32
        %c1_i32_10 = arith.constant 1 : i32
        %c0_i32_11 = arith.constant 0 : i32
        func.call @conv2dk3_ui8(%8, %9, %9, %1, %11, %c32_i32_5, %c64_i32_6, %c32_i32_7, %c3_i32_8, %c3_i32_9, %c2_i32, %c1_i32_10, %c0_i32_11) : (memref<32x1x64xui8>, memref<32x1x64xui8>, memref<32x1x64xui8>, memref<36864xi8>, memref<32x1x32xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
        aie.objectfifo.release @act2_02_03_05(Consume, 2)
        aie.objectfifo.release @act3_05_04(Produce, 1)
        aie.objectfifo.release @wts_buf_01(Consume, 1)
      }
      aie.end
    } {link_with = "conv2dk3.o"}
    %core_1_5 = aie.core(%tile_1_5) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %0 = aie.objectfifo.acquire @wts_buf_10(Consume, 1) : !aie.objectfifosubview<memref<16384xi8>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<16384xi8>> -> memref<16384xi8>
        %c0_0 = arith.constant 0 : index
        %2 = memref.load %rtpComputeTile15[%c0_0] : memref<16xi32>
        %c0_1 = arith.constant 0 : index
        %c32 = arith.constant 32 : index
        %c1_2 = arith.constant 1 : index
        scf.for %arg1 = %c0_1 to %c32 step %c1_2 {
          %3 = aie.objectfifo.acquire @act1_04_15_11(Consume, 1) : !aie.objectfifosubview<memref<32x1x256xui8>>
          %4 = aie.objectfifo.subview.access %3[0] : !aie.objectfifosubview<memref<32x1x256xui8>> -> memref<32x1x256xui8>
          %5 = aie.objectfifo.acquire @act2_15_12_14(Produce, 1) : !aie.objectfifosubview<memref<32x1x64xui8>>
          %6 = aie.objectfifo.subview.access %5[0] : !aie.objectfifosubview<memref<32x1x64xui8>> -> memref<32x1x64xui8>
          %c32_i32 = arith.constant 32 : i32
          %c256_i32 = arith.constant 256 : i32
          %c64_i32 = arith.constant 64 : i32
          func.call @conv2dk1_ui8(%4, %1, %6, %c32_i32, %c256_i32, %c64_i32, %2) : (memref<32x1x256xui8>, memref<16384xi8>, memref<32x1x64xui8>, i32, i32, i32, i32) -> ()
          aie.objectfifo.release @act1_04_15_11(Consume, 1)
          aie.objectfifo.release @act2_15_12_14(Produce, 1)
        }
        aie.objectfifo.release @wts_buf_10(Consume, 1)
      }
      aie.end
    } {link_with = "conv2dk1_ui8.o"}
    %core_1_4 = aie.core(%tile_1_4) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %0 = aie.objectfifo.acquire @wts_buf_11(Consume, 1) : !aie.objectfifosubview<memref<36864xi8>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<36864xi8>> -> memref<36864xi8>
        %2 = aie.objectfifo.acquire @act2_15_12_14(Consume, 2) : !aie.objectfifosubview<memref<32x1x64xui8>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<32x1x64xui8>> -> memref<32x1x64xui8>
        %4 = aie.objectfifo.subview.access %2[1] : !aie.objectfifosubview<memref<32x1x64xui8>> -> memref<32x1x64xui8>
        %5 = aie.objectfifo.acquire @act3_14_13(Produce, 1) : !aie.objectfifosubview<memref<32x1x32xui8>>
        %6 = aie.objectfifo.subview.access %5[0] : !aie.objectfifosubview<memref<32x1x32xui8>> -> memref<32x1x32xui8>
        %c32_i32 = arith.constant 32 : i32
        %c64_i32 = arith.constant 64 : i32
        %c32_i32_0 = arith.constant 32 : i32
        %c3_i32 = arith.constant 3 : i32
        %c3_i32_1 = arith.constant 3 : i32
        %c0_i32 = arith.constant 0 : i32
        %c1_i32 = arith.constant 1 : i32
        %c32_i32_2 = arith.constant 32 : i32
        func.call @conv2dk3_ui8(%3, %3, %4, %1, %6, %c32_i32, %c64_i32, %c32_i32_0, %c3_i32, %c3_i32_1, %c0_i32, %c1_i32, %c32_i32_2) : (memref<32x1x64xui8>, memref<32x1x64xui8>, memref<32x1x64xui8>, memref<36864xi8>, memref<32x1x32xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
        aie.objectfifo.release @act3_14_13(Produce, 1)
        %c0_3 = arith.constant 0 : index
        %c30 = arith.constant 30 : index
        %c1_4 = arith.constant 1 : index
        scf.for %arg1 = %c0_3 to %c30 step %c1_4 {
          %12 = aie.objectfifo.acquire @act2_15_12_14(Consume, 3) : !aie.objectfifosubview<memref<32x1x64xui8>>
          %13 = aie.objectfifo.subview.access %12[0] : !aie.objectfifosubview<memref<32x1x64xui8>> -> memref<32x1x64xui8>
          %14 = aie.objectfifo.subview.access %12[1] : !aie.objectfifosubview<memref<32x1x64xui8>> -> memref<32x1x64xui8>
          %15 = aie.objectfifo.subview.access %12[2] : !aie.objectfifosubview<memref<32x1x64xui8>> -> memref<32x1x64xui8>
          %16 = aie.objectfifo.acquire @act3_14_13(Produce, 1) : !aie.objectfifosubview<memref<32x1x32xui8>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<32x1x32xui8>> -> memref<32x1x32xui8>
          %c32_i32_12 = arith.constant 32 : i32
          %c64_i32_13 = arith.constant 64 : i32
          %c32_i32_14 = arith.constant 32 : i32
          %c3_i32_15 = arith.constant 3 : i32
          %c3_i32_16 = arith.constant 3 : i32
          %c1_i32_17 = arith.constant 1 : i32
          %c1_i32_18 = arith.constant 1 : i32
          %c32_i32_19 = arith.constant 32 : i32
          func.call @conv2dk3_ui8(%13, %14, %15, %1, %17, %c32_i32_12, %c64_i32_13, %c32_i32_14, %c3_i32_15, %c3_i32_16, %c1_i32_17, %c1_i32_18, %c32_i32_19) : (memref<32x1x64xui8>, memref<32x1x64xui8>, memref<32x1x64xui8>, memref<36864xi8>, memref<32x1x32xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
          aie.objectfifo.release @act2_15_12_14(Consume, 1)
          aie.objectfifo.release @act3_14_13(Produce, 1)
        }
        %7 = aie.objectfifo.acquire @act2_15_12_14(Consume, 2) : !aie.objectfifosubview<memref<32x1x64xui8>>
        %8 = aie.objectfifo.subview.access %7[0] : !aie.objectfifosubview<memref<32x1x64xui8>> -> memref<32x1x64xui8>
        %9 = aie.objectfifo.subview.access %7[1] : !aie.objectfifosubview<memref<32x1x64xui8>> -> memref<32x1x64xui8>
        %10 = aie.objectfifo.acquire @act3_14_13(Produce, 1) : !aie.objectfifosubview<memref<32x1x32xui8>>
        %11 = aie.objectfifo.subview.access %10[0] : !aie.objectfifosubview<memref<32x1x32xui8>> -> memref<32x1x32xui8>
        %c32_i32_5 = arith.constant 32 : i32
        %c64_i32_6 = arith.constant 64 : i32
        %c32_i32_7 = arith.constant 32 : i32
        %c3_i32_8 = arith.constant 3 : i32
        %c3_i32_9 = arith.constant 3 : i32
        %c2_i32 = arith.constant 2 : i32
        %c1_i32_10 = arith.constant 1 : i32
        %c32_i32_11 = arith.constant 32 : i32
        func.call @conv2dk3_ui8(%8, %9, %9, %1, %11, %c32_i32_5, %c64_i32_6, %c32_i32_7, %c3_i32_8, %c3_i32_9, %c2_i32, %c1_i32_10, %c32_i32_11) : (memref<32x1x64xui8>, memref<32x1x64xui8>, memref<32x1x64xui8>, memref<36864xi8>, memref<32x1x32xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
        aie.objectfifo.release @act2_15_12_14(Consume, 2)
        aie.objectfifo.release @act3_14_13(Produce, 1)
        aie.objectfifo.release @wts_buf_11(Consume, 1)
      }
      aie.end
    } {link_with = "conv2dk3.o"}
    %core_1_3 = aie.core(%tile_1_3) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %0 = aie.objectfifo.acquire @wts_buf_12(Consume, 1) : !aie.objectfifosubview<memref<16384xi8>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<16384xi8>> -> memref<16384xi8>
        %c0_0 = arith.constant 0 : index
        %2 = memref.load %rtpComputeTile13[%c0_0] : memref<16xi32>
        %c1_1 = arith.constant 1 : index
        %3 = memref.load %rtpComputeTile13[%c1_1] : memref<16xi32>
        %c0_2 = arith.constant 0 : index
        %c32 = arith.constant 32 : index
        %c1_3 = arith.constant 1 : index
        scf.for %arg1 = %c0_2 to %c32 step %c1_3 {
          %4 = aie.objectfifo.acquire @act3_14_13(Consume, 1) : !aie.objectfifosubview<memref<32x1x32xui8>>
          %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<32x1x32xui8>> -> memref<32x1x32xui8>
          %6 = aie.objectfifo.acquire @act3_12_13(Consume, 1) : !aie.objectfifosubview<memref<32x1x32xui8>>
          %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<32x1x32xui8>> -> memref<32x1x32xui8>
          %8 = aie.objectfifo.acquire @act1_13_22_21(Produce, 1) : !aie.objectfifosubview<memref<32x1x256xui8>>
          %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<32x1x256xui8>> -> memref<32x1x256xui8>
          %10 = aie.objectfifo.acquire @skip_1(Consume, 1) : !aie.objectfifosubview<memref<32x1x256xui8>>
          %11 = aie.objectfifo.subview.access %10[0] : !aie.objectfifosubview<memref<32x1x256xui8>> -> memref<32x1x256xui8>
          %c32_i32 = arith.constant 32 : i32
          %c64_i32 = arith.constant 64 : i32
          %c256_i32 = arith.constant 256 : i32
          func.call @conv2dk1_skip_ui8(%5, %7, %1, %9, %11, %c32_i32, %c64_i32, %c256_i32, %2, %3) : (memref<32x1x32xui8>, memref<32x1x32xui8>, memref<16384xi8>, memref<32x1x256xui8>, memref<32x1x256xui8>, i32, i32, i32, i32, i32) -> ()
          aie.objectfifo.release @act3_14_13(Consume, 1)
          aie.objectfifo.release @act3_12_13(Consume, 1)
          aie.objectfifo.release @act1_13_22_21(Produce, 1)
          aie.objectfifo.release @skip_1(Consume, 1)
        }
        aie.objectfifo.release @wts_buf_12(Consume, 1)
      }
      aie.end
    } {link_with = "conv2dk1_skip.o"}
    %core_1_2 = aie.core(%tile_1_2) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %0 = aie.objectfifo.acquire @wts_buf_11(Consume, 1) : !aie.objectfifosubview<memref<36864xi8>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<36864xi8>> -> memref<36864xi8>
        %2 = aie.objectfifo.acquire @act2_15_12_14(Consume, 2) : !aie.objectfifosubview<memref<32x1x64xui8>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<32x1x64xui8>> -> memref<32x1x64xui8>
        %4 = aie.objectfifo.subview.access %2[1] : !aie.objectfifosubview<memref<32x1x64xui8>> -> memref<32x1x64xui8>
        %5 = aie.objectfifo.acquire @act3_12_13(Produce, 1) : !aie.objectfifosubview<memref<32x1x32xui8>>
        %6 = aie.objectfifo.subview.access %5[0] : !aie.objectfifosubview<memref<32x1x32xui8>> -> memref<32x1x32xui8>
        %c32_i32 = arith.constant 32 : i32
        %c64_i32 = arith.constant 64 : i32
        %c32_i32_0 = arith.constant 32 : i32
        %c3_i32 = arith.constant 3 : i32
        %c3_i32_1 = arith.constant 3 : i32
        %c0_i32 = arith.constant 0 : i32
        %c1_i32 = arith.constant 1 : i32
        %c0_i32_2 = arith.constant 0 : i32
        func.call @conv2dk3_ui8(%3, %3, %4, %1, %6, %c32_i32, %c64_i32, %c32_i32_0, %c3_i32, %c3_i32_1, %c0_i32, %c1_i32, %c0_i32_2) : (memref<32x1x64xui8>, memref<32x1x64xui8>, memref<32x1x64xui8>, memref<36864xi8>, memref<32x1x32xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
        aie.objectfifo.release @act3_12_13(Produce, 1)
        %c0_3 = arith.constant 0 : index
        %c30 = arith.constant 30 : index
        %c1_4 = arith.constant 1 : index
        scf.for %arg1 = %c0_3 to %c30 step %c1_4 {
          %12 = aie.objectfifo.acquire @act2_15_12_14(Consume, 3) : !aie.objectfifosubview<memref<32x1x64xui8>>
          %13 = aie.objectfifo.subview.access %12[0] : !aie.objectfifosubview<memref<32x1x64xui8>> -> memref<32x1x64xui8>
          %14 = aie.objectfifo.subview.access %12[1] : !aie.objectfifosubview<memref<32x1x64xui8>> -> memref<32x1x64xui8>
          %15 = aie.objectfifo.subview.access %12[2] : !aie.objectfifosubview<memref<32x1x64xui8>> -> memref<32x1x64xui8>
          %16 = aie.objectfifo.acquire @act3_12_13(Produce, 1) : !aie.objectfifosubview<memref<32x1x32xui8>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<32x1x32xui8>> -> memref<32x1x32xui8>
          %c32_i32_12 = arith.constant 32 : i32
          %c64_i32_13 = arith.constant 64 : i32
          %c32_i32_14 = arith.constant 32 : i32
          %c3_i32_15 = arith.constant 3 : i32
          %c3_i32_16 = arith.constant 3 : i32
          %c1_i32_17 = arith.constant 1 : i32
          %c1_i32_18 = arith.constant 1 : i32
          %c0_i32_19 = arith.constant 0 : i32
          func.call @conv2dk3_ui8(%13, %14, %15, %1, %17, %c32_i32_12, %c64_i32_13, %c32_i32_14, %c3_i32_15, %c3_i32_16, %c1_i32_17, %c1_i32_18, %c0_i32_19) : (memref<32x1x64xui8>, memref<32x1x64xui8>, memref<32x1x64xui8>, memref<36864xi8>, memref<32x1x32xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
          aie.objectfifo.release @act2_15_12_14(Consume, 1)
          aie.objectfifo.release @act3_12_13(Produce, 1)
        }
        %7 = aie.objectfifo.acquire @act2_15_12_14(Consume, 2) : !aie.objectfifosubview<memref<32x1x64xui8>>
        %8 = aie.objectfifo.subview.access %7[0] : !aie.objectfifosubview<memref<32x1x64xui8>> -> memref<32x1x64xui8>
        %9 = aie.objectfifo.subview.access %7[1] : !aie.objectfifosubview<memref<32x1x64xui8>> -> memref<32x1x64xui8>
        %10 = aie.objectfifo.acquire @act3_12_13(Produce, 1) : !aie.objectfifosubview<memref<32x1x32xui8>>
        %11 = aie.objectfifo.subview.access %10[0] : !aie.objectfifosubview<memref<32x1x32xui8>> -> memref<32x1x32xui8>
        %c32_i32_5 = arith.constant 32 : i32
        %c64_i32_6 = arith.constant 64 : i32
        %c32_i32_7 = arith.constant 32 : i32
        %c3_i32_8 = arith.constant 3 : i32
        %c3_i32_9 = arith.constant 3 : i32
        %c2_i32 = arith.constant 2 : i32
        %c1_i32_10 = arith.constant 1 : i32
        %c0_i32_11 = arith.constant 0 : i32
        func.call @conv2dk3_ui8(%8, %9, %9, %1, %11, %c32_i32_5, %c64_i32_6, %c32_i32_7, %c3_i32_8, %c3_i32_9, %c2_i32, %c1_i32_10, %c0_i32_11) : (memref<32x1x64xui8>, memref<32x1x64xui8>, memref<32x1x64xui8>, memref<36864xi8>, memref<32x1x32xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
        aie.objectfifo.release @act2_15_12_14(Consume, 2)
        aie.objectfifo.release @act3_12_13(Produce, 1)
        aie.objectfifo.release @wts_buf_11(Consume, 1)
      }
      aie.end
    } {link_with = "conv2dk3.o"}
    %core_2_2 = aie.core(%tile_2_2) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %0 = aie.objectfifo.acquire @wts_buf_20(Consume, 1) : !aie.objectfifosubview<memref<16384xi8>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<16384xi8>> -> memref<16384xi8>
        %c0_0 = arith.constant 0 : index
        %2 = memref.load %rtpComputeTile22[%c0_0] : memref<16xi32>
        %c0_1 = arith.constant 0 : index
        %c32 = arith.constant 32 : index
        %c1_2 = arith.constant 1 : index
        scf.for %arg1 = %c0_1 to %c32 step %c1_2 {
          %3 = aie.objectfifo.acquire @act1_13_22_21(Consume, 1) : !aie.objectfifosubview<memref<32x1x256xui8>>
          %4 = aie.objectfifo.subview.access %3[0] : !aie.objectfifosubview<memref<32x1x256xui8>> -> memref<32x1x256xui8>
          %5 = aie.objectfifo.acquire @act2_22_23_25(Produce, 1) : !aie.objectfifosubview<memref<32x1x64xui8>>
          %6 = aie.objectfifo.subview.access %5[0] : !aie.objectfifosubview<memref<32x1x64xui8>> -> memref<32x1x64xui8>
          %c32_i32 = arith.constant 32 : i32
          %c256_i32 = arith.constant 256 : i32
          %c64_i32 = arith.constant 64 : i32
          func.call @conv2dk1_ui8(%4, %1, %6, %c32_i32, %c256_i32, %c64_i32, %2) : (memref<32x1x256xui8>, memref<16384xi8>, memref<32x1x64xui8>, i32, i32, i32, i32) -> ()
          aie.objectfifo.release @act1_13_22_21(Consume, 1)
          aie.objectfifo.release @act2_22_23_25(Produce, 1)
        }
        aie.objectfifo.release @wts_buf_20(Consume, 1)
      }
      aie.end
    } {link_with = "conv2dk1_ui8.o"}
    %core_2_3 = aie.core(%tile_2_3) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %0 = aie.objectfifo.acquire @wts_buf_21(Consume, 1) : !aie.objectfifosubview<memref<36864xi8>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<36864xi8>> -> memref<36864xi8>
        %2 = aie.objectfifo.acquire @act2_22_23_25(Consume, 2) : !aie.objectfifosubview<memref<32x1x64xui8>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<32x1x64xui8>> -> memref<32x1x64xui8>
        %4 = aie.objectfifo.subview.access %2[1] : !aie.objectfifosubview<memref<32x1x64xui8>> -> memref<32x1x64xui8>
        %5 = aie.objectfifo.acquire @act3_23_24(Produce, 1) : !aie.objectfifosubview<memref<32x1x32xui8>>
        %6 = aie.objectfifo.subview.access %5[0] : !aie.objectfifosubview<memref<32x1x32xui8>> -> memref<32x1x32xui8>
        %c32_i32 = arith.constant 32 : i32
        %c64_i32 = arith.constant 64 : i32
        %c32_i32_0 = arith.constant 32 : i32
        %c3_i32 = arith.constant 3 : i32
        %c3_i32_1 = arith.constant 3 : i32
        %c0_i32 = arith.constant 0 : i32
        %c1_i32 = arith.constant 1 : i32
        %c32_i32_2 = arith.constant 32 : i32
        func.call @conv2dk3_ui8(%3, %3, %4, %1, %6, %c32_i32, %c64_i32, %c32_i32_0, %c3_i32, %c3_i32_1, %c0_i32, %c1_i32, %c32_i32_2) : (memref<32x1x64xui8>, memref<32x1x64xui8>, memref<32x1x64xui8>, memref<36864xi8>, memref<32x1x32xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
        aie.objectfifo.release @act3_23_24(Produce, 1)
        %c0_3 = arith.constant 0 : index
        %c30 = arith.constant 30 : index
        %c1_4 = arith.constant 1 : index
        scf.for %arg1 = %c0_3 to %c30 step %c1_4 {
          %12 = aie.objectfifo.acquire @act2_22_23_25(Consume, 3) : !aie.objectfifosubview<memref<32x1x64xui8>>
          %13 = aie.objectfifo.subview.access %12[0] : !aie.objectfifosubview<memref<32x1x64xui8>> -> memref<32x1x64xui8>
          %14 = aie.objectfifo.subview.access %12[1] : !aie.objectfifosubview<memref<32x1x64xui8>> -> memref<32x1x64xui8>
          %15 = aie.objectfifo.subview.access %12[2] : !aie.objectfifosubview<memref<32x1x64xui8>> -> memref<32x1x64xui8>
          %16 = aie.objectfifo.acquire @act3_23_24(Produce, 1) : !aie.objectfifosubview<memref<32x1x32xui8>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<32x1x32xui8>> -> memref<32x1x32xui8>
          %c32_i32_12 = arith.constant 32 : i32
          %c64_i32_13 = arith.constant 64 : i32
          %c32_i32_14 = arith.constant 32 : i32
          %c3_i32_15 = arith.constant 3 : i32
          %c3_i32_16 = arith.constant 3 : i32
          %c1_i32_17 = arith.constant 1 : i32
          %c1_i32_18 = arith.constant 1 : i32
          %c32_i32_19 = arith.constant 32 : i32
          func.call @conv2dk3_ui8(%13, %14, %15, %1, %17, %c32_i32_12, %c64_i32_13, %c32_i32_14, %c3_i32_15, %c3_i32_16, %c1_i32_17, %c1_i32_18, %c32_i32_19) : (memref<32x1x64xui8>, memref<32x1x64xui8>, memref<32x1x64xui8>, memref<36864xi8>, memref<32x1x32xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
          aie.objectfifo.release @act2_22_23_25(Consume, 1)
          aie.objectfifo.release @act3_23_24(Produce, 1)
        }
        %7 = aie.objectfifo.acquire @act2_22_23_25(Consume, 2) : !aie.objectfifosubview<memref<32x1x64xui8>>
        %8 = aie.objectfifo.subview.access %7[0] : !aie.objectfifosubview<memref<32x1x64xui8>> -> memref<32x1x64xui8>
        %9 = aie.objectfifo.subview.access %7[1] : !aie.objectfifosubview<memref<32x1x64xui8>> -> memref<32x1x64xui8>
        %10 = aie.objectfifo.acquire @act3_23_24(Produce, 1) : !aie.objectfifosubview<memref<32x1x32xui8>>
        %11 = aie.objectfifo.subview.access %10[0] : !aie.objectfifosubview<memref<32x1x32xui8>> -> memref<32x1x32xui8>
        %c32_i32_5 = arith.constant 32 : i32
        %c64_i32_6 = arith.constant 64 : i32
        %c32_i32_7 = arith.constant 32 : i32
        %c3_i32_8 = arith.constant 3 : i32
        %c3_i32_9 = arith.constant 3 : i32
        %c2_i32 = arith.constant 2 : i32
        %c1_i32_10 = arith.constant 1 : i32
        %c32_i32_11 = arith.constant 32 : i32
        func.call @conv2dk3_ui8(%8, %9, %9, %1, %11, %c32_i32_5, %c64_i32_6, %c32_i32_7, %c3_i32_8, %c3_i32_9, %c2_i32, %c1_i32_10, %c32_i32_11) : (memref<32x1x64xui8>, memref<32x1x64xui8>, memref<32x1x64xui8>, memref<36864xi8>, memref<32x1x32xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
        aie.objectfifo.release @act2_22_23_25(Consume, 2)
        aie.objectfifo.release @act3_23_24(Produce, 1)
        aie.objectfifo.release @wts_buf_21(Consume, 1)
      }
      aie.end
    } {link_with = "conv2dk3.o"}
    %core_2_4 = aie.core(%tile_2_4) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %0 = aie.objectfifo.acquire @wts_buf_22(Consume, 1) : !aie.objectfifosubview<memref<16384xi8>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<16384xi8>> -> memref<16384xi8>
        %c0_0 = arith.constant 0 : index
        %2 = memref.load %rtpComputeTile24[%c0_0] : memref<16xi32>
        %c1_1 = arith.constant 1 : index
        %3 = memref.load %rtpComputeTile24[%c1_1] : memref<16xi32>
        %c0_2 = arith.constant 0 : index
        %c32 = arith.constant 32 : index
        %c1_3 = arith.constant 1 : index
        scf.for %arg1 = %c0_2 to %c32 step %c1_3 {
          %4 = aie.objectfifo.acquire @act3_23_24(Consume, 1) : !aie.objectfifosubview<memref<32x1x32xui8>>
          %5 = aie.objectfifo.subview.access %4[0] : !aie.objectfifosubview<memref<32x1x32xui8>> -> memref<32x1x32xui8>
          %6 = aie.objectfifo.acquire @act3_25_24(Consume, 1) : !aie.objectfifosubview<memref<32x1x32xui8>>
          %7 = aie.objectfifo.subview.access %6[0] : !aie.objectfifosubview<memref<32x1x32xui8>> -> memref<32x1x32xui8>
          %8 = aie.objectfifo.acquire @outOFL2L3(Produce, 1) : !aie.objectfifosubview<memref<32x1x256xui8>>
          %9 = aie.objectfifo.subview.access %8[0] : !aie.objectfifosubview<memref<32x1x256xui8>> -> memref<32x1x256xui8>
          %10 = aie.objectfifo.acquire @skip_2(Consume, 1) : !aie.objectfifosubview<memref<32x1x256xui8>>
          %11 = aie.objectfifo.subview.access %10[0] : !aie.objectfifosubview<memref<32x1x256xui8>> -> memref<32x1x256xui8>
          %c32_i32 = arith.constant 32 : i32
          %c64_i32 = arith.constant 64 : i32
          %c256_i32 = arith.constant 256 : i32
          func.call @conv2dk1_skip_ui8(%5, %7, %1, %9, %11, %c32_i32, %c64_i32, %c256_i32, %2, %3) : (memref<32x1x32xui8>, memref<32x1x32xui8>, memref<16384xi8>, memref<32x1x256xui8>, memref<32x1x256xui8>, i32, i32, i32, i32, i32) -> ()
          aie.objectfifo.release @act3_23_24(Consume, 1)
          aie.objectfifo.release @act3_25_24(Consume, 1)
          aie.objectfifo.release @outOFL2L3(Produce, 1)
          aie.objectfifo.release @skip_2(Consume, 1)
        }
        aie.objectfifo.release @wts_buf_22(Consume, 1)
      }
      aie.end
    } {link_with = "conv2dk1_skip.o"}
    %core_2_5 = aie.core(%tile_2_5) {
      %c0 = arith.constant 0 : index
      %c9223372036854775807 = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %arg0 = %c0 to %c9223372036854775807 step %c1 {
        %0 = aie.objectfifo.acquire @wts_buf_21(Consume, 1) : !aie.objectfifosubview<memref<36864xi8>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<36864xi8>> -> memref<36864xi8>
        %2 = aie.objectfifo.acquire @act2_22_23_25(Consume, 2) : !aie.objectfifosubview<memref<32x1x64xui8>>
        %3 = aie.objectfifo.subview.access %2[0] : !aie.objectfifosubview<memref<32x1x64xui8>> -> memref<32x1x64xui8>
        %4 = aie.objectfifo.subview.access %2[1] : !aie.objectfifosubview<memref<32x1x64xui8>> -> memref<32x1x64xui8>
        %5 = aie.objectfifo.acquire @act3_25_24(Produce, 1) : !aie.objectfifosubview<memref<32x1x32xui8>>
        %6 = aie.objectfifo.subview.access %5[0] : !aie.objectfifosubview<memref<32x1x32xui8>> -> memref<32x1x32xui8>
        %c32_i32 = arith.constant 32 : i32
        %c64_i32 = arith.constant 64 : i32
        %c32_i32_0 = arith.constant 32 : i32
        %c3_i32 = arith.constant 3 : i32
        %c3_i32_1 = arith.constant 3 : i32
        %c0_i32 = arith.constant 0 : i32
        %c1_i32 = arith.constant 1 : i32
        %c0_i32_2 = arith.constant 0 : i32
        func.call @conv2dk3_ui8(%3, %3, %4, %1, %6, %c32_i32, %c64_i32, %c32_i32_0, %c3_i32, %c3_i32_1, %c0_i32, %c1_i32, %c0_i32_2) : (memref<32x1x64xui8>, memref<32x1x64xui8>, memref<32x1x64xui8>, memref<36864xi8>, memref<32x1x32xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
        aie.objectfifo.release @act3_25_24(Produce, 1)
        %c0_3 = arith.constant 0 : index
        %c30 = arith.constant 30 : index
        %c1_4 = arith.constant 1 : index
        scf.for %arg1 = %c0_3 to %c30 step %c1_4 {
          %12 = aie.objectfifo.acquire @act2_22_23_25(Consume, 3) : !aie.objectfifosubview<memref<32x1x64xui8>>
          %13 = aie.objectfifo.subview.access %12[0] : !aie.objectfifosubview<memref<32x1x64xui8>> -> memref<32x1x64xui8>
          %14 = aie.objectfifo.subview.access %12[1] : !aie.objectfifosubview<memref<32x1x64xui8>> -> memref<32x1x64xui8>
          %15 = aie.objectfifo.subview.access %12[2] : !aie.objectfifosubview<memref<32x1x64xui8>> -> memref<32x1x64xui8>
          %16 = aie.objectfifo.acquire @act3_25_24(Produce, 1) : !aie.objectfifosubview<memref<32x1x32xui8>>
          %17 = aie.objectfifo.subview.access %16[0] : !aie.objectfifosubview<memref<32x1x32xui8>> -> memref<32x1x32xui8>
          %c32_i32_12 = arith.constant 32 : i32
          %c64_i32_13 = arith.constant 64 : i32
          %c32_i32_14 = arith.constant 32 : i32
          %c3_i32_15 = arith.constant 3 : i32
          %c3_i32_16 = arith.constant 3 : i32
          %c1_i32_17 = arith.constant 1 : i32
          %c1_i32_18 = arith.constant 1 : i32
          %c0_i32_19 = arith.constant 0 : i32
          func.call @conv2dk3_ui8(%13, %14, %15, %1, %17, %c32_i32_12, %c64_i32_13, %c32_i32_14, %c3_i32_15, %c3_i32_16, %c1_i32_17, %c1_i32_18, %c0_i32_19) : (memref<32x1x64xui8>, memref<32x1x64xui8>, memref<32x1x64xui8>, memref<36864xi8>, memref<32x1x32xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
          aie.objectfifo.release @act2_22_23_25(Consume, 1)
          aie.objectfifo.release @act3_25_24(Produce, 1)
        }
        %7 = aie.objectfifo.acquire @act2_22_23_25(Consume, 2) : !aie.objectfifosubview<memref<32x1x64xui8>>
        %8 = aie.objectfifo.subview.access %7[0] : !aie.objectfifosubview<memref<32x1x64xui8>> -> memref<32x1x64xui8>
        %9 = aie.objectfifo.subview.access %7[1] : !aie.objectfifosubview<memref<32x1x64xui8>> -> memref<32x1x64xui8>
        %10 = aie.objectfifo.acquire @act3_25_24(Produce, 1) : !aie.objectfifosubview<memref<32x1x32xui8>>
        %11 = aie.objectfifo.subview.access %10[0] : !aie.objectfifosubview<memref<32x1x32xui8>> -> memref<32x1x32xui8>
        %c32_i32_5 = arith.constant 32 : i32
        %c64_i32_6 = arith.constant 64 : i32
        %c32_i32_7 = arith.constant 32 : i32
        %c3_i32_8 = arith.constant 3 : i32
        %c3_i32_9 = arith.constant 3 : i32
        %c2_i32 = arith.constant 2 : i32
        %c1_i32_10 = arith.constant 1 : i32
        %c0_i32_11 = arith.constant 0 : i32
        func.call @conv2dk3_ui8(%8, %9, %9, %1, %11, %c32_i32_5, %c64_i32_6, %c32_i32_7, %c3_i32_8, %c3_i32_9, %c2_i32, %c1_i32_10, %c0_i32_11) : (memref<32x1x64xui8>, memref<32x1x64xui8>, memref<32x1x64xui8>, memref<36864xi8>, memref<32x1x32xui8>, i32, i32, i32, i32, i32, i32, i32, i32) -> ()
        aie.objectfifo.release @act2_22_23_25(Consume, 2)
        aie.objectfifo.release @act3_25_24(Produce, 1)
        aie.objectfifo.release @wts_buf_21(Consume, 1)
      }
      aie.end
    } {link_with = "conv2dk3.o"}
    aiex.runtime_sequence(%arg0: memref<65536xi8>, %arg1: memref<212992xi8>, %arg2: memref<262144xi8>) {
      aiex.npu.rtp_write(@rtpComputeTile02, 0, 1)
      aiex.npu.rtp_write(@rtpComputeTile05, 0, 1)
      aiex.npu.rtp_write(@rtpComputeTile05, 1, 0)
      aiex.npu.rtp_write(@rtpComputeTile05, 2, 1)
      aiex.npu.rtp_write(@rtpComputeTile15, 0, 1)
      aiex.npu.rtp_write(@rtpComputeTile13, 0, 1)
      aiex.npu.rtp_write(@rtpComputeTile13, 1, 0)
      aiex.npu.rtp_write(@rtpComputeTile22, 0, 1)
      aiex.npu.rtp_write(@rtpComputeTile24, 0, 1)
      aiex.npu.rtp_write(@rtpComputeTile24, 1, 0)
      %0 = aiex.dma_configure_task_for @act1_00_02_01 {
        aie.dma_bd(%arg0 : memref<65536xi8>, 0, 65536, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 65536, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%0)
      %1 = aiex.dma_configure_task_for @wts_0_L3L2 {
        aie.dma_bd(%arg1 : memref<212992xi8>, 0, 73728, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 73728, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%1)
      %2 = aiex.dma_configure_task_for @wts_1_L3L2 {
        aie.dma_bd(%arg1 : memref<212992xi8>, 73728, 69632, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 69632, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%2)
      %3 = aiex.dma_configure_task_for @wts_2_L3L2 {
        aie.dma_bd(%arg1 : memref<212992xi8>, 143360, 69632, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 69632, stride = 1>])
        aie.end
      }
      aiex.dma_start_task(%3)
      %4 = aiex.dma_configure_task_for @outOFL2L3 {
        aie.dma_bd(%arg2 : memref<262144xi8>, 0, 262144, [<size = 1, stride = 0>, <size = 1, stride = 0>, <size = 1, stride = 0>, <size = 262144, stride = 1>])
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%4)
      aiex.dma_await_task(%4)
      aiex.dma_free_task(%0)
      aiex.dma_free_task(%1)
      aiex.dma_free_task(%2)
      aiex.dma_free_task(%3)
    }
  }
}

