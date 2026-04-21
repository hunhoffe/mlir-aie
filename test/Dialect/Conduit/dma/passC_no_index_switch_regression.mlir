// RUN: aie-opt --conduit-check-channels --conduit-to-dma %s | FileCheck %s
//
// Regression test: Pass C generates scf.index_switch for dynamic buffer
// selection in depth>1 channels. This test uses depth=4 to verify the
// scf.index_switch has 4 cases.
//
// CHECK-LABEL: module @passC_no_index_switch_regression
// CHECK: aie.core
// CHECK: scf.index_switch
// CHECK: aie.end

module @passC_no_index_switch_regression {
  aie.device(npu1_1col) {
    %shim = aie.tile(0, 0)
    %tile = aie.tile(0, 2)

    // depth=4 channel: triggers rotation counter + buffer selection
    conduit.create @fifo {depth = 4 : i64,
                    element_type = memref<32xi32>
                    }

    aie.shim_dma_allocation @fifo_shim_alloc(%shim, MM2S, 0)

    %core = aie.core(%tile) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c8 = arith.constant 8 : index
      %val = arith.constant 42 : i32

      scf.for %i = %c0 to %c8 step %c1 {
        %win = conduit.acquire {name = @fifo, count = 1 : i64,
                                port = #conduit.port<Consume>}
                   : !conduit.window<memref<32xi32>>
        %buf = conduit.subview_access %win {index = 0 : i64}
                   : !conduit.window<memref<32xi32>> -> memref<32xi32>
        memref.store %val, %buf[%c0] : memref<32xi32>
        conduit.release %win {count = 1 : i64, port = #conduit.port<Consume>}
            : !conduit.window<memref<32xi32>>
      }

      aie.end
    }

    aie.runtime_sequence(%in: memref<256xi32>) {
      aiex.npu.dma_memcpy_nd (%in[0,0,0,0][1,1,1,256][0,0,0,1])
          {metadata = @fifo_shim_alloc, id = 0 : i64} : memref<256xi32>
    }
  }
}
