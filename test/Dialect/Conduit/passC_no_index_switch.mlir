// RUN: aie-opt --conduit-check-channels --conduit-to-dma %s | FileCheck %s
//
// Regression test: Pass C generates scf.if chains for buffer selection
// in depth>1 channels. This test verifies that scf.if is used (not
// scf.index_switch) and that the rotation counter is memref.alloca().
//
// The buffer selection pattern for depth=2:
//   %idx = arith.index_cast %counter : i32 to index
//   %cond = arith.cmpi eq, %idx, %c0 : index
//   %buf = scf.if %cond -> memref<T> {
//     scf.yield %buf_0
//   } else {
//     scf.yield %buf_1
//   }
//
// CHECK-LABEL: module @passC_no_index_switch
// CHECK: aie.core
// CHECK: %alloca = memref.alloca() : memref<1xi32>
// CHECK: arith.cmpi eq
// CHECK: scf.if
// CHECK-NOT: scf.index_switch
// CHECK: aie.end

module @passC_no_index_switch {
  aie.device(npu1_1col) {
    %shim = aie.tile(0, 0)
    %tile = aie.tile(0, 2)

    // depth=2: triggers dynamic rotation counter + buffer selection
    conduit.create @fifo {capacity = 64 : i64, depth = 2 : i64,
                    element_type = memref<32xi32>,
                    producer_tile = array<i64: 0, 0>,
                    consumer_tiles = array<i64: 0, 2>}

    aie.shim_dma_allocation @fifo_shim_alloc(%shim, MM2S, 0)

    %core = aie.core(%tile) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c4 = arith.constant 4 : index
      %val = arith.constant 42 : i32

      scf.for %i = %c0 to %c4 step %c1 {
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

    aie.runtime_sequence(%in: memref<128xi32>) {
      aiex.npu.dma_memcpy_nd (%in[0,0,0,0][1,1,1,128][0,0,0,1])
          {metadata = @fifo_shim_alloc, id = 0 : i64} : memref<128xi32>
    }
  }
}
