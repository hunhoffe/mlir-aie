// RUN: aie-opt --conduit-check-channels --conduit-to-dma %s | FileCheck %s
//
// Regression test: Pass C must NOT generate scf.index_switch for dynamic
// buffer selection in depth>1 channels.
//
// Bug: PEANO (llvm-aie v20.0.0, commit 0e7cfc0e) generates incorrect lookup
// tables for scf.index_switch when the modular buffer index wraps around.
// Example: depth=4, (counter+1)%4=0 at counter=3 → table produces buff_3
// instead of buff_0. This causes the core to access the wrong buffer without
// lock protection → concurrent DMA+core write → hardware fault.
//
// Confirmed by inspecting the .data section of the compiled ELF:
//   [0x7bc58] = 0x0007ac00 (buff_3) — WRONG for counter=3, idx+1
//   correct:   = 0x00079400 (buff_0)
//
// Hardware symptom: N_middle=6 sliding window with conv2dk3 fails with
// "qds_device::wait() unexpected command state" while N_middle<=5 passes.
// The oracle (fully unrolled, no scf.index_switch) always passes.
//
// Fix: use scf.if chain (→ cf.cond_br) instead of scf.index_switch.
// cf.cond_br is correctly compiled by PEANO for all counter values.
//
// CHECK-LABEL: module @passC_no_index_switch_regression
// CHECK: aie.core
// CHECK-NOT: scf.index_switch
// CHECK: scf.if
// CHECK: aie.end

module @passC_no_index_switch_regression {
  aie.device(npu1_1col) {
    %shim = aie.tile(0, 0)
    %tile = aie.tile(0, 2)

    // depth=4 channel: triggers rotation counter + buffer selection
    conduit.create @fifo {capacity = 128 : i64, depth = 4 : i64,
                    element_type = memref<32xi32>,
                    producer_tile = array<i64: 0, 0>,
                    consumer_tiles = array<i64: 0, 2>}

    aie.shim_dma_allocation @fifo_shim_alloc(%shim, MM2S, 0)

    %core = aie.core(%tile) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c8 = arith.constant 8 : index
      %val = arith.constant 42 : i32

      scf.for %i = %c0 to %c8 step %c1 {
        %win = conduit.acquire {name = "fifo", count = 1 : i64,
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
