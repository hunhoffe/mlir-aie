// RUN: aie-opt --conduit-check-channels --conduit-to-dma %s | FileCheck %s
//
// Regression test: rotation counter is allocated as aie.buffer at device
// level, NOT as memref.alloca inside the core body.
//
// Background: Pass C uses aie.buffer for the rotation counter. When
// multiple conduits target the same compute tile, a single shared buffer
// (memref<Nxi32>) is used with non-overlapping slot indices.
//
// This test verifies: (1) aie.buffer with sym_name "rotation_counter_0_2"
// IS present at device level, (2) no memref.alloca inside aie.core for the
// rotation counter, (3) both conduits share the same buffer with distinct slots.
//
// CHECK-LABEL: module @passC_rotation_counter_aie_buffer
// CHECK: aie.device
// Rotation counter must appear as aie.buffer at device level, not memref.alloca.
// CHECK: %[[ROTBUF:.*]] = aie.buffer({{.*}}) {sym_name = "rotation_counter_0_2"} : memref<2xi32>
// Init stores happen at device level (using aie.buffer).
// CHECK: memref.store {{.*}} %[[ROTBUF]]
// CHECK: memref.store {{.*}} %[[ROTBUF]]
// CHECK: aie.core
// CHECK-NOT: memref.alloca
// CHECK: aie.end

module @passC_rotation_counter_aie_buffer {
  aie.device(npu1_1col) {
    %shim = aie.tile(0, 0)
    %tile = aie.tile(0, 2)

    // Two depth-2 channels from shim → compute tile.
    // Both need rotation counters (depth=2 requires runtime index tracking).
    conduit.create @fifoA {depth = 2 : i64,
                    element_type = memref<32xi32>
                    }

    conduit.create @fifoB {depth = 2 : i64,
                    element_type = memref<32xi32>
                    }

    aie.shim_dma_allocation @fifoA_shim_alloc(%shim, MM2S, 0)
    aie.shim_dma_allocation @fifoB_shim_alloc(%shim, MM2S, 1)

    %core = aie.core(%tile) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c4 = arith.constant 4 : index
      %val = arith.constant 42 : i32

      scf.for %i = %c0 to %c4 step %c1 {
        %wa = conduit.acquire {name = @fifoA, count = 1 : i64,
                               port = #conduit.port<Consume>}
                  : !conduit.window<memref<32xi32>>
        %ea = conduit.subview_access %wa {index = 0 : i64}
                  : !conduit.window<memref<32xi32>> -> memref<32xi32>

        %wb = conduit.acquire {name = @fifoB, count = 1 : i64,
                               port = #conduit.port<Consume>}
                  : !conduit.window<memref<32xi32>>
        %eb = conduit.subview_access %wb {index = 0 : i64}
                  : !conduit.window<memref<32xi32>> -> memref<32xi32>

        memref.store %val, %ea[%c0] : memref<32xi32>

        conduit.release %wa {count = 1 : i64, port = #conduit.port<Consume>}
            : !conduit.window<memref<32xi32>>
        conduit.release %wb {count = 1 : i64, port = #conduit.port<Consume>}
            : !conduit.window<memref<32xi32>>
      }

      aie.end
    }

    aie.runtime_sequence(%inA: memref<128xi32>, %inB: memref<128xi32>) {
      aiex.npu.dma_memcpy_nd (%inA[0,0,0,0][1,1,1,128][0,0,0,1])
          {metadata = @fifoA_shim_alloc, id = 0 : i64} : memref<128xi32>
      aiex.npu.dma_memcpy_nd (%inB[0,0,0,0][1,1,1,128][0,0,0,1])
          {metadata = @fifoB_shim_alloc, id = 1 : i64} : memref<128xi32>
    }
  }
}
