// RUN: aie-opt --objectfifo-to-conduit --dma-task-to-conduit --conduit-fuse-operators %s | FileCheck %s
//
// Test for --conduit-fuse-operators: spatial IRON operator fusion.
//
// Two aie.device(npu2) blocks represent independent operators:
//   Device A (producer): reads @ext_in from LPDDR5, writes @inter_out to LPDDR5.
//     @inter_out has fusion_group = "fg0" marking it as a fusible intermediate.
//   Device B (consumer): reads @inter_in from LPDDR5, writes @ext_out to LPDDR5.
//     @inter_in has fusion_group = "fg0" matching device A's output.
//
// After --conduit-fuse-operators:
//   1. The two devices should be merged into one aie.device.
//   2. Device B's tiles should be offset (col 0 → col 1).
//   3. The intermediate channels (@inter_out, @inter_in) should be erased and
//      replaced by a fused_intermediate conduit.create.
//   4. External I/O channels (@ext_in, @ext_out) should survive.
//   5. Core body channel references renamed to @fused_intermediate_0.
//   6. Intermediate DMA task ops erased from runtime sequence.

// CHECK-LABEL: module @fuse_operators_basic

// Only one device remains after fusion (devB merged into devA):
// CHECK:       aie.device(npu2)

// Device B's tile(0,2) should be offset to tile(1,2) in the merged device:
// CHECK-DAG:   aie.tile(0, 2)
// CHECK-DAG:   aie.tile(1, 2)

// External I/O channels survive; fused intermediate replaces @inter_out/@inter_in:
// CHECK-DAG:   conduit.create @ext_in
// CHECK-DAG:   conduit.create @ext_out
// CHECK-DAG:   conduit.create @fused_intermediate_0

// The intermediate channels should NOT appear:
// CHECK-NOT:   conduit.create @inter_out
// CHECK-NOT:   conduit.create @inter_in

// Producer core references fused intermediate (not @inter_out):
// CHECK:       aie.core(%{{.*}}tile_0_2)
// CHECK:         name = @fused_intermediate_0{{.*}}port = #conduit.port<Produce>

// Consumer core references fused intermediate (not @inter_in):
// CHECK:       aie.core(%{{.*}}tile_1_2)
// CHECK:         name = @fused_intermediate_0{{.*}}port = #conduit.port<Consume>

// Second aie.device should NOT exist:
// CHECK-NOT:   aie.device(npu2)

module @fuse_operators_basic {
  aie.device(npu2) @devA {
    %shim_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)

    // External input: LPDDR5 → compute tile.
    aie.objectfifo @ext_in(%shim_0, {%tile_0_2}, 2 : i32)
        : !aie.objectfifo<memref<128xbf16>>

    // Intermediate output: compute tile → LPDDR5 (fusible).
    aie.objectfifo @inter_out(%tile_0_2, {%shim_0}, 2 : i32)
        {fusion_group = "fg0"}
        : !aie.objectfifo<memref<128xbf16>>

    func.func private @producer_kernel(memref<128xbf16>, memref<128xbf16>)

    %core = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %cmax = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %i = %c0 to %cmax step %c1 {
        %in = aie.objectfifo.acquire @ext_in(Consume, 1)
            : !aie.objectfifosubview<memref<128xbf16>>
        %in_buf = aie.objectfifo.subview.access %in[0]
            : !aie.objectfifosubview<memref<128xbf16>> -> memref<128xbf16>
        %out = aie.objectfifo.acquire @inter_out(Produce, 1)
            : !aie.objectfifosubview<memref<128xbf16>>
        %out_buf = aie.objectfifo.subview.access %out[0]
            : !aie.objectfifosubview<memref<128xbf16>> -> memref<128xbf16>
        func.call @producer_kernel(%in_buf, %out_buf)
            : (memref<128xbf16>, memref<128xbf16>) -> ()
        aie.objectfifo.release @inter_out(Produce, 1)
        aie.objectfifo.release @ext_in(Consume, 1)
      }
      aie.end
    } {link_with = "producer.a"}

    aie.runtime_sequence(%arg0: memref<128xbf16>, %intermediate: memref<128xbf16>) {
      %t0 = aiex.dma_configure_task_for @ext_in {
        aie.dma_bd(%arg0 : memref<128xbf16>, 0, 128,
          [<size = 1, stride = 0>, <size = 1, stride = 0>,
           <size = 1, stride = 0>, <size = 128, stride = 1>])
            {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%t0)
      %t1 = aiex.dma_configure_task_for @inter_out {
        aie.dma_bd(%intermediate : memref<128xbf16>, 0, 128,
          [<size = 1, stride = 0>, <size = 1, stride = 0>,
           <size = 1, stride = 0>, <size = 128, stride = 1>])
            {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%t1)
      aiex.dma_await_task(%t1)
      aiex.dma_free_task(%t0)
    }
  }

  aie.device(npu2) @devB {
    // shim at column 1 (distinct from devA's column-0 shim) so devB's
    // tile set is NOT a subset of devA's; this preserves the offset path
    // (devB → +colMaxA+1 = +1) that this fixture's CHECKs encode.
    %shim_0 = aie.tile(1, 0)
    %tile_0_2 = aie.tile(0, 2)

    // Intermediate input: LPDDR5 → compute tile (fusible, matching fusion_group).
    aie.objectfifo @inter_in(%shim_0, {%tile_0_2}, 2 : i32)
        {fusion_group = "fg0"}
        : !aie.objectfifo<memref<128xbf16>>

    // External output: compute tile → LPDDR5.
    aie.objectfifo @ext_out(%tile_0_2, {%shim_0}, 2 : i32)
        : !aie.objectfifo<memref<128xbf16>>

    func.func private @consumer_kernel(memref<128xbf16>, memref<128xbf16>)

    %core = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %cmax = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %i = %c0 to %cmax step %c1 {
        %in = aie.objectfifo.acquire @inter_in(Consume, 1)
            : !aie.objectfifosubview<memref<128xbf16>>
        %in_buf = aie.objectfifo.subview.access %in[0]
            : !aie.objectfifosubview<memref<128xbf16>> -> memref<128xbf16>
        %out = aie.objectfifo.acquire @ext_out(Produce, 1)
            : !aie.objectfifosubview<memref<128xbf16>>
        %out_buf = aie.objectfifo.subview.access %out[0]
            : !aie.objectfifosubview<memref<128xbf16>> -> memref<128xbf16>
        func.call @consumer_kernel(%in_buf, %out_buf)
            : (memref<128xbf16>, memref<128xbf16>) -> ()
        aie.objectfifo.release @ext_out(Produce, 1)
        aie.objectfifo.release @inter_in(Consume, 1)
      }
      aie.end
    } {link_with = "consumer.a"}

    aie.runtime_sequence(%intermediate: memref<128xbf16>, %arg1: memref<128xbf16>) {
      %t0 = aiex.dma_configure_task_for @inter_in {
        aie.dma_bd(%intermediate : memref<128xbf16>, 0, 128,
          [<size = 1, stride = 0>, <size = 1, stride = 0>,
           <size = 1, stride = 0>, <size = 128, stride = 1>])
            {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%t0)
      %t1 = aiex.dma_configure_task_for @ext_out {
        aie.dma_bd(%arg1 : memref<128xbf16>, 0, 128,
          [<size = 1, stride = 0>, <size = 1, stride = 0>,
           <size = 1, stride = 0>, <size = 128, stride = 1>])
            {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%t1)
      aiex.dma_await_task(%t1)
      aiex.dma_free_task(%t0)
    }
  }
}
