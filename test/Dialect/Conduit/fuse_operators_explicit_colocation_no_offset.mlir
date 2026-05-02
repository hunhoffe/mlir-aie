// RUN: aie-opt --objectfifo-to-conduit --dma-task-to-conduit --conduit-fuse-operators --aie-combine-device=same-tile=true --conduit-fuse-core-bodies %s | FileCheck %s
//
// Test: --conduit-fuse-operators must NOT offset device B's tile coordinates
// when devB's tiles are a subset of devA's (explicit co-location for spatial
// + core-body fusion of two ops on the same physical tile).
//
// Pipeline:
//   1. --conduit-fuse-operators merges devA + devB (same-tile co-located).
//      With the deviceTilesSubsetOf gate, colOffset = 0; both producer and
//      consumer cores stay on tile(0,2) and the intermediate channel is
//      emitted as @fused_intermediate_0.
//   2. --aie-combine-device + --conduit-fuse-core-bodies then merges the two
//      cores on tile(0,2) into a single core (mul + sink in one body).
//
// Without the no-offset gate, devB's tile(0,2) would have been offset to
// tile(1,2) — separating the two cores onto different columns and preventing
// the core-body merge.

// CHECK-LABEL: module @fuse_explicit_colocation
// CHECK:       aie.device(npu2)

// Both devB tiles must remain at column 0 (no offset applied):
// CHECK-DAG:   aie.tile(0, 2)
// CHECK-NOT:   aie.tile(1, 2)

// External I/O survives; intermediate channel erased after core-body fusion:
// CHECK-DAG:   conduit.create @ext_in
// CHECK-DAG:   conduit.create @ext_out
// CHECK-NOT:   conduit.create @inter_out
// CHECK-NOT:   conduit.create @inter_in
// CHECK-NOT:   conduit.create @fused_intermediate_0

// Exactly one aie.core remains, on tile(0,2), with merged link_files
// containing both producer.o and consumer.o:
// CHECK:       aie.core(%{{.*}}tile_0_2)
// CHECK:       link_files = [{{.*}}"producer.o"{{.*}}"consumer.o"{{.*}}]
// CHECK-NOT:   aie.core

module @fuse_explicit_colocation {
  aie.device(npu2) @devA {
    %shim_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)

    aie.objectfifo @ext_in(%shim_0, {%tile_0_2}, 2 : i32)
        : !aie.objectfifo<memref<128xbf16>>

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
    } {link_with = "producer.o"}

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

  // Device B uses the SAME tile coordinates as device A — explicit
  // co-location so spatial fusion + a follow-up core-body fusion can
  // merge the two cores into a single core on tile(0,2).
  aie.device(npu2) @devB {
    %shim_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)

    aie.objectfifo @inter_in(%shim_0, {%tile_0_2}, 2 : i32)
        {fusion_group = "fg0"}
        : !aie.objectfifo<memref<128xbf16>>

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
    } {link_with = "consumer.o"}

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
