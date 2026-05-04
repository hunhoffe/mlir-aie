// RUN: aie-opt --objectfifo-to-conduit --dma-task-to-conduit --conduit-fuse-operators --conduit-depth-promote --conduit-to-dma %s | FileCheck %s
//
// Pinning the cross-column compute-to-compute shared-memory guard.
//
// Two operators are fused into a single device.  After --conduit-fuse-operators,
// devB's tile_0_2 is offset to tile_1_2, so the fused intermediate channel goes
// from tile(0,2) → tile(1,2) — same row, different column ("W-neighbor").
//
// AIE2's TargetModel::isLegalMemAffinity reports W-neighbor as a legal shared-
// memory pairing, but the Conduit lowering path has never been NPU-validated
// for cross-column shared memory.  The fix in ConduitToDMAAlloc.cpp /
// ConduitToDMARoute.cpp / ConduitInferModes.cpp guards against this case so
// the intermediate falls through to the DMA flow path (which is exercised and
// known good) when the user did not explicitly request shared_memory.
//
// Expected lowering for the fused intermediate (named fused_intermediate_0):
//   * an aie.flow is emitted between %tile_0_2 and %tile_1_2 over DMA
//   * consumer-side buffer is allocated on tile(1,2) (DMA path), in addition
//     to producer-side buffers on tile(0,2)
//   * locks live on BOTH producer and consumer tiles (separate DMA endpoints,
//     not the single producer-tile lock pair that the shared-memory path emits)
//   * a memtile/aie.mem region appears on BOTH tiles (DMA endpoints)

// CHECK-LABEL: module @cross_column_shmem_falls_through_to_dma

// Tile decls.
// CHECK-DAG: %tile_0_2 = aie.tile(0, 2)
// CHECK-DAG: %tile_1_2 = aie.tile(1, 2)

// DMA flow MUST be emitted for the fused intermediate (cross-col same-row).
// CHECK-DAG: aie.flow(%tile_0_2, DMA : {{[0-9]+}}, %tile_1_2, DMA : {{[0-9]+}})

// Producer-side buffers on tile_0_2.
// CHECK-DAG: aie.buffer(%tile_0_2) {sym_name = "fused_intermediate_0_buff_0"}
// CHECK-DAG: aie.buffer(%tile_0_2) {sym_name = "fused_intermediate_0_buff_1"}

// Consumer-side buffers on tile_1_2 — KEY: this is what the shared-memory
// path would NOT emit.
// CHECK-DAG: aie.buffer(%tile_1_2) {sym_name = "fused_intermediate_0_cons_buff_0"}
// CHECK-DAG: aie.buffer(%tile_1_2) {sym_name = "fused_intermediate_0_cons_buff_1"}

// Lock pairs on BOTH tiles (DMA endpoints, not a single producer-tile pair).
// CHECK-DAG: aie.lock(%tile_0_2, {{[0-9]+}}) {{.*}}sym_name = "fused_intermediate_0_prod_lock_0"
// CHECK-DAG: aie.lock(%tile_0_2, {{[0-9]+}}) {{.*}}sym_name = "fused_intermediate_0_cons_lock_0"
// CHECK-DAG: aie.lock(%tile_1_2, {{[0-9]+}}) {{.*}}sym_name = "fused_intermediate_0_cons_prod_lock_0"
// CHECK-DAG: aie.lock(%tile_1_2, {{[0-9]+}}) {{.*}}sym_name = "fused_intermediate_0_cons_cons_lock_0"

// aie.mem regions on BOTH tiles confirm DMA path (shared-memory would emit none).
// CHECK-DAG: aie.mem(%tile_0_2)
// CHECK-DAG: aie.mem(%tile_1_2)

// All Conduit ops gone (Pass C consumed them).
// CHECK-NOT: conduit.create
// CHECK-NOT: conduit.acquire
// CHECK-NOT: conduit.release

module @cross_column_shmem_falls_through_to_dma {
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
    // tile set is NOT a subset of devA's; preserves the offset path that
    // pushes devB's tile_0_2 to tile_1_2 (the cross-column shmem case
    // this fixture pins).
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
