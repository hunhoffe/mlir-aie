// RUN: aie-opt --objectfifo-to-conduit --dma-task-to-conduit --conduit-fuse-operators %s | FileCheck %s
//
// Regression test for Bug B (Matrix Row #1 4col_med compile FAIL on
// `aiex.run` arg-type mismatch, task #78 / #81).
//
// Root cause: `--conduit-fuse-operators`'s pre-fix `buildArgGroupsFromSeq`
// used an `offset == 0` heuristic to detect arg-group boundaries.  In a
// 4-column lowering whose BD ordering interleaves columns (col0_argA,
// col0_argB, col1_argA, col1_argB, ...) every column's first BD hits
// `offset == 0` and starts a bogus new group, while non-zero-offset
// column-1+ BDs absorb into the wrong neighbour's group.  The downstream
// `computeFullBufferType` then projected the merged sequence's block-arg
// type from the truncated `groups[i].maxExtent` (= per-column extent =
// 2048) instead of the recorded `memref<8192xbf16>` (= full host buffer).
//
// Fix: arg_index-driven grouping (every BD whose `arg_index = N` lands in
// the same group regardless of source order) + max(BD-derived extent,
// origType extent) in `computeFullBufferType`.
//
// Concrete reproducer here: 4-column producer that writes one host buffer
// %arg0 (memref<8192xbf16>) using 4 BDs at offsets 0/2048/4096/6144, with
// the column-major BD ordering that triggered the heuristic failure.
// devB consumes the fused intermediate and writes a final output buffer.

// CHECK-LABEL: module @fs81_4col_type_preservation

// Single surviving device after fusion:
// CHECK:       aie.device(npu2) @devA
// CHECK-NOT:   aie.device(npu2) @devB

// Merged runtime_sequence preserves the full 8192 element extent for the
// host input buffer (NOT shrunk to per-column 2048).  Note that the
// merged runtime_sequence emits anonymously — the @sequence symbol from
// the input devA/devB sequences is dropped during fusion.
// CHECK:       aie.runtime_sequence(
// CHECK-SAME:    memref<8192xbf16>

// All four post-fusion BDs targeting the input must reference the SAME
// merged-seq block arg (arg_index = 0), with their original column-major
// offsets preserved.  arg_index = 0 must appear at least four times.
// CHECK-DAG:   arg_index = 0
// CHECK-DAG:   arg_index = 0
// CHECK-DAG:   arg_index = 0
// CHECK-DAG:   arg_index = 0

// The discardable provenance tag must not leak past --conduit-fuse-operators.
// CHECK-NOT:   _origin_device

module @fs81_4col_type_preservation {
  // DevA: 4-column producer.  Reads %arg0 (full 8192-element host buffer)
  // via 4 column-distributed BDs at offsets 0/2048/4096/6144, writes a
  // fusible intermediate per column.
  aie.device(npu2) @devA {
    %shim_0 = aie.tile(0, 0)
    %shim_1 = aie.tile(1, 0)
    %shim_2 = aie.tile(2, 0)
    %shim_3 = aie.tile(3, 0)
    %tile_0_2 = aie.tile(0, 2)
    %tile_1_2 = aie.tile(1, 2)
    %tile_2_2 = aie.tile(2, 2)
    %tile_3_2 = aie.tile(3, 2)

    aie.objectfifo @ext_in_0(%shim_0, {%tile_0_2}, 2 : i32)
        : !aie.objectfifo<memref<2048xbf16>>
    aie.objectfifo @ext_in_1(%shim_1, {%tile_1_2}, 2 : i32)
        : !aie.objectfifo<memref<2048xbf16>>
    aie.objectfifo @ext_in_2(%shim_2, {%tile_2_2}, 2 : i32)
        : !aie.objectfifo<memref<2048xbf16>>
    aie.objectfifo @ext_in_3(%shim_3, {%tile_3_2}, 2 : i32)
        : !aie.objectfifo<memref<2048xbf16>>

    aie.objectfifo @inter_out_0(%tile_0_2, {%shim_0}, 2 : i32)
        {fusion_group = "fg0"}
        : !aie.objectfifo<memref<2048xbf16>>
    aie.objectfifo @inter_out_1(%tile_1_2, {%shim_1}, 2 : i32)
        {fusion_group = "fg0"}
        : !aie.objectfifo<memref<2048xbf16>>
    aie.objectfifo @inter_out_2(%tile_2_2, {%shim_2}, 2 : i32)
        {fusion_group = "fg0"}
        : !aie.objectfifo<memref<2048xbf16>>
    aie.objectfifo @inter_out_3(%tile_3_2, {%shim_3}, 2 : i32)
        {fusion_group = "fg0"}
        : !aie.objectfifo<memref<2048xbf16>>

    func.func private @prod_kernel(memref<2048xbf16>, memref<2048xbf16>)

    %core0 = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %cmax = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %i = %c0 to %cmax step %c1 {
        %a = aie.objectfifo.acquire @ext_in_0(Consume, 1)
            : !aie.objectfifosubview<memref<2048xbf16>>
        %a_buf = aie.objectfifo.subview.access %a[0]
            : !aie.objectfifosubview<memref<2048xbf16>> -> memref<2048xbf16>
        %o = aie.objectfifo.acquire @inter_out_0(Produce, 1)
            : !aie.objectfifosubview<memref<2048xbf16>>
        %o_buf = aie.objectfifo.subview.access %o[0]
            : !aie.objectfifosubview<memref<2048xbf16>> -> memref<2048xbf16>
        func.call @prod_kernel(%a_buf, %o_buf)
            : (memref<2048xbf16>, memref<2048xbf16>) -> ()
        aie.objectfifo.release @inter_out_0(Produce, 1)
        aie.objectfifo.release @ext_in_0(Consume, 1)
      }
      aie.end
    } {link_with = "prod.a"}

    %core1 = aie.core(%tile_1_2) {
      %c0 = arith.constant 0 : index
      %cmax = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %i = %c0 to %cmax step %c1 {
        %a = aie.objectfifo.acquire @ext_in_1(Consume, 1)
            : !aie.objectfifosubview<memref<2048xbf16>>
        %a_buf = aie.objectfifo.subview.access %a[0]
            : !aie.objectfifosubview<memref<2048xbf16>> -> memref<2048xbf16>
        %o = aie.objectfifo.acquire @inter_out_1(Produce, 1)
            : !aie.objectfifosubview<memref<2048xbf16>>
        %o_buf = aie.objectfifo.subview.access %o[0]
            : !aie.objectfifosubview<memref<2048xbf16>> -> memref<2048xbf16>
        func.call @prod_kernel(%a_buf, %o_buf)
            : (memref<2048xbf16>, memref<2048xbf16>) -> ()
        aie.objectfifo.release @inter_out_1(Produce, 1)
        aie.objectfifo.release @ext_in_1(Consume, 1)
      }
      aie.end
    } {link_with = "prod.a"}

    %core2 = aie.core(%tile_2_2) {
      %c0 = arith.constant 0 : index
      %cmax = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %i = %c0 to %cmax step %c1 {
        %a = aie.objectfifo.acquire @ext_in_2(Consume, 1)
            : !aie.objectfifosubview<memref<2048xbf16>>
        %a_buf = aie.objectfifo.subview.access %a[0]
            : !aie.objectfifosubview<memref<2048xbf16>> -> memref<2048xbf16>
        %o = aie.objectfifo.acquire @inter_out_2(Produce, 1)
            : !aie.objectfifosubview<memref<2048xbf16>>
        %o_buf = aie.objectfifo.subview.access %o[0]
            : !aie.objectfifosubview<memref<2048xbf16>> -> memref<2048xbf16>
        func.call @prod_kernel(%a_buf, %o_buf)
            : (memref<2048xbf16>, memref<2048xbf16>) -> ()
        aie.objectfifo.release @inter_out_2(Produce, 1)
        aie.objectfifo.release @ext_in_2(Consume, 1)
      }
      aie.end
    } {link_with = "prod.a"}

    %core3 = aie.core(%tile_3_2) {
      %c0 = arith.constant 0 : index
      %cmax = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %i = %c0 to %cmax step %c1 {
        %a = aie.objectfifo.acquire @ext_in_3(Consume, 1)
            : !aie.objectfifosubview<memref<2048xbf16>>
        %a_buf = aie.objectfifo.subview.access %a[0]
            : !aie.objectfifosubview<memref<2048xbf16>> -> memref<2048xbf16>
        %o = aie.objectfifo.acquire @inter_out_3(Produce, 1)
            : !aie.objectfifosubview<memref<2048xbf16>>
        %o_buf = aie.objectfifo.subview.access %o[0]
            : !aie.objectfifosubview<memref<2048xbf16>> -> memref<2048xbf16>
        func.call @prod_kernel(%a_buf, %o_buf)
            : (memref<2048xbf16>, memref<2048xbf16>) -> ()
        aie.objectfifo.release @inter_out_3(Produce, 1)
        aie.objectfifo.release @ext_in_3(Consume, 1)
      }
      aie.end
    } {link_with = "prod.a"}

    // Critical: %arg0 has the FULL host-buffer extent (8192), but BDs are
    // ordered column-major: ext_in_0 (off=0) then ext_in_1 (off=2048) etc.
    // Pre-fix the offset==0 heuristic also tracked the inter_out_* BDs as
    // their own group every time their offset hit 0, then absorbed the
    // 2048+ offsets into the wrong neighbour.
    aie.runtime_sequence @sequence(%a0: memref<8192xbf16>,
                                    %a1: memref<8192xbf16>) {
      %t0 = aiex.dma_configure_task_for @ext_in_0 {
        aie.dma_bd(%a0 : memref<8192xbf16>, 0, 2048,
          [<size = 1, stride = 0>, <size = 1, stride = 0>,
           <size = 1, stride = 0>, <size = 2048, stride = 1>])
            {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%t0)
      %t1 = aiex.dma_configure_task_for @ext_in_1 {
        aie.dma_bd(%a0 : memref<8192xbf16>, 2048, 2048,
          [<size = 1, stride = 0>, <size = 1, stride = 0>,
           <size = 1, stride = 0>, <size = 2048, stride = 1>])
            {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%t1)
      %t2 = aiex.dma_configure_task_for @ext_in_2 {
        aie.dma_bd(%a0 : memref<8192xbf16>, 4096, 2048,
          [<size = 1, stride = 0>, <size = 1, stride = 0>,
           <size = 1, stride = 0>, <size = 2048, stride = 1>])
            {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%t2)
      %t3 = aiex.dma_configure_task_for @ext_in_3 {
        aie.dma_bd(%a0 : memref<8192xbf16>, 6144, 2048,
          [<size = 1, stride = 0>, <size = 1, stride = 0>,
           <size = 1, stride = 0>, <size = 2048, stride = 1>])
            {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%t3)
      // Fusible intermediate output BDs (offset=0 each — same 4-column
      // pattern that was the trigger for the heuristic mis-grouping).
      %t4 = aiex.dma_configure_task_for @inter_out_0 {
        aie.dma_bd(%a1 : memref<8192xbf16>, 0, 2048,
          [<size = 1, stride = 0>, <size = 1, stride = 0>,
           <size = 1, stride = 0>, <size = 2048, stride = 1>])
            {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%t4)
      %t5 = aiex.dma_configure_task_for @inter_out_1 {
        aie.dma_bd(%a1 : memref<8192xbf16>, 2048, 2048,
          [<size = 1, stride = 0>, <size = 1, stride = 0>,
           <size = 1, stride = 0>, <size = 2048, stride = 1>])
            {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%t5)
      %t6 = aiex.dma_configure_task_for @inter_out_2 {
        aie.dma_bd(%a1 : memref<8192xbf16>, 4096, 2048,
          [<size = 1, stride = 0>, <size = 1, stride = 0>,
           <size = 1, stride = 0>, <size = 2048, stride = 1>])
            {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%t6)
      %t7 = aiex.dma_configure_task_for @inter_out_3 {
        aie.dma_bd(%a1 : memref<8192xbf16>, 6144, 2048,
          [<size = 1, stride = 0>, <size = 1, stride = 0>,
           <size = 1, stride = 0>, <size = 2048, stride = 1>])
            {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%t7)
      aiex.dma_await_task(%t4)
      aiex.dma_await_task(%t5)
      aiex.dma_await_task(%t6)
      aiex.dma_await_task(%t7)
      aiex.dma_free_task(%t0)
      aiex.dma_free_task(%t1)
      aiex.dma_free_task(%t2)
      aiex.dma_free_task(%t3)
    }
  }

  // DevB: 4-column consumer.  Consumes the fusible intermediate in 4
  // columns and writes the final output to a full 8192 host buffer.
  aie.device(npu2) @devB {
    %shim_0 = aie.tile(0, 0)
    %shim_1 = aie.tile(1, 0)
    %shim_2 = aie.tile(2, 0)
    %shim_3 = aie.tile(3, 0)
    %tile_0_2 = aie.tile(0, 2)
    %tile_1_2 = aie.tile(1, 2)
    %tile_2_2 = aie.tile(2, 2)
    %tile_3_2 = aie.tile(3, 2)

    aie.objectfifo @inter_in_0(%shim_0, {%tile_0_2}, 2 : i32)
        {fusion_group = "fg0"}
        : !aie.objectfifo<memref<2048xbf16>>
    aie.objectfifo @inter_in_1(%shim_1, {%tile_1_2}, 2 : i32)
        {fusion_group = "fg0"}
        : !aie.objectfifo<memref<2048xbf16>>
    aie.objectfifo @inter_in_2(%shim_2, {%tile_2_2}, 2 : i32)
        {fusion_group = "fg0"}
        : !aie.objectfifo<memref<2048xbf16>>
    aie.objectfifo @inter_in_3(%shim_3, {%tile_3_2}, 2 : i32)
        {fusion_group = "fg0"}
        : !aie.objectfifo<memref<2048xbf16>>

    aie.objectfifo @ext_out_0(%tile_0_2, {%shim_0}, 2 : i32)
        : !aie.objectfifo<memref<2048xbf16>>
    aie.objectfifo @ext_out_1(%tile_1_2, {%shim_1}, 2 : i32)
        : !aie.objectfifo<memref<2048xbf16>>
    aie.objectfifo @ext_out_2(%tile_2_2, {%shim_2}, 2 : i32)
        : !aie.objectfifo<memref<2048xbf16>>
    aie.objectfifo @ext_out_3(%tile_3_2, {%shim_3}, 2 : i32)
        : !aie.objectfifo<memref<2048xbf16>>

    func.func private @cons_kernel(memref<2048xbf16>, memref<2048xbf16>)

    %core0 = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %cmax = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %i = %c0 to %cmax step %c1 {
        %a = aie.objectfifo.acquire @inter_in_0(Consume, 1)
            : !aie.objectfifosubview<memref<2048xbf16>>
        %a_buf = aie.objectfifo.subview.access %a[0]
            : !aie.objectfifosubview<memref<2048xbf16>> -> memref<2048xbf16>
        %o = aie.objectfifo.acquire @ext_out_0(Produce, 1)
            : !aie.objectfifosubview<memref<2048xbf16>>
        %o_buf = aie.objectfifo.subview.access %o[0]
            : !aie.objectfifosubview<memref<2048xbf16>> -> memref<2048xbf16>
        func.call @cons_kernel(%a_buf, %o_buf)
            : (memref<2048xbf16>, memref<2048xbf16>) -> ()
        aie.objectfifo.release @ext_out_0(Produce, 1)
        aie.objectfifo.release @inter_in_0(Consume, 1)
      }
      aie.end
    } {link_with = "cons.a"}

    %core1 = aie.core(%tile_1_2) {
      %c0 = arith.constant 0 : index
      %cmax = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %i = %c0 to %cmax step %c1 {
        %a = aie.objectfifo.acquire @inter_in_1(Consume, 1)
            : !aie.objectfifosubview<memref<2048xbf16>>
        %a_buf = aie.objectfifo.subview.access %a[0]
            : !aie.objectfifosubview<memref<2048xbf16>> -> memref<2048xbf16>
        %o = aie.objectfifo.acquire @ext_out_1(Produce, 1)
            : !aie.objectfifosubview<memref<2048xbf16>>
        %o_buf = aie.objectfifo.subview.access %o[0]
            : !aie.objectfifosubview<memref<2048xbf16>> -> memref<2048xbf16>
        func.call @cons_kernel(%a_buf, %o_buf)
            : (memref<2048xbf16>, memref<2048xbf16>) -> ()
        aie.objectfifo.release @ext_out_1(Produce, 1)
        aie.objectfifo.release @inter_in_1(Consume, 1)
      }
      aie.end
    } {link_with = "cons.a"}

    %core2 = aie.core(%tile_2_2) {
      %c0 = arith.constant 0 : index
      %cmax = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %i = %c0 to %cmax step %c1 {
        %a = aie.objectfifo.acquire @inter_in_2(Consume, 1)
            : !aie.objectfifosubview<memref<2048xbf16>>
        %a_buf = aie.objectfifo.subview.access %a[0]
            : !aie.objectfifosubview<memref<2048xbf16>> -> memref<2048xbf16>
        %o = aie.objectfifo.acquire @ext_out_2(Produce, 1)
            : !aie.objectfifosubview<memref<2048xbf16>>
        %o_buf = aie.objectfifo.subview.access %o[0]
            : !aie.objectfifosubview<memref<2048xbf16>> -> memref<2048xbf16>
        func.call @cons_kernel(%a_buf, %o_buf)
            : (memref<2048xbf16>, memref<2048xbf16>) -> ()
        aie.objectfifo.release @ext_out_2(Produce, 1)
        aie.objectfifo.release @inter_in_2(Consume, 1)
      }
      aie.end
    } {link_with = "cons.a"}

    %core3 = aie.core(%tile_3_2) {
      %c0 = arith.constant 0 : index
      %cmax = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %i = %c0 to %cmax step %c1 {
        %a = aie.objectfifo.acquire @inter_in_3(Consume, 1)
            : !aie.objectfifosubview<memref<2048xbf16>>
        %a_buf = aie.objectfifo.subview.access %a[0]
            : !aie.objectfifosubview<memref<2048xbf16>> -> memref<2048xbf16>
        %o = aie.objectfifo.acquire @ext_out_3(Produce, 1)
            : !aie.objectfifosubview<memref<2048xbf16>>
        %o_buf = aie.objectfifo.subview.access %o[0]
            : !aie.objectfifosubview<memref<2048xbf16>> -> memref<2048xbf16>
        func.call @cons_kernel(%a_buf, %o_buf)
            : (memref<2048xbf16>, memref<2048xbf16>) -> ()
        aie.objectfifo.release @ext_out_3(Produce, 1)
        aie.objectfifo.release @inter_in_3(Consume, 1)
      }
      aie.end
    } {link_with = "cons.a"}

    aie.runtime_sequence @sequence(%b0: memref<8192xbf16>,
                                    %b1: memref<8192xbf16>) {
      %t0 = aiex.dma_configure_task_for @inter_in_0 {
        aie.dma_bd(%b0 : memref<8192xbf16>, 0, 2048,
          [<size = 1, stride = 0>, <size = 1, stride = 0>,
           <size = 1, stride = 0>, <size = 2048, stride = 1>])
            {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%t0)
      %t1 = aiex.dma_configure_task_for @inter_in_1 {
        aie.dma_bd(%b0 : memref<8192xbf16>, 2048, 2048,
          [<size = 1, stride = 0>, <size = 1, stride = 0>,
           <size = 1, stride = 0>, <size = 2048, stride = 1>])
            {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%t1)
      %t2 = aiex.dma_configure_task_for @inter_in_2 {
        aie.dma_bd(%b0 : memref<8192xbf16>, 4096, 2048,
          [<size = 1, stride = 0>, <size = 1, stride = 0>,
           <size = 1, stride = 0>, <size = 2048, stride = 1>])
            {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%t2)
      %t3 = aiex.dma_configure_task_for @inter_in_3 {
        aie.dma_bd(%b0 : memref<8192xbf16>, 6144, 2048,
          [<size = 1, stride = 0>, <size = 1, stride = 0>,
           <size = 1, stride = 0>, <size = 2048, stride = 1>])
            {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%t3)
      %t4 = aiex.dma_configure_task_for @ext_out_0 {
        aie.dma_bd(%b1 : memref<8192xbf16>, 0, 2048,
          [<size = 1, stride = 0>, <size = 1, stride = 0>,
           <size = 1, stride = 0>, <size = 2048, stride = 1>])
            {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%t4)
      %t5 = aiex.dma_configure_task_for @ext_out_1 {
        aie.dma_bd(%b1 : memref<8192xbf16>, 2048, 2048,
          [<size = 1, stride = 0>, <size = 1, stride = 0>,
           <size = 1, stride = 0>, <size = 2048, stride = 1>])
            {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%t5)
      %t6 = aiex.dma_configure_task_for @ext_out_2 {
        aie.dma_bd(%b1 : memref<8192xbf16>, 4096, 2048,
          [<size = 1, stride = 0>, <size = 1, stride = 0>,
           <size = 1, stride = 0>, <size = 2048, stride = 1>])
            {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%t6)
      %t7 = aiex.dma_configure_task_for @ext_out_3 {
        aie.dma_bd(%b1 : memref<8192xbf16>, 6144, 2048,
          [<size = 1, stride = 0>, <size = 1, stride = 0>,
           <size = 1, stride = 0>, <size = 2048, stride = 1>])
            {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%t7)
      aiex.dma_await_task(%t4)
      aiex.dma_await_task(%t5)
      aiex.dma_await_task(%t6)
      aiex.dma_await_task(%t7)
      aiex.dma_free_task(%t0)
      aiex.dma_free_task(%t1)
      aiex.dma_free_task(%t2)
      aiex.dma_free_task(%t3)
    }
  }
}
