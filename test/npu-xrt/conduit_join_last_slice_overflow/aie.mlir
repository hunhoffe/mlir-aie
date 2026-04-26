//===- aie.mlir ------------------------------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2026 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//
//
// JOIN last-slice BD-length overflow regression — npu-xrt e2e companion to
// the lit-only pin at
//   test/Dialect/Conduit/conduit_to_dma_join_last_slice_bd_overflow.mlir.
//
// Bug:
//   In Pass C JOIN lowering (lib/Dialect/Conduit/Transforms/ConduitToDMALink.cpp,
//   the lastInd fall-through arms at lines ~1107 (S2MM ingest) and ~1186
//   (MM2S send)) the LAST slice's BD length is computed as
//     srcLens[srcIdx] = joinDstPerBufForLen - srcOffsets[srcIdx];   // S2MM
//     mm2sLens[s]     = joinDstPerBufForLen - mm2sOffsets[s];       // MM2S
//   Pre-fix joinDstPerBufForLen was sourced from the SHIM consumer's
//   get_memref_async num_elems (the per-DISPATCH window), which can be a
//   large multiple of the per-slice memtile JOIN buffer.  Post-fix
//   (ConduitToDMALink.cpp:574-595, the joinDstPerBufForLen derivation block)
//   it is taken from the JOIN buffer's MemRefType element count
//   (intBufTy = jDstInfo->elemType, getNumElements()), with numElems used
//   only when the elemType is non-MemRef.
//
// Topology (matches the lit pin):
//   compute(0,2) ─┐
//                 ├──> memtile(0,1) ──> shim(0,0)
//   compute(0,3) ─┘
//   join_src_a:  compute(0,2) -> memtile, 64 bf16 (worker_a writes 0..63)
//   join_src_b:  compute(0,3) -> memtile, 64 bf16 (worker_b writes 64..127)
//   join_dst:    memtile -> shim,        128 bf16 (= 64 + 64, slice offsets [0, 64])
//   shim consumer: aie.dma_bd(%arg0 : memref<512xbf16>, 0, 512)
//                  -> get_memref_async num_elems = 512 (= 4 × per-slice 128)
//
// Correct vs buggy emit on the LAST slice (= slice 1, source compute(0,3)):
//   correct (post-fix):  joinDstPerBufForLen = 128 → BD len = 128 - 64 = 64
//   buggy   (pre-fix):   joinDstPerBufForLen = 512 → BD len = 512 - 64 = 448
//   The 448 writes against memref<128xbf16> JOIN buffers — a 4× overflow.
//
// Behavior at this shape:
//   - Stateful (no --use-conduit, stateful.lit):
//       Stateful objectfifo→aie lowering computes per-slice BD lengths
//       directly from the ofifo element type (memref<64xbf16> per source).
//       PASSES byte-equivalent today on npu1_1col and npu2_1col.
//   - Conduit pre-fix (--use-conduit):
//       Pass C emits dma_bd len = 448 on the LAST-slice JOIN BDs.  Firmware
//       programs the BD; the resulting cross-buffer memtile write hangs the
//       dispatch (no completion token, the host wait would TIMEOUT).
//   - Conduit post-fix (--use-conduit, this working tree):
//       Pass C emits dma_bd len = 64 on every JOIN BD.  Dispatch completes;
//       output bytes match the stateful reference.

module {
  aie.device(NPUDEVICE) {
    %shim_0   = aie.tile(0, 0)
    %mem_0_1  = aie.tile(0, 1)
    %tile_0_2 = aie.tile(0, 2)
    %tile_0_3 = aie.tile(0, 3)

    aie.objectfifo @join_src_a (%tile_0_2, {%mem_0_1}, 2 : i32)
        : !aie.objectfifo<memref<64xbf16>>
    aie.objectfifo @join_src_b (%tile_0_3, {%mem_0_1}, 2 : i32)
        : !aie.objectfifo<memref<64xbf16>>
    aie.objectfifo @join_dst (%mem_0_1, {%shim_0}, 2 : i32)
        : !aie.objectfifo<memref<128xbf16>>

    // JOIN: 2 sources -> 1 destination through MemTile(0,1), per-source
    // byte-offsets [0, 64] (each source contributes 64 bf16 to the
    // 128-bf16 dst).
    aie.objectfifo.link [@join_src_a, @join_src_b] -> [@join_dst] ([0, 64][])

    // worker_a: each acquired 64-bf16 buffer is filled with float(j) for
    // j in [0, 64).  All values are bf16-exact (integers in [0, 256) fit
    // in bf16 sign + 8 exp + 7 mantissa).
    %core_0_2 = aie.core(%tile_0_2) {
      %c0   = arith.constant 0 : index
      %c1   = arith.constant 1 : index
      %c64  = arith.constant 64 : index
      %cmax = arith.constant 0xFFFFFE : index
      scf.for %niter = %c0 to %cmax step %c1 {
        %sv = aie.objectfifo.acquire @join_src_a(Produce, 1)
                 : !aie.objectfifosubview<memref<64xbf16>>
        %elem = aie.objectfifo.subview.access %sv[0]
                 : !aie.objectfifosubview<memref<64xbf16>> -> memref<64xbf16>
        scf.for %i = %c0 to %c64 step %c1 {
          %i_i32  = arith.index_cast %i : index to i32
          %i_f32  = arith.sitofp %i_i32 : i32 to f32
          %i_bf16 = arith.truncf %i_f32 : f32 to bf16
          memref.store %i_bf16, %elem[%i] : memref<64xbf16>
        }
        aie.objectfifo.release @join_src_a(Produce, 1)
      }
      aie.end
    }

    // worker_b: each acquired 64-bf16 buffer is filled with float(64 + j)
    // for j in [0, 64), i.e. values 64..127.  Also bf16-exact.
    %core_0_3 = aie.core(%tile_0_3) {
      %c0      = arith.constant 0 : index
      %c1      = arith.constant 1 : index
      %c64     = arith.constant 64 : index
      %c64_i32 = arith.constant 64 : i32
      %cmax    = arith.constant 0xFFFFFE : index
      scf.for %niter = %c0 to %cmax step %c1 {
        %sv = aie.objectfifo.acquire @join_src_b(Produce, 1)
                 : !aie.objectfifosubview<memref<64xbf16>>
        %elem = aie.objectfifo.subview.access %sv[0]
                 : !aie.objectfifosubview<memref<64xbf16>> -> memref<64xbf16>
        scf.for %i = %c0 to %c64 step %c1 {
          %i_i32  = arith.index_cast %i : index to i32
          %v_i32  = arith.addi %i_i32, %c64_i32 : i32
          %v_f32  = arith.sitofp %v_i32 : i32 to f32
          %v_bf16 = arith.truncf %v_f32 : f32 to bf16
          memref.store %v_bf16, %elem[%i] : memref<64xbf16>
        }
        aie.objectfifo.release @join_src_b(Produce, 1)
      }
      aie.end
    }

    // Shim consumer: per-dispatch transfer of 512 bf16 (4× the per-slice
    // memtile JOIN buffer of 128).  --dma-task-to-conduit lowers the
    // dma_bd transfer length into conduit.get_memref_async num_elems = 512,
    // which Pass C's collect phase records as info.numElems for @join_dst.
    // Pre-fix that value (incorrectly) became joinDstPerBufForLen for the
    // JOIN last-slice BD length calc.
    aie.runtime_sequence @join_overflow(%arg0: memref<512xbf16>) {
      %t = aiex.dma_configure_task_for @join_dst {
        aie.dma_bd(%arg0 : memref<512xbf16>, 0, 512) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%t)
      aiex.dma_await_task(%t)
      aiex.dma_free_task(%t)
    }
  }
}
