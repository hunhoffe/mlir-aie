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
// HW smoke for the canon refuse-to-collapse-on-link rule
// (--conduit-canonicalize-channel-puts on a shim-MM2S → memtile LINKED
// channel; lit-only pin lives at
//   test/Dialect/Conduit/conduit_to_dma_b_channel_consolidation.mlir).
//
// Pattern (mirrors the geometry that wedges the Llama prefill attn_scores
// GEMM at M=2048 K=2048 N=512 num_invocations=16, scaled down):
//   shim(0,0) ──@in_L3L2──> memtile(0,1) ──@in_L2L1──> compute(0,2) ──@out──> shim(0,0)
//   aie.objectfifo.link [@in_L3L2] -> [@in_L2L1]([] [0])
//
// Runtime sequence: 4 IDENTICAL MM2S puts on @in_L3L2 (all at offset 0,
// len 64 bf16) — exactly the structural shape that
// HomogeneousRepeatPattern matches.  Pre-fix, canon collapsed those 4
// puts into 1 put + dma_repeat = 3 on @in_L3L2; Pass C took the LINK
// path (ConduitToDMALink.cpp Case B) and surfaced the consolidated
// `repeat_count = 3` onto the runtime_sequence shim configure.  On
// the linked path, that consolidated form does NOT compose with
// multi-round consumer pacing on the memtile relay (verified
// 2026-05-03 by hand-patch on the Llama GEMM IR — collapsed form
// hangs at HW dispatch with XRT timeout, paced form completes
// correctly), so HW dispatch hangs.
//
// Post-fix (this working tree): canon's HomogeneousRepeatPattern (and
// ArithProgressionPattern, symmetric) calls
// `xilinx::conduit::detail::isLinkedChannel(scope, chanName)` (declared
// in CanonicalizeChannelPutsUtils.h) which walks the scope for
// conduit.scatter / gather / transpose ops (Pass A's lowered form of
// aie.objectfifo.link) and returns true if @in_L3L2 appears in any
// `src` / `srcs` / `dst` / `dsts` symbol-ref attr.  Canon emits a
// remark and returns false from tryCollapsePuts.  Pass C therefore
// emits 4 separate paced shim configures, HW dispatch completes, and
// the per-byte byte-equivalence check in test.cpp passes.
//
// Compute kernel: in-core bf16-by-bf16 copy from the acquired input
// slice to the acquired output slot.  Each iteration consumes one input
// slice and produces one output slot, so 4 puts → 4 dispatches → 4
// output slots populated.
//
// Reference (test.cpp computes this):
//   output[i*64 + j] = input[j]  for i in [0,4), j in [0,64)
// All input values 0..63 are bf16-exact (integers fit in sign + 8 exp +
// 7 mantissa).  Output is 4 identical replicas of the input ramp.
//
// HW-shape note: slice = 64 bf16 (128 bytes) clears npu2 shim BD
// min-length comfortably.  N = 4 is well under any tile BD cap so the
// only canon refusal that fires is the link-refusal (NOT the cap-refusal
// which is also present in HomogeneousRepeatPattern).

module {
  aie.device(NPUDEVICE) {
    %shim_0   = aie.tile(0, 0)
    %mem_0_1  = aie.tile(0, 1)
    %tile_0_2 = aie.tile(0, 2)

    // Shim → memtile → compute (LINKED), depth=2 throughout.  The link
    // is what triggers Pass C's Case B (linked-MM2S → memtile relay).
    aie.objectfifo @in_L3L2 (%shim_0,   {%mem_0_1},  2 : i32)
        : !aie.objectfifo<memref<64xbf16>>
    aie.objectfifo @in_L2L1 (%mem_0_1,  {%tile_0_2}, 2 : i32)
        : !aie.objectfifo<memref<64xbf16>>
    aie.objectfifo.link [@in_L3L2] -> [@in_L2L1]([] [0])

    // Direct compute → shim output (NOT linked); this exercises the
    // existing canon path (which would also collapse 4 identical gets,
    // but with no link the collapse is fine and Pass C handles it).
    aie.objectfifo @out (%tile_0_2, {%shim_0}, 2 : i32)
        : !aie.objectfifo<memref<64xbf16>>

    // Compute core: per-iteration copy from acquired @in_L2L1 slice to
    // acquired @out slot.  Trip = ∞ sentinel; the host runtime sequence
    // controls the actual dispatch count via shim puts/gets.
    %core_0_2 = aie.core(%tile_0_2) {
      %c0   = arith.constant 0 : index
      %c1   = arith.constant 1 : index
      %c64  = arith.constant 64 : index
      %cmax = arith.constant 0xFFFFFE : index
      scf.for %niter = %c0 to %cmax step %c1 {
        %sv_in = aie.objectfifo.acquire @in_L2L1 (Consume, 1)
            : !aie.objectfifosubview<memref<64xbf16>>
        %elem_in = aie.objectfifo.subview.access %sv_in[0]
            : !aie.objectfifosubview<memref<64xbf16>> -> memref<64xbf16>

        %sv_out = aie.objectfifo.acquire @out (Produce, 1)
            : !aie.objectfifosubview<memref<64xbf16>>
        %elem_out = aie.objectfifo.subview.access %sv_out[0]
            : !aie.objectfifosubview<memref<64xbf16>> -> memref<64xbf16>

        scf.for %j = %c0 to %c64 step %c1 {
          %v = memref.load %elem_in[%j]  : memref<64xbf16>
          memref.store %v, %elem_out[%j] : memref<64xbf16>
        }

        aie.objectfifo.release @in_L2L1 (Consume, 1)
        aie.objectfifo.release @out     (Produce, 1)
      }
      aie.end
    }

    // Runtime sequence: 4 STRUCTURALLY IDENTICAL puts on @in_L3L2
    // (the LINKED channel) — matches HomogeneousRepeatPattern's match
    // condition exactly.  Post-fix, canon refuses and Pass C emits 4
    // separate paced shim configures.
    //
    // Output: 4 S2MM gets on @out at offsets [0, 64, 128, 192] —
    // arith-progression-shaped on a NON-linked channel; if collapse
    // happens here that's fine.
    aie.runtime_sequence @canon_no_collapse_on_link(%input  : memref<64xbf16>,
                                                    %output : memref<256xbf16>) {
      // ---- 4x identical MM2S puts on the LINKED input channel ----
      %ti0 = aiex.dma_configure_task_for @in_L3L2 {
        aie.dma_bd(%input : memref<64xbf16>, 0, 64) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%ti0)
      aiex.dma_free_task(%ti0)

      %ti1 = aiex.dma_configure_task_for @in_L3L2 {
        aie.dma_bd(%input : memref<64xbf16>, 0, 64) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%ti1)
      aiex.dma_free_task(%ti1)

      %ti2 = aiex.dma_configure_task_for @in_L3L2 {
        aie.dma_bd(%input : memref<64xbf16>, 0, 64) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%ti2)
      aiex.dma_free_task(%ti2)

      %ti3 = aiex.dma_configure_task_for @in_L3L2 {
        aie.dma_bd(%input : memref<64xbf16>, 0, 64) {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%ti3)
      aiex.dma_free_task(%ti3)

      // ---- 4x S2MM gets on the (non-linked) output channel ----
      %to0 = aiex.dma_configure_task_for @out {
        aie.dma_bd(%output : memref<256xbf16>, 0,   64) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%to0)
      aiex.dma_await_task(%to0)
      aiex.dma_free_task(%to0)

      %to1 = aiex.dma_configure_task_for @out {
        aie.dma_bd(%output : memref<256xbf16>, 64,  64) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%to1)
      aiex.dma_await_task(%to1)
      aiex.dma_free_task(%to1)

      %to2 = aiex.dma_configure_task_for @out {
        aie.dma_bd(%output : memref<256xbf16>, 128, 64) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%to2)
      aiex.dma_await_task(%to2)
      aiex.dma_free_task(%to2)

      %to3 = aiex.dma_configure_task_for @out {
        aie.dma_bd(%output : memref<256xbf16>, 192, 64) {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%to3)
      aiex.dma_await_task(%to3)
      aiex.dma_free_task(%to3)
    }
  }
}
