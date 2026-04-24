// RUN: aie-opt --objectfifo-to-conduit --dma-task-to-conduit %s | FileCheck %s
//
// =============================================================================
// BUG ISOLATION (NOT YET FIXED)
// =============================================================================
//
// `--dma-task-to-conduit` silently drops the IRON-emitted
// `aiex.dma_configure_task_for {repeat_count = N : i32}` attribute.
//
// Source:  lib/Dialect/Conduit/Transforms/ConduitDmaTaskToConduit.cpp:236-336
//          Walks runtime_sequence converting `aiex.dma_configure_task_for` ops
//          into `conduit.put_memref` / `conduit.get_memref`. Reads BD offset,
//          len, dimensions, buffer block-arg → never calls
//          `op->getRepeatCount()`. The companion `repeat_count` attribute is
//          dropped on the floor at lines 312-327.
//
// Trigger: IRON's GEMM operator emits a per-col N-tile broadcast as
//          `aie.dma_bd(... [<size = R, stride = 0>, ...])` PLUS a sibling
//          `{repeat_count = R-1 : i32}` on the configure_task. The shim DMA
//          must fire the BD R times per dispatch (once per N-tile). With the
//          attr dropped, Pass C cannot regenerate the configure_task with
//          repeat_count > 0, so the shim queue fires the BD ONCE, the cores
//          stall after 1/R of the work, and the kernel TIMEs OUT.
//
// Surfaces as: #79 — Llama 3.2 1B prefill `attn_query` GEMM TIMEOUT under
//              bare `--use-conduit` at any prompt_len (N=2048 axis is
//              config-driven: 2048 / (8 cols * 64 tile_n) = 4 N-tiles/col).
//              `gemm_pattern_d_smoke.py` reproduces the TIMEOUT in <10s.
//
// =============================================================================
// THIS TEST PINS THE WRONG (CURRENT) BEHAVIOR.
// When the fix lands, flip the CHECK-NOT below to a CHECK-SAME pinning the
// surfaced attribute (likely `dma_repeat = 3` on the conduit.create or a new
// `repeat_count` / `bd_repeat` attr on the conduit.put_memref — see open
// questions for the fix author below) and rename this file to drop _BUG.
// =============================================================================
//
// -----------------------------------------------------------------------------
// Open questions for the fix author
// -----------------------------------------------------------------------------
//
// Q1. N vs N+1 convention. IRON emits `repeat_count = 3` paired with
//     `<size = 4, stride = 0>` (4 BD firings) — so the field encodes
//     "additional firings" (value N → N+1 total fires).
//     Evidence:
//       * `lib/Dialect/AIEX/Transforms/AIEDmaToNpu.cpp:180-183` packs
//         `repeat_cnt` VERBATIM into the NPU command word
//         (`cmd |= (repeat_cnt & 0xFF) << 16;`) — no +1 transformation
//         in the npu_inst writer.
//       * Prior bug_c byte-patch session: patching `repeat_count: 4 → 0`
//         shifted behavior from "5 firings (overfire)" to "1 firing" —
//         consistent with firmware reading the field as N+1 firings.
//       * IRON's GEMM with 4 N-tiles emits `repeat_count = 3` (= 3+1 = 4 fires).
//       * IRON's GEMM with 8 N-tiles emits `repeat_count = 7` (= 7+1 = 8 fires).
//     Verdict for the fix: when surfacing this onto the conduit op, decide
//     whether to keep IRON's "additional firings" convention (then Pass C
//     re-emits the same value verbatim) OR convert to total-firings to match
//     the existing `dma_repeat` semantics on conduit.create (where
//     `dma_repeat = 4` in `passC_shim_bd_dma_repeat_uses_configure_task_repeat.mlir`
//     means "fire 4 times"). The Pass C surfacing test currently uses the
//     total-firings convention. If we keep that, the fix must do
//     `dma_repeat = repeat_count + 1`.
//
// Q2. Conflict with Pass A's existing `dma_repeat` stamp. Pass A
//     (`ObjectFifoToConduit.cpp`) already infers dma_repeat from
//     producer/consumer emit counts (and skips the stamp on `emit.count == 1`
//     per `dc792ebbd5`). For the GEMM A_L3L2 channel here, emit.count == 1 so
//     Pass A skips, leaving the stamp to dma-task-to-conduit. But for channels
//     where Pass A DOES stamp (emit.count > 1), the new dma-task-to-conduit
//     stamp could either (a) collide with Pass A's value, or (b) supersede it.
//     Recommended: assert equivalence (collision → emitError) since IRON
//     would not normally emit both signals on the same channel; investigate
//     if a real conflict is observed in the wild.
//
// Q3. S2MM symmetry. The branch at ConduitDmaTaskToConduit.cpp:315-328
//     handles the S2MM (output) direction → conduit.get_memref. IRON's
//     GEMM output channel for `issue_token=true` paths emits
//     `repeat_count = 1` on the S2MM configure_task (= 2 total fires).
//     The fix should symmetrically surface repeat_count on get_memref OR
//     onto the conduit.create's `bd_repeat`/`dma_repeat` (the create attr
//     is bidirectional). Decide where the canonical home for this signal
//     lives.
//
// Style: mirrors `dma_task_to_conduit_repeat_bd.mlir` (FS5 broadcast-only
//        regression — same shim BD shape, no `repeat_count` attribute).
//        This test ADDS the `{repeat_count = 3 : i32}` attribute to make
//        the dropped-attr bug observable.

// CHECK-LABEL: module @dma_task_to_conduit_drops_repeat_count_BUG
module @dma_task_to_conduit_drops_repeat_count_BUG {
  aie.device(npu2) {
    %shim_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)

    // External A-input from LPDDR5: per-col N-tile broadcast pattern
    // (4 BD firings per dispatch). Mirrors the A_L3L2 shim channel of
    // IRON GEMM at the failing shape.
    aie.objectfifo @A_L3L2_0(%shim_0, {%tile_0_2}, 2 : i32)
        : !aie.objectfifo<memref<4096xbf16>>

    func.func private @gemm_kernel(memref<4096xbf16>)

    %core = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %cmax = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %i = %c0 to %cmax step %c1 {
        %in = aie.objectfifo.acquire @A_L3L2_0(Consume, 1)
            : !aie.objectfifosubview<memref<4096xbf16>>
        %in_buf = aie.objectfifo.subview.access %in[0]
            : !aie.objectfifosubview<memref<4096xbf16>> -> memref<4096xbf16>
        func.call @gemm_kernel(%in_buf) : (memref<4096xbf16>) -> ()
        aie.objectfifo.release @A_L3L2_0(Consume, 1)
      }
      aie.end
    } {link_with = "gemm.a"}

    // Runtime sequence: IRON-shaped MM2S configure_task with the per-col
    // N-tile broadcast.
    //
    //   {repeat_count = 3 : i32}                    ← IRON encodes "N+1 firings"
    //   aie.dma_bd(... [<size = 4, stride = 0>, ...])   ← matches: 4 firings
    //
    // Together: shim fires the BD 4× per dispatch (once per N-tile).
    // The dropped `repeat_count` attribute is the bug.
    aie.runtime_sequence(%arg0: memref<16384xbf16>) {
      %t0 = aiex.dma_configure_task_for @A_L3L2_0 {
        aie.dma_bd(%arg0 : memref<16384xbf16>, 0, 4096,
          [<size = 4, stride = 0>,
           <size = 1, stride = 0>,
           <size = 64, stride = 64>,
           <size = 64, stride = 1>])
            {burst_length = 0 : i32}
        aie.end
      } {repeat_count = 3 : i32}
      aiex.dma_start_task(%t0)
      aiex.dma_free_task(%t0)
    }
  }
}

// -----------------------------------------------------------------------------
// CURRENT (buggy) behavior: the conduit.create has no `dma_repeat` attribute,
// and the lowered conduit.put_memref carries no repeat-related field. The
// `<size=4, stride=0>` broadcast dim survives in `producer_dimensions` (the
// FS5 fix preserves it) but the companion firing count is gone.
//
// Pass A's emit.count==1 skip (commit dc792ebbd5) means `dma_repeat` is
// correctly absent on the create from the Pass A side; the bug is that
// dma-task-to-conduit doesn't restamp it from the dropped `repeat_count`.
// -----------------------------------------------------------------------------

// CHECK:       conduit.create @A_L3L2_0
// CHECK-NOT:   dma_repeat
// CHECK-NOT:   bd_repeat
// CHECK-NOT:   repeat_count

// CHECK:       conduit.put_memref
// CHECK-SAME:  name = @A_L3L2_0
// CHECK-NOT:   dma_repeat
// CHECK-NOT:   bd_repeat
// CHECK-NOT:   repeat_count
