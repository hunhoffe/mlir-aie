// RUN: aie-opt --objectfifo-to-conduit %s | FileCheck %s
//
// Task #38 — Pass A `dma_repeat` over-counts when the acquire site sits
// inside an `scf.if` branch whose condition is not provably constant.
//
// This test ISOLATES the bug — it pins the WRONG current behavior so the
// eventual fix MUST update this CHECK (the test goes from pinning the bug
// to pinning correct behavior, per the
// `isolate-bug-with-lit-test-BEFORE-fixing` working convention added
// 2026-04-24, see `CLAUDE.md` Working Conventions).
//
// ---------------------------------------------------------------------------
// Bug shape
// ---------------------------------------------------------------------------
// Helpers `tripCountOfLoop` and `productOfEnclosingLoops` in
// `mlir-aie/lib/Dialect/Conduit/Transforms/ObjectFifoToConduit.cpp:340-437`
// walk parent ops of the acquire and multiply in static trip counts of every
// enclosing `scf.for` / `scf.parallel`.  Anything that is NOT a recognized
// loop returns `TripStatus::NotALoop` and is skipped silently (see
// `enum class TripStatus` at line 209 and the loop at lines 425-435).
// `scf.if` therefore falls into the NotALoop bucket — the walk steps
// transparently past it and continues up to the enclosing `scf.for`,
// folding in its full trip count as if no conditional existed.
//
// Same dispatch-stall failure mode as the bug_c `emit.count == 1`
// over-stamp that the Task #74 skip block addresses (lines 609-638): the
// shim BD `dma_repeat` ends up larger than the consumer's actual
// per-dispatch acquire count, so the consumer cores stop releasing after
// the conditional misses, the producer waits on a release that never
// comes, and the NPU stalls.
//
// ---------------------------------------------------------------------------
// Geometry pinned here
// ---------------------------------------------------------------------------
//   * Shim → core, single channel `@chan`.
//   * Producer (shim): emit.count = 2 — two `aiex.dma_configure_task_for`
//     ops on `@chan` (IRON gemv-style multi-batch host fan-out, see
//     `infer_iter_count_multi_emission_gemv_pattern.mlir` for the canonical
//     two-emission shape).  BD len = 8 elements; fifo elem
//     = `memref<8xbf16>` → `acquires_per_BD = 8 / 8 = 1`.
//   * Consumer (core 0,2): outer `scf.for` trip = 8.  Body is `scf.if %cond
//     { acquire @chan; release @chan }`.  `%cond` is `arith.cmpi eq` against
//     `arith.remui %i, %c2` — TRUE on even iterations, FALSE on odd —
//     so the runtime acquire count per dispatch is 4, not 8.
//   * Three-factor formula (`inferDmaRepeatForChannel`, lines 595-665) sees
//     trip = 8 (transparent over scf.if), emissions = 2, per-BD = 1:
//         dma_repeat = (8 / 2) / 1 = 4   ← OVER-COUNTED.
//     True per-dispatch acquires = (4 even-iters) → real `dma_repeat` should
//     be 2 if the cond is provably 50%, but Pass A cannot in general prove
//     the firing rate of an `scf.if` body.
//
// emit.count > 1 is required to exercise the bug — emit.count == 1 hits the
// Task #74 shim-BD skip block and never reaches the formula.  The existing
// `infer_iter_count_inside_if_branch.mlir` test exercises the parallel
// "compute-to-compute, emissions = 0" code path with the SAME root cause
// (transparent scf.if walk) and pins `dma_repeat = 6`; both tests must flip
// together when the fix lands.
//
// ---------------------------------------------------------------------------
// Expected post-fix behavior
// ---------------------------------------------------------------------------
// Pass A treats an enclosing `scf.if` whose `%cond` is not provably
// constant as introducing dynamic per-iteration acquire counts and SKIPS
// `dma_repeat` inference with a remark, e.g.:
//   "conduit-objectfifo: dma_repeat inference skipped: conditional acquire
//    under scf.if; per-iteration count not statically known"
// This re-uses the existing skip-with-remark machinery already wired
// through `inferDmaRepeatForChannel` (see the
// `SideStatus::Dynamic` branch at lines 560-563 which calls
// `productOfEnclosingLoops` → `TripStatus::Dynamic` propagation).
//
// ---------------------------------------------------------------------------
// Fix design (where the source change lands)
// ---------------------------------------------------------------------------
// Cleanest home: `productOfEnclosingLoops` at
// `ObjectFifoToConduit.cpp:419-437`.  Add a check: when `cur` is an
// `scf::IfOp` AND the previous step's child sits inside `cur`'s then- or
// else-region, attempt a constant-fold on `cur.getCondition()`:
//   * fold == true  AND child is in then-region  → continue walking
//     (preserves the always-taken sub-case, which is exactly what
//     `infer_iter_count_inside_if_branch.mlir` pins via
//     `arith.constant true`).
//   * fold == false AND child is in else-region  → continue walking
//     (symmetric always-taken).
//   * otherwise → return `TripStatus::Dynamic` (caller emits skip remark).
// Putting it in `productOfEnclosingLoops` (not `tripCountOfLoop`) is right
// because the decision depends on which region the descending child is in,
// and that information is naturally available during the parent-walk.
//
// ---------------------------------------------------------------------------
// Cross-impact when the fix lands
// ---------------------------------------------------------------------------
//   * `test/Dialect/Conduit/infer_iter_count_inside_if_branch.mlir`: also
//     pins the over-count today (`dma_repeat = 6`), but uses
//     `%true = arith.constant true` as the guard.  Under the constant-fold-
//     aware fix proposed above the existing test continues to pass (the
//     always-taken sub-case survives).  If the fix lands as the simpler
//     "any scf.if = Dynamic" variant (no constant fold), the existing
//     test must be updated to expect a Dynamic skip remark + no
//     `dma_repeat` attribute.  Decide at fix time.
//   * No other `infer_iter_count_*.mlir` test exercises an `scf.if`
//     inside a core (verified via Glob over the
//     `test/Dialect/Conduit/infer_iter_count_*.mlir` set 2026-04-24).
//   * Real-world cross-ref: IRON `gemm` uses Python-level
//     `if rtp_n_tiles_per_core > 1: loop = range_(...)` which can emit
//     conditional IR shaped like this if the Python branch is replaced
//     with an MLIR-level `scf.if`.

// CHECK-LABEL: module @infer_scf_if_conditional_acquire_overcount
// CHECK: conduit.create @chan
// CHECK-SAME: dma_repeat = 4

module @infer_scf_if_conditional_acquire_overcount {
  aie.device(npu1) {
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)

    aie.objectfifo @chan(%tile_0_0, {%tile_0_2}, 2 : i32)
        : !aie.objectfifo<memref<8xbf16>>

    // Consumer: outer trip = 8; acquire fires only on even iterations
    // (4 of 8) — but Pass A walks past scf.if transparently and folds in
    // the full outer trip = 8, then divides by emissions = 2 → stamps
    // dma_repeat = 4 (over-counted; should be Dynamic-skip).
    aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %c8 = arith.constant 8 : index
      scf.for %i = %c0 to %c8 step %c1 {
        %parity = arith.remui %i, %c2 : index
        %cond = arith.cmpi eq, %parity, %c0 : index
        scf.if %cond {
          %sub = aie.objectfifo.acquire @chan (Consume, 1)
              : !aie.objectfifosubview<memref<8xbf16>>
          %elem = aie.objectfifo.subview.access %sub[0]
              : !aie.objectfifosubview<memref<8xbf16>> -> memref<8xbf16>
          aie.objectfifo.release @chan (Consume, 1)
        }
      }
      aie.end
    }

    // emit.count = 2 — two aiex.dma_configure_task_for on @chan, each with
    // BD len = 8 elements (fifo elem = 8 → acquires_per_BD = 1).  Choosing
    // emit.count > 1 deliberately: emit.count == 1 hits the Task #74
    // single-shim-BD skip block and never reaches the three-factor formula
    // where the bug manifests.
    aie.runtime_sequence(%batch0: memref<8xbf16>, %batch1: memref<8xbf16>) {
      %t0 = aiex.dma_configure_task_for @chan {
        aie.dma_bd(%batch0 : memref<8xbf16>, 0, 8,
          [<size = 1, stride = 0>, <size = 1, stride = 0>,
           <size = 1, stride = 0>, <size = 8, stride = 1>])
            {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%t0)
      aiex.dma_await_task(%t0)
      aiex.dma_free_task(%t0)
      %t1 = aiex.dma_configure_task_for @chan {
        aie.dma_bd(%batch1 : memref<8xbf16>, 0, 8,
          [<size = 1, stride = 0>, <size = 1, stride = 0>,
           <size = 1, stride = 0>, <size = 8, stride = 1>])
            {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%t1)
      aiex.dma_await_task(%t1)
      aiex.dma_free_task(%t1)
    }
  }
}
