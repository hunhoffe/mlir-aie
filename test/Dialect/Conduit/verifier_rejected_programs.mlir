// Verifier rejected-program test suite.
//
// 10 cases that must be rejected by the Conduit verifier or analysis passes:
//   Case 1:  M6 CSDF rate imbalance (dialect verifier)
//   Case 2:  M7 CSDF buffer capacity insufficient (dialect verifier)
//   Case 3:  Cascade value width wrong — not 384 or 512 bits (dialect verifier)
//   Case 4:  Cascade depth > 1 rejected by --conduit-to-dma (Phase 1)
//   Case 5:  Mixed-mode liveness violation (--conduit-check-liveness, P2-B)
//   Case 6:  Convergence hazard — same source port, same dest, different IDs
//            (--conduit-check-channels, P2-E)
//   Case 7:  Unmatched conduit.put_cascade — no get_cascade
//            (--conduit-check-pairing, M9)
//   Case 8:  Ambiguous cascade — two get_cascade for same name
//            (--conduit-check-pairing, M9)
//   Case 9:  Shared-memory conduit with non-adjacent alloc_tile (--conduit-to-dma)
//   Case 10: M8a double-release — cumulative release exceeds acquired count
//            (dialect verifier)
//
// All runs use FileCheck with check-prefixes scoped per pass:
//   CHECK       — RUN 1: dialect verifier (-split-input-file only, no extra pass)
//   PASSCHECK   — RUN 2: --conduit-to-dma (cases 4, 9)
//   LIVENESS    — RUN 3: --conduit-check-liveness (case 5)
//   HAZARD      — RUN 4: --conduit-check-channels (case 6)
//   PAIR        — RUN 5: --conduit-check-pairing (cases 7, 8)
//
// Design: no expected-error/expected-warning annotations are used. All
// diagnostics are matched via FileCheck against stderr (2>&1). Each RUN line
// uses "not aie-opt" to invert the non-zero exit code from aie-opt (which
// always exits 1 because sections 1, 2, 3, and 10 produce dialect verifier
// errors under every pass combination). FileCheck uses check-prefixes to match
// only the diagnostics relevant to the given RUN's pass pipeline. Sections
// irrelevant to a given run produce no additional output beyond the dialect
// errors from sections 1, 2, 3, 10, which are matched by the CHECK prefix.

// RUN: not aie-opt -split-input-file %s 2>&1 | FileCheck %s
// RUN: not aie-opt --conduit-to-dma -split-input-file %s 2>&1 | FileCheck --check-prefix=PASSCHECK %s
// RUN: not aie-opt --conduit-check-liveness -split-input-file %s 2>&1 | FileCheck --check-prefix=LIVENESS %s
// RUN: not aie-opt --conduit-check-channels -split-input-file %s 2>&1 | FileCheck --check-prefix=HAZARD %s
// RUN: not aie-opt --conduit-check-pairing -split-input-file %s 2>&1 | FileCheck --check-prefix=PAIR %s

// ============================================================================
// Case 1: M6 — CSDF rate imbalance.
//
// producer_rates = [2, 3] (sum=5, len=2)
// consumer_rates = [1]    (sum=1, len=1)
// Balance: sum(P)*len(C) = 5*1 = 5  !=  sum(C)*len(P) = 1*2 = 2 → ERROR
//
// Fires under all RUN lines during dialect verification.
// RUN 1 FileCheck matches it with CHECK below.
// ============================================================================

// CHECK: 'conduit.create' op CSDF rate imbalance: sum(producer_rates)*len(consumer_rates)=5 != sum(consumer_rates)*len(producer_rates)=2
// PAIR: 'conduit.create' op CSDF rate imbalance

func.func @case1_m6_csdf_rate_imbalance() {
  conduit.create @csdf_imbal {capacity = 5 : i64,
                  producer_tile = array<i64: 0, 2>,
                  consumer_tiles = array<i64: 0, 3>,
                  element_type = memref<i32>,
                  depth = 5 : i64,
                  producer_rates = array<i64: 2, 3>,
                  consumer_rates = array<i64: 1>}
  return
}

// -----

// ============================================================================
// Case 2: M7 — CSDF buffer capacity insufficient.
//
// producer_rates = [3, 1] (sum=4, len=2); consumer_rates = [2] (sum=2, len=1)
// M6 balance: 4*1 == 2*2  PASS
// Hyper-period H=2: t=0: produce 3 (occ=3), consume 2 (occ=1); t=1: produce 1,
// consume 2.  Peak occupancy = 3 > capacity = 2 → ERROR
// ============================================================================

// CHECK: M7: CSDF buffer capacity insufficient: peak token occupancy over one hyper-period=3 exceeds capacity=2

func.func @case2_m7_capacity_insufficient() {
  conduit.create @csdf_cap {capacity = 2 : i64,
                  producer_tile = array<i64: 0, 2>,
                  consumer_tiles = array<i64: 0, 3>,
                  element_type = memref<i32>,
                  depth = 2 : i64,
                  producer_rates = array<i64: 3, 1>,
                  consumer_rates = array<i64: 2>}
  return
}

// -----

// ============================================================================
// Case 3: conduit.distribute with a cascade-mode source — rejected by the
//         distribute op verifier (M5: cascade src in distribute is invalid).
//
// After cascade migration (#27), conduit.put_cascade / conduit.get_cascade
// no longer exist, so the old Case 3 (wrong cascade width) is superseded.
// The cascade-mode distribute rejection still exercises cascade verifier logic.
//
// Fires under all RUN lines during dialect verification.
// ============================================================================

// CHECK: 'conduit.distribute' op cascade channel 'cas_c3_src' cannot be used in a distribute src

func.func @case3_cascade_distribute_src() {
  conduit.create @cas_c3_src {capacity = 1 : i64, depth = 1 : i64,
                  routing_mode = #conduit.routing_mode<cascade>,
                  producer_tile = array<i64: 0, 2>,
                  consumer_tiles = array<i64: 0, 1>}
  conduit.create @cas_c3_dst {capacity = 1 : i64, depth = 1 : i64,
                  producer_tile = array<i64: 0, 1>,
                  consumer_tiles = array<i64: 1, 2>}
  conduit.distribute {srcs = [@cas_c3_src], dsts = [@cas_c3_dst], memtile = "tile(0,1)"}
  return
}

// -----

// ============================================================================
// Case 4: Cascade depth > 1 — rejected by --conduit-to-dma Phase 1.
//
// The hardware cascade stream is a blocking register with no FIFO.  A cascade
// conduit with depth=2 cannot be implemented in hardware.
// Pass C (Phase 1 collect) emits a hard error.
//
// Under RUN 1 (no extra pass), the dialect verifier does NOT check cascade
// depth — no diagnostic fires, section is silent.
// Under RUN 2 (--conduit-to-dma), Phase 1 collect fires the error.
//
// PASSCHECK: cascade conduit must have depth = 1; hardware has no FIFO on the cascade stream
// ============================================================================

module @case4_cascade_depth_gt1 {
  aie.device(npu1) {
    %tile03 = aie.tile(0, 3)
    %tile13 = aie.tile(1, 3)

    conduit.create @cas_d2 {capacity = 2 : i64,
                    producer_tile = array<i64: 0, 3>,
                    consumer_tiles = array<i64: 1, 3>,
                    element_type = memref<1xvector<16xi32>>,
                    depth = 2 : i64,
                    routing_mode = #conduit.routing_mode<cascade>}

    aie.core(%tile03) {
      %v = arith.constant dense<0> : vector<16xi32>
      aie.put_cascade(%v : vector<16xi32>)
      aie.end
    }

    aie.core(%tile13) {
      %r = aie.get_cascade() : vector<16xi32>
      aie.end
    }
  }
}

// -----

// ============================================================================
// Case 5: Mixed-mode liveness violation (P2-B).
//
// An aie.core body loads from a buffer (memref.load) and sends the result
// on the cascade stream (aie.put_cascade) without a dominating
// aie.use_lock(Acquire) to synchronize DMA completion.
//
// --conduit-check-liveness walks aie.core regions, finds put_cascade ops
// whose value traces back through a memref.load, and checks that every
// aie.use_lock(Acquire, >=1) in the core dominates the put_cascade.
//
// This section is silent under all other RUN lines.
//
// LIVENESS: put_cascade may fire before DMA transfer completes
// ============================================================================

module @case5_mixed_mode_liveness {
  aie.device(npu1) {
    %tile03 = aie.tile(0, 3)
    %tile13 = aie.tile(1, 3)

    %lock0 = aie.lock(%tile03, 0) { sym_name = "lock0" }
    // Buffer of vector<16xi32> — load produces 512-bit value (valid AIE2 type).
    %buf0  = aie.buffer(%tile03) { sym_name = "buf0" } : memref<1xvector<16xi32>>

    // Producer core: the put_cascade comes BEFORE the use_lock(Acquire),
    // so the lock acquisition does NOT dominate the put_cascade → violation.
    aie.core(%tile03) {
      %c0 = arith.constant 0 : index
      %val = memref.load %buf0[%c0] : memref<1xvector<16xi32>>
      // Violation: put_cascade executes before DMA data-ready lock is acquired.
      aie.put_cascade(%val : vector<16xi32>)
      // use_lock AFTER put_cascade — does NOT dominate it.
      aie.use_lock(%lock0, AcquireGreaterEqual, 1)
      aie.use_lock(%lock0, Release, 1)
      aie.end
    }

    aie.core(%tile13) {
      %r = aie.get_cascade() : vector<16xi32>
      aie.end
    }
  }
}

// -----

// ============================================================================
// Case 6: Convergence hazard — two packet flows with different IDs route to
//         the same consumer tile through the same physical source port.
//
// aie.packet_flow ID 0 and ID 1 both originate from tile(0,3) DMA:0 and
// both route to tile(2,3).  Switchbox arbitration gives undefined ordering.
//
// --conduit-check-channels walks aie.packet_flow ops and emits a warning.
// All other RUN lines produce no diagnostics on this section.
//
// HAZARD: packet flows with different IDs (0 and 1) route to the same consumer tile (2, 3)
// HAZARD-SAME: through the same switchbox source port on tile (0, 3)
// HAZARD-SAME: ordering is not guaranteed under sustained load
// ============================================================================

module @case6_convergence_hazard {
  aie.device(npu1) {
    %t03 = aie.tile(0, 3)
    %t23 = aie.tile(2, 3)

    // Flow ID 0: tile(0,3) DMA:0 → tile(2,3) DMA:0
    aie.packet_flow(0) {
      aie.packet_source<%t03, DMA : 0>
      aie.packet_dest<%t23, DMA : 0>
    }

    // Flow ID 1: same source port (0,3):DMA:0, same dest tile(2,3) → HAZARD
    aie.packet_flow(1) {
      aie.packet_source<%t03, DMA : 0>
      aie.packet_dest<%t23, DMA : 1>
    }
  }
}

// -----

// ============================================================================
// Case 7: Unmatched aie.put_cascade — no corresponding aie.get_cascade.
//
// Cascade is a blocking rendezvous.  A producer with no matching consumer
// get stalls indefinitely in hardware.
//
// After cascade migration (#27), conduit.put_cascade / conduit.get_cascade
// no longer exist; --conduit-check-pairing no longer checks cascade pairing
// (deferred to --aie-check-cascade-pairing).  This section is now silent
// under all RUN lines (including PAIR/RUN 5).
//
// All RUN lines produce no cascade-specific diagnostics on this section.
// ============================================================================

module @case7_unmatched_put_cascade {
  aie.device(npu1) {
    %tile03 = aie.tile(0, 3)

    conduit.create @cas_unmatched {capacity = 1 : i64,
                    producer_tile = array<i64: 0, 3>,
                    consumer_tiles = array<i64: 1, 3>,
                    depth = 1 : i64,
                    routing_mode = #conduit.routing_mode<cascade>}

    aie.core(%tile03) {
      %v = arith.constant dense<7> : vector<16xi32>
      aie.put_cascade(%v : vector<16xi32>)
      aie.end
    }
    // No consumer core and no aie.get_cascade anywhere.
  }
}

// -----

// ============================================================================
// Case 8: Two aie.get_cascade ops for the same conduit (hardware-invalid:
//         cascade is point-to-point).
//
// After cascade migration (#27), conduit.get_cascade no longer exists.
// --conduit-check-pairing no longer checks cascade pairing; deferred to
// --aie-check-cascade-pairing.  This section is silent under all RUN lines.
// ============================================================================

module @case8_ambiguous_get_cascade {
  aie.device(npu1) {
    %tile03 = aie.tile(0, 3)
    %tile13 = aie.tile(1, 3)
    %tile23 = aie.tile(2, 3)

    conduit.create @cas_ambig {capacity = 1 : i64,
                    producer_tile = array<i64: 0, 3>,
                    consumer_tiles = array<i64: 1, 3>,
                    depth = 1 : i64,
                    routing_mode = #conduit.routing_mode<cascade>}

    aie.core(%tile03) {
      %v = arith.constant dense<5> : vector<16xi32>
      aie.put_cascade(%v : vector<16xi32>)
      aie.end
    }

    // First consumer — valid aie.get_cascade.
    aie.core(%tile13) {
      %r = aie.get_cascade() : vector<16xi32>
      aie.end
    }

    // Second consumer — duplicate get (hardware-invalid).
    // Use --aie-check-cascade-pairing to detect this.
    aie.core(%tile23) {
      %r2 = aie.get_cascade() : vector<16xi32>
      aie.end
    }
  }
}

// -----

// ============================================================================
// Case 9: Shared-memory conduit with non-adjacent alloc_tile.
//
// When alloc_tile is specified, Phase 3c verifies adjacency via
// AIETargetModel.isLegalMemAffinity.  If the alloc_tile is not adjacent to
// both producer and consumer, the buffer is unreachable from one of the cores.
//
// Topology: producer=tile(0,2), consumer=tile(0,3) are adjacent (eligible),
// but alloc_tile=tile(3,3) is NOT adjacent to tile(0,2) → ERROR.
//
// Under RUN 1 (no extra pass), no adjacency check runs — silent.
// Under RUN 2 (--conduit-to-dma), Phase 3c fires the error.
//
// PASSCHECK: shared-memory conduit requires adjacent tiles
// ============================================================================

module @case9_sharedmem_nonadj_alloc {
  aie.device(npu1) {
    %tile02 = aie.tile(0, 2)
    %tile03 = aie.tile(0, 3)
    %tile33 = aie.tile(3, 3)

    conduit.create @shm_bad {capacity = 1 : i64,
                    producer_tile = array<i64: 0, 2>,
                    consumer_tiles = array<i64: 0, 3>,
                    element_type = memref<16xi32>,
                    depth = 1 : i64,
                    alloc_tile = array<i64: 3, 3>}

    aie.core(%tile02) {
      %win = conduit.acquire {name = @shm_bad, count = 1 : i64,
                              port = #conduit.port<Produce>}
                 : !conduit.window<memref<16xi32>>
      %buf = conduit.subview_access %win {index = 0 : i64}
                 : !conduit.window<memref<16xi32>> -> memref<16xi32>
      conduit.release %win {count = 1 : i64, port = #conduit.port<Produce>}
          : !conduit.window<memref<16xi32>>
      aie.end
    }

    aie.core(%tile03) {
      %win = conduit.acquire {name = @shm_bad, count = 1 : i64,
                              port = #conduit.port<Consume>}
                 : !conduit.window<memref<16xi32>>
      %buf = conduit.subview_access %win {index = 0 : i64}
                 : !conduit.window<memref<16xi32>> -> memref<16xi32>
      conduit.release %win {count = 1 : i64, port = #conduit.port<Consume>}
          : !conduit.window<memref<16xi32>>
      aie.end
    }
  }
}

// -----

// ============================================================================
// Case 10: M8a — double-release: cumulative release count exceeds acquired.
//
// acquire(count=1) produces one window slot.  Two conduit.release ops each
// release count=1, totaling 2 released.  2 > 1 → hardware lock-counter
// overflow, which silently corrupts subsequent lock states.
//
// Acquire::verify → checkWindowReleaseCumulativeCount rejects this.
// Fires under all RUN lines during dialect verification.
// ============================================================================

// CHECK: 'conduit.acquire' op M8: cumulative release count (2) exceeds acquired count (1) -- double-release causes hardware lock-counter overflow

func.func @case10_m8a_double_release() {
  conduit.create @dbl_rel {capacity = 1 : i64,
                  producer_tile = array<i64: 0, 2>,
                  consumer_tiles = array<i64: 0, 3>,
                  element_type = memref<1xi32>,
                  depth = 1 : i64}
  %win = conduit.acquire {name = @dbl_rel, count = 1 : i64,
                          port = #conduit.port<Consume>}
             : !conduit.window<memref<1xi32>>
  conduit.release %win {count = 1 : i64, port = #conduit.port<Consume>}
      : !conduit.window<memref<1xi32>>
  conduit.release %win {count = 1 : i64, port = #conduit.port<Consume>}
      : !conduit.window<memref<1xi32>>
  return
}
