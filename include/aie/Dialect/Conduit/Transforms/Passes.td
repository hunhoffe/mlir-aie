//===- Passes.td - Conduit dialect passes ------------------*- tablegen -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2026 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//
//
// Declares the Conduit transformation passes:
//   Pass A: --objectfifo-to-conduit  (ObjectFIFO → Conduit IR)
//   Pass C: --conduit-to-dma         (Conduit IR → aie.dma_bd / aie.lock)
//
//===----------------------------------------------------------------------===//

#ifndef CONDUIT_PASSES
#define CONDUIT_PASSES

include "mlir/Pass/PassBase.td"

//===----------------------------------------------------------------------===//
// Pass A — ObjectFIFO → Conduit
//===----------------------------------------------------------------------===//

def ObjectFifoToConduit : Pass<"objectfifo-to-conduit", "::mlir::ModuleOp"> {
  let summary = "Lift aie.objectfifo.* ops into Conduit IR";
  let description = [{
    Converts the ObjectFIFO abstraction layer into Conduit IR:

      aie.objectfifo @name (prod, {cons}, depth)
        → conduit.create (tile coords as typed attributes on the op)

      aie.objectfifo.link [@srcs] -> [@dsts] ([jo][do])
        → conduit.scatter or conduit.gather {src/srcs, dst/dsts}

      aie.objectfifo.acquire @name (Role, count)
        → conduit.acquire {name, count}

      aie.objectfifo.release @name (Role, count)
        → conduit.release {name, count}

      aie.objectfifo.subview.access %sv[i]
        → conduit.subview_access {name, index}

    Tile coordinates are carried as typed attributes on conduit.create ops
    so that Pass C (--conduit-to-dma) can reconstruct the hardware placement.

    This pass is a text-level structural transformation; it does not require
    the AIE ObjectFIFO stateful transform to have run first.
  }];

  let constructor = "xilinx::conduit::createObjectFifoToConduitPass()";

  let options = [
    Option<"inferRates", "infer-rates", "bool", /*default=*/"false",
           "Attach CSDF producer_rates/consumer_rates from objectfifo acquire "
           "patterns (single-consumer fifos only). Multi-consumer fifos are "
           "skipped with a remark. Default false to preserve existing behavior.">,
  ];

  let dependentDialects = [
    "xilinx::conduit::ConduitDialect",
    "xilinx::AIE::AIEDialect",
    "::mlir::memref::MemRefDialect",
  ];
}

//===----------------------------------------------------------------------===//
// Pass C — Conduit → BD/DMA
//===----------------------------------------------------------------------===//

def ConduitToDMA : Pass<"conduit-to-dma", "::mlir::ModuleOp"> {
  let summary = "Lower Conduit IR to aie.dma_bd / aie.lock / aie.buffer hardware ops";
  let description = [{
    Converts Conduit IR directly to the raw AIE hardware programming model:

      conduit.create {name, capacity} + tile annotations
        → aie.buffer (depth-many) + aie.lock (prod + cons) per tile

      conduit.scatter / conduit.gather
        → MemTile DMA BD chain: aie.dma_start, aie.dma_bd, aie.next_bd,
          aie.use_lock inside aie.memtile_dma

      conduit.acquire {name, count}
        → aie.use_lock(%cons_lock, AcquireGreaterEqual, count)

      conduit.release {name, count}
        → aie.use_lock(%prod_lock, Release, count)

      conduit.subview_access {name, index}
        → buffer reference via index into buffersPerConduit[name]

      conduit.create (shim-tile producer)
        → aie.shim_dma_allocation + aie.flow

    Performs LockAnalysis + DMAChannelAnalysis + buffer allocation on Conduit
    IR.
  }];

  let constructor = "xilinx::conduit::createConduitToDMAPass()";

  let dependentDialects = [
    "xilinx::conduit::ConduitDialect",
    "xilinx::AIE::AIEDialect",
  ];
}

//===----------------------------------------------------------------------===//
// Depth Promotion — conduit-depth-promote
//===----------------------------------------------------------------------===//

def ConduitDepthPromote : Pass<"conduit-depth-promote", "::mlir::ModuleOp"> {
  let summary = "Promote eligible depth-1 conduits to depth-2 for compute-DMA overlap";
  let description = [{
    Experimental: not validated on hardware. Use as opt-in flag only.

    Scans all `conduit.create` ops with depth == 1 and promotes eligible ones
    to depth == 2 (double-buffering), enabling compute-DMA overlap.

    When `csdf=true` and `producer_rates`/`consumer_rates` attributes are present
    on a conduit.create, the pass uses the CSDFa minimum buffer depth formula
    (Denolf 2007, Koek 2016 §4) instead of the fixed depth-2 heuristic:

      min_depth = ceil(max(P, C) * eta / min(P, C))

    where P = sum(producer_rates), C = sum(consumer_rates), and eta is the target
    efficiency (default 1.0 = stall-free).

    Exclusion criteria (any one disqualifies):
      1. CSDF / cyclostatic access pattern (when csdf=false)
      2. Linked conduit (link source or destination)
      3. No surrounding loop (no overlap benefit without iteration)
      4. Passthrough-only (acquire immediately followed by release, no compute)
      5. Non-uniform acquire/release counts across uses
      6. Memory budget exceeded on target tile
      7. AIE1 lock budget exceeded (AIE1 has 16 locks per tile)
      8. BD budget exceeded (each tile has limited BD slots)

    This pass runs AFTER Pass A or Pass B and BEFORE Pass C.
  }];

  let constructor = "xilinx::conduit::createConduitDepthPromotePass()";

  let options = [
    Option<"csdfa", "csdf", "bool", /*default=*/"false",
           "Use CSDFa minimum buffer depth formula (Denolf 2007) when "
           "producer_rates/consumer_rates are present, instead of skipping "
           "CSDF-annotated conduits. Default false (guarded until "
           "conduit-infer-rates is stable).">,
    Option<"eta", "eta", "double", /*default=*/"1.0",
           "Target efficiency for CSDFa depth formula: "
           "min_depth = ceil(max(P,C) * eta / min(P,C)). "
           "1.0 = stall-free; >1.0 = over-provisioning.">,
  ];

  let dependentDialects = [
    "xilinx::conduit::ConduitDialect",
    "xilinx::AIE::AIEDialect",
  ];
}

//===----------------------------------------------------------------------===//
// Pairing Check — conduit-check-pairing (M9 Phase 2 analysis)
//===----------------------------------------------------------------------===//

def ConduitPairingCheck : Pass<"conduit-check-pairing", "::mlir::ModuleOp"> {
  let summary = "M9 Phase 2: warn when conduit.acquire has no matching release in same block";
  let description = [{
    Optional analysis pass that checks same-block acquire-release pairing for
    Conduit IR window tokens (M9 Phase 2).

    For each conduit.acquire and conduit.wait_window op, the pass checks whether
    a matching conduit.release or conduit.release_async appears in the same basic
    block.  If not, a warning is emitted via op.emitWarning() (capturable by
    FileCheck and suppressed by --verify-diagnostics).

    This pass is OPT-IN and NOT part of the default pipeline.  It fires warnings
    only and never signals pass failure.

    Background: This logic was previously implemented as llvm::errs() output
    inside Acquire::verify() / WaitWindow::verify() to work around MLIR
    diagnostic infinite recursion.  Moving it to a separate pass eliminates
    the recursion risk and makes diagnostics capturable.
  }];

  let constructor = "xilinx::conduit::createConduitPairingCheckPass()";

  let dependentDialects = [
    "xilinx::conduit::ConduitDialect",
  ];
}

//===----------------------------------------------------------------------===//
// Liveness Check — conduit-check-liveness (M11 analysis)
//===----------------------------------------------------------------------===//

def ConduitLivenessCheck : Pass<"conduit-check-liveness", "::mlir::ModuleOp"> {
  let summary = "M11: error when a conduit window lock grant is never released";
  let description = [{
    Analysis pass that verifies every !conduit.window<T> SSA value produced by
    conduit.acquire or conduit.wait_window is eventually released.

    A window value with no release path represents a permanently held hardware
    lock.  The peer (producer waiting to reclaim the buffer slot) will block
    forever, causing hardware deadlock.

    Two release paths are accepted:
      1. Direct SSA release: conduit.release with the window as operand.
      2. Async name release: conduit.release_async on the same channel name
         anywhere in the enclosing function (async releases reference the
         channel by name rather than by SSA value).

    Unlike M9 (conduit-check-pairing), which warns when no release is in the
    SAME BASIC BLOCK, M11 checks the entire function body.  M11 does not fire
    when a cross-block release exists; M9 may fire in that case.

    M11 signals pass failure (not just a warning) because a leaked lock grant
    always causes hardware deadlock — there is no valid program that acquires
    and never releases.

    Limitations:
      - Loop-carried conditional skips (M12 / PostDominatorTree — deferred).
      - Indirect release via name aliasing may suppress a false negative.

    Run with:  aie-opt --conduit-check-liveness <input.mlir>

    This pass is OPT-IN and NOT part of the default pipeline.
  }];

  let constructor = "xilinx::conduit::createConduitLivenessCheckPass()";

  let dependentDialects = [
    "xilinx::conduit::ConduitDialect",
    "::mlir::func::FuncDialect",
  ];
}

//===----------------------------------------------------------------------===//
// Channel Fusion — conduit-fuse-channels
//===----------------------------------------------------------------------===//

def ConduitFuseChannels : Pass<"conduit-fuse-channels", "::mlir::ModuleOp"> {
  let summary = "Annotate non-overlapping conduits on the same tile for DMA channel sharing";
  let description = [{
    Experimental: not validated on hardware. Use as opt-in flag only.

    Addresses the DMA channel exhaustion gap: AIE compute tiles have at most 2
    MM2S and 2 S2MM DMA channels.  When more than two conduits share a tile,
    Pass C cannot assign unique hardware channels.

    Two conduits on the same producer tile can time-share one DMA channel when
    their buffer-window accesses do not overlap in program order.  Formally, if
    every Conduit op for channel A appears before (or after) every Conduit op
    for channel B within a common basic block, they are sequentially
    non-overlapping and can share one hardware channel slot.

    Algorithm (linear-scan interval coloring):
      1. Group conduit.create ops by producer_tile coordinate.
      2. For each tile with >= 2 conduits, compute per-block live intervals
         [first_op_index, last_op_index] for each conduit name.
      3. Run a greedy interval-coloring pass (analogous to linear-scan register
         allocation) to assign non-overlapping conduits to the same group.
      4. Annotate each eligible conduit.create with:
             fused_dma_channel_group = "groupN"
         where N identifies the shared hardware channel group.

    Pass C (--conduit-to-dma) is responsible for interpreting this attribute
    and assigning conduits in the same group to the same hardware channel slot.
    This pass is annotation-only; it does not restructure the IR.

    Correctness guarantee: two conduits annotated with the same group are
    guaranteed to have non-overlapping live intervals in at least one basic
    block.  Pass C must ensure the DMA channel is reprogrammed between uses.

    Limitations:
      - Only single-block live intervals are considered.  Cross-block CFG
        liveness analysis is deferred.
      - Only MM2S (producer-side) grouping is implemented; S2MM consumer-side
        fusion is a symmetric future extension.
      - Shim tiles (row == 0) are excluded; they use a separate shim DMA model.

    Run with:  aie-opt --conduit-fuse-channels <input.mlir>
  }];

  let constructor = "xilinx::conduit::createConduitFuseChannelsPass()";

  let dependentDialects = [
    "xilinx::conduit::ConduitDialect",
  ];
}

//===----------------------------------------------------------------------===//
// Relay Fusion — conduit-fuse-relay
//===----------------------------------------------------------------------===//

def ConduitFuseRelay : Pass<"conduit-fuse-relay", "::mlir::ModuleOp"> {
  let summary = "Fuse gather→scatter relay chains into conduit.transpose";
  let description = [{
    Fuses a gather→scatter relay pair through the same MemTile into a single
    conduit.transpose op, eliminating the intermediate channel and halving the
    MemTile DMA channel usage.

    Match condition:
      1. gather.dst == scatter.src  (connected via the same intermediate channel)
      2. gather.memtile == scatter.memtile  (same MemTile relay point)
      3. The intermediate channel has no other users (no acquire/release/
         put_memref/get_memref ops reference that channel name)
      4. srcs.size() + dsts.size() <= 12  (hardware port budget)
      5. srcs.size() * dsts.size() <= 32  (AIE2 packet ID budget)

    The fused conduit.transpose carries the cross-product of gather and scatter
    offsets (N×M entries).  lock_id and sync_mode are copied from the gather.

    This pass is OPT-IN and NOT part of the default pipeline.

    Run with:  aie-opt --conduit-fuse-relay <input.mlir>
  }];

  let constructor = "xilinx::conduit::createConduitFuseRelayPass()";

  let dependentDialects = [
    "xilinx::conduit::ConduitDialect",
  ];
}

//===----------------------------------------------------------------------===//
// Channel Check — conduit-check-channels (P1-B validation)
//===----------------------------------------------------------------------===//

def ConduitCheckChannels : Pass<"conduit-check-channels", "::mlir::ModuleOp"> {
  let summary = "Validate that no tile exceeds its hardware DMA channel limit";
  let description = [{
    Validation pass that checks hardware DMA channel slot_elems constraints.

    Each AIE tile has a fixed number of DMA channels:
      - AIE2 compute tiles: 2 MM2S + 2 S2MM
      - AIE2 MemTiles:      6 MM2S + 6 S2MM
      - AIE1 compute tiles: 2 MM2S + 2 S2MM

    Each conduit requires one MM2S channel on its producer tile and one S2MM
    channel on each of its consumer tiles.  This pass walks all conduit.create
    ops, groups them by tile, queries the target model for hardware limits, and
    emits an error if any tile exceeds its capacity.

    When --conduit-fuse-channels has been run first, fusion annotations
    (fused_dma_channel_group attribute) are respected: conduits in the same
    group count as one channel, not N.

    Shim tiles (row == 0) are excluded — they use a separate shim DMA model.

    Run AFTER Pass A or Pass B and optionally AFTER --conduit-fuse-channels.
    Run BEFORE Pass C to catch over-commitment early.

    This pass emits hard errors and signals pass failure on violation.
    It is OPT-IN and NOT part of the default pipeline.
  }];

  let constructor = "xilinx::conduit::createConduitCheckChannelsPass()";

  let dependentDialects = [
    "xilinx::conduit::ConduitDialect",
    "xilinx::AIE::AIEDialect",
  ];
}

//===----------------------------------------------------------------------===//
// Mode Inference — conduit-infer-modes (P2-C)
//===----------------------------------------------------------------------===//

def ConduitInferModes : Pass<"conduit-infer-modes", "::mlir::ModuleOp"> {
  let summary = "Resolve routing_mode=\"any\" conduits to \"circuit\" or \"packet\"";
  let description = [{
    Walks conduit.create ops with routing_mode="any" and resolves them to a
    concrete routing mode by applying the R3 + Step 3.5 decision procedure:

      R3a. Adjacent tiles (isLegalMemAffinity) with no via_DMA override:
           resolve to "circuit".  Pass C Phase 3c will use shared memory,
           consuming no DMA channel.

      R3b. Circuit-mode MM2S DMA channel available on the producer tile:
           resolve to "circuit".  Channel budget is tracked per-tile in
           source order, matching Pass C Phase 4.5a allocation order.

      Step 3.5. All circuit MM2S channels exhausted on the producer tile:
           resolve to "packet" if global packet flow ID budget remains.
           Each consumer endpoint costs one packet ID.

      Step 4. Both circuit and packet modes exhausted: emit a hard error.

    The inference order within a module matches the conduit.create source
    order (same as Pass C), ensuring the circuit/packet split is consistent
    between this pass and subsequent Pass C execution.

    Cascade conduits (routing_mode="cascade") and conduits that already have
    a concrete routing mode ("circuit" or "packet") are skipped.

    Non-"any" conduits that consume DMA channels (circuit-mode, non-adjacent,
    non-shim) are pre-accounted before "any" conduits are resolved, so the
    budget seen by "any" conduits is accurate.

    This pass runs AFTER Pass A or Pass B and BEFORE Pass C.
    It is OPT-IN and NOT part of the default pipeline.

    Run with:  aie-opt --conduit-infer-modes <input.mlir>
  }];

  let constructor = "xilinx::conduit::createConduitInferModesPass()";

  let dependentDialects = [
    "xilinx::conduit::ConduitDialect",
    "xilinx::AIE::AIEDialect",
  ];
}

//===----------------------------------------------------------------------===//
// Dep-Token DAG Check — conduit-check-deps (M12 analysis)
//===----------------------------------------------------------------------===//

def ConduitCheckDeps : Pass<"conduit-check-deps", "::mlir::ModuleOp"> {
  let summary = "M12: error when the dep-token DAG contains a cycle (static deadlock)";
  let description = [{
    Analysis pass that checks the !conduit.dma.token dependency DAG for cycles.

    The $deps operand on conduit.put_memref_async, conduit.get_memref_async, and
    conduit.wait_all_async carries completion tokens that form a dependency DAG:
    an op with %tok in its $deps cannot begin until the op that produced %tok
    has completed.  A cycle in this DAG is a static deadlock — no scheduling
    can resolve a circular completion dependency.

    Algorithm:
      1. Walk all token-producing ops; build adjacency list: dep → successor.
      2. DFS with gray/black coloring over all nodes.
      3. Any back edge (reaching a gray node) → emit hard error and continue
         to report all cycles (not just the first).

    Nodes: every !conduit.dma.token SSA value.
    Edges: $deps operand of put_memref_async / get_memref_async / wait_all_async.

    IMPORTANT — PASSB-DEP-001 coverage gap:
      This pass requires Task #24 (PASSB-DEP-001 fix) for complete coverage.
      Until that fix lands, cycles routed through conduit.wait_all_async may
      not be detected because the $deps operand list may be incomplete.

    This pass emits hard errors and signals pass failure on violation.
    It is OPT-IN and NOT part of the default pipeline.

    Run AFTER Pass B (which emits the dep tokens) and BEFORE Pass C.
    Run with:  aie-opt --conduit-check-deps <input.mlir>
  }];

  let constructor = "xilinx::conduit::createConduitCheckDepsPass()";

  let dependentDialects = [
    "xilinx::conduit::ConduitDialect",
  ];
}

//===----------------------------------------------------------------------===//
// Rate Inference — conduit-infer-rates
//===----------------------------------------------------------------------===//

def ConduitInferRates : Pass<"conduit-infer-rates", "::mlir::ModuleOp"> {
  let summary = "Infer CSDF producer_rates/consumer_rates from put/get num_elems";
  let description = [{
    Walks conduit.create ops that have no producer_rates or consumer_rates
    attribute and infers them from the num_elems attributes of all
    conduit.put_memref_async and conduit.get_memref_async ops that reference
    the same channel name.

    After attachment, the M6 (CSDF balance) and M7 (hyper-period simulation)
    verifiers in Create::verify() fire automatically on the next mlir verify
    step, catching rate imbalances and buffer capacity violations at
    compile time.

    For uniform channels (all puts move P elements, all gets move C elements),
    producer_rates = [P] and consumer_rates = [C] (period-1 SDF).
    For true CSDF, the full phase sequence is attached.

    Skips conduits where:
      - rates are already explicitly set
      - any num_elems is non-constant (remark emitted, no error)
      - only put ops or only get ops are present (asymmetric coverage)

    This pass runs AFTER Pass B (--air-channel-to-conduit) and BEFORE Pass C.
    It is OPT-IN and NOT part of the default pipeline.

    Run with:  aie-opt --conduit-infer-rates <input.mlir>
  }];

  let constructor = "xilinx::conduit::createConduitInferRatesPass()";

  let dependentDialects = [
    "xilinx::conduit::ConduitDialect",
    "xilinx::AIE::AIEDialect",
  ];
}

//===----------------------------------------------------------------------===//
// Tier Check — conduit-check-tiers (M-12 mixed-tier verifier)
//===----------------------------------------------------------------------===//

def ConduitCheckTiers : Pass<"conduit-check-tiers", "::mlir::ModuleOp"> {
  let summary = "M-12: error when Tier 2 and Tier 3 ops reference the same channel in the same aie.core";
  let description = [{
    Analysis pass that rejects programs mixing Tier 2 (acquire/release) and
    Tier 3 (get_memref/put_memref) ops for the same channel name within the
    same aie.core region.

    The bug: Tier 2 ops maintain a rotation counter inside the core body that
    selects the current physical buffer.  Tier 3 ops bypass this counter — they
    issue DMA transfers without advancing it.  If both tiers reference the same
    channel in the same core, the next Tier 2 acquire reads from the wrong
    physical buffer, producing a silent hardware correctness bug.

    Valid cross-endpoint mixing is NOT rejected:
      - Tier 3 on the shim producer + Tier 2 on a compute core consumer is the
        canonical ObjectFIFO shim-to-core pattern and remains valid.
      - Same channel name in different aie.core regions is valid (no shared
        rotation counter across cores).

    This pass emits hard errors and signals pass failure on violation.
    It is OPT-IN and NOT part of the default pipeline.

    Run with:  aie-opt --conduit-check-tiers <input.mlir>
  }];

  let constructor = "xilinx::conduit::createConduitCheckTiersPass()";

  let dependentDialects = [
    "xilinx::conduit::ConduitDialect",
    "xilinx::AIE::AIEDialect",
  ];
}

//===----------------------------------------------------------------------===//
// Ordering Check — conduit-check-ordering (CSDFa static-ordering verifier)
//===----------------------------------------------------------------------===//

def ConduitCheckOrdering : Pass<"conduit-check-ordering", "::mlir::ModuleOp"> {
  let summary = "Verify CSDFa static token ordering for DMA-only programs with rate annotations";
  let description = [{
    Analysis pass verifying CSDFa static token ordering for DMA-only programs
    with rate annotations.  Complements --conduit-check-deps (acyclicity) and
    M6/M7 (balance).  Together: acyclicity + balance + ordering = formally
    verified deadlock-freedom for the DMA-only subset, grounded in Denolf 2007
    Theorems 4-5.

    Scope: DMA-only channels only.  Silently skips any conduit.create whose
    use-def chain contains conduit.acquire or conduit.acquire_async ops —
    cross-tier programs are outside CSDFa scope by design.

    Algorithm:
      1. Walk all conduit.create ops with both producer_rates and consumer_rates.
      2. Filter to DMA-only channels (no window-token ops in the use chain).
      3. Group DMA-only channels by tile (producer and consumer tiles).
      4. For each tile with 2+ DMA-only rated channels, check whether pairs
         of channels fire at overlapping CSDF phases.  If so, the relative
         ordering of their DMA events is undefined — emit a warning.
      5. Provably cyclic ordering (if detected) causes a hard error.

    This pass is OPT-IN and NOT part of the default pipeline.

    Run with:  aie-opt --conduit-check-ordering <input.mlir>
  }];

  let constructor = "xilinx::conduit::createConduitCheckOrderingPass()";

  let dependentDialects = [
    "xilinx::conduit::ConduitDialect",
  ];
}

//===----------------------------------------------------------------------===//
// Operator Fusion — conduit-fuse-operators
//===----------------------------------------------------------------------===//

def ConduitFuseOperators : Pass<"conduit-fuse-operators", "::mlir::ModuleOp"> {
  let summary = "Spatial IRON operator fusion: replace LPDDR5 intermediates with shared-memory conduits";
  let description = [{
    Given two aie.device ops in one module (e.g., GEMV at col=0 and ReLU at
    col=1), eliminates the LPDDR5 intermediate by:

      1. Identifying the output channel of device A (conduit.create whose only
         consumers are shim tiles — exits to LPDDR5).
      2. Identifying the input channel of device B (conduit.create whose
         producer tile is a shim tile — reads from LPDDR5).
      3. Matching channels by element_type (unambiguous for single-output/
         single-input operators).
      4. Offsetting all tile coordinates in device B by +(maxColA + 1) to
         ensure no tile coordinate conflicts when Pass C processes both devices.
      5. Emitting a module-level conduit.create @fused_intermediate_N with:
           routing_mode = "any"  (resolved to shared-memory by --conduit-infer-modes)
           depth = 0             (sentinel; resolved by --conduit-depth-promote)
           producer_tile from A's matched channel
           consumer_tiles from B's matched channel (after offset)
           element_type and producer_rates/consumer_rates propagated from both channels
      6. Deleting the matched conduit.create ops and their aie.shim_dma_allocation ops.
      7. Deleting runtime aiex.dma_configure_task_for/start/await/free ops for the
         intermediate channels (shared-memory path needs no runtime sequence ops).

    The result is a single-PDI design where the GEMV output flows directly to
    the ReLU input through shared tile memory (zero DMA channels, zero shim
    crossings for the intermediate).

    Prerequisites: --objectfifo-to-conduit (Pass A must run first).

    Run before: --conduit-infer-modes --conduit-depth-promote --conduit-to-dma.

    Limitations (POC):
      - Tile column offset is simple +maxColA+1 (no multi-operator chain optimization).
      - Handles 2 devices; N>2 is future work.
      - Shared-memory routing assumes col distance == 1; MemTile relay for
        larger distances is not implemented.
  }];

  let constructor = "xilinx::conduit::createConduitFuseOperatorsPass()";

  let dependentDialects = [
    "xilinx::conduit::ConduitDialect",
    "xilinx::AIE::AIEDialect",
  ];
}

//===----------------------------------------------------------------------===//
// Loop Balance Check — conduit-check-loop-balance (MVE-1)
//===----------------------------------------------------------------------===//

def ConduitCheckLoopBalance : Pass<"conduit-check-loop-balance", "::mlir::ModuleOp"> {
  let summary = "Check that consumer loop trip counts do not exceed conduit dma_repeat";
  let description = [{
    Emits a warning when a conduit.create channel has a finite total DMA send
    count (dma_repeat attribute) and a consumer-side conduit.acquire for that
    channel is inside a statically bounded scf.for whose trip count exceeds
    dma_repeat.

    Background:
      dma_repeat=N on conduit.create means the DMA fires exactly N times total
      (maps to DMAStartOp.repeat_count = N-1 with a non-circular BD chain).
      If the consumer loop iterates T > N times, it will stall after N
      iterations waiting for tokens that never arrive — hardware deadlock.

      Note: bd_repeat (per-BD unroll factor) is NOT checked here because it
      is not the total send count; its effect depends on depth and BD structure.

    The check is conservative: only statically bounded loops (constant lb, ub,
    step) are checked.  Dynamic loops are silently skipped.

    This pass emits warnings only and never signals pass failure.
    It is OPT-IN and NOT part of the default pipeline.

    Run with:  aie-opt --conduit-check-loop-balance <input.mlir>
  }];
  let constructor = "xilinx::conduit::createConduitCheckLoopBalancePass()";
  let dependentDialects = [
    "xilinx::conduit::ConduitDialect",
    "::mlir::scf::SCFDialect",
  ];
}

//===----------------------------------------------------------------------===//
// --conduit-place-buffers — DMA-aware mem_bank assignment
//===----------------------------------------------------------------------===//

def ConduitPlaceBuffers
    : Pass<"conduit-place-buffers", "::mlir::ModuleOp"> {
  let summary = "Assign mem_bank attributes on aie.buffer ops for DMA-aware "
                "bank staggering";
  let description = [{
    Runs between --conduit-to-dma and --aie-assign-buffer-addresses.
    For each group of aie.buffer ops emitted by Pass C (ConduitToDMAAlloc)
    (named `<chan>_cons_buff_<N>`), assigns:

      buf[i].mem_bank = i % numBanks

    so that consecutive FIFO slots land in different SRAM banks.  This
    eliminates the DMA↔core bank conflicts that degrade throughput when
    producer and consumer access the same bank simultaneously.

    Requires --conduit-to-dma to have run first.
  }];
  let constructor = "xilinx::conduit::createConduitPlaceBuffersPass()";
  let dependentDialects = [
    "xilinx::conduit::ConduitDialect",
    "xilinx::AIE::AIEDialect",
  ];
}

//===----------------------------------------------------------------------===//
// DMA Task → Conduit — dma-task-to-conduit
//===----------------------------------------------------------------------===//

def ConduitDmaTaskToConduit
    : Pass<"dma-task-to-conduit", "::mlir::ModuleOp"> {
  let summary = "Convert IRON DMATask ops in aie.runtime_sequence to "
                "conduit.put_memref / conduit.get_memref";
  let description = [{
    Converts DMA task programming ops inside aie.runtime_sequence into
    Conduit IR Tier 3 memref-DMA ops:

      aiex.dma_configure_task_for @chan (MM2S) + aiex.dma_start_task
        → conduit.put_memref {name = @chan, num_elems, offsets, sizes, strides}

      aiex.dma_configure_task_for @chan (S2MM) + aiex.dma_start_task
        → conduit.get_memref {name = @chan, num_elems, offsets, sizes, strides}

      aiex.dma_await_task / aiex.dma_free_task → erased

    Direction (MM2S vs S2MM) is determined by looking up the
    aie.shim_dma_allocation that the dma_configure_task_for references.
    The conduit channel name is read from the shim_dma_allocation's
    conduit_channel attribute (set by Pass A).

    This pass runs AFTER Pass A (--objectfifo-to-conduit) and converts the
    remaining runtime sequence DMA ops into the Conduit abstraction level.
  }];

  let constructor = "xilinx::conduit::createConduitDmaTaskToConduitPass()";

  let dependentDialects = [
    "xilinx::conduit::ConduitDialect",
    "xilinx::AIE::AIEDialect",
  ];
}

//===----------------------------------------------------------------------===//
// Core-Body Fusion — conduit-fuse-core-bodies
//===----------------------------------------------------------------------===//

def ConduitFuseCoreBodies : Pass<"conduit-fuse-core-bodies", "::mlir::ModuleOp"> {
  let summary = "Loop-body IRON operator fusion: compose aie.core bodies connected by intermediate conduits";
  let description = [{
    Given two aie.core ops on the same tile connected by an intermediate
    conduit.create channel, this pass fuses their loop bodies:

      1. Identifies fusable core pairs (same tile, connected by intermediate
         conduit, 1:1 rate-matched, matching loop structure).
      2. Decides intermediate routing: L1 (memref.alloc) if intermediate fits
         in tile L1 SRAM, MemTile relay (conduit.scatter/gather) if L1 pressure,
         skip if neither suffices.
      3. Composes the two core bodies into a single aie.core body by cloning
         Core B's loop body into Core A after the intermediate produce/release.
      4. Replaces intermediate conduit.acquire/release/subview_access ops with
         direct memref.alloc references (L1) or relay buffer references (MemTile).
      5. Deletes the intermediate conduit.create, dead Core B, and associated
         runtime sequence DMA ops.

    Only 1:1 rate-matched (SDF period 1) edges are supported.
    Multi-consumer (broadcast) intermediates are not fused.

    Prerequisites: --objectfifo-to-conduit (Pass A) + --dma-task-to-conduit.
    IRON operators must emit explicit scf.for tile loops.

    Run before: --conduit-infer-modes --conduit-depth-promote --conduit-to-dma.

    This pass is OPT-IN and NOT part of the default pipeline.

    Run with:  aie-opt --conduit-fuse-core-bodies <input.mlir>
  }];

  let constructor = "xilinx::conduit::createConduitFuseCoreBodyPass()";

  let dependentDialects = [
    "xilinx::conduit::ConduitDialect",
    "xilinx::AIE::AIEDialect",
    "::mlir::memref::MemRefDialect",
    "::mlir::scf::SCFDialect",
  ];
}

//===----------------------------------------------------------------------===//
// Device Merge — aie-combine-device
//===----------------------------------------------------------------------===//

def ConduitCombineDevice : Pass<"aie-combine-device", "::mlir::ModuleOp"> {
  let summary = "Merge two aie.device ops connected by matching fusion_group into one";
  let description = [{
    General-purpose device merge pass for IRON operator fusion.

    IRON operators emit separate aie.device blocks per operator. Downstream
    fusion passes (--conduit-fuse-operators, --conduit-fuse-core-bodies) need
    all fusable tiles to reside in a single aie.device block. This pass merges
    two consecutive aie.device ops connected by a conduit.create channel with
    matching `fusion_group` attributes.

    Two modes:
      - **tile-offset** (default, `same-tile=false`): Offsets all tile
        coordinates in device B by `max_col(devA) + 1`. This ensures no tile
        coordinate conflicts when the merged device is processed by downstream
        passes. Used by spatial fusion (--conduit-fuse-operators).
      - **same-tile** (`same-tile=true`): Keeps original tile coordinates.
        Used as a pre-merge step for loop-body fusion
        (--conduit-fuse-core-bodies) where both cores target the same tile.

    The pass:
      1. Finds two consecutive aie.device ops connected by a conduit channel
         with matching `fusion_group` attributes.
      2. Applies tile offset to device B (tile-offset mode only).
      3. Physically moves all ops from device B's body into device A's body.
      4. Merges runtime_sequence bodies (appends devB's sequence args and ops
         to devA's sequence).
      5. Sinks aie.core, aie.mem, and aie.runtime_sequence ops to the end of
         the merged body (before aie.end terminator) to maintain dominance.
      6. Erases the empty device B.

    This pass does NOT erase or rewrite any conduit channels. Channel
    rewriting (renaming, erasing intermediate channels, creating fused
    channels) is left to the downstream fusion pass.

    Run with:  aie-opt --aie-combine-device <input.mlir>
    Same-tile: aie-opt --aie-combine-device="same-tile=true" <input.mlir>
  }];

  let constructor = "xilinx::conduit::createConduitCombineDevicePass()";

  let options = [
    Option<"sameTile", "same-tile", "bool", /*default=*/"false",
           "Keep original tile coordinates (same-tile mode) instead of "
           "offsetting device B tiles by max_col(devA)+1 (tile-offset mode). "
           "Use same-tile=true as a pre-merge for loop-body fusion.">,
  ];

  let dependentDialects = [
    "xilinx::conduit::ConduitDialect",
    "xilinx::AIE::AIEDialect",
  ];
}

#endif // CONDUIT_PASSES
