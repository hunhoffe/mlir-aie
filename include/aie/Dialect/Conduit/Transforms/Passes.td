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
// Declares the three Conduit transformation passes:
//   Pass A: --objectfifo-to-conduit  (ObjectFIFO → Conduit IR)
//   Pass B: --air-channel-to-conduit (AIR Channel → Conduit IR)
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
        → conduit.create + conduit.annotate (producer_tile, consumer_tile_N)

      aie.objectfifo.link [@srcs] -> [@dsts] ([jo][do])
        → conduit.link {srcs, dsts, mode, offsets, memtile}

      aie.objectfifo.acquire @name (Role, count)
        → conduit.acquire {name, count}

      aie.objectfifo.release @name (Role, count)
        → conduit.release {name, count}

      aie.objectfifo.subview.access %sv[i]
        → conduit.subview_access {name, index}

    Tile coordinate annotations are carried forward as conduit.annotate ops
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
// Pass B — AIR Channel → Conduit
//===----------------------------------------------------------------------===//

def AirChannelToConduit : Pass<"air-channel-to-conduit", "::mlir::ModuleOp"> {
  let summary = "Lift air.channel.put/get ops into Conduit IR (Tier 3 memref-DMA ops)";
  let description = [{
    Converts the AIR Channel abstraction layer into Conduit IR Tier 3 ops:

      air.channel @name [1, 1]
        → conduit.create {name, slot_elems =1, element_type=<inferred>,
                          depth=1}

      air.channel.put async [%deps] @chan(%buf[offsets][sizes][strides])
        → conduit.put_memref_async {name, num_elems, offsets, sizes, strides}
                : !conduit.dma.token

      air.channel.get async [%deps] @chan(%buf[offsets][sizes][strides])
        → conduit.get_memref_async {name, num_elems, offsets, sizes, strides}
                : !conduit.dma.token

      air.wait_all async [%deps]  →  conduit.wait_all_async %deps
      air.wait_all       [%deps]  →  conduit.wait_all %deps

    The AIR dialect operates on generic unregistered ops when parsed via
    --allow-unregistered-dialect.  This pass matches by op name string so it
    does not require the AIR dialect library to be linked into aie-opt.

    When `infer-rates` is true (the default), the pass automatically runs
    --conduit-infer-rates at the end to attach producer_rates/consumer_rates
    to conduit.create ops, enabling M6/M7 CSDF verification.  Pass
    `--air-channel-to-conduit-infer-rates=false` to disable.

    Known limitations:
    - Only scalar [1,1] channels are handled (Task #32 established that
      SPMD specialization always produces [1,1] before lowering).
    - Channel size/shape information is extracted from the first put/get
      memref operand type; if no put/get is found, capacity defaults to 1.
    - The air.async.token → !conduit.dma.token mapping relies on SSA
      value threading; all users of the old token must be updated.
  }];

  let constructor = "xilinx::conduit::createAirChannelToConduitPass()";

  let options = [
    Option<"inferRates", "infer-rates", "bool", /*default=*/"false",
           "Run --conduit-infer-rates automatically at the end of the pass "
           "to attach CSDF producer_rates/consumer_rates to conduit.create "
           "ops, enabling M6/M7 verification. Default is false because "
           "air.channel slot_elems (broadcast fan-out) does not match the CSDF "
           "buffer-slot semantics that M7 checks; enabling rates on most "
           "air.channel programs triggers spurious M7 failures.">,
  ];

  let dependentDialects = [
    "xilinx::conduit::ConduitDialect",
    "::mlir::memref::MemRefDialect",
  ];
}

//===----------------------------------------------------------------------===//
// Pass C — Conduit → BD/DMA
//===----------------------------------------------------------------------===//

def ConduitToDMA : Pass<"conduit-to-dma", "::mlir::ModuleOp"> {
  let summary = "Lower Conduit IR to aie.dma_bd / aie.lock / aie.buffer hardware ops";
  let description = [{
    Converts Conduit IR directly to the raw AIE hardware programming model,
    replacing the existing aie-objectFifoStatefulTransform path:

      conduit.create {name, capacity} + tile annotations
        → aie.buffer (depth-many) + aie.lock (prod + cons) per tile

      conduit.link {srcs, dsts, mode, memtile, offsets}
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

    Implements the same LockAnalysis + DMAChannelAnalysis + buffer-allocation
    logic as AIEObjectFifoStatefulTransform, operating on Conduit IR instead
    of ObjectFIFO IR.

    A correct implementation produces identical aie.dma_bd / aie.lock /
    aie.buffer / aie.flow counts as the existing stateful transform pass.
    Differences reveal either bugs or optimization opportunities.
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
// Air Channel Index Flattener — air-channel-flatten-indices
//===----------------------------------------------------------------------===//

def AirChannelIndexFlattener : Pass<"air-channel-flatten-indices", "::mlir::ModuleOp"> {
  let summary = "Flatten multi-dimensional air.channel indices to scalar channels";
  let description = [{
    Rewrites multi-dimensional air.channel declarations and their put/get ops
    to use scalar channels.

    air.channel @foo [M, N] declares M×N independent channels.  SPMD
    specialization (air-specialize-channel-broadcast) is supposed to reduce
    these to [1,1] before Pass B runs, but for programs that still carry
    multi-dimensional channels, this pass performs the flattening:

      air.channel @foo [M, N]
        → M×N air.channel declarations @foo_0_0, @foo_0_1, ..., @foo_{M-1}_{N-1}

      air.channel.put @foo[%c0, %c1]  (with arith.constant indices)
        → air.channel.put @foo_0_1

      air.channel.get @foo[%ci, %cj]  (with arith.constant indices)
        → air.channel.get @foo_{i}_{j}

    Dynamic indices (loop induction variables, block arguments, etc.) cause a
    hard error because the destination channel cannot be determined statically.

    This pass is a prerequisite for --air-channel-to-conduit when multi-
    dimensional channels are present.  It should run BEFORE Pass B.

    Run with:  aie-opt --air-channel-flatten-indices <input.mlir>
  }];

  let constructor = "xilinx::conduit::createAirChannelIndexFlattenerPass()";

  let dependentDialects = [];
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
// --conduit-materialize-buffers — emit aie.buffer + conduit.register_buffers
//===----------------------------------------------------------------------===//

def ConduitMaterializeBuffers
    : Pass<"conduit-materialize-buffers", "::mlir::ModuleOp"> {
  let summary = "Emit aie.buffer ops and conduit.register_buffers for each "
                "Conduit channel";
  let description = [{
    Runs after --conduit-depth-promote.  For each conduit.create whose consumer
    tile can be inferred from conduit.acquire{port=Consume} ops inside aie.core
    bodies:

      1. Computes buffer count = max(depth, window_size + 1).
      2. Emits `aie.buffer × buffer_count` on the consumer tile.
      3. Emits `conduit.register_buffers {name = @chan, buffers = [...]}`.

    Enables --conduit-place-buffers to set mem_bank before Pass C runs, which
    is impossible in Pass C's single-pass buffer allocation.  Also allows
    expert users to supply hand-allocated buffers via conduit.register_buffers
    before running --conduit-to-dma.

    Channels that already have a conduit.register_buffers op are skipped.
    Channels with no compute consumer (shim-only or DDR) are skipped.
    Channels with depth = 0 are skipped (run --conduit-depth-promote first).

    Standard pipeline position:
      --conduit-depth-promote → --conduit-materialize-buffers
        → [--conduit-place-buffers] → --conduit-to-dma
        → --aie-assign-buffer-addresses
  }];
  let constructor =
      "xilinx::conduit::createConduitMaterializeBuffersPass()";
  let dependentDialects = [
    "xilinx::conduit::ConduitDialect",
    "xilinx::AIE::AIEDialect",
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
    Runs between --conduit-materialize-buffers and --aie-assign-buffer-addresses.
    For each set of buffers registered via conduit.register_buffers, assigns:

      buf[i].mem_bank = i % numBanks

    so that consecutive FIFO slots land in different SRAM banks.  This
    eliminates the DMA↔core bank conflicts that degrade throughput when
    producer and consumer access the same bank simultaneously.

    Requires --conduit-materialize-buffers to have run first.

    Enables the Exp A2 three-way comparison:
      basic-sequential vs bank-aware vs conduit-place-buffers.
  }];
  let constructor = "xilinx::conduit::createConduitPlaceBuffersPass()";
  let dependentDialects = [
    "xilinx::conduit::ConduitDialect",
    "xilinx::AIE::AIEDialect",
  ];
}

#endif // CONDUIT_PASSES
