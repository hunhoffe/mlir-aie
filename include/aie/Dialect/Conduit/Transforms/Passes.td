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
        → conduit.objectfifo_link {srcs, dsts, mode, offsets, memtile}

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
        → conduit.create {name, capacity=1, element_type=<inferred>,
                          depth=1, link_mode=<empty>}

      air.channel.put async [%deps] @chan(%buf[offsets][sizes][strides])
        → conduit.put_memref_async {name, num_elems, offsets, sizes, strides}
                : !conduit.async.token

      air.channel.get async [%deps] @chan(%buf[offsets][sizes][strides])
        → conduit.get_memref_async {name, num_elems, offsets, sizes, strides}
                : !conduit.async.token

      air.wait_all async [%deps]  →  conduit.wait_all_async %deps
      air.wait_all       [%deps]  →  conduit.wait_all %deps

    The AIR dialect operates on generic unregistered ops when parsed via
    --allow-unregistered-dialect.  This pass matches by op name string so it
    does not require the AIR dialect library to be linked into aie-opt.

    Known limitations:
    - Only scalar [1,1] channels are handled (Task #32 established that
      SPMD specialization always produces [1,1] before lowering).
    - Channel size/shape information is extracted from the first put/get
      memref operand type; if no put/get is found, capacity defaults to 1.
    - The air.async.token → !conduit.async.token mapping relies on SSA
      value threading; all users of the old token must be updated.
  }];

  let constructor = "xilinx::conduit::createAirChannelToConduitPass()";

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

      conduit.objectfifo_link {srcs, dsts, mode, memtile, offsets}
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

    Exclusion criteria (any one disqualifies):
      1. CSDF / cyclostatic access pattern
      2. Linked conduit (objectfifo_link source or destination)
      3. No surrounding loop (no overlap benefit without iteration)
      4. Passthrough-only (acquire immediately followed by release, no compute)
      5. Non-uniform acquire/release counts across uses
      6. Memory budget exceeded on target tile
      7. AIE1 lock budget exceeded (AIE1 has 16 locks per tile)
      8. BD budget exceeded (each tile has limited BD slots)

    This pass runs AFTER Pass A or Pass B and BEFORE Pass C.
  }];

  let constructor = "xilinx::conduit::createConduitDepthPromotePass()";

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

#endif // CONDUIT_PASSES
