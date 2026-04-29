//===- ConduitToDMALower.cpp - Phase 6-8: acquire/release + erasure *-C++-*-=//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2026 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//
//
// Phase 6: Lower conduit.acquire/release → aie.use_lock.
//   Step 1: SubviewAccess → buffer replacement (static or dynamic
//   index_switch). Step 2: Release → use_lock + counter increment (collect for
//   deferred erase). Step 3: Erase Release ops. Step 4: Acquire → use_lock +
//   counter init; erase.
//
// Phase 7: Erase remaining Conduit ops (create, wait, wait_all_async).
//
// Phase 8: Lower async acquire/release/wait_window/wait_all.
//   Step 8a: Record acquire_async metadata.
//   Step 8b: Lower wait_window → use_lock.
//   Step 8c: Lower wait_all → use_lock.
//   Step 8a-erase: Erase acquire_async ops.
//   Step 8d: Lower release_async → use_lock + counter increment.
//   Steps 8e-8f: Lower put/get_memref_async inside aie.core → use_lock pair.
//   Steps 8g-8h: Erase put/get memref (sync) ops.
//
//===----------------------------------------------------------------------===//

#include "ConduitToDMACommon.h"

namespace xilinx::conduit {

void lowerPhase(ConduitToDMAState &state) {
  if (!state.deviceOp)
    return;

  mlir::OpBuilder &builder = *state.builder;
  mlir::MLIRContext *ctx = state.ctx;
  mlir::ModuleOp module = state.module;
  const AIE::LockAction acqAction = state.acqAction;

  // -----------------------------------------------------------------------
  // Phase 6: Lower acquire/release inside func bodies → aie.use_lock.
  // -----------------------------------------------------------------------

  // Step 1: SubviewAccess → buffer replacement.
  {
    llvm::SmallVector<SubviewAccess> subviewsToErase;
    module.walk([&](SubviewAccess op) {
      llvm::StringRef conduitName;
      Port acquirePort = Port::Consume;
      if (auto acqOp =
              mlir::dyn_cast_or_null<Acquire>(op.getWindow().getDefiningOp())) {
        conduitName = acqOp.getName();
        acquirePort = acqOp.getPort();
      } else if (auto waitOp = mlir::dyn_cast_or_null<WaitWindow>(
                     op.getWindow().getDefiningOp())) {
        conduitName = waitOp.getName();
        // Derive port from whether the current core is the producer.
        // Default to Consume; switch to Produce if the enclosing CoreOp's
        // tile matches the conduit's producerTileCoord.
        acquirePort = Port::Consume;
        ConduitInfo *winCinfo = state.lookupConduit(conduitName, op);
        if (winCinfo) {
          mlir::Operation *coreParent = op.getOperation()->getParentOp();
          while (coreParent && !mlir::isa<AIE::CoreOp>(coreParent))
            coreParent = coreParent->getParentOp();
          if (coreParent) {
            mlir::Value coreTileVal =
                mlir::cast<AIE::CoreOp>(coreParent).getTile();
            auto coreTileOp = coreTileVal.getDefiningOp<AIE::TileOp>();
            if (coreTileOp) {
              int64_t cCol = static_cast<int64_t>(coreTileOp.getCol());
              int64_t cRow = static_cast<int64_t>(coreTileOp.getRow());
              auto [pCol, pRow] = winCinfo->producerTileCoord;
              if (cCol == pCol && cRow == pRow)
                acquirePort = Port::Produce;
            }
          }
        }
      }

      bool replaced = false;
      if (!conduitName.empty()) {
        ConduitInfo *cinfo = state.lookupConduit(conduitName, op);
        if (cinfo && !cinfo->buffers.empty()) {
          int64_t idx = op.getIndex();

          // Resolve per-tile buffers and rotation counter.
          auto resolved = cinfo->resolveForTile(op);
          llvm::SmallVector<AIE::BufferOp> *tileBuffers = resolved.buffers;
          // Use the port-appropriate rotation counter:
          // - Consume port uses consumerTileRotationBufs (rotationBuf)
          // - Produce port uses producerTileRotationBufs (producerRotationBuf)
          mlir::Value tileRotationBuf = (acquirePort == Port::Produce)
                                            ? resolved.producerRotationBuf
                                            : resolved.rotationBuf;

          {
            int64_t bufIdx =
                static_cast<int64_t>(tileBuffers->size()) > 1
                    ? idx % static_cast<int64_t>(tileBuffers->size())
                    : 0;
            int64_t numBufs = static_cast<int64_t>(tileBuffers->size());
            // Defense-in-depth: at numBufs>1 with no rotation counter,
            // we'd silently emit static buff[0] selection — the
            // multi-device-rotation bug class fixed by c09973b389. ERROR
            // here so future regressions surface at compile time instead
            // of as wrong-data on hardware.
            if (numBufs > 1 && !tileRotationBuf) {
              op.emitError(
                  "conduit-to-dma: depth>1 buffer rotation requires a "
                  "rotation counter, but none was allocated. This is the "
                  "multi-device rotation bug class — check "
                  "ConduitToDMAAlloc.cpp prescanAndCreateRotationBufs.");
              state.passFailed = true;
              return;
            }
            bool useStaticSelection = (numBufs <= 1 || !tileRotationBuf);
            if (useStaticSelection) {
              mlir::Value bufVal = (*tileBuffers)[bufIdx].getResult();
              if (bufVal.getType() == op.getResult().getType()) {
                op.getResult().replaceAllUsesWith(bufVal);
                replaced = true;
              }
            } else {
              // Dynamic selection via rotation counter + index_switch.
              builder.setInsertionPoint(op);
              mlir::Location loc = op.getLoc();

              int64_t tileRotationBufSlot =
                  (acquirePort == Port::Produce)
                      ? resolved.producerRotationBufSlot
                      : resolved.rotationBufSlot;
              mlir::Value slotIdx =
                  builder.create<mlir::arith::ConstantIndexOp>(
                      loc, tileRotationBufSlot);
              mlir::Value ctrI32 = builder.create<mlir::memref::LoadOp>(
                  loc, tileRotationBuf, mlir::ValueRange{slotIdx});
              mlir::Value ctrIdx = builder.create<mlir::arith::IndexCastOp>(
                  loc, builder.getIndexType(), ctrI32);

              mlir::Value absIdx = ctrIdx;
              if (idx > 0) {
                mlir::Value idxConst =
                    builder.create<mlir::arith::ConstantIndexOp>(loc, idx);
                mlir::Value sum =
                    builder.create<mlir::arith::AddIOp>(loc, ctrIdx, idxConst);
                // Branchless modulo: sum < 2*numBufs, so one conditional
                // subtract suffices.  Avoids arith.remui (software divide on
                // AIE2 which has no hardware divide instruction).
                mlir::Value depthConst =
                    builder.create<mlir::arith::ConstantIndexOp>(loc, numBufs);
                mlir::Value cond = builder.create<mlir::arith::CmpIOp>(
                    loc, mlir::arith::CmpIPredicate::uge, sum, depthConst);
                mlir::Value sub =
                    builder.create<mlir::arith::SubIOp>(loc, sum, depthConst);
                absIdx =
                    builder.create<mlir::arith::SelectOp>(loc, cond, sub, sum);
              }

              // Use scf::IndexSwitchOp, matching stateful transform.
              mlir::Type bufTy = op.getResult().getType();

              // Build case values [0, 1, ..., numBufs-1].
              llvm::SmallVector<int64_t, 4> caseValues;
              for (int64_t i = 0; i < numBufs; ++i)
                caseValues.push_back(i);
              auto cases = mlir::DenseI64ArrayAttr::get(ctx, caseValues);
              auto switchOp = mlir::scf::IndexSwitchOp::create(
                  builder, loc, mlir::TypeRange({bufTy}), absIdx, cases,
                  static_cast<unsigned>(numBufs));
              // Default case: yield buf[(idx % numBufs)].
              builder.createBlock(&switchOp.getDefaultRegion());
              builder.setInsertionPointToStart(&switchOp.getDefaultBlock());
              builder.create<mlir::scf::YieldOp>(
                  loc, (*tileBuffers)[bufIdx].getResult());
              // Case regions: case i yields buf[(idx + i) % numBufs].
              for (int64_t i = 0; i < numBufs; ++i) {
                builder.createBlock(&switchOp.getCaseRegions()[i]);
                builder.setInsertionPoint(&switchOp.getCaseBlock(i),
                                          switchOp.getCaseBlock(i).begin());
                int64_t bufferToAccess = (idx + i) % numBufs;
                builder.create<mlir::scf::YieldOp>(
                    loc, (*tileBuffers)[bufferToAccess].getResult());
              }

              op.getResult().replaceAllUsesWith(switchOp.getResult(0));
              replaced = true;
            }
          }
        }
      }
      if (!replaced) {
        op.emitError("conduit-to-dma: SubviewAccess could not be resolved to "
                     "an allocated aie.buffer — type mismatch, index out of "
                     "range, or conduit not in map");
        state.passFailed = true;
        return;
      }
      subviewsToErase.push_back(op);
    });
    for (auto op : subviewsToErase)
      op.erase();
    if (state.passFailed)
      return;
  }

  // Helper: emit fast modulo for rotation counter updates.
  //
  // arith.remui is a software divide on AIE2 (no hardware divide instruction).
  // Since the counter is always in [0, depth-1] and delta ≤ depth, the sum
  // newVal = counter + delta satisfies newVal < 2*depth, so one conditional
  // subtract always suffices.
  //
  // For power-of-2 depth d: andi(newVal, d-1) — single instruction.
  // For general depth d:    cmpi sge + subi + select — branchless.
  auto emitFastModulo = [&](mlir::Location loc, mlir::Value newVal,
                            int64_t depth) -> mlir::Value {
    mlir::Type i32Ty = mlir::IntegerType::get(ctx, 32);
    if (depth > 1 && (depth & (depth - 1)) == 0) {
      // Power-of-2: single AND.
      mlir::Value mask =
          mlir::arith::ConstantIntOp::create(builder, loc, i32Ty, depth - 1);
      return builder.create<mlir::arith::AndIOp>(loc, newVal, mask);
    }
    // General: branchless conditional subtract.
    mlir::Value depthVal =
        mlir::arith::ConstantIntOp::create(builder, loc, i32Ty, depth);
    mlir::Value cond = builder.create<mlir::arith::CmpIOp>(
        loc, mlir::arith::CmpIPredicate::uge, newVal, depthVal);
    mlir::Value sub =
        builder.create<mlir::arith::SubIOp>(loc, newVal, depthVal);
    return builder.create<mlir::arith::SelectOp>(loc, cond, sub, newVal);
  };

  // Step 2: Release → use_lock; collect for deferred erase.
  llvm::SmallVector<Release> releasesToErase;
  module.walk([&](Release op) {
    llvm::StringRef conduitName;
    if (auto acqOp =
            mlir::dyn_cast_or_null<Acquire>(op.getWindow().getDefiningOp()))
      conduitName = acqOp.getName();
    else if (auto waitOp = mlir::dyn_cast_or_null<WaitWindow>(
                 op.getWindow().getDefiningOp()))
      conduitName = waitOp.getName();

    if (conduitName.empty()) {
      releasesToErase.push_back(op);
      return;
    }
    ConduitInfo *cinfo = state.lookupConduit(conduitName, op);
    if (!cinfo) {
      releasesToErase.push_back(op);
      return;
    }
    builder.setInsertionPoint(op);
    int64_t count = static_cast<int64_t>(op.getCount());
    Port port = op.getPort();

    // Resolve per-tile lock pair and rotation counters.
    auto resolved = cinfo->resolveForTile(op);
    AIE::LockOp resolvedProdLock = resolved.prodLock;
    AIE::LockOp resolvedConsLock = resolved.consLock;
    mlir::Value resolvedRotationBuf = resolved.rotationBuf;
    mlir::Value resolvedProducerRotationBuf = resolved.producerRotationBuf;

    AIE::LockOp lock =
        (port == Port::Consume) ? resolvedProdLock : resolvedConsLock;
    if (lock) {
      // Scale release count by bd_repeat for Produce port: the producer
      // core releases N lock units per buffer (one per DMA repetition).
      int64_t effectiveCount = count;
      if (cinfo->bdRepeat > 1 && port == Port::Produce)
        effectiveCount *= cinfo->bdRepeat;
      int32_t relVal =
          state.lockRelValue(port, static_cast<int32_t>(effectiveCount));
      builder.create<AIE::UseLockOp>(op.getLoc(), lock.getResult(),
                                     AIE::LockAction::Release, relVal);
    }
    // Counter increment for depth>1 Consume port.
    // Use nConsumerBuffers() as the ring modulus: for sliding-window patterns,
    // nConsumerBuffers() > depth to accommodate extra held slots.
    if (resolvedRotationBuf && port == Port::Consume && cinfo->depth > 1) {
      int64_t consModulus = cinfo->nConsumerBuffers();
      if (count > consModulus) {
        op.emitError("conduit-to-dma: release count (")
            << count << ") exceeds consumer buffer count (" << consModulus
            << ") — rotation counter increment would be incorrect";
        state.passFailed = true;
        return;
      }
      mlir::Location loc = op.getLoc();
      mlir::Type i32Ty = mlir::IntegerType::get(ctx, 32);
      int64_t resolvedRotationBufSlot = resolved.rotationBufSlot;
      mlir::Value slotIdx = builder.create<mlir::arith::ConstantIndexOp>(
          loc, resolvedRotationBufSlot);
      mlir::Value curI32 = builder.create<mlir::memref::LoadOp>(
          loc, resolvedRotationBuf, mlir::ValueRange{slotIdx});
      mlir::Value incI32 =
          mlir::arith::ConstantIntOp::create(builder, loc, i32Ty, count);
      mlir::Value newVal =
          builder.create<mlir::arith::AddIOp>(loc, curI32, incI32);
      mlir::Value result = emitFastModulo(loc, newVal, consModulus);
      builder.create<mlir::memref::StoreOp>(loc, result, resolvedRotationBuf,
                                            mlir::ValueRange{slotIdx});
    }
    // Counter increment for depth>1 Produce port (producer buffer rotation).
    if (resolvedProducerRotationBuf && port == Port::Produce &&
        cinfo->depth > 1) {
      int64_t prodModulus =
          (cinfo->effectiveDepth > 0) ? cinfo->effectiveDepth : cinfo->depth;
      if (count > prodModulus) {
        op.emitError("conduit-to-dma: release count (")
            << count << ") exceeds conduit depth (" << prodModulus
            << ") — rotation counter increment would be incorrect";
        state.passFailed = true;
        return;
      }
      mlir::Location loc = op.getLoc();
      mlir::Type i32Ty = mlir::IntegerType::get(ctx, 32);
      int64_t resolvedProducerRotationBufSlot =
          resolved.producerRotationBufSlot;
      mlir::Value slotIdx = builder.create<mlir::arith::ConstantIndexOp>(
          loc, resolvedProducerRotationBufSlot);
      mlir::Value curI32 = builder.create<mlir::memref::LoadOp>(
          loc, resolvedProducerRotationBuf, mlir::ValueRange{slotIdx});
      mlir::Value incI32 =
          mlir::arith::ConstantIntOp::create(builder, loc, i32Ty, count);
      mlir::Value newVal =
          builder.create<mlir::arith::AddIOp>(loc, curI32, incI32);
      mlir::Value result = emitFastModulo(loc, newVal, prodModulus);
      builder.create<mlir::memref::StoreOp>(
          loc, result, resolvedProducerRotationBuf, mlir::ValueRange{slotIdx});
    }
    releasesToErase.push_back(op);
  });

  // Step 4: Acquire → use_lock + counter init; erase (collect-then-erase).
  // NOTE: The delta inference walkBlock runs BEFORE Step 3's Release erasure
  // because it needs to see Release ops to track heldCount correctly.
  // The walkBlock is defined and seeded here; Release erasure follows.
  //
  // Delta inference: conduit.acquire{count=N} means "I need a window of N
  // elements total."  The hardware AcquireGreaterEqual value is a DELTA —
  // how many new DMA-delivered elements to wait for beyond those already
  // available from a prior acquire in a dominating scope.
  //
  // State tracked per channel (StringAttr key) during the walk:
  //   lastAcquireCount: count of the most recent acquire for this channel.
  //                     Set on acquire.  NOT decremented by release.
  //   heldCount:        count currently held.  Set on acquire to count.
  //                     Decremented by release.  May be < lastAcquireCount
  //                     if a partial release occurred.
  //
  // Delta rules:
  //   Same-block acquire (no live parent scope for this channel):
  //     delta = count - heldCount
  //     (accounts for elements that were partially released in this block)
  //
  //   Cross-block acquire (in a nested scf.for/scf.if body, parent block
  //   has an open acquire for this channel):
  //     delta = count - parent.lastAcquireCount
  //     (uses lastAcquireCount, NOT heldCount — partial releases in the
  //     parent do NOT reduce the number of DMA slots already claimed; the
  //     DMA eagerly pre-fills those slots, so they are already in the buffer
  //     by the time the child acquire runs)
  //
  // When entering a nested block: child inherits parent.lastAcquireCount
  // as its starting heldCount (and lastAcquireCount), so cross-block acquires
  // compute the correct delta without additional annotation.
  //
  // delta == 0 → no AcquireGreaterEqual emitted (window already large enough).
  // delta > 0  → emit AcquireGreaterEqual(delta).

  // Track (conduitName, tileCoord) pairs to avoid double-initializing
  // rotation counters.  Uses tile coordinates instead of Operation* to
  // ensure deterministic behavior across runs.
  std::set<std::tuple<std::string, int64_t, int64_t, bool>> counterInitialized;

  // Per-channel live-window state, keyed by (channel name, port).
  // Propagated through nested blocks via the recursive walk below.
  struct ChannelState {
    int64_t lastAcquireCount =
        0; // count of most recent acquire (not reduced by release)
    int64_t heldCount = 0; // currently held (lastAcquireCount - sum releases)
  };
  using StateMap =
      llvm::DenseMap<std::pair<mlir::StringAttr, int>, ChannelState>;

  llvm::SmallVector<Acquire> acquiresToErase;

  // Recursive per-block walker.  `parentState` is the state map inherited
  // from the enclosing scope; entries not present default to zero.
  // The function processes the given block's ops in program order and
  // recurses into nested regions, passing `lastAcquireCount` (not heldCount)
  // as the inherited state for child blocks.
  // walkBlock returns the final StateMap after processing the block.
  // Callers use the returned map to propagate child held-counts back to the
  // parent scope (cross-block held-count tracking for tail partial-release).
  std::function<StateMap(mlir::Block *, StateMap)> walkBlock;
  walkBlock = [&](mlir::Block *block, StateMap liveState) -> StateMap {
    for (mlir::Operation &rawOp : *block) {
      if (auto op = mlir::dyn_cast<Acquire>(rawOp)) {
        ConduitInfo *cinfo = state.lookupConduit(op.getName(), op);
        if (!cinfo) {
          acquiresToErase.push_back(op);
          continue;
        }
        builder.setInsertionPoint(op);
        int64_t count = static_cast<int64_t>(op.getCount());

        // Compute delta from live-window state.
        Port port = op.getPort();
        auto key = std::make_pair(mlir::StringAttr::get(ctx, op.getName()),
                                  static_cast<int>(port));
        int64_t held = 0;
        if (auto it = liveState.find(key); it != liveState.end())
          held = it->second.heldCount;
        int64_t delta = (count > held) ? count - held : 0;

        // Update liveState: this acquire now owns `count` elements.
        liveState[key].lastAcquireCount = count;
        liveState[key].heldCount = count;

        auto resolved = cinfo->resolveForTile(op);
        AIE::LockOp resolvedProdLock = resolved.prodLock;
        AIE::LockOp resolvedConsLock = resolved.consLock;
        mlir::Value resolvedRotationBuf = resolved.rotationBuf;
        mlir::Value resolvedProducerRotationBuf = resolved.producerRotationBuf;
        mlir::Operation *acquireCoreOp = resolved.coreOp;

        AIE::LockOp lock =
            (port == Port::Produce) ? resolvedProdLock : resolvedConsLock;

        // Counter init for depth>1 Consume acquires.
        if (resolvedRotationBuf && port == Port::Consume && cinfo->depth > 1 &&
            acquireCoreOp) {
          mlir::Value coreTileVal =
              mlir::cast<AIE::CoreOp>(acquireCoreOp).getTile();
          auto coreTileOp = coreTileVal.getDefiningOp<AIE::TileOp>();
          int64_t col = static_cast<int64_t>(coreTileOp.getCol());
          int64_t row = static_cast<int64_t>(coreTileOp.getRow());
          auto ctrKey = std::make_tuple(op.getName().str(), col, row, false);
          if (!counterInitialized.count(ctrKey)) {
            counterInitialized.insert(ctrKey);
            mlir::OpBuilder initBuilder(ctx);
            // Insert init store at start of core body (rotation buffer is an
            // aie.buffer at device level, so setInsertionPointAfterValue would
            // place the store outside the core).
            auto &coreBody =
                mlir::cast<AIE::CoreOp>(acquireCoreOp).getBody().front();
            initBuilder.setInsertionPointToStart(&coreBody);
            mlir::Location loc = op.getLoc();
            mlir::Type i32Ty = mlir::IntegerType::get(ctx, 32);
            mlir::Value zero =
                mlir::arith::ConstantIntOp::create(initBuilder, loc, i32Ty, 0);
            int64_t rotationBufSlot = resolved.rotationBufSlot;
            mlir::Value slotIdx =
                initBuilder.create<mlir::arith::ConstantIndexOp>(
                    loc, rotationBufSlot);
            initBuilder.create<mlir::memref::StoreOp>(
                loc, zero, resolvedRotationBuf, mlir::ValueRange{slotIdx});
          }
        }

        // Counter init for depth>1 Produce acquires (producer buffer rotation).
        if (resolvedProducerRotationBuf && port == Port::Produce &&
            cinfo->depth > 1 && acquireCoreOp) {
          mlir::Value coreTileVal =
              mlir::cast<AIE::CoreOp>(acquireCoreOp).getTile();
          auto coreTileOp = coreTileVal.getDefiningOp<AIE::TileOp>();
          int64_t col = static_cast<int64_t>(coreTileOp.getCol());
          int64_t row = static_cast<int64_t>(coreTileOp.getRow());
          auto ctrKey = std::make_tuple(op.getName().str(), col, row, true);
          if (!counterInitialized.count(ctrKey)) {
            counterInitialized.insert(ctrKey);
            mlir::OpBuilder initBuilder(ctx);
            auto &coreBody =
                mlir::cast<AIE::CoreOp>(acquireCoreOp).getBody().front();
            initBuilder.setInsertionPointToStart(&coreBody);
            mlir::Location loc = op.getLoc();
            mlir::Type i32Ty = mlir::IntegerType::get(ctx, 32);
            mlir::Value zero =
                mlir::arith::ConstantIntOp::create(initBuilder, loc, i32Ty, 0);
            int64_t producerRotationBufSlot = resolved.producerRotationBufSlot;
            mlir::Value slotIdx =
                initBuilder.create<mlir::arith::ConstantIndexOp>(
                    loc, producerRotationBufSlot);
            initBuilder.create<mlir::memref::StoreOp>(
                loc, zero, resolvedProducerRotationBuf,
                mlir::ValueRange{slotIdx});
          }
        }

        // Emit AcquireGreaterEqual only when delta > 0.
        // delta == 0 means the window is already satisfied by a dominating
        // acquire — no additional lock grant needed.
        if (lock && delta > 0) {
          // Scale delta by bd_repeat for Produce acquires.
          int64_t effectiveDelta = delta;
          if (cinfo->bdRepeat > 1 && port == Port::Produce)
            effectiveDelta *= cinfo->bdRepeat;
          int32_t acqVal =
              state.lockAcqValue(port, static_cast<int32_t>(effectiveDelta));
          builder.create<AIE::UseLockOp>(op.getLoc(), lock.getResult(),
                                         acqAction, acqVal);
        }
        acquiresToErase.push_back(op);
        continue;
      }

      if (auto op = mlir::dyn_cast<Release>(rawOp)) {
        // Update heldCount for this channel: partial or full release.
        // lastAcquireCount is NOT changed — it reflects the number of DMA
        // slots claimed, which persists even when the core logically releases
        // a slot back to the producer lock.
        // The channel name comes from the window's defining acquire op.
        mlir::StringAttr nameAttr;
        if (auto acqOp =
                mlir::dyn_cast_or_null<Acquire>(op.getWindow().getDefiningOp()))
          nameAttr = mlir::StringAttr::get(ctx, acqOp.getName());
        else if (auto waitOp = mlir::dyn_cast_or_null<WaitWindow>(
                     op.getWindow().getDefiningOp()))
          nameAttr = mlir::StringAttr::get(ctx, waitOp.getName());
        if (!nameAttr)
          continue;
        Port port = op.getPort();
        auto key = std::make_pair(nameAttr, static_cast<int>(port));
        int64_t relCount = static_cast<int64_t>(op.getCount());
        if (auto it = liveState.find(key); it != liveState.end()) {
          it->second.heldCount -= relCount;
          if (it->second.heldCount < 0)
            it->second.heldCount = 0;
        }
        // Release has its own lowering in Step 2; we only update
        // liveState here for delta inference.
        continue;
      }

      // For ops with nested regions (scf.for, scf.if, etc.), recurse into
      // each region's blocks, propagating lastAcquireCount as the inherited
      // heldCount.  This implements the cross-block rule: child blocks see
      // the parent's lastAcquireCount (not heldCount), so partial releases
      // in the parent don't reduce the delta for the first child acquire.
      // Rationale: the DMA eagerly pre-fills slots up to lastAcquireCount,
      // so those slots are already in the buffer when the child block runs.
      //
      // IMPORTANT: this reset only applies to Consume channels (port=1).
      // For Produce channels (port=0), no DMA pre-fills output slots — the
      // core must acquire each slot from the lock.  If the parent released
      // all held slots (heldCount=0), the child block must acquire fresh
      // slots; resetting heldCount=lastAcquireCount would incorrectly
      // suppress the delta, causing the inner loop to write output without
      // owning the lock → hardware deadlock.
      for (mlir::Region &region : rawOp.getRegions()) {
        StateMap childState = liveState;
        for (auto &[k, cs] : childState) {
          int port = k.second; // 0=Produce, 1=Consume
          if (port == static_cast<int>(Port::Consume)) {
            // Consume: inherit lastAcquireCount as heldCount.
            // DMA eagerly pre-fills slots up to lastAcquireCount, so those
            // slots are already in the buffer when the child block runs.
            cs.heldCount = cs.lastAcquireCount;
          } else {
            // B-1 fix: Produce port — reset heldCount to 0 at loop body entry.
            // Each loop iteration starts fresh: the parent may have released a
            // non-uniform fraction (acquire N, release M<N, enter loop), but
            // the child should start from 0 held — the child acquire must wait
            // for a new slot from the lock, not assume the parent's partial
            // hold persists.  Keeping the parent's heldCount would suppress the
            // AcquireGreaterEqual delta, causing the core to write without
            // owning the lock → hardware deadlock.
            cs.heldCount = 0;
          }
        }

        // Process each child block; track the final state from the last block.
        // For scf.for, this is the loop body's exit state.
        StateMap childFinalState = childState;
        for (mlir::Block &childBlock : region)
          childFinalState = walkBlock(&childBlock, childState);

        // Propagate child's post-loop heldCount back to parent for Consume
        // channels.  After an scf.for with acquire=K/release=1 per iteration,
        // the child exits with heldCount=K-1 (the sliding-window tail).  The
        // parent must know about these still-held slots so that a subsequent
        // tail acquire in the outer block computes delta=0 (no new slots
        // needed) instead of delta=count-parentHeld (wrong
        // AcquireGreaterEqual).
        //
        // Only Consume port: Produce port holds are reset to 0 at loop entry
        // (B-1 fix), so no slots are inherited from the child Produce side.
        //
        // lastAcquireCount is also updated to reflect the child's last acquire,
        // enabling correct delta computation for any further nested regions.
        for (auto &[k, childCs] : childFinalState) {
          int port = k.second;
          if (port == static_cast<int>(Port::Consume)) {
            liveState[k].heldCount = childCs.heldCount;
            liveState[k].lastAcquireCount = childCs.lastAcquireCount;
          }
          // Produce: do not propagate — parent retains its pre-loop state.
          // The Produce port heldCount in the parent is managed by its own
          // acquire/release ops; child loop iterations operate independently.
        }
      }
    }
    return liveState;
  };

  // Seed the walk from each aie.core region with empty initial state.
  module.walk([&](AIE::CoreOp coreOp) {
    StateMap initialState;
    for (mlir::Block &block : coreOp.getBody())
      initialState = walkBlock(&block, initialState);
  });

  // Step 3: Erase Release ops (after walkBlock has seen them for delta
  // inference).
  for (auto op : releasesToErase)
    op.erase();

  for (auto op : acquiresToErase)
    op.erase();

  // -----------------------------------------------------------------------
  // Phase 7: Erase remaining Conduit ops.
  // -----------------------------------------------------------------------

  // Clear dep operands from PutMemrefAsync/GetMemrefAsync before erasure.
  module.walk([&](PutMemrefAsync op) { op.getDepsMutable().clear(); });
  module.walk([&](GetMemrefAsync op) { op.getDepsMutable().clear(); });

  // Erase conduit.scatter ops (Sprint 1 relay op, lowered in earlier phases).
  {
    llvm::SmallVector<ScatterOp> toErase;
    module.walk([&](ScatterOp op) { toErase.push_back(op); });
    for (auto op : llvm::reverse(toErase))
      op.erase();
  }

  // Erase conduit.gather ops (Sprint 1 relay op, lowered in earlier phases).
  {
    llvm::SmallVector<GatherOp> toErase;
    module.walk([&](GatherOp op) { toErase.push_back(op); });
    for (auto op : llvm::reverse(toErase))
      op.erase();
  }

  // Collect-then-erase Create ops.
  // Note: conduit.wait was removed from the dialect (absorbed into wait_all).
  {
    llvm::SmallVector<Create> createsToErase;
    module.walk([&](Create op) { createsToErase.push_back(op); });
    for (auto op : llvm::reverse(createsToErase))
      op.erase();
  }

  // WaitAllAsync: collect-then-reverse-erase (defs before uses).
  {
    llvm::SmallVector<WaitAllAsync> waitAllAsyncsToErase;
    module.walk([&](WaitAllAsync op) { waitAllAsyncsToErase.push_back(op); });
    for (auto op : llvm::reverse(waitAllAsyncsToErase))
      op.erase();
  }

  // -----------------------------------------------------------------------
  // Phase 8: Lower async acquire/release/wait_window/wait_all.
  // -----------------------------------------------------------------------

  // Step 8a: Record acquire_async metadata.
  llvm::SmallVector<AcquireAsync> asyncAcquiresToErase;
  module.walk([&](AcquireAsync op) {
    AsyncAcquireInfo info;
    info.conduitName = op.getName().str();
    info.port = op.getPort();
    info.count = static_cast<int64_t>(op.getCount());
    state.asyncAcquireMap[op.getToken()] = info;
    asyncAcquiresToErase.push_back(op);
  });

  // Step 8b: Lower wait_window.
  llvm::SmallVector<WaitWindow> waitWindowsToErase;
  module.walk([&](WaitWindow op) {
    llvm::StringRef conduitName = op.getName();
    ConduitInfo *cinfo = state.lookupConduit(conduitName, op);
    if (!cinfo) {
      op.emitError("conduit-to-dma: wait_window references unknown conduit '")
          << conduitName << "'";
      state.passFailed = true;
      return;
    }

    Port port = Port::Consume;
    int64_t count = 1;
    {
      auto ait = state.asyncAcquireMap.find(op.getToken());
      if (ait != state.asyncAcquireMap.end()) {
        port = ait->second.port;
        count = ait->second.count;
        state.asyncAcquireMap.erase(ait);
      }
    }

    auto resolved = cinfo->resolveForTile(op);
    AIE::LockOp resolvedProdLock = resolved.prodLock;
    AIE::LockOp resolvedConsLock = resolved.consLock;

    // Rotation counter init for async acquires (via wait_window).
    // Step 4 initializes counters for sync AcquireOp but async paths
    // (acquire_async → wait_window) were missed, leaving rotation
    // counters uninitialized for channels accessed only via async acquire.
    {
      mlir::Value resolvedRotationBuf = resolved.rotationBuf;
      mlir::Value resolvedProducerRotationBuf = resolved.producerRotationBuf;
      mlir::Operation *acquireCoreOp = resolved.coreOp;

      if (resolvedRotationBuf && port == Port::Consume && cinfo->depth > 1 &&
          acquireCoreOp) {
        mlir::Value coreTileVal =
            mlir::cast<AIE::CoreOp>(acquireCoreOp).getTile();
        auto coreTileOp = coreTileVal.getDefiningOp<AIE::TileOp>();
        int64_t col = static_cast<int64_t>(coreTileOp.getCol());
        int64_t row = static_cast<int64_t>(coreTileOp.getRow());
        auto ctrKey = std::make_tuple(conduitName.str(), col, row, false);
        if (!counterInitialized.count(ctrKey)) {
          counterInitialized.insert(ctrKey);
          mlir::OpBuilder initBuilder(ctx);
          auto &coreBody =
              mlir::cast<AIE::CoreOp>(acquireCoreOp).getBody().front();
          initBuilder.setInsertionPointToStart(&coreBody);
          mlir::Location loc = op.getLoc();
          mlir::Type i32Ty = mlir::IntegerType::get(ctx, 32);
          mlir::Value zero =
              mlir::arith::ConstantIntOp::create(initBuilder, loc, i32Ty, 0);
          int64_t rotationBufSlot = resolved.rotationBufSlot;
          mlir::Value slotIdx =
              initBuilder.create<mlir::arith::ConstantIndexOp>(loc,
                                                               rotationBufSlot);
          initBuilder.create<mlir::memref::StoreOp>(
              loc, zero, resolvedRotationBuf, mlir::ValueRange{slotIdx});
        }
      }

      if (resolvedProducerRotationBuf && port == Port::Produce &&
          cinfo->depth > 1 && acquireCoreOp) {
        mlir::Value coreTileVal =
            mlir::cast<AIE::CoreOp>(acquireCoreOp).getTile();
        auto coreTileOp = coreTileVal.getDefiningOp<AIE::TileOp>();
        int64_t col = static_cast<int64_t>(coreTileOp.getCol());
        int64_t row = static_cast<int64_t>(coreTileOp.getRow());
        auto ctrKey = std::make_tuple(conduitName.str(), col, row, true);
        if (!counterInitialized.count(ctrKey)) {
          counterInitialized.insert(ctrKey);
          mlir::OpBuilder initBuilder(ctx);
          auto &coreBody =
              mlir::cast<AIE::CoreOp>(acquireCoreOp).getBody().front();
          initBuilder.setInsertionPointToStart(&coreBody);
          mlir::Location loc = op.getLoc();
          mlir::Type i32Ty = mlir::IntegerType::get(ctx, 32);
          mlir::Value zero =
              mlir::arith::ConstantIntOp::create(initBuilder, loc, i32Ty, 0);
          int64_t producerRotationBufSlot = resolved.producerRotationBufSlot;
          mlir::Value slotIdx =
              initBuilder.create<mlir::arith::ConstantIndexOp>(
                  loc, producerRotationBufSlot);
          initBuilder.create<mlir::memref::StoreOp>(loc, zero,
                                                    resolvedProducerRotationBuf,
                                                    mlir::ValueRange{slotIdx});
        }
      }
    }

    builder.setInsertionPoint(op);
    AIE::LockOp lock =
        (port == Port::Produce) ? resolvedProdLock : resolvedConsLock;
    if (lock) {
      int32_t acqVal = state.lockAcqValue(port, static_cast<int32_t>(count));
      builder.create<AIE::UseLockOp>(op.getLoc(), lock.getResult(), acqAction,
                                     acqVal);
    }

    if (!op.getResult().use_empty()) {
      op.emitError(
          "conduit-to-dma: wait_window result has surviving users after "
          "Phase 6 SubviewAccess replacement — type mismatch or missed op");
      state.passFailed = true;
      return;
    }

    waitWindowsToErase.push_back(op);
  });
  for (auto op : waitWindowsToErase)
    op.erase();

  if (state.passFailed)
    return;

  // Step 8c: Lower wait_all.
  //
  // wait_all ops inside an aie.runtime_sequence are handled by Step 8g
  // (basic Path C, 2026-04-25): they reference !conduit.dma.token values
  // produced by put/get_memref_async (rebuilt from IRON's
  // aiex.dma_configure_task_for + dma_start_task), and Step 8g lowers
  // them to aiex.dma_await_task / aiex.dma_free_task at their source
  // locations to preserve IRON's per-launch BD release boundaries.
  // Skipping rtSeq-scoped wait_all here prevents this pass from
  // erasing them before Step 8g sees them.
  llvm::SmallVector<WaitAll> waitAllToErase;
  module.walk([&](WaitAll op) {
    if (op->getParentOfType<AIE::RuntimeSequenceOp>())
      return;
    builder.setInsertionPoint(op);
    for (mlir::Value tok : op.getTokens()) {
      auto ait = state.asyncAcquireMap.find(tok);
      if (ait == state.asyncAcquireMap.end())
        continue;

      const AsyncAcquireInfo &ainfo = ait->second;
      ConduitInfo *cinfo = state.lookupConduit(ainfo.conduitName, op);
      if (!cinfo)
        continue;

      auto resolved = cinfo->resolveForTile(op);
      AIE::LockOp resolvedProdLock = resolved.prodLock;
      AIE::LockOp resolvedConsLock = resolved.consLock;

      AIE::LockOp lock =
          (ainfo.port == Port::Produce) ? resolvedProdLock : resolvedConsLock;
      if (lock) {
        int32_t acqVal =
            state.lockAcqValue(ainfo.port, static_cast<int32_t>(ainfo.count));
        builder.create<AIE::UseLockOp>(op.getLoc(), lock.getResult(), acqAction,
                                       acqVal);
      }
    }
    waitAllToErase.push_back(op);
  });
  for (auto op : waitAllToErase)
    op.erase();

  // Step 8a-erase: Erase AcquireAsync ops.
  for (auto op : asyncAcquiresToErase)
    op.erase();

  // Step 8d: Lower release_async → use_lock + counter increment.
  llvm::SmallVector<ReleaseAsync> releaseAsyncsToErase;
  module.walk([&](ReleaseAsync op) {
    llvm::StringRef conduitName = op.getName();
    ConduitInfo *cinfo = state.lookupConduit(conduitName, op);
    if (!cinfo) {
      releaseAsyncsToErase.push_back(op);
      return;
    }

    Port port = op.getPort();
    auto resolved = cinfo->resolveForTile(op);
    AIE::LockOp resolvedProdLock = resolved.prodLock;
    AIE::LockOp resolvedConsLock = resolved.consLock;
    mlir::Value resolvedRotationBuf = resolved.rotationBuf;
    mlir::Value resolvedProducerRotationBuf = resolved.producerRotationBuf;

    builder.setInsertionPoint(op);
    int64_t count = static_cast<int64_t>(op.getCount());
    AIE::LockOp lock =
        (port == Port::Consume) ? resolvedProdLock : resolvedConsLock;
    if (lock) {
      // Scale release count by bd_repeat for Produce port (async).
      int64_t effectiveCount = count;
      if (cinfo->bdRepeat > 1 && port == Port::Produce)
        effectiveCount *= cinfo->bdRepeat;
      int32_t relVal =
          state.lockRelValue(port, static_cast<int32_t>(effectiveCount));
      builder.create<AIE::UseLockOp>(op.getLoc(), lock.getResult(),
                                     AIE::LockAction::Release, relVal);
    }
    // Counter increment for depth>1 Consume port (release_async path).
    // Use nConsumerBuffers() as the ring modulus (same as sync Release path).
    if (resolvedRotationBuf && port == Port::Consume && cinfo->depth > 1) {
      int64_t consModulus = cinfo->nConsumerBuffers();
      if (count > consModulus) {
        op.emitError("conduit-to-dma: release_async count (")
            << count << ") exceeds consumer buffer count (" << consModulus
            << ") — rotation counter increment would be incorrect";
        state.passFailed = true;
        return;
      }
      mlir::Location loc = op.getLoc();
      mlir::Type i32Ty = mlir::IntegerType::get(ctx, 32);
      int64_t resolvedRotationBufSlot = resolved.rotationBufSlot;
      mlir::Value slotIdx = builder.create<mlir::arith::ConstantIndexOp>(
          loc, resolvedRotationBufSlot);
      mlir::Value curI32 = builder.create<mlir::memref::LoadOp>(
          loc, resolvedRotationBuf, mlir::ValueRange{slotIdx});
      mlir::Value incI32 =
          mlir::arith::ConstantIntOp::create(builder, loc, i32Ty, count);
      mlir::Value newVal =
          builder.create<mlir::arith::AddIOp>(loc, curI32, incI32);
      mlir::Value result = emitFastModulo(loc, newVal, consModulus);
      builder.create<mlir::memref::StoreOp>(loc, result, resolvedRotationBuf,
                                            mlir::ValueRange{slotIdx});
    }
    // Counter increment for depth>1 Produce port (producer buffer rotation).
    if (resolvedProducerRotationBuf && port == Port::Produce &&
        cinfo->depth > 1) {
      int64_t prodModulus =
          (cinfo->effectiveDepth > 0) ? cinfo->effectiveDepth : cinfo->depth;
      if (count > prodModulus) {
        op.emitError("conduit-to-dma: release_async count (")
            << count << ") exceeds conduit depth (" << prodModulus
            << ") — rotation counter increment would be incorrect";
        state.passFailed = true;
        return;
      }
      mlir::Location loc = op.getLoc();
      mlir::Type i32Ty = mlir::IntegerType::get(ctx, 32);
      int64_t resolvedProducerRotationBufSlot =
          resolved.producerRotationBufSlot;
      mlir::Value slotIdx = builder.create<mlir::arith::ConstantIndexOp>(
          loc, resolvedProducerRotationBufSlot);
      mlir::Value curI32 = builder.create<mlir::memref::LoadOp>(
          loc, resolvedProducerRotationBuf, mlir::ValueRange{slotIdx});
      mlir::Value incI32 =
          mlir::arith::ConstantIntOp::create(builder, loc, i32Ty, count);
      mlir::Value newVal =
          builder.create<mlir::arith::AddIOp>(loc, curI32, incI32);
      mlir::Value result = emitFastModulo(loc, newVal, prodModulus);
      builder.create<mlir::memref::StoreOp>(
          loc, result, resolvedProducerRotationBuf, mlir::ValueRange{slotIdx});
    }
    releaseAsyncsToErase.push_back(op);
  });
  for (auto op : releaseAsyncsToErase)
    op.erase();

  // Steps 8e-8f: Lower put/get_memref_async inside aie.core → use_lock pair.
  //
  // When hierarchy-produced IR (via --air-hierarchy-to-aie → Pass B) places
  // conduit.put_memref_async / conduit.get_memref_async inside aie.core
  // bodies, the core must synchronize with the DMA engine via use_lock.
  // The DMA BD chain (Phase 5.5) handles DMA-side locking; these use_lock
  // ops handle core-side locking:
  //
  //   put_memref_async (producer): acquire prodLock → release consLock
  //   get_memref_async (consumer): acquire consLock → release prodLock
  //
  // This mirrors the Tier 2 acquire/release protocol (Steps 2+4).

  // Step 8e: Lower PutMemrefAsync.
  //
  // Skip async ops inside an aie.runtime_sequence — Step 8g rebuilds them
  // into aiex.dma_configure_task_for + dma_start_task and pairs each with
  // its wait_all consumer.  Erasing them here would lose the source-order
  // metadata Step 8g requires.
  {
    llvm::SmallVector<PutMemrefAsync> toErase;
    module.walk([&](PutMemrefAsync op) {
      if (op->getParentOfType<AIE::RuntimeSequenceOp>())
        return;
      llvm::StringRef conduitName = op.getName();
      ConduitInfo *cinfo = state.lookupConduit(conduitName, op);
      if (cinfo) {
        auto resolved = cinfo->resolveForTile(op);
        if (resolved.coreOp) {
          // Inside aie.core — emit use_lock pair for producer synchronization.
          // Skip for shim-producer channels (row==0): the shim DMA is managed
          // by the host runtime via aiex.npu.dma_memcpy_nd; core-side use_lock
          // is not needed and the shim lock lives outside the core's region.
          auto [prodCol, prodRow] = cinfo->producerTileCoord;
          (void)prodCol;
          bool shimProducer = (prodRow == 0);
          if (!shimProducer) {
            builder.setInsertionPoint(op);
            int64_t count = 1;
            if (cinfo->bdRepeat > 1)
              count *= cinfo->bdRepeat;

            // Acquire prodLock: wait for empty buffer slot.
            if (resolved.prodLock) {
              int32_t acqVal = state.lockAcqValue(Port::Produce,
                                                  static_cast<int32_t>(count));
              builder.create<AIE::UseLockOp>(op.getLoc(),
                                             resolved.prodLock.getResult(),
                                             acqAction, acqVal);
            }
            // Release consLock: signal data ready for DMA.
            if (resolved.consLock) {
              int32_t relVal = state.lockRelValue(Port::Produce,
                                                  static_cast<int32_t>(count));
              builder.create<AIE::UseLockOp>(op.getLoc(),
                                             resolved.consLock.getResult(),
                                             AIE::LockAction::Release, relVal);
            }
          }
        }
      }
      toErase.push_back(op);
    });
    for (auto op : llvm::reverse(toErase))
      op.erase();
  }

  // Step 8f: Lower GetMemrefAsync.
  //
  // Skip async ops inside an aie.runtime_sequence — Step 8g handles them.
  {
    llvm::SmallVector<GetMemrefAsync> toErase;
    module.walk([&](GetMemrefAsync op) {
      if (op->getParentOfType<AIE::RuntimeSequenceOp>())
        return;
      llvm::StringRef conduitName = op.getName();
      ConduitInfo *cinfo = state.lookupConduit(conduitName, op);
      if (cinfo) {
        auto resolved = cinfo->resolveForTile(op);
        if (resolved.coreOp) {
          // Inside aie.core — emit use_lock pair for consumer synchronization.
          // Unlike the put_memref_async handler (Step 8e), the consumer ALWAYS
          // needs use_lock regardless of where the producer is.  Even when the
          // producer is a shim tile (row==0), the compute consumer must
          // synchronize with its local DMA engine via locks.
          builder.setInsertionPoint(op);
          int64_t count = 1;

          // Acquire consLock: wait for data to arrive.
          if (resolved.consLock) {
            // Debug: verify lock belongs to same device as the core.
            auto coreDev = resolved.coreOp->getParentOfType<AIE::DeviceOp>();
            auto lockDev = resolved.consLock->getParentOfType<AIE::DeviceOp>();
            if (coreDev && lockDev && coreDev != lockDev) {
              op.emitWarning("DEBUG: consLock for '" + conduitName.str() +
                             "' is from a different device!");
            }
            int32_t acqVal =
                state.lockAcqValue(Port::Consume, static_cast<int32_t>(count));
            builder.create<AIE::UseLockOp>(
                op.getLoc(), resolved.consLock.getResult(), acqAction, acqVal);
          }
          // Release prodLock: signal buffer slot is empty.
          if (resolved.prodLock) {
            auto coreDev = resolved.coreOp->getParentOfType<AIE::DeviceOp>();
            auto lockDev = resolved.prodLock->getParentOfType<AIE::DeviceOp>();
            if (coreDev && lockDev && coreDev != lockDev) {
              op.emitWarning("DEBUG: prodLock for '" + conduitName.str() +
                             "' is from a different device!");
            }
            int32_t relVal =
                state.lockRelValue(Port::Consume, static_cast<int32_t>(count));
            builder.create<AIE::UseLockOp>(op.getLoc(),
                                           resolved.prodLock.getResult(),
                                           AIE::LockAction::Release, relVal);
          }
        }
      }
      toErase.push_back(op);
    });
    for (auto op : llvm::reverse(toErase))
      op.erase();
  }

  // Step 8g: Lower conduit.put_memref / get_memref inside
  // aie.runtime_sequence → aiex.dma_configure_task_for + dma_start/await/free.
  //
  // This is the reverse of --dma-task-to-conduit.  Each put/get_memref op
  // carries an explicit `arg_index` attribute set by --dma-task-to-conduit
  // from the original aie.dma_bd's BlockArgument index.  We use that
  // attribute as the authoritative binding when re-emitting aie.dma_bd.
  //
  // History: an earlier version of this step grouped ops by an
  // `offsets[0] == 0` heuristic ("a new arg group starts when offset is
  // zero").  That heuristic was the bug — it mis-binds ops whose first
  // offset is a non-zero patch marker (e.g. IRON's `0xDEADBEE0` runtime
  // patch markers used by StridedCopy/Repeat) to the previous group's
  // block arg, silently routing OUTPUT BDs to INPUT block args.  The
  // heuristic has been deleted; arg_index is required, and missing
  // arg_index fails the pass loudly to surface upstream pipeline bugs.
  //
  // Mapping:
  //   conduit.put_memref {name=@chan, arg_index=N} →
  //       aiex.dma_configure_task_for @chan_shim_alloc {
  //         aie.dma_bd(%argN, ...) / aie.end }
  //   conduit.get_memref {name=@chan, arg_index=N} → same but with
  //       {issue_token = true} + aiex.dma_await_task after dma_start_task
  //
  // The shim_dma_allocation symbol is looked up via the conduit_channel attr
  // or by the "{name}_shim_alloc" naming convention established by routePhase.
  {
    // Build conduit_name → shim_alloc info from aie.shim_dma_allocation ops.
    llvm::StringMap<std::string> conduitToAllocSym;
    llvm::StringMap<AIE::DMAChannelDir> conduitToDir;
    module.walk([&](AIE::ShimDMAAllocationOp alloc) {
      llvm::StringRef conduitName;
      if (auto cc =
              alloc->getAttrOfType<mlir::FlatSymbolRefAttr>("conduit_channel"))
        conduitName = cc.getValue();
      else {
        // Fallback: strip _shim_alloc suffix.
        conduitName = alloc.getSymName();
        if (conduitName.ends_with("_shim_alloc"))
          conduitName = conduitName.drop_back(strlen("_shim_alloc"));
      }
      conduitToAllocSym[conduitName] = alloc.getSymName().str();
      conduitToDir[conduitName] = alloc.getChannelDir();
    });

    module.walk([&](AIE::RuntimeSequenceOp rtSeq) {
      // Collect put/get_memref AND put/get_memref_async ops in source order.
      // Async variants are emitted by --dma-task-to-conduit (basic Path C,
      // 2026-04-25) when IRON's dma_await_task / dma_free_task need to be
      // preserved as conduit.wait_all{token=...} for per-launch BD release.
      llvm::SmallVector<mlir::Operation *> memrefOps;
      // wait_all ops in source order — Step 8g lowers them to
      // aiex.dma_await_task / aiex.dma_free_task at their source locations
      // so IRON's per-launch BD release boundaries survive.
      llvm::SmallVector<WaitAll> waitAllsInSeq;
      for (auto &op : rtSeq.getBody().front()) {
        if (mlir::isa<PutMemref>(op) || mlir::isa<GetMemref>(op) ||
            mlir::isa<PutMemrefAsync>(op) || mlir::isa<GetMemrefAsync>(op))
          memrefOps.push_back(&op);
        else if (auto wa = mlir::dyn_cast<WaitAll>(op))
          waitAllsInSeq.push_back(wa);
      }
      if (memrefOps.empty())
        return;

      auto blockArgs = rtSeq.getBody().front().getArguments();
      auto indexTy = mlir::IndexType::get(ctx);

      // Per-channel live-task tracking for the trailing release.  Emit
      // no inline release between same-channel configures; every
      // configured task gets a matching release at end-of-rtSeq instead
      // (S2MM with issue_token=true → aiex.dma_await_task; MM2S →
      // aiex.dma_free_task).  AIEAssignRuntimeSequenceBDIDs's interval
      // analysis treats each task as [configure ... trailing-release]
      // and assigns distinct BD IDs from the per-channel pool (16 on
      // shim); pool exhaustion surfaces a clear diagnostic.
      struct LiveTask {
        mlir::Value task;
        bool isS2MM;
        mlir::Location loc;
      };
      // All configured tasks per channel, in source order.  Trailing
      // release emits one release per stored task that was NOT already
      // released by an explicit wait_all consumer.
      llvm::StringMap<llvm::SmallVector<LiveTask, 4>> tasksPerChannel;
      // Insertion-ordered channel keys so trailing release order is
      // deterministic (StringMap iteration order is hash-dependent and
      // would make lit tests flaky).  StringRefs point into
      // FlatSymbolRefAttr storage, which outlives this loop (attrs are
      // not mutated here).  Channels are emitted in first-seen order;
      // tasks within a channel are emitted in source order.
      llvm::SmallVector<llvm::StringRef, 16> liveOrder;
      // Map: !conduit.dma.token from put/get_memref_async → emitted task
      // SSA value.  Used to lower wait_all ops to dma_await/free_task with
      // the right operand.
      llvm::DenseMap<mlir::Value, mlir::Value> conduitTokenToTask;
      // Map: emitted task SSA → emitted aiex.dma_configure_task_for op.
      // Lets us stamp `issue_token = true` after the fact when an MM2S
      // task is awaited by a wait_all{token=true} consumer.  We use
      // `Operation *` (not LiveTask *) to avoid dangling-pointer hazards
      // when tasksPerChannel SmallVectors reallocate.
      llvm::DenseMap<mlir::Value, mlir::Operation *> taskToConfigOp;
      // Map: emitted task SSA → direction (for picking await vs free).
      llvm::DenseMap<mlir::Value, bool> taskIsS2MM;
      // Tasks that have been released by an explicit wait_all consumer —
      // skipped by the trailing-release loop to avoid double-release.
      llvm::DenseSet<mlir::Value> releasedTasks;

      for (unsigned i = 0; i < memrefOps.size(); ++i) {
        mlir::Operation *op = memrefOps[i];
        auto nameRef = op->getAttrOfType<mlir::FlatSymbolRefAttr>("name");
        if (!nameRef)
          continue;

        llvm::StringRef conduitName = nameRef.getValue();
        auto allocIt = conduitToAllocSym.find(conduitName);
        if (allocIt == conduitToAllocSym.end())
          continue;

        std::string allocSym = allocIt->second;
        bool isS2MM = (conduitToDir[conduitName] == AIE::DMAChannelDir::S2MM);

        // Resolve block arg via the explicit arg_index attribute set by
        // --dma-task-to-conduit.  An earlier offsets[0]==0 grouping
        // heuristic silently mis-bound output ops whose first offset is
        // a non-zero patch marker to the input block arg; arg_index is
        // now the authoritative binding; missing arg_index is a hard
        // error (signals upstream pipeline bug).
        auto argIdxAttr = op->getAttrOfType<mlir::IntegerAttr>("arg_index");
        if (!argIdxAttr) {
          op->emitError("conduit-to-dma Step 8g: put/get_memref op is "
                        "missing the required `arg_index` attribute — must "
                        "be set by --dma-task-to-conduit (or any other "
                        "producer of put/get_memref ops inside an "
                        "aie.runtime_sequence) so the BD can be re-bound "
                        "to the correct block argument");
          state.passFailed = true;
          return;
        }
        int64_t argIdxSigned = argIdxAttr.getInt();
        if (argIdxSigned < 0 ||
            static_cast<size_t>(argIdxSigned) >= blockArgs.size()) {
          op->emitError("conduit-to-dma Step 8g: arg_index ")
              << argIdxSigned << " is out of range (runtime_sequence has "
              << blockArgs.size() << " block arguments)";
          state.passFailed = true;
          return;
        }
        unsigned argIdx = static_cast<unsigned>(argIdxSigned);
        mlir::Value bufArg = blockArgs[argIdx];

        int64_t numElems =
            op->getAttrOfType<mlir::IntegerAttr>("num_elems").getInt();

        // Use the op's offsets[0] as the DMA BD offset (not hardcoded 0).
        auto offsetsAttr =
            op->getAttrOfType<mlir::DenseI64ArrayAttr>("offsets");
        int bdOffset = (offsetsAttr && !offsetsAttr.empty())
                           ? static_cast<int>(offsetsAttr[0])
                           : 0;

        // Get BDDimLayout dimensions for the BD chain.
        //   * put_memref (MM2S) carries producer_dimensions: a single
        //     BDDimLayoutArrayAttr matching aie.dma_bd's dimensions slot.
        //   * get_memref (S2MM) carries consumer_dimensions: a
        //     BDDimLayoutArrayArrayAttr (one entry per consumer tile).
        //     For shim S2MM rebuild, there is exactly one consumer (the
        //     shim DMA), so we use the first inner array.  This restores
        //     the strided write geometry (e.g. StridedCopy's 8×131072
        //     scatter) that was previously dropped to nullptr.
        AIE::BDDimLayoutArrayAttr dims;
        if (isS2MM) {
          if (auto consDimsAttr =
                  op->getAttrOfType<AIE::BDDimLayoutArrayArrayAttr>(
                      "consumer_dimensions")) {
            if (!consDimsAttr.empty()) {
              dims = mlir::dyn_cast<AIE::BDDimLayoutArrayAttr>(
                  consDimsAttr.getValue().front());
            }
          }
        } else {
          if (auto dimsAttr = op->getAttrOfType<AIE::BDDimLayoutArrayAttr>(
                  "producer_dimensions"))
            dims = dimsAttr;
        }

        // Bug C / iter_count fix: consume `dma_repeat = N` from the source
        // conduit.create on the shim BD.  Pass A's iter_count inference
        // stamps `dma_repeat = N` on shim-facing channels when the per-core
        // outer loop fires the BD chain N times per host dispatch (e.g.
        // num_invocations=4 with cores running while_true=False produces
        // dma_repeat=4).  Without consuming it here, each shim dispatch
        // only fires the BD once and the cores stall after exhausting
        // 1/N-th of the work — Bug C root cause for the .bin runtime path.
        //
        // We surface dma_repeat to the shim DMA via the
        // `aiex.dma_configure_task_for.repeat_count` attribute (AIEX.td:1088),
        // which lowers directly into NpuPushQueueOp.repeat_count
        // (AIEDMATasksToNPU.cpp:56) and from there into the NPU command
        // word's repeat field (AIEDmaToNpu.cpp:180-183).  This is the
        // hardware's purpose-built mechanism for shim DMA replay and avoids
        // mutating the BD's data-layout transformation, which (a) would
        // collide with the AIE2p 4-dim cap on IRON-emitted BDs of the form
        // [<1,0>, <1,0>, <1,0>, <inner, stride>], and (b) has unverified
        // lock-semantics interaction.
        //
        // Mirrors the memtile path in ConduitToDMALink.cpp:1804+ which
        // surfaces dma_repeat via DMAStartOp.repeat_count (with libxaie's
        // 0=once, N-1 convention for that op).  For the shim NPU command
        // path the value is passed through verbatim — set repeat_count = N.
        // BD len and dims are left untouched.
        int64_t channelDmaRepeat = 0;
        if (auto *info = state.lookupConduit(conduitName, op))
          channelDmaRepeat = info->dmaRepeat;

        // Derive an additional repeat_count contribution from the BD's
        // outer wrap+stride dim, when canon's ArithProgressionPattern
        // stamped a multi-cycle outer dim.  BD descriptor's iteration_size
        // alone is insufficient on the shim NPU command path: the
        // push-queue's repeat_count must also be N.  This mirrors the
        // npu.dma_memcpy_nd convention (AIEDmaToNpu.cpp:380-393), which
        // sets BOTH iteration_size = sizes[3] AND push-queue
        // repeat_count = sizes[3].  Empirically (homogeneous_repeat NPU
        // smoke), without this only 1 of N iterations fires even though
        // the BD descriptor's iteration_size is correct.
        //
        // Canon's ArithProgressionPattern deliberately does NOT also stamp
        // dma_repeat=N on the channel (would double-multiply with the
        // outer dim's iteration on the channel-DMA / memtile paths), so
        // channelDmaRepeat and outerRepeat cannot both be > 0 for the
        // same configure on the shim path here.  Combine via max() as a
        // belt-and-braces guard.
        int64_t outerRepeat = 0;
        if (dims && !dims.getValue().empty()) {
          uint32_t outerSize = dims.getValue().front().getSize();
          if (outerSize > 1)
            outerRepeat = static_cast<int64_t>(outerSize);
        }
        int64_t effectiveRepeat = std::max(channelDmaRepeat, outerRepeat);

        builder.setInsertionPoint(op);
        mlir::Location loc = op->getLoc();

        // No inline release between same-channel configures: releases
        // are batched at end-of-rtSeq (see trailing-release loop below)
        // so AIEAssignRuntimeSequenceBDIDs sees per-task intervals
        // [configure ... trailing-release] and assigns distinct BD IDs
        // from the per-channel pool.  Pool exhaustion surfaces a clear
        // diagnostic instructing the producer to interleave await/free
        // for finer-grained release.

        // Build aiex.dma_configure_task_for.
        mlir::OperationState configState(loc, "aiex.dma_configure_task_for");
        configState.addAttribute("alloc",
                                 mlir::FlatSymbolRefAttr::get(ctx, allocSym));
        if (isS2MM)
          configState.addAttribute("issue_token", builder.getBoolAttr(true));
        // Surface dma_repeat to the shim DMA via the configure_task's
        // repeat_count attribute.  Any positive dma_repeat surfaces
        // verbatim — including dma_repeat = 1, which IRON / firmware read
        // as 2 fires per call (see CLAUDE.md "Convention-divergence" entry
        // and AIEDmaToNpu.cpp packing).  Skipping `> 0` (default / absent)
        // preserves the correct no-attr emission for unstamped channels.
        if (effectiveRepeat > 0)
          configState.addAttribute(
              "repeat_count",
              builder.getI32IntegerAttr(static_cast<int32_t>(effectiveRepeat)));
        configState.addTypes(indexTy);
        configState.addRegion();
        mlir::Operation *configOp = builder.create(configState);

        // Build body: aie.dma_bd + aie.end.  BD len + dims are taken
        // verbatim from the source put/get_memref op; dma_repeat is
        // applied via the configure_task attribute above, NOT by mutating
        // the BD's data layout.
        mlir::Region &bodyRegion = configOp->getRegion(0);
        mlir::Block *bdBlock = new mlir::Block();
        bodyRegion.push_back(bdBlock);
        builder.setInsertionPointToEnd(bdBlock);

        // AIEX runtime-sequence verifier (AIEDMATasksToNPU.cpp:432-454)
        // requires lower-3 dim sizes' product == BD len.  When canon's
        // ArithProgressionPattern stamps a single outer wrap+stride dim,
        // the lone dim lands in input_sizes[0] (innermost) under the
        // verifier's reverse-index mapping (j = K-1-i), failing the lower-3
        // == len check (e.g. canon emits dimensions=[<size=4,stride=64>] +
        // len=64 → verifier sees lower-3 product=4, len=128 bytes for bf16).
        //
        // Pad to K=4 so the canon-prepended outer dim lands in
        // input_sizes[3] (iteration slot), with size-1 stride-1 padding
        // dims filling the gap and the existing innermost dim (or a
        // fabricated <num_elems,1> when no existing inner dim) at
        // input_sizes[0].  Lower-3 product becomes:
        //   K_existing >= 1: product of existing.sizes (preserved invariant
        //                    from pre-canon valid input)
        //   K_existing == 0: 1 × 1 × num_elems (fabricated inner)
        // both equal num_elems.
        //
        // Safe to pad here because runtime-sequence dma_bd ops bypass the
        // AIEDialect dma_bd verifier (AIEDialect.cpp:2194 parent-class
        // skip), so the AIE 3-dim cap on non-MemTile parents doesn't
        // apply.  Compute-tile / memtile dma_bd emit sites in Link.cpp
        // must NOT use this padding — canon's defensive cap check refuses
        // arith collapse when the producer (puts) or consumer (gets) is
        // a compute tile, preventing the 4-dim form from reaching those
        // paths.
        if (dims && !dims.getValue().empty()) {
          llvm::ArrayRef<AIE::BDDimLayoutAttr> existingDims = dims.getValue();
          size_t K = existingDims.size();
          AIE::BDDimLayoutArrayAttr emitDims = dims;
          if (K < 4) {
            llvm::SmallVector<AIE::BDDimLayoutAttr> newDims;
            newDims.reserve(4);
            // Outer (canon-prepended iteration dim, or user's outermost).
            newDims.push_back(existingDims.front());
            // Inner block size: existing inner dims (K-1) when K > 1,
            // else 1 fabricated <num_elems, 1>. Total must equal 4.
            // Padding fills the gap between outer and inner: 4 - 1 (outer)
            // - innerCount.
            size_t innerCount = (K == 1) ? 1 : (K - 1);
            size_t padCount = 4 - 1 - innerCount;
            // AIE HW requires stride × elem_size_bytes divisible by 4.
            // Padding dims have size=1 (no effective addressing), so stride
            // value is mathematically irrelevant — but verifier still checks.
            // Pick the smallest stride satisfying the constraint for the
            // current element type.
            auto memrefTy = mlir::cast<mlir::MemRefType>(bufArg.getType());
            unsigned elemBits =
                memrefTy.getElementType().getIntOrFloatBitWidth();
            unsigned elemBytes = std::max(1U, elemBits / 8);
            uint32_t padStride = std::max(1U, 4U / elemBytes);
            for (size_t i = 0; i < padCount; ++i)
              newDims.push_back(AIE::BDDimLayoutAttr::get(
                  ctx, /*size=*/1, /*stride=*/padStride));
            if (K == 1) {
              // No existing inner dim — fabricate <num_elems, 1> as
              // innermost so lower-3 product == num_elems.
              newDims.push_back(AIE::BDDimLayoutAttr::get(
                  ctx, static_cast<uint32_t>(numElems), /*stride=*/1));
            } else {
              // Preserve existing inner block (existingDims[1..K-1]) at
              // the inner end so their lower-3 product invariant holds.
              for (size_t i = 1; i < K; ++i)
                newDims.push_back(existingDims[i]);
            }
            emitDims = AIE::BDDimLayoutArrayAttr::get(ctx, newDims);
          }
          builder.create<AIE::DMABDOp>(loc, bufArg, bdOffset,
                                       static_cast<int>(numElems), emitDims);
        } else {
          builder.create<AIE::DMABDOp>(loc, bufArg, bdOffset,
                                       static_cast<int>(numElems));
        }
        // Set burst_length = 0 on the dma_bd.
        bdBlock->back().setAttr("burst_length", builder.getI32IntegerAttr(0));
        builder.create<AIE::EndOp>(loc);

        // Emit aiex.dma_start_task(%task).
        builder.setInsertionPointAfter(configOp);
        mlir::Value taskResult = configOp->getResult(0);
        {
          mlir::OperationState startState(loc, "aiex.dma_start_task");
          startState.addOperands(taskResult);
          builder.create(startState);
        }

        // Basic Path C tracking: record the task's configure op + direction
        // so wait_all consumer lowering (below) can (a) emit the right
        // release op (await for S2MM, await-or-free for MM2S depending on
        // wait_all.token), and (b) stamp `issue_token = true` on the MM2S
        // configure when a wait_all{token = true} consumer needs to await it.
        // For the async variants, also map the conduit token result → task
        // SSA so wait_all operands resolve correctly.
        taskToConfigOp[taskResult] = configOp;
        taskIsS2MM[taskResult] = isS2MM;
        if (mlir::isa<PutMemrefAsync>(op) || mlir::isa<GetMemrefAsync>(op)) {
          mlir::Value conduitToken = op->getResult(0);
          conduitTokenToTask[conduitToken] = taskResult;
        }

        // Append this task to the per-channel list (Path B: every
        // configured task gets a matching trailing release).  First time
        // this channel is seen, also record its key in liveOrder for
        // deterministic trailing-release channel ordering at end-of-body.
        auto it = tasksPerChannel.find(conduitName);
        if (it == tasksPerChannel.end()) {
          auto inserted = tasksPerChannel.try_emplace(
              conduitName, llvm::SmallVector<LiveTask, 4>{});
          inserted.first->second.push_back(LiveTask{taskResult, isS2MM, loc});
          liveOrder.push_back(conduitName);
        } else {
          it->second.push_back(LiveTask{taskResult, isS2MM, loc});
        }
      }

      // wait_all consumer lowering inside aie.runtime_sequence.
      // For each conduit.wait_all in the runtime sequence, emit an explicit
      // per-task release op (aiex.dma_await_task or aiex.dma_free_task) at
      // the wait_all's source location.  This preserves the per-launch BD
      // release boundaries that IRON emits, so the
      // AIEAssignRuntimeSequenceBDIDs allocator's per-channel live intervals
      // shrink to [configure ... explicit-release] instead of spanning the
      // whole runtime sequence.  Without this preservation, BD-pool
      // exhaustion fires on workloads that exceed the per-shim 16-BD pool
      // (e.g. Llama LM-head GEMM), even though IRON had already provided
      // adequate release points in the source IR.
      //
      // Lowering rules:
      //   wait_all{token = true}  → aiex.dma_await_task (one per operand);
      //                             stamp `issue_token = true` on the
      //                             corresponding configure_task (required
      //                             by aiecc legalization for MM2S; S2MM
      //                             already gets issue_token = true above).
      //   wait_all{token = false} → aiex.dma_free_task (one per operand).
      //
      // Tasks released here are recorded in `releasedTasks` so the
      // trailing-release loop below skips them (avoids double-release).
      // Operands that don't resolve to a tracked task (e.g. window tokens
      // mixed in, or tokens from a configure that was not lowered here)
      // are skipped silently — those will fall back to the trailing-release
      // path or be diagnosed by downstream verifiers.
      for (WaitAll wa : waitAllsInSeq) {
        bool wantAwait = wa.getToken();
        builder.setInsertionPoint(wa);
        for (mlir::Value tokenOperand : wa.getTokens()) {
          auto tokIt = conduitTokenToTask.find(tokenOperand);
          if (tokIt == conduitTokenToTask.end())
            continue;
          mlir::Value task = tokIt->second;

          // For MM2S tasks awaited by wait_all{token = true}, ensure the
          // corresponding configure_task was emitted with issue_token = true
          // — the firmware requires it before dma_await_task may target the
          // BD.  S2MM configures already get issue_token = true at emission
          // time (see line ~1296 above), so this only affects MM2S.
          if (wantAwait) {
            auto s2mmIt = taskIsS2MM.find(task);
            if (s2mmIt != taskIsS2MM.end() && !s2mmIt->second) {
              auto cfgIt = taskToConfigOp.find(task);
              if (cfgIt != taskToConfigOp.end())
                cfgIt->second->setAttr("issue_token",
                                       builder.getBoolAttr(true));
            }
          }

          mlir::OperationState rel(wa.getLoc(), wantAwait
                                                    ? "aiex.dma_await_task"
                                                    : "aiex.dma_free_task");
          rel.addOperands(task);
          builder.create(rel);
          releasedTasks.insert(task);
        }
      }

      // Trailing release: emit one release per configured task in
      // (channel-first-seen-order, then source-order within each channel).
      // S2MM (issue_token=true) → aiex.dma_await_task; MM2S →
      // aiex.dma_free_task.  Matches stateful's emission pattern.  Tasks
      // already released by an explicit wait_all consumer are skipped to
      // avoid double-release.
      builder.setInsertionPointToEnd(&rtSeq.getBody().front());
      for (llvm::StringRef channelKey : liveOrder) {
        auto trailIt = tasksPerChannel.find(channelKey);
        assert(trailIt != tasksPerChannel.end() &&
               "liveOrder out of sync with tasksPerChannel");
        for (const LiveTask &live : trailIt->second) {
          if (releasedTasks.contains(live.task))
            continue;
          mlir::OperationState rel(rtSeq.getLoc(), live.isS2MM
                                                       ? "aiex.dma_await_task"
                                                       : "aiex.dma_free_task");
          rel.addOperands(live.task);
          builder.create(rel);
        }
      }

      // Erase the conduit wait_all ops first (they reference the async
      // memref op results), then the put/get_memref ops.
      for (WaitAll wa : waitAllsInSeq)
        wa.erase();
      for (auto *op : llvm::reverse(memrefOps))
        op->erase();
    });
  }

  // Step 8h: Erase remaining sync put/get memref ops NOT inside
  // runtime_sequence (e.g., stale ops in other contexts).
  {
    llvm::SmallVector<PutMemref> toErase;
    module.walk([&](PutMemref op) { toErase.push_back(op); });
    for (auto op : llvm::reverse(toErase))
      op.erase();
  }
  {
    llvm::SmallVector<GetMemref> toErase;
    module.walk([&](GetMemref op) { toErase.push_back(op); });
    for (auto op : llvm::reverse(toErase))
      op.erase();
  }
}

} // namespace xilinx::conduit
