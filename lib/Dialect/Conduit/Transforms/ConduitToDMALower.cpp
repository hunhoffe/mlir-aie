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
//   Step 1: SubviewAccess → buffer replacement (static or dynamic rotation).
//   Step 2: Release → use_lock + counter increment (collect for deferred
//   erase). Step 3: Erase Release ops. Step 4: Acquire → use_lock + counter
//   init; erase.
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

              // IMPORTANT: Do NOT use scf::IndexSwitchOp here.
              //
              // PEANO (llvm-aie) has a code generation bug where it generates
              // incorrect lookup tables for scf.index_switch when the modular
              // index wraps around (e.g., (counter+1)%4 = 0 at counter=3).
              // The erroneous table causes the wrong buffer address to be
              // selected, resulting in concurrent DMA+core access to the same
              // buffer without lock protection → hardware fault (unexpected
              // command state) on AIE2 npu1.
              //
              // Confirmed: for depth=4 with 6 middle iterations, the table at
              // .data[0x7bc4c+12] contained buff_3 instead of buff_0, causing
              // the N=6 sliding window to fail while N=5 passed.
              //
              // Fix: use a chain of scf::IfOp (→ cf.cond_br), which PEANO
              // generates correctly. This avoids the lookup table entirely.
              mlir::Type bufTy = op.getResult().getType();

              // Build a nested if/else chain: if absIdx==0 yield buf[0]
              // else if absIdx==1 yield buf[1] else ... else yield buf[N-1].
              // The outermost if wraps the whole expression.
              mlir::Value result = (*tileBuffers)[numBufs - 1].getResult();
              for (int64_t i = numBufs - 2; i >= 0; --i) {
                mlir::Value caseConst =
                    builder.create<mlir::arith::ConstantIndexOp>(loc, i);
                mlir::Value cond = builder.create<mlir::arith::CmpIOp>(
                    loc, mlir::arith::CmpIPredicate::eq, absIdx, caseConst);
                mlir::Value innerResult = result; // capture for lambda
                auto ifOp = builder.create<mlir::scf::IfOp>(
                    loc, bufTy, cond, /*withElseRegion=*/true);
                // Then block: yield buf[i]
                {
                  mlir::OpBuilder::InsertionGuard g(builder);
                  builder.setInsertionPointToStart(
                      &ifOp.getThenRegion().front());
                  builder.create<mlir::scf::YieldOp>(
                      loc, (*tileBuffers)[i].getResult());
                }
                // Else block: yield the result from the inner chain
                {
                  mlir::OpBuilder::InsertionGuard g(builder);
                  builder.setInsertionPointToStart(
                      &ifOp.getElseRegion().front());
                  builder.create<mlir::scf::YieldOp>(loc, innerResult);
                }
                result = ifOp.getResult(0);
              }

              op.getResult().replaceAllUsesWith(result);
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
            initBuilder.setInsertionPointAfterValue(resolvedRotationBuf);
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
            initBuilder.setInsertionPointAfterValue(
                resolvedProducerRotationBuf);
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
        auto ctrKey =
            std::make_tuple(conduitName.str(), col, row, false);
        if (!counterInitialized.count(ctrKey)) {
          counterInitialized.insert(ctrKey);
          mlir::OpBuilder initBuilder(ctx);
          initBuilder.setInsertionPointAfterValue(resolvedRotationBuf);
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

      if (resolvedProducerRotationBuf && port == Port::Produce &&
          cinfo->depth > 1 && acquireCoreOp) {
        mlir::Value coreTileVal =
            mlir::cast<AIE::CoreOp>(acquireCoreOp).getTile();
        auto coreTileOp = coreTileVal.getDefiningOp<AIE::TileOp>();
        int64_t col = static_cast<int64_t>(coreTileOp.getCol());
        int64_t row = static_cast<int64_t>(coreTileOp.getRow());
        auto ctrKey =
            std::make_tuple(conduitName.str(), col, row, true);
        if (!counterInitialized.count(ctrKey)) {
          counterInitialized.insert(ctrKey);
          mlir::OpBuilder initBuilder(ctx);
          initBuilder.setInsertionPointAfterValue(
              resolvedProducerRotationBuf);
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
  llvm::SmallVector<WaitAll> waitAllToErase;
  module.walk([&](WaitAll op) {
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
  {
    llvm::SmallVector<PutMemrefAsync> toErase;
    module.walk([&](PutMemrefAsync op) {
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
  {
    llvm::SmallVector<GetMemrefAsync> toErase;
    module.walk([&](GetMemrefAsync op) {
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
  // This is the reverse of --dma-task-to-conduit. After --conduit-fuse-operators
  // merges runtime_sequences and eliminates dead block args, the remaining
  // put/get ops correspond 1:1 (positionally) to the runtime_sequence block
  // args.  Each Nth conduit.put_memref/get_memref maps to block arg N.
  //
  // Mapping:
  //   conduit.put_memref {name=@chan} → aiex.dma_configure_task_for
  //       @chan_shim_alloc { aie.dma_bd(%argN, ...) / aie.end }
  //   conduit.get_memref {name=@chan} → same but with {issue_token = true}
  //       + aiex.dma_await_task after dma_start_task
  //
  // The shim_dma_allocation symbol is looked up via the conduit_channel attr
  // or by the "{name}_shim_alloc" naming convention established by routePhase.
  {
    // Build conduit_name → shim_alloc info from aie.shim_dma_allocation ops.
    llvm::StringMap<std::string> conduitToAllocSym;
    llvm::StringMap<AIE::DMAChannelDir> conduitToDir;
    module.walk([&](AIE::ShimDMAAllocationOp alloc) {
      llvm::StringRef conduitName;
      if (auto cc = alloc->getAttrOfType<mlir::FlatSymbolRefAttr>(
              "conduit_channel"))
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
      // Collect put/get_memref ops in source order.
      llvm::SmallVector<mlir::Operation *> memrefOps;
      for (auto &op : rtSeq.getBody().front()) {
        if (mlir::isa<PutMemref>(op) || mlir::isa<GetMemref>(op))
          memrefOps.push_back(&op);
      }
      if (memrefOps.empty())
        return;

      auto blockArgs = rtSeq.getBody().front().getArguments();
      auto indexTy = mlir::IndexType::get(ctx);

      // Task SSA values for free/await at end.
      llvm::SmallVector<mlir::Value> putTasks;
      llvm::SmallVector<mlir::Value> awaitTasks;

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
        bool isS2MM =
            (conduitToDir[conduitName] == AIE::DMAChannelDir::S2MM);

        if (i >= blockArgs.size())
          continue;
        mlir::Value bufArg = blockArgs[i];

        int64_t numElems =
            op->getAttrOfType<mlir::IntegerAttr>("num_elems").getInt();

        // Get BDDimLayout dimensions (put_memref carries producer_dimensions).
        AIE::BDDimLayoutArrayAttr dims;
        if (auto dimsAttr =
                op->getAttrOfType<AIE::BDDimLayoutArrayAttr>(
                    "producer_dimensions"))
          dims = dimsAttr;

        builder.setInsertionPoint(op);
        mlir::Location loc = op->getLoc();

        // Build aiex.dma_configure_task_for.
        mlir::OperationState configState(loc,
                                         "aiex.dma_configure_task_for");
        configState.addAttribute(
            "alloc", mlir::FlatSymbolRefAttr::get(ctx, allocSym));
        if (isS2MM)
          configState.addAttribute("issue_token",
                                   builder.getBoolAttr(true));
        configState.addTypes(indexTy);
        configState.addRegion();
        mlir::Operation *configOp = builder.create(configState);

        // Build body: aie.dma_bd + aie.end.
        mlir::Region &bodyRegion = configOp->getRegion(0);
        mlir::Block *bdBlock = new mlir::Block();
        bodyRegion.push_back(bdBlock);
        builder.setInsertionPointToEnd(bdBlock);

        if (dims && !dims.getValue().empty())
          builder.create<AIE::DMABDOp>(
              loc, bufArg, /*offset=*/0,
              static_cast<int>(numElems), dims);
        else
          builder.create<AIE::DMABDOp>(
              loc, bufArg, /*offset=*/0,
              static_cast<int>(numElems));
        // Set burst_length = 0 on the dma_bd.
        bdBlock->back().setAttr("burst_length",
                                builder.getI32IntegerAttr(0));
        builder.create<AIE::EndOp>(loc);

        // Emit aiex.dma_start_task(%task).
        builder.setInsertionPointAfter(configOp);
        mlir::Value taskResult = configOp->getResult(0);
        {
          mlir::OperationState startState(loc, "aiex.dma_start_task");
          startState.addOperands(taskResult);
          builder.create(startState);
        }

        if (isS2MM)
          awaitTasks.push_back(taskResult);
        else
          putTasks.push_back(taskResult);
      }

      // Emit await + free at the end of the runtime_sequence body.
      builder.setInsertionPointToEnd(&rtSeq.getBody().front());

      for (auto task : awaitTasks) {
        mlir::OperationState awaitState(rtSeq.getLoc(),
                                        "aiex.dma_await_task");
        awaitState.addOperands(task);
        builder.create(awaitState);
      }

      // Free put tasks in reverse order.
      for (auto task : llvm::reverse(putTasks)) {
        mlir::OperationState freeState(rtSeq.getLoc(),
                                       "aiex.dma_free_task");
        freeState.addOperands(task);
        builder.create(freeState);
      }

      // Erase the conduit put/get_memref ops.
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
