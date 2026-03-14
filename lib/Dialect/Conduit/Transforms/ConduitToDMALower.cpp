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
//   Step 2: Release → use_lock + counter increment (collect for deferred erase).
//   Step 3: Erase Release ops.
//   Step 4: Acquire → use_lock + counter init; erase.
//
// Phase 7: Erase remaining Conduit ops (create, wait, wait_all_async).
//
// Phase 8: Lower async acquire/release/wait_window/wait_all.
//   Step 8a: Record acquire_async metadata.
//   Step 8b: Lower wait_window → use_lock.
//   Step 8c: Lower wait_all → use_lock.
//   Step 8a-erase: Erase acquire_async ops.
//   Step 8d: Lower release_async → use_lock + counter increment.
//   Steps 8e-8h: Erase put/get memref ops.
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
  const bool isAIE2 = state.isAIE2;
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
      if (auto acqOp = mlir::dyn_cast_or_null<Acquire>(
              op.getWindow().getDefiningOp())) {
        conduitName = acqOp.getName();
        acquirePort = acqOp.getPort();
      } else if (auto waitOp = mlir::dyn_cast_or_null<WaitWindow>(
                     op.getWindow().getDefiningOp())) {
        conduitName = waitOp.getName();
        acquirePort = Port::Consume;
      }

      bool replaced = false;
      if (!conduitName.empty()) {
        auto it = state.conduitMap.find(conduitName.str());
        if (it != state.conduitMap.end() && !it->second.buffers.empty()) {
          int64_t idx = op.getIndex();
          ConduitInfo &cinfo = it->second;

          // Resolve per-tile buffers and rotation counter.
          auto resolved = cinfo.resolveForTile(op);
          llvm::SmallVector<AIE::BufferOp> *tileBuffers = resolved.buffers;
          AIE::BufferOp tileRotationBuf = resolved.rotationBuf;

          {
            int64_t bufIdx = static_cast<int64_t>(tileBuffers->size()) > 1
                                 ? idx % static_cast<int64_t>(tileBuffers->size())
                                 : 0;
            int64_t numBufs = static_cast<int64_t>(tileBuffers->size());
            bool useStaticSelection =
                (numBufs <= 1 || !tileRotationBuf || acquirePort == Port::Produce);
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

              mlir::Value c0Idx = builder.create<mlir::arith::ConstantIndexOp>(loc, 0);
              mlir::Value ctrI32 = builder.create<mlir::memref::LoadOp>(
                  loc, tileRotationBuf.getResult(), mlir::ValueRange{c0Idx});
              mlir::Value ctrIdx = builder.create<mlir::arith::IndexCastOp>(
                  loc, builder.getIndexType(), ctrI32);

              mlir::Value absIdx = ctrIdx;
              if (idx > 0) {
                mlir::Value idxConst =
                    builder.create<mlir::arith::ConstantIndexOp>(loc, idx);
                mlir::Value sum = builder.create<mlir::arith::AddIOp>(
                    loc, ctrIdx, idxConst);
                mlir::Value depthConst =
                    builder.create<mlir::arith::ConstantIndexOp>(loc, numBufs);
                absIdx = builder.create<mlir::arith::RemUIOp>(loc, sum, depthConst);
              }

              mlir::Type bufTy = op.getResult().getType();
              llvm::SmallVector<int64_t> caseVals;
              for (int64_t i = 0; i < numBufs; ++i)
                caseVals.push_back(i);

              auto switchOp = builder.create<mlir::scf::IndexSwitchOp>(
                  loc, mlir::TypeRange{bufTy}, absIdx, caseVals,
                  static_cast<int>(numBufs));

              for (int64_t i = 0; i < numBufs; ++i) {
                mlir::Block *caseBlock =
                    &switchOp.getCaseRegions()[i].emplaceBlock();
                mlir::OpBuilder caseBuilder(ctx);
                caseBuilder.setInsertionPointToEnd(caseBlock);
                caseBuilder.create<mlir::scf::YieldOp>(
                    loc, (*tileBuffers)[i].getResult());
              }
              {
                mlir::Block *defBlock =
                    &switchOp.getDefaultRegion().emplaceBlock();
                mlir::OpBuilder defBuilder(ctx);
                defBuilder.setInsertionPointToEnd(defBlock);
                defBuilder.create<mlir::scf::YieldOp>(
                    loc, (*tileBuffers)[0].getResult());
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
    if (state.passFailed) return;
  }

  // Step 2: Release → use_lock; collect for deferred erase.
  llvm::SmallVector<Release> releasesToErase;
  module.walk([&](Release op) {
    llvm::StringRef conduitName;
    if (auto acqOp = mlir::dyn_cast_or_null<Acquire>(
            op.getWindow().getDefiningOp()))
      conduitName = acqOp.getName();
    else if (auto waitOp = mlir::dyn_cast_or_null<WaitWindow>(
                 op.getWindow().getDefiningOp()))
      conduitName = waitOp.getName();

    if (conduitName.empty()) {
      releasesToErase.push_back(op);
      return;
    }
    auto it = state.conduitMap.find(conduitName.str());
    if (it == state.conduitMap.end()) {
      releasesToErase.push_back(op);
      return;
    }
    ConduitInfo &cinfo = it->second;
    builder.setInsertionPoint(op);
    int64_t count = static_cast<int64_t>(op.getCount());
    Port port = op.getPort();

    // Resolve per-tile lock pair and rotation counter.
    auto resolved = cinfo.resolveForTile(op);
    AIE::LockOp resolvedProdLock = resolved.prodLock;
    AIE::LockOp resolvedConsLock = resolved.consLock;
    AIE::BufferOp resolvedRotationBuf = resolved.rotationBuf;

    AIE::LockOp lock =
        (port == Port::Consume) ? resolvedProdLock : resolvedConsLock;
    if (lock) {
      int32_t relVal = isAIE2 ? static_cast<int32_t>(count)
                              : (port == Port::Consume ? 0 : 1);
      builder.create<AIE::UseLockOp>(op.getLoc(), lock.getResult(),
                                     AIE::LockAction::Release, relVal);
    }
    // Counter increment for depth>1 Consume port.
    if (resolvedRotationBuf && port == Port::Consume && cinfo.depth > 1) {
      mlir::Location loc = op.getLoc();
      mlir::Type i32Ty = mlir::IntegerType::get(ctx, 32);
      mlir::Value c0 = builder.create<mlir::arith::ConstantIndexOp>(loc, 0);
      mlir::Value curI32 = builder.create<mlir::memref::LoadOp>(
          loc, resolvedRotationBuf.getResult(), mlir::ValueRange{c0});
      mlir::Value incI32 = mlir::arith::ConstantIntOp::create(
          builder, loc, i32Ty, count);
      mlir::Value newVal =
          builder.create<mlir::arith::AddIOp>(loc, curI32, incI32);
      mlir::Value depthI32 = mlir::arith::ConstantIntOp::create(
          builder, loc, i32Ty, cinfo.depth);
      mlir::Value result =
          builder.create<mlir::arith::RemUIOp>(loc, newVal, depthI32);
      builder.create<mlir::memref::StoreOp>(
          loc, result, resolvedRotationBuf.getResult(), mlir::ValueRange{c0});
    }
    releasesToErase.push_back(op);
  });

  // Step 3: Erase Release ops before erasing Acquire.
  for (auto op : releasesToErase)
    op.erase();

  // Step 4: Acquire → use_lock + counter init; erase (collect-then-erase).
  // Track (conduitName, tileCoord) pairs to avoid double-initializing
  // rotation counters.  Uses tile coordinates instead of Operation* to
  // ensure deterministic behavior across runs.
  std::set<std::pair<std::string, std::pair<int64_t, int64_t>>>
      counterInitialized;

  llvm::SmallVector<Acquire> acquiresToErase;
  module.walk([&](Acquire op) {
    auto it = state.conduitMap.find(op.getName().str());
    if (it == state.conduitMap.end()) {
      acquiresToErase.push_back(op);
      return;
    }
    ConduitInfo &cinfo = it->second;
    builder.setInsertionPoint(op);
    int64_t count = static_cast<int64_t>(op.getCount());
    Port port = op.getPort();

    auto resolved = cinfo.resolveForTile(op);
    AIE::LockOp resolvedProdLock = resolved.prodLock;
    AIE::LockOp resolvedConsLock = resolved.consLock;
    AIE::BufferOp resolvedRotationBuf = resolved.rotationBuf;
    mlir::Operation *acquireCoreOp = resolved.coreOp;

    AIE::LockOp lock =
        (port == Port::Produce) ? resolvedProdLock : resolvedConsLock;

    // Counter init for depth>1 Consume acquires.
    if (resolvedRotationBuf && port == Port::Consume && cinfo.depth > 1 &&
        acquireCoreOp) {
      mlir::Value coreTileVal =
          mlir::cast<AIE::CoreOp>(acquireCoreOp).getTile();
      auto coreTileOp = coreTileVal.getDefiningOp<AIE::TileOp>();
      auto tileCoord = std::make_pair(
          static_cast<int64_t>(coreTileOp.getCol()),
          static_cast<int64_t>(coreTileOp.getRow()));
      auto key = std::make_pair(op.getName().str(), tileCoord);
      if (!counterInitialized.count(key)) {
        counterInitialized.insert(key);
        mlir::Block *entryBlock = &acquireCoreOp->getRegion(0).front();
        mlir::OpBuilder initBuilder(entryBlock, entryBlock->begin());
        mlir::Location loc = op.getLoc();
        mlir::Type i32Ty = mlir::IntegerType::get(ctx, 32);
        mlir::Value zero = mlir::arith::ConstantIntOp::create(
            initBuilder, loc, i32Ty, 0);
        mlir::Value c0 =
            initBuilder.create<mlir::arith::ConstantIndexOp>(loc, 0);
        initBuilder.create<mlir::memref::StoreOp>(
            loc, zero, resolvedRotationBuf.getResult(),
            mlir::ValueRange{c0});
      }
    }

    if (lock) {
      int32_t acqVal = isAIE2 ? static_cast<int32_t>(count)
                              : (port == Port::Produce ? 0 : 1);
      builder.create<AIE::UseLockOp>(op.getLoc(), lock.getResult(),
                                     acqAction, acqVal);
    }
    acquiresToErase.push_back(op);
  });
  for (auto op : acquiresToErase)
    op.erase();

  // -----------------------------------------------------------------------
  // Phase 7: Erase remaining Conduit ops.
  // -----------------------------------------------------------------------

  // Clear dep operands from PutMemrefAsync/GetMemrefAsync before erasure.
  module.walk([&](PutMemrefAsync op) { op.getDepsMutable().clear(); });
  module.walk([&](GetMemrefAsync op) { op.getDepsMutable().clear(); });

  // Collect-then-erase for Wait and Create ops.
  {
    llvm::SmallVector<Wait> waitsToErase;
    module.walk([&](Wait op) { waitsToErase.push_back(op); });
    for (auto op : llvm::reverse(waitsToErase))
      op.erase();
  }
  {
    llvm::SmallVector<Create> createsToErase;
    module.walk([&](Create op) { createsToErase.push_back(op); });
    for (auto op : llvm::reverse(createsToErase))
      op.erase();
  }

  // WaitAllAsync: collect-then-reverse-erase (defs before uses).
  {
    llvm::SmallVector<WaitAllAsync> waitAllAsyncsToErase;
    module.walk([&](WaitAllAsync op) {
      waitAllAsyncsToErase.push_back(op);
    });
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
    info.port = Port::Consume;
    info.count = static_cast<int64_t>(op.getCount());
    state.asyncAcquireMap[op.getToken()] = info;
    asyncAcquiresToErase.push_back(op);
  });

  // Step 8b: Lower wait_window.
  llvm::SmallVector<WaitWindow> waitWindowsToErase;
  module.walk([&](WaitWindow op) {
    llvm::StringRef conduitName = op.getName();
    auto it = state.conduitMap.find(conduitName.str());
    if (it == state.conduitMap.end()) {
      op.emitError("conduit-to-dma: wait_window references unknown conduit '")
          << conduitName << "'";
      state.passFailed = true;
      return;
    }
    ConduitInfo &cinfo = it->second;

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

    auto resolved = cinfo.resolveForTile(op);
    AIE::LockOp resolvedProdLock = resolved.prodLock;
    AIE::LockOp resolvedConsLock = resolved.consLock;

    builder.setInsertionPoint(op);
    AIE::LockOp lock = (port == Port::Produce) ? resolvedProdLock : resolvedConsLock;
    if (lock) {
      int32_t acqVal = isAIE2 ? static_cast<int32_t>(count)
                              : (port == Port::Produce ? 0 : 1);
      builder.create<AIE::UseLockOp>(op.getLoc(), lock.getResult(),
                                     acqAction, acqVal);
    }

    if (!op.getResult().use_empty()) {
      op.emitError("conduit-to-dma: wait_window result has surviving users after "
                   "Phase 6 SubviewAccess replacement — type mismatch or missed op");
      state.passFailed = true;
      return;
    }

    waitWindowsToErase.push_back(op);
  });
  for (auto op : waitWindowsToErase)
    op.erase();

  if (state.passFailed) return;

  // Step 8c: Lower wait_all.
  llvm::SmallVector<WaitAll> waitAllToErase;
  module.walk([&](WaitAll op) {
    builder.setInsertionPoint(op);
    for (mlir::Value tok : op.getTokens()) {
      auto ait = state.asyncAcquireMap.find(tok);
      if (ait == state.asyncAcquireMap.end())
        continue;

      const AsyncAcquireInfo &ainfo = ait->second;
      auto it = state.conduitMap.find(ainfo.conduitName);
      if (it == state.conduitMap.end())
        continue;
      ConduitInfo &cinfo = it->second;

      auto resolved = cinfo.resolveForTile(op);
      AIE::LockOp resolvedProdLock = resolved.prodLock;
      AIE::LockOp resolvedConsLock = resolved.consLock;

      AIE::LockOp lock =
          (ainfo.port == Port::Produce) ? resolvedProdLock : resolvedConsLock;
      if (lock) {
        int32_t acqVal = isAIE2 ? static_cast<int32_t>(ainfo.count)
                                : (ainfo.port == Port::Produce ? 0 : 1);
        builder.create<AIE::UseLockOp>(op.getLoc(), lock.getResult(),
                                       acqAction, acqVal);
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
    auto it = state.conduitMap.find(conduitName.str());
    if (it == state.conduitMap.end()) {
      releaseAsyncsToErase.push_back(op);
      return;
    }
    ConduitInfo &cinfo = it->second;

    Port port = op.getPort();
    auto resolved = cinfo.resolveForTile(op);
    AIE::LockOp resolvedProdLock = resolved.prodLock;
    AIE::LockOp resolvedConsLock = resolved.consLock;
    AIE::BufferOp resolvedRotationBuf = resolved.rotationBuf;

    builder.setInsertionPoint(op);
    int64_t count = static_cast<int64_t>(op.getCount());
    AIE::LockOp lock =
        (port == Port::Consume) ? resolvedProdLock : resolvedConsLock;
    if (lock) {
      int32_t relVal = isAIE2 ? static_cast<int32_t>(count)
                              : (port == Port::Consume ? 0 : 1);
      builder.create<AIE::UseLockOp>(op.getLoc(), lock.getResult(),
                                     AIE::LockAction::Release, relVal);
    }
    // Counter increment for depth>1 Consume port.
    if (resolvedRotationBuf && port == Port::Consume && cinfo.depth > 1) {
      mlir::Location loc = op.getLoc();
      mlir::Type i32Ty = mlir::IntegerType::get(ctx, 32);
      mlir::Value c0 = builder.create<mlir::arith::ConstantIndexOp>(loc, 0);
      mlir::Value curI32 = builder.create<mlir::memref::LoadOp>(
          loc, resolvedRotationBuf.getResult(), mlir::ValueRange{c0});
      mlir::Value incI32 = mlir::arith::ConstantIntOp::create(
          builder, loc, i32Ty, count);
      mlir::Value newVal =
          builder.create<mlir::arith::AddIOp>(loc, curI32, incI32);
      mlir::Value depthI32 = mlir::arith::ConstantIntOp::create(
          builder, loc, i32Ty, cinfo.depth);
      mlir::Value result =
          builder.create<mlir::arith::RemUIOp>(loc, newVal, depthI32);
      builder.create<mlir::memref::StoreOp>(
          loc, result, resolvedRotationBuf.getResult(), mlir::ValueRange{c0});
    }
    releaseAsyncsToErase.push_back(op);
  });
  for (auto op : releaseAsyncsToErase)
    op.erase();

  // Steps 8e-8h: Erase put/get memref ops (collect-then-erase).
  {
    llvm::SmallVector<PutMemrefAsync> toErase;
    module.walk([&](PutMemrefAsync op) { toErase.push_back(op); });
    for (auto op : llvm::reverse(toErase))
      op.erase();
  }
  {
    llvm::SmallVector<GetMemrefAsync> toErase;
    module.walk([&](GetMemrefAsync op) { toErase.push_back(op); });
    for (auto op : llvm::reverse(toErase))
      op.erase();
  }
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
