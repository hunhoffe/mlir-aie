//===- ConduitToDMARoute.cpp - Phase 4-4.5a: flow + shim DMA ----*-C++-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2026 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//
//
// Phase 4: Handle shim-tile endpoints → aie.shim_dma_allocation + aie.flow.
//   4a: Producer is shim (row==0): shim sends MM2S to compute consumer.
//   4b: Consumer is shim (row==0): compute producer sends MM2S to shim S2MM.
//
// Phase 4.5: Rewrite aiex.npu.dma_wait / dma_memcpy_nd symbol references.
//
// Phase 4.5a: Emit aie.flow for non-adjacent conduits (compute→compute,
//             compute→MemTile, MemTile→compute).
//
//===----------------------------------------------------------------------===//

#include "ConduitToDMACommon.h"

namespace xilinx::conduit {

void routePhase(ConduitToDMAState &state) {
  if (!state.deviceOp)
    return;

  mlir::OpBuilder &builder = *state.builder;
  mlir::MLIRContext *ctx = state.ctx;
  const bool isAIE2 = state.isAIE2Plus();

  // -----------------------------------------------------------------------
  // Phase 4: Shim endpoints.
  // -----------------------------------------------------------------------

  for (auto &[name, info] : state.conduitMap) {
    auto [prodCol, prodRow] = info.producerTileCoord;
    if (prodCol < 0)
      continue;

    // --- Sub-case 4a: producer is a shim tile (row==0) ---
    // Broadcast: emit one aie.flow per consumer tile.
    if (prodRow == 0) {
      if (info.consumerTileCoords.empty())
        continue;

      AIE::TileOp shimTile = state.lookupTileByCoord(prodCol, prodRow);
      if (!shimTile)
        continue;

      builder.setInsertionPoint(state.deviceBody->getTerminator());

      // Shim-side prod/cons locks (AIE2 only).
      if (isAIE2) {
        int lockIdx = state.lockIdCounter[shimTile.getResult()]++;
        std::string symName = name + "_prod_lock_0";
        // prod_lock init=depth: all slots initially free (host can write).
        int64_t shimDepth = info.depth > 0 ? info.depth : 1;
        AIE::LockOp lk = builder.create<AIE::LockOp>(
            state.deviceOp.getLoc(), shimTile.getResult(), lockIdx,
            static_cast<int>(shimDepth));
        lk.setSymNameAttr(mlir::StringAttr::get(ctx, symName));
      }
      if (isAIE2) {
        int lockIdx = state.lockIdCounter[shimTile.getResult()]++;
        std::string symName = name + "_cons_lock_0";
        AIE::LockOp lk = builder.create<AIE::LockOp>(
            state.deviceOp.getLoc(), shimTile.getResult(), lockIdx,
            static_cast<int>(0));
        lk.setSymNameAttr(mlir::StringAttr::get(ctx, symName));
      }

      // aie.shim_dma_allocation (MM2S, channel 0).
      std::string allocSym = name + "_shim_alloc";
      state.shimConduitNames.insert(name);
      if (!mlir::SymbolTable::lookupSymbolIn(
              state.deviceOp, mlir::StringAttr::get(ctx, allocSym)))
        builder.create<AIE::ShimDMAAllocationOp>(
            state.deviceOp.getLoc(), allocSym, shimTile.getResult(),
            AIE::DMAChannelDir::MM2S,
            /*channel_index=*/static_cast<int64_t>(0),
            /*plio=*/false,
            /*packet=*/nullptr);

      // One flow per consumer tile (broadcast).
      for (unsigned consIdx = 0;
           consIdx < info.consumerTileCoords.size(); ++consIdx) {
        auto [consCol, consRow] = info.consumerTileCoords[consIdx];
        AIE::TileOp consTile = state.lookupTileByCoord(consCol, consRow);
        if (!consTile)
          continue;
        state.emitFlow(info.routingMode, shimTile.getResult(),
                       AIE::WireBundle::DMA, static_cast<int32_t>(consIdx),
                       consTile.getResult(), AIE::WireBundle::DMA,
                       static_cast<int32_t>(0));
      }
    }

    // --- Sub-case 4b: consumer is a shim tile (row==0) ---
    for (auto [shimCol, shimRow] : info.shimConsumerTileCoords) {
      if (shimRow != 0)
        continue;

      AIE::TileOp prodTile = state.lookupTileByCoord(prodCol, prodRow);
      if (!prodTile)
        continue;

      AIE::TileOp shimTile = state.lookupTileByCoord(shimCol, shimRow);
      if (!shimTile)
        continue;

      builder.setInsertionPoint(state.deviceBody->getTerminator());

      std::string allocSym = name + "_shim_alloc";
      state.shimConduitNames.insert(name);
      if (!mlir::SymbolTable::lookupSymbolIn(
              state.deviceOp, mlir::StringAttr::get(ctx, allocSym)))
        builder.create<AIE::ShimDMAAllocationOp>(
            state.deviceOp.getLoc(), allocSym, shimTile.getResult(),
            AIE::DMAChannelDir::S2MM,
            /*channel_index=*/static_cast<int64_t>(0),
            /*plio=*/false,
            /*packet=*/nullptr);

      state.emitFlow(info.routingMode, prodTile.getResult(),
                     AIE::WireBundle::DMA, static_cast<int32_t>(0),
                     shimTile.getResult(), AIE::WireBundle::DMA,
                     static_cast<int32_t>(0));
    }
  }

  // -----------------------------------------------------------------------
  // Phase 4.5: Rewrite symbol references from @<name> to @<name>_shim_alloc.
  // -----------------------------------------------------------------------
  if (!state.shimConduitNames.empty()) {
    state.module.walk([&](mlir::Operation *op) {
      llvm::StringRef opName = op->getName().getStringRef();

      if (opName == "aiex.npu.dma_wait") {
        if (auto symAttr =
                op->getAttrOfType<mlir::FlatSymbolRefAttr>("symbol")) {
          llvm::StringRef ref = symAttr.getValue();
          if (state.shimConduitNames.count(ref)) {
            op->setAttr("symbol",
                        mlir::FlatSymbolRefAttr::get(
                            ctx, (ref + "_shim_alloc").str()));
          }
        }
        return;
      }

      if (opName == "aiex.npu.dma_memcpy_nd") {
        if (auto symAttr =
                op->getAttrOfType<mlir::FlatSymbolRefAttr>("metadata")) {
          llvm::StringRef ref = symAttr.getValue();
          if (state.shimConduitNames.count(ref)) {
            op->setAttr("metadata",
                        mlir::FlatSymbolRefAttr::get(
                            ctx, (ref + "_shim_alloc").str()));
          }
        } else if (auto symAttr =
                       op->getAttrOfType<mlir::SymbolRefAttr>("metadata")) {
          llvm::StringRef ref = symAttr.getRootReference().getValue();
          if (!state.shimConduitNames.count(ref))
            return;
          op->setAttr("metadata",
                      mlir::SymbolRefAttr::get(
                          mlir::StringAttr::get(
                              ctx, (ref + "_shim_alloc").str()),
                          symAttr.getNestedReferences()));
        }
        return;
      }
    });
  }

  // -----------------------------------------------------------------------
  // Phase 4.5a: Emit aie.flow for non-adjacent conduits.
  //
  // Fused channel groups: conduits with the same fused_dma_channel_group
  // label share one hardware MM2S channel slot.
  // -----------------------------------------------------------------------

  for (auto &[name, info] : state.conduitMap) {
    if (info.sharedMemory)
      continue;
    if (state.linkSrcNamesEarly.count(name) || state.linkJoinSrcNames.count(name))
      continue;
    auto [prodCol, prodRow] = info.producerTileCoord;
    if (prodCol < 0 || prodRow == 0)
      continue;
    if (info.consumerTileCoords.empty())
      continue;

    AIE::TileOp prodTile = state.lookupTileByCoord(prodCol, prodRow);
    if (!prodTile)
      continue;
    mlir::Value prodTileVal = prodTile.getResult();

    if (!info.consumerTileBuffers.count(prodTileVal))
      continue;

    builder.setInsertionPoint(state.deviceBody->getTerminator());

    // Assign MM2S channel (fused groups share a channel).
    int32_t mm2sChannel;
    if (!info.fuseGroup.empty()) {
      auto it = state.fuseGroupMM2SChannel.find(info.fuseGroup);
      if (it != state.fuseGroupMM2SChannel.end()) {
        mm2sChannel = it->second;
      } else {
        mm2sChannel = state.tileNextMM2SChannel[prodTileVal]++;
        state.fuseGroupMM2SChannel[info.fuseGroup] = mm2sChannel;
      }
      state.fuseGroupMembers[info.fuseGroup].push_back(name);
    } else {
      mm2sChannel = state.tileNextMM2SChannel[prodTileVal]++;
    }
    state.conduitMM2SChannel[name] = mm2sChannel;

    for (unsigned consIdx = 0; consIdx < info.consumerTileCoords.size();
         ++consIdx) {
      auto [consCol, consRow] = info.consumerTileCoords[consIdx];
      if (consRow == 0)
        continue;

      // For single-consumer conduits, adjacent tiles use shared memory
      // (Phase 3c) — no DMA flow needed.  For broadcast (multi-consumer),
      // Phase 3c is skipped; all consumers use DMA, so flows are needed
      // for every consumer regardless of adjacency.
      if (info.consumerTileCoords.size() == 1) {
        bool rightAdj = state.targetModel->isLegalMemAffinity(
            prodCol, prodRow, consCol, consRow);
        bool leftAdj = state.targetModel->isLegalMemAffinity(
            consCol, consRow, prodCol, prodRow);
        if (rightAdj || leftAdj)
          continue;
      }

      AIE::TileOp consTile = state.lookupTileByCoord(consCol, consRow);
      if (!consTile)
        continue;

      mlir::Value consTileVal = consTile.getResult();
      int32_t s2mmChannel = state.tileNextS2MMChannel[consTileVal]++;
      state.conduitConsS2MMChannel[{name, consIdx}] = s2mmChannel;

      state.emitFlow(info.routingMode, prodTileVal,
                     AIE::WireBundle::DMA, mm2sChannel,
                     consTileVal, AIE::WireBundle::DMA, s2mmChannel);
    }
  }
}

} // namespace xilinx::conduit
