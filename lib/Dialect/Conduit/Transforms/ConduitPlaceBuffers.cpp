//===- ConduitPlaceBuffers.cpp - DMA-aware mem_bank assignment ----*-C++-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2026 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//
//
// --conduit-place-buffers: for each group of aie.buffer ops emitted by
// Pass C (ConduitToDMAAlloc) (named <chan>_cons_buff_N), assign
// mem_bank = i % numBanks so that consecutive FIFO slots land in different
// SRAM banks.
//
// Runs between --conduit-to-dma and --aie-assign-buffer-addresses.
// Eliminates DMA↔core bank conflicts that degrade throughput when producer
// and consumer access the same bank simultaneously.
//
// AIE2 compute tiles have 4 SRAM banks (banks 0-3).
//===----------------------------------------------------------------------===//

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/Conduit/Transforms/ConduitPasses.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"

using namespace xilinx::AIE;

namespace xilinx::conduit {

#define GEN_PASS_DEF_CONDUITPLACEBUFFERS
#include "aie/Dialect/Conduit/Transforms/ConduitPasses.h.inc"

namespace {

// AIE2 compute tiles: 4 SRAM banks.
static constexpr int64_t kDefaultNumBanks = 4;

struct ConduitPlaceBuffersPass
    : impl::ConduitPlaceBuffersBase<ConduitPlaceBuffersPass> {

  void runOnOperation() override {
    mlir::ModuleOp module = getOperation();
    mlir::MLIRContext *ctx = module.getContext();

    // Collect consumer buffers per tile using the naming convention emitted
    // by Pass C (ConduitToDMAAlloc): "<chan>_cons_buff_<N>".
    // Assign mem_bank per-tile across all channel groups so that buffers
    // from different channels don't collide in the same bank.
    llvm::DenseMap<mlir::Operation *, llvm::SmallVector<AIE::BufferOp>>
        tileBuffers;
    module.walk([&](AIE::BufferOp bufOp) {
      auto symName = bufOp.getSymName();
      if (!symName)
        return;
      // Match the suffix "_cons_buff_<N>".
      llvm::StringRef name = *symName;
      static constexpr llvm::StringLiteral kSuffix = "_cons_buff_";
      auto pos = name.rfind(kSuffix);
      if (pos == llvm::StringRef::npos)
        return;
      // Verify everything after the suffix is digits (the slot index).
      llvm::StringRef rest = name.drop_front(pos + kSuffix.size());
      if (rest.empty() ||
          rest.find_first_not_of("0123456789") != llvm::StringRef::npos)
        return;
      tileBuffers[bufOp.getTile().getDefiningOp()].push_back(bufOp);
    });

    for (auto &[tile, bufs] : tileBuffers) {
      // Sort by name for deterministic ordering. This naturally groups
      // buffers from the same channel together (same prefix), keeping
      // consecutive FIFO slots in different banks for double-buffering.
      llvm::sort(bufs, [](AIE::BufferOp a, AIE::BufferOp b) {
        return a.getSymName()->compare(*b.getSymName()) < 0;
      });
      int bankIdx = 0;
      for (auto bufOp : bufs) {
        if (!bufOp.getMemBank()) {
          bufOp.setMemBankAttr(mlir::IntegerAttr::get(
              mlir::IntegerType::get(ctx, 32),
              static_cast<int64_t>(bankIdx) % kDefaultNumBanks));
          ++bankIdx;
        }
      }
    }
  }
};

} // namespace

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createConduitPlaceBuffersPass() {
  return std::make_unique<ConduitPlaceBuffersPass>();
}

} // namespace xilinx::conduit
