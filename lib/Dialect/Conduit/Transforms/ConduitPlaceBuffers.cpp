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
// --conduit-materialize-buffers (named <chan>_cons_buff_N), assign
// mem_bank = i % numBanks so that consecutive FIFO slots land in different
// SRAM banks.
//
// Runs between --conduit-materialize-buffers and --aie-assign-buffer-addresses.
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

    // Group aie.buffer ops by channel name using the naming convention
    // emitted by --conduit-materialize-buffers: "<chan>_cons_buff_<N>".
    // For each group, assign mem_bank = slot_index % numBanks.
    llvm::StringMap<llvm::SmallVector<AIE::BufferOp>> channelBuffers;
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
      if (rest.empty() || rest.find_first_not_of("0123456789") != llvm::StringRef::npos)
        return;
      llvm::StringRef chanName = name.take_front(pos);
      channelBuffers[chanName].push_back(bufOp);
    });

    for (auto &[chanName, bufs] : channelBuffers) {
      // Sort by slot index (the numeric suffix) so bank assignment is stable.
      llvm::sort(bufs, [](AIE::BufferOp a, AIE::BufferOp b) {
        llvm::StringRef na = *a.getSymName(), nb = *b.getSymName();
        static constexpr llvm::StringLiteral kSuffix = "_cons_buff_";
        int64_t ia = 0, ib = 0;
        na.drop_front(na.rfind(kSuffix) + kSuffix.size()).getAsInteger(10, ia);
        nb.drop_front(nb.rfind(kSuffix) + kSuffix.size()).getAsInteger(10, ib);
        return ia < ib;
      });
      for (auto [i, bufOp] : llvm::enumerate(bufs)) {
        if (!bufOp.getMemBank()) {
          bufOp.setMemBankAttr(mlir::IntegerAttr::get(
              mlir::IntegerType::get(ctx, 32),
              static_cast<int64_t>(i) % kDefaultNumBanks));
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
