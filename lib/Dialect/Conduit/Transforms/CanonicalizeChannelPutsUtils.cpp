//===- CanonicalizeChannelPutsUtils.cpp - shared canon helpers --*-C++-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2026 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#include "CanonicalizeChannelPutsUtils.h"
#include "ConduitTileInference.h"

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIE/IR/AIETargetModel.h"
#include "aie/Dialect/Conduit/IR/ConduitDialect.h"
#include "aie/Dialect/Conduit/Transforms/ConduitPasses.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringMap.h"

using namespace mlir;
using namespace xilinx;
using namespace xilinx::conduit;

namespace {

// Two DenseI64ArrayAttrs are structurally equal iff both null or same array.
bool denseI64ArrayEq(DenseI64ArrayAttr a, DenseI64ArrayAttr b) {
  if (!a && !b)
    return true;
  if (!a || !b)
    return false;
  return a == b;
}

// Producer-dimensions optional attr equality (cheap pointer eq via uniqued
// attrs).
bool optionalAttrEq(Attribute a, Attribute b) {
  if (!a && !b)
    return true;
  if (!a || !b)
    return false;
  return a == b;
}

} // namespace

namespace xilinx::conduit::detail {

int64_t getDmaRepeatOr0(Create createOp) {
  // 0-indexed convention (Bug #98 / Task #39): dma_repeat encodes
  // "additional fires beyond the initial one."  Absent attribute = 0 =
  // single fire.  Pass A's IRON path stamps via
  // ConduitDmaTaskToConduit.cpp:511 with the value IRON computed as
  // `sizes[0] - 1` per aiex.py:289-291; canon's HomogeneousRepeatPattern
  // stamps `N - 1` for N collapsed puts; Pass C surfaces the value
  // verbatim onto configure_task.repeat_count → firmware fires
  // `value + 1` times (AIEDmaToNpu.cpp:180-183 packs verbatim into the
  // NPU push-queue command word).
  if (auto rep = createOp.getDmaRepeat())
    return static_cast<int64_t>(*rep);
  return 0;
}

bool putsAreStructurallyIdentical(PutMemrefAsync a, PutMemrefAsync b) {
  if (a.getNameAttr() != b.getNameAttr())
    return false;
  if (a.getNumElems() != b.getNumElems())
    return false;
  if (!denseI64ArrayEq(a.getOffsetsAttr(), b.getOffsetsAttr()))
    return false;
  if (!denseI64ArrayEq(a.getSizesAttr(), b.getSizesAttr()))
    return false;
  if (!denseI64ArrayEq(a.getStridesAttr(), b.getStridesAttr()))
    return false;
  if (!optionalAttrEq(a.getProducerDimensionsAttr(),
                      b.getProducerDimensionsAttr()))
    return false;
  return true;
}

bool getsAreStructurallyIdentical(GetMemrefAsync a, GetMemrefAsync b) {
  if (a.getNameAttr() != b.getNameAttr())
    return false;
  if (a.getNumElems() != b.getNumElems())
    return false;
  if (!denseI64ArrayEq(a.getOffsetsAttr(), b.getOffsetsAttr()))
    return false;
  if (!denseI64ArrayEq(a.getSizesAttr(), b.getSizesAttr()))
    return false;
  if (!denseI64ArrayEq(a.getStridesAttr(), b.getStridesAttr()))
    return false;
  if (!optionalAttrEq(a.getConsumerDimensionsAttr(),
                      b.getConsumerDimensionsAttr()))
    return false;
  return true;
}

std::optional<llvm::SmallVector<WaitAll>> collectSyncChain(Value tok) {
  llvm::SmallVector<WaitAll> chain;
  for (Operation *user : tok.getUsers()) {
    auto wa = dyn_cast<WaitAll>(user);
    if (!wa)
      return std::nullopt;
    if (wa.getTokens().size() != 1)
      return std::nullopt;
    if (wa.getTokens()[0] != tok)
      return std::nullopt;
    chain.push_back(wa);
  }
  // Sort in IR order for deterministic chain shape comparison.
  llvm::sort(chain, [](WaitAll a, WaitAll b) {
    return a.getOperation()->isBeforeInBlock(b.getOperation());
  });
  return chain;
}

llvm::SmallVector<bool> chainShape(ArrayRef<WaitAll> chain) {
  llvm::SmallVector<bool> shape;
  shape.reserve(chain.size());
  for (WaitAll w : chain)
    shape.push_back(w.getToken());
  return shape;
}

AIE::DeviceOp findEnclosingDevice(Operation *op) {
  return op->getParentOfType<AIE::DeviceOp>();
}

Value lookupProducerTile(Operation *scope, StringRef channelName) {
  llvm::StringMap<InferredTiles> inferredMap = inferAllTiles(scope);
  auto it = inferredMap.find(channelName);
  if (it == inferredMap.end())
    return nullptr;
  return it->second.producerTile;
}

Value lookupConsumerTile(Operation *scope, StringRef channelName) {
  llvm::StringMap<InferredTiles> inferredMap = inferAllTiles(scope);
  auto it = inferredMap.find(channelName);
  if (it == inferredMap.end())
    return nullptr;
  // Prefer a non-shim consumer; if absent, fall back to the first shim
  // consumer (e.g., compute → shim sink).
  if (!it->second.consumerTiles.empty())
    return it->second.consumerTiles.front();
  if (!it->second.shimConsumerTiles.empty())
    return it->second.shimConsumerTiles.front();
  return nullptr;
}

std::optional<uint32_t> tileBDCap(Operation *scope, Value tile) {
  AIE::DeviceOp dev = scope->getParentOfType<AIE::DeviceOp>();
  if (!dev)
    dev = dyn_cast<AIE::DeviceOp>(scope);
  if (!dev)
    return std::nullopt;
  if (!tile)
    return std::nullopt;
  auto tileOp = tile.getDefiningOp<AIE::TileOp>();
  if (!tileOp)
    return std::nullopt;
  const AIE::AIETargetModel &tm = AIE::getTargetModel(dev);
  return tm.getNumBDs(static_cast<int>(tileOp.getCol()),
                      static_cast<int>(tileOp.getRow()));
}

bool isLinkedChannel(Operation *scope, StringRef chanName) {
  if (!scope)
    return false;
  // Inspection pattern mirrors ConduitDepthPromotion.cpp:79-103
  // (`collectLinkedConduitNames`).  Walk Scatter/Gather/Transpose ops
  // (the lowered form of aie.objectfifo.link emitted by Pass A) and
  // check whether `chanName` appears as a `src`/`srcs`/`dst`/`dsts`
  // symbol-ref attribute.  We scope the filter to the three relay ops
  // rather than every op (collectLinkedConduitNames does the same)
  // because the attribute names `src`/`dst`/`srcs`/`dsts` are also
  // used elsewhere in MLIR and could otherwise produce false positives.
  bool found = false;
  auto match = [&](Operation *op) {
    if (auto srcsAttr = op->getAttrOfType<ArrayAttr>("srcs")) {
      for (Attribute s : srcsAttr)
        if (auto sym = dyn_cast<FlatSymbolRefAttr>(s))
          if (sym.getValue() == chanName) {
            found = true;
            return;
          }
    }
    if (auto dstsAttr = op->getAttrOfType<ArrayAttr>("dsts")) {
      for (Attribute d : dstsAttr)
        if (auto sym = dyn_cast<FlatSymbolRefAttr>(d))
          if (sym.getValue() == chanName) {
            found = true;
            return;
          }
    }
    if (auto srcAttr = op->getAttrOfType<FlatSymbolRefAttr>("src"))
      if (srcAttr.getValue() == chanName) {
        found = true;
        return;
      }
    if (auto dstAttr = op->getAttrOfType<FlatSymbolRefAttr>("dst"))
      if (dstAttr.getValue() == chanName) {
        found = true;
        return;
      }
  };
  scope->walk([&](Operation *op) {
    if (!isa<ScatterOp, GatherOp, TransposeOp>(op))
      return WalkResult::advance();
    match(op);
    if (found)
      return WalkResult::interrupt();
    return WalkResult::advance();
  });
  return found;
}

std::optional<uint32_t> tileBDDimCap(Operation *scope, Value tile) {
  AIE::DeviceOp dev = scope->getParentOfType<AIE::DeviceOp>();
  if (!dev)
    dev = dyn_cast<AIE::DeviceOp>(scope);
  if (!dev)
    return std::nullopt;
  if (!tile)
    return std::nullopt;
  auto tileOp = tile.getDefiningOp<AIE::TileOp>();
  if (!tileOp)
    return std::nullopt;
  const AIE::AIETargetModel &tm = AIE::getTargetModel(dev);
  int col = static_cast<int>(tileOp.getCol());
  int row = static_cast<int>(tileOp.getRow());
  // Authoritative caps:
  //   compute/core tile dma_bd  -> 3 (AIEDialect.cpp:2233-2236, default branch)
  //   MemTile dma_bd            -> 4 (AIEDialect.cpp:2233-2236, MemTileDMAOp)
  //   Shim runtime-sequence BD  -> 4 (AIEDMATasksToNPU.cpp:347-350)
  if (tm.isCoreTile(col, row))
    return 3;
  if (tm.isMemTile(col, row))
    return 4;
  if (tm.isShimNOCorPLTile(col, row))
    return 4;
  return std::nullopt;
}

} // namespace xilinx::conduit::detail

//===----------------------------------------------------------------------===//
// Public utility helpers (declared in ConduitPasses.h)
//===----------------------------------------------------------------------===//

namespace xilinx::conduit {

int64_t getEffectivePutCount(Create channel) {
  if (!channel)
    return 0;
  StringRef name = channel.getName();
  AIE::DeviceOp dev = channel->getParentOfType<AIE::DeviceOp>();
  Operation *scope = dev ? dev.getOperation()
                         : channel->getParentOfType<ModuleOp>().getOperation();
  if (!scope)
    return 0;
  int64_t raw = 0;
  scope->walk([&](PutMemrefAsync p) {
    if (p.getName() == name)
      ++raw;
  });
  // dma_repeat is 0-indexed ("additional fires"); total fires = 1 + value.
  // See getDmaRepeatOr0 docstring.  Bug #98 / Task #39.
  return raw * (1 + detail::getDmaRepeatOr0(channel));
}

int64_t getEffectiveGetCount(Create channel) {
  if (!channel)
    return 0;
  StringRef name = channel.getName();
  AIE::DeviceOp dev = channel->getParentOfType<AIE::DeviceOp>();
  Operation *scope = dev ? dev.getOperation()
                         : channel->getParentOfType<ModuleOp>().getOperation();
  if (!scope)
    return 0;
  int64_t raw = 0;
  scope->walk([&](GetMemrefAsync g) {
    if (g.getName() == name)
      ++raw;
  });
  // dma_repeat is 0-indexed ("additional fires"); total fires = 1 + value.
  // See getDmaRepeatOr0 docstring.  Bug #98 / Task #39.
  return raw * (1 + detail::getDmaRepeatOr0(channel));
}

} // namespace xilinx::conduit
