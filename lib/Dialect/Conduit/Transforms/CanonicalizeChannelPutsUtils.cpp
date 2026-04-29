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

int64_t getDmaRepeatOr1(Create createOp) {
  if (auto rep = createOp.getDmaRepeat())
    return static_cast<int64_t>(*rep);
  return 1;
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
  return raw * detail::getDmaRepeatOr1(channel);
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
  return raw * detail::getDmaRepeatOr1(channel);
}

} // namespace xilinx::conduit
