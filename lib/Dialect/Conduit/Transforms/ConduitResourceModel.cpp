//===- ConduitResourceModel.cpp - Per-tile resource accounting ---*-C++-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2026 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#include "ConduitResourceModel.h"

#include "aie/Dialect/Conduit/IR/ConduitDialect.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"

#include "llvm/ADT/SmallVector.h"

namespace xilinx::conduit {

int64_t estimateSingleSlotBytes(mlir::Type elemType) {
  auto mref = mlir::dyn_cast<mlir::MemRefType>(elemType);
  if (!mref)
    return 4; // default 4 bytes per element
  int64_t elemBits = mref.getElementTypeBitWidth();
  int64_t elemCount = 1;
  for (int64_t d : mref.getShape()) {
    if (mlir::ShapedType::isDynamic(d))
      return 4; // can't determine statically
    elemCount *= d;
  }
  return (elemBits / 8) * elemCount;
}

void populateConduitResourceModel(
    mlir::ModuleOp module,
    const llvm::StringMap<InferredTiles> &inferredMap,
    ConduitResourceModel &model) {
  module.walk([&](Create op) {
    // Cascade conduits use no buffers, locks, or BDs — skip resource counting.
    if (auto rm = op.getRoutingMode())
      if (*rm == RoutingMode::Cascade)
        return;

    auto depthAttr = op->getAttrOfType<mlir::IntegerAttr>("depth");
    int64_t depth = depthAttr ? depthAttr.getInt() : 1;
    auto elemTypeAttr = op->getAttrOfType<mlir::TypeAttr>("element_type");

    // Get consumer tile coords: prefer inference, fallback to attribute
    // for channels outside aie.core (e.g. in func.func or hand-written IR).
    llvm::SmallVector<std::pair<int64_t, int64_t>> consCoords;
    auto tileIt = inferredMap.find(op.getName().str());
    if (tileIt != inferredMap.end() &&
        !tileIt->second.consumerTiles.empty()) {
      for (mlir::Value tv : tileIt->second.consumerTiles) {
        auto [col, row] = extractCoord(tv);
        if (col >= 0)
          consCoords.push_back({col, row});
      }
    }

    // Estimate per-consumer resources.
    for (auto [col, row] : consCoords) {
      int64_t key = tileKey(col, row);
      model.lockCount[key] += 2; // prod + cons lock pair
      model.bdCount[key] += depth;
      if (elemTypeAttr) {
        int64_t perSlotBytes =
            estimateSingleSlotBytes(elemTypeAttr.getValue());
        model.memUsed[key] += perSlotBytes * depth;
      }
    }
    // Producer tile also uses resources for non-shim.
    std::pair<int64_t, int64_t> prodCoord = {-1, -1};
    if (tileIt != inferredMap.end() && tileIt->second.producerTile) {
      prodCoord = extractCoord(tileIt->second.producerTile);
    }
    if (prodCoord.first >= 0 && prodCoord.second != 0) { // non-shim
      int64_t key = tileKey(prodCoord.first, prodCoord.second);
      model.lockCount[key] += 2;
      model.bdCount[key] += depth;
      if (elemTypeAttr) {
        int64_t perSlotBytes =
            estimateSingleSlotBytes(elemTypeAttr.getValue());
        model.memUsed[key] += perSlotBytes * depth;
      }
    }
  });
}

} // namespace xilinx::conduit
