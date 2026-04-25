//===- LoopAnalysisUtils.cpp - Loop / affine-map analysis helpers ---------===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2026 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//
//
// Ported verbatim from mlir-air `air::evaluateConstantsInMap`
// (mlir-air/mlir/lib/Util/Util.cpp).  Namespace adapted to xilinx::conduit.
//
//===----------------------------------------------------------------------===//

#include "LoopAnalysisUtils.h"

#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"

using namespace mlir;

namespace xilinx {
namespace conduit {

std::optional<int64_t>
evaluateConstantsInMap(AffineMap map,
                       llvm::ArrayRef<std::optional<int64_t>> symAndDimInputs,
                       MLIRContext *ctx) {
  std::optional<int64_t> output = std::nullopt;
  if (map.getNumSymbols() == symAndDimInputs.size()) {
    return evaluateConstantsInMap(map, symAndDimInputs,
                                  llvm::ArrayRef<std::optional<int64_t>>{},
                                  ctx);
  } else if (map.getNumDims() == symAndDimInputs.size()) {
    return evaluateConstantsInMap(
        map, llvm::ArrayRef<std::optional<int64_t>>{}, symAndDimInputs, ctx);
  } else
    return output;
}

std::optional<int64_t>
evaluateConstantsInMap(AffineMap map,
                       llvm::ArrayRef<std::optional<int64_t>> symbolInputs,
                       llvm::ArrayRef<std::optional<int64_t>> dimInputs,
                       MLIRContext *ctx) {
  std::optional<int64_t> output = std::nullopt;
  if (map.getNumSymbols() != symbolInputs.size())
    return output;
  if (map.getNumDims() != dimInputs.size())
    return output;
  auto newmap = map;
  for (unsigned i = 0; i < map.getNumSymbols(); i++) {
    if (!symbolInputs[i])
      continue;
    auto c = getAffineConstantExpr(*symbolInputs[i], ctx);
    newmap =
        newmap.replace(getAffineSymbolExpr(i, ctx), c, 0, map.getNumSymbols());
  }
  for (unsigned i = 0; i < map.getNumDims(); i++) {
    if (!dimInputs[i])
      continue;
    auto c = getAffineConstantExpr(*dimInputs[i], ctx);
    newmap = newmap.replace(getAffineDimExpr(i, ctx), c, map.getNumDims(), 0);
  }
  output = simplifyAffineMap(newmap).getSingleConstantResult();
  return output;
}

} // namespace conduit
} // namespace xilinx
