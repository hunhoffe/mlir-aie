//===- LoopAnalysisUtils.h - Loop / affine-map analysis helpers -----------===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2026 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//
//
// Generic loop / affine analysis utilities ported from mlir-air's
// `air::Util` for reuse inside Conduit transforms.  Pure affine; no
// dialect-specific dependencies.
//
//===----------------------------------------------------------------------===//

#ifndef AIE_DIALECT_CONDUIT_TRANSFORMS_LOOPANALYSISUTILS_H
#define AIE_DIALECT_CONDUIT_TRANSFORMS_LOOPANALYSISUTILS_H

#include "mlir/IR/AffineMap.h"
#include "llvm/ADT/ArrayRef.h"

#include <cstdint>
#include <optional>

namespace xilinx {
namespace conduit {

// Evaluate the affine expression of `map` on a sparse vector of constant
// inputs.  The single-input overload infers whether the inputs bind to the
// map's symbols or its dims based on whichever count matches.  Returns
// std::nullopt if the simplified map does not collapse to a single constant
// (or the input arity does not match symbol/dim count).
std::optional<int64_t>
evaluateConstantsInMap(mlir::AffineMap map,
                       llvm::ArrayRef<std::optional<int64_t>> symAndDimInputs,
                       mlir::MLIRContext *ctx);

std::optional<int64_t>
evaluateConstantsInMap(mlir::AffineMap map,
                       llvm::ArrayRef<std::optional<int64_t>> symbolInputs,
                       llvm::ArrayRef<std::optional<int64_t>> dimInputs,
                       mlir::MLIRContext *ctx);

} // namespace conduit
} // namespace xilinx

#endif // AIE_DIALECT_CONDUIT_TRANSFORMS_LOOPANALYSISUTILS_H
