//===- HomogeneousRepeatPattern.h --------------------------------*-C++-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2026 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//
//
// Same-offset / structurally-identical N-put (or N-get) collapse pattern
// for --conduit-canonicalize-channel-puts.
//
//===----------------------------------------------------------------------===//

#ifndef AIE_DIALECT_CONDUIT_TRANSFORMS_PATTERNS_HOMOGENEOUSREPEATPATTERN_H
#define AIE_DIALECT_CONDUIT_TRANSFORMS_PATTERNS_HOMOGENEOUSREPEATPATTERN_H

namespace mlir {
class MLIRContext;
class RewritePatternSet;
} // namespace mlir

namespace xilinx::conduit::detail {

// Register the homogeneous-repeat collapse patterns (puts + gets) on
// `patterns`.  These are file-local OpRewritePattern subclasses defined in
// HomogeneousRepeatPattern.cpp.
void populateHomogeneousRepeatPatterns(mlir::RewritePatternSet &patterns,
                                       mlir::MLIRContext *ctx);

} // namespace xilinx::conduit::detail

#endif // AIE_DIALECT_CONDUIT_TRANSFORMS_PATTERNS_HOMOGENEOUSREPEATPATTERN_H
