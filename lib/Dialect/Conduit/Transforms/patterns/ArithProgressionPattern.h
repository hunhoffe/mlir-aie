//===- ArithProgressionPattern.h ---------------------------------*-C++-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2026 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//
//
// 1D arith-progression collapse pattern for
// --conduit-canonicalize-channel-puts.  Detects N otherwise-identical
// put_memref_async (or get_memref_async) ops on a single channel whose
// leading offset entry forms an arithmetic progression
// (offsets[i][0] = base + i × stride).  Rewrites to 1 surviving op + an
// outer wrap+stride dim prepended to producer_dimensions /
// consumer_dimensions and dma_repeat = N stamped on conduit.create.
//
//===----------------------------------------------------------------------===//

#ifndef AIE_DIALECT_CONDUIT_TRANSFORMS_PATTERNS_ARITHPROGRESSIONPATTERN_H
#define AIE_DIALECT_CONDUIT_TRANSFORMS_PATTERNS_ARITHPROGRESSIONPATTERN_H

namespace mlir {
class MLIRContext;
class RewritePatternSet;
} // namespace mlir

namespace xilinx::conduit::detail {

// Register the arith-progression collapse patterns (puts + gets) on
// `patterns`.  These are file-local OpRewritePattern subclasses defined in
// ArithProgressionPattern.cpp.
void populateArithProgressionPatterns(mlir::RewritePatternSet &patterns,
                                      mlir::MLIRContext *ctx);

} // namespace xilinx::conduit::detail

#endif // AIE_DIALECT_CONDUIT_TRANSFORMS_PATTERNS_ARITHPROGRESSIONPATTERN_H
