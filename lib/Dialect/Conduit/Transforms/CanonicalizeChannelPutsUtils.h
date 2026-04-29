//===- CanonicalizeChannelPutsUtils.h - shared canon helpers --*-C++-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2026 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//
//
// Shared analysis helpers for --conduit-canonicalize-channel-puts and
// --conduit-expand-channel-puts.  Behavior is unchanged from the legacy
// monolithic implementation; this header simply makes the helpers
// reachable from the patterns/ subdirectory.
//
//===----------------------------------------------------------------------===//

#ifndef AIE_DIALECT_CONDUIT_TRANSFORMS_CANONICALIZECHANNELPUTSUTILS_H
#define AIE_DIALECT_CONDUIT_TRANSFORMS_CANONICALIZECHANNELPUTSUTILS_H

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/Conduit/IR/ConduitDialect.h"

#include "mlir/IR/Value.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"

#include <cstdint>
#include <optional>

namespace mlir {
class Operation;
} // namespace mlir

namespace xilinx::conduit::detail {

// Read the optional channel-level dma_repeat as int64.  Absent → 1.
int64_t getDmaRepeatOr1(::xilinx::conduit::Create createOp);

// Returns true iff two PutMemrefAsync ops are structurally identical for the
// purpose of loop-unroll collapse.
bool putsAreStructurallyIdentical(::xilinx::conduit::PutMemrefAsync a,
                                  ::xilinx::conduit::PutMemrefAsync b);

// Returns true iff two GetMemrefAsync ops are structurally identical for the
// purpose of loop-unroll collapse.
bool getsAreStructurallyIdentical(::xilinx::conduit::GetMemrefAsync a,
                                  ::xilinx::conduit::GetMemrefAsync b);

// Collect the WaitAll ops that consume a token (a put/get's getToken()).
std::optional<llvm::SmallVector<::xilinx::conduit::WaitAll>>
collectSyncChain(mlir::Value tok);

// Return the bool-attr "shape" of a sync chain — list of `token` attribute
// values.  Two chains with the same shape can be collapsed together.
llvm::SmallVector<bool>
chainShape(llvm::ArrayRef<::xilinx::conduit::WaitAll> chain);

// Find the parent DeviceOp for a Create op.
::xilinx::AIE::DeviceOp findEnclosingDevice(mlir::Operation *op);

// Look up the producer tile SSA Value for a channel using inferAllTiles.
mlir::Value lookupProducerTile(mlir::Operation *scope,
                               llvm::StringRef channelName);

// Look up the first non-shim consumer tile SSA Value for a channel.
mlir::Value lookupConsumerTile(mlir::Operation *scope,
                               llvm::StringRef channelName);

// Return the BD cap for `tile` on the active target model, or nullopt when
// the cap cannot be determined.
std::optional<uint32_t> tileBDCap(mlir::Operation *scope, mlir::Value tile);

} // namespace xilinx::conduit::detail

#endif // AIE_DIALECT_CONDUIT_TRANSFORMS_CANONICALIZECHANNELPUTSUTILS_H
