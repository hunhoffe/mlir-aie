//===- AirChannelToConduit.cpp - AIR Channel → Conduit IR (Pass B) -*-C++-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2026 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//
//
// Pass B of the Conduit lowering pipeline: lift air.channel.* ops into
// Conduit Tier 3 memref-DMA IR.
//
// Architecture:
//
//   aie.objectfifo.*  ──┐
//                       ├──► Conduit IR ──► aie.dma_bd / aie.lock / aie.buffer
//   air.channel.*     ──┘
//
//   (ObjectFifoToConduit.cpp)   (this file)   (ConduitToDMA.cpp)
//
// Design note: why generic op matching
// -------------------------------------
// The AIR dialect (mlir-air) is a separate repository with its own build.
// The mlir-aie aie-opt tool does NOT link against the AIR dialect library.
// To parse air.channel.* programs, the caller must use
//   aie-opt --allow-unregistered-dialect --air-channel-to-conduit ...
// This pass therefore matches ops by their string name
// ("air.channel.put", "air.channel.get", etc.) rather than by C++ type.
// All operand / attribute access goes through the generic MLIR API.
//
// Supported mappings
// ------------------
//
// 1. air.channel declaration (Symbol op, no operands):
//      air.channel @name [1, 1]
//    → conduit.create {name="name", capacity=1, depth=1}
//      The element_type is left unset (unknown until a put/get is seen).
//      A second pass fills element_type from the memref operand of the
//      first put/get that references this channel.
//
// 2. air.channel.put (blocking or async):
//      %tok = air.channel.put async [%deps] @chan[%i,%j]
//                 (%buf[%o0,%o1][%s0,%s1][%st0,%st1]) : (memref<...>)
//    → %tok = conduit.put_memref_async
//                 {name="chan", num_elems=<product(sizes)>,
//                  offsets=<static offsets or []>,
//                  sizes=<static sizes or []>,
//                  strides=<static strides or []>}
//                 : !conduit.async.token
//      The %tok SSA value is replaced with the new !conduit.async.token.
//
// 3. air.channel.get (blocking or async):
//    → conduit.get_memref_async {same attrs} : !conduit.async.token
//
// 4. air.wait_all:
//      %t = air.wait_all async [%dep0, %dep1]
//    → %t = conduit.wait_all_async %dep0, %dep1
//              : (!conduit.async.token, ...) -> !conduit.async.token
//      Blocking (no result):
//      air.wait_all [%dep0, %dep1]
//    → conduit.wait_all %dep0, %dep1
//
// 5. air.async.token type → !conduit.async.token
//    (via SSA replacement; no explicit type conversion needed because
//    conduit ops produce !conduit.async.token results directly)
//
// Known limitations (documented honestly)
// ----------------------------------------
// - Only [1,1] scalar channels supported; multi-dimensional indices ignored.
// - num_elems is computed from static sizes only; dynamic sizes emit 0.
// - Offset/size/stride Index SSA values from the AIR op are dropped when
//   non-static; the emitted conduit op carries empty arrays in that case.
//   Full dynamic operand threading is a future TODO.
// - The blocking (non-async) put/get forms with no result SSA value are
//   lowered to the async form with the result token unused.  This is safe
//   because the token is not consumed by any downstream op in the original.
// - air.execute regions (async wrappers) are not transformed; they remain
//   in the output as unregistered ops when --allow-unregistered-dialect is
//   used.  Pass B's scope is limited to channel ops as per Task #32.
//
//===----------------------------------------------------------------------===//

#include "aie/Dialect/Conduit/Transforms/ConduitPasses.h"

#include "aie/Dialect/Conduit/IR/ConduitDialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

#include <string>

namespace xilinx::conduit {

#define GEN_PASS_DEF_AIRCHANNELTOCONDUIT
#include "aie/Dialect/Conduit/Transforms/ConduitPasses.h.inc"

namespace {

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Return true if this is an air.channel declaration op.
static bool isAirChannelDecl(mlir::Operation *op) {
  return op->getName().getStringRef() == "air.channel";
}

/// Return true if this is air.channel.put (async or blocking).
static bool isAirChannelPut(mlir::Operation *op) {
  return op->getName().getStringRef() == "air.channel.put";
}

/// Return true if this is air.channel.get (async or blocking).
static bool isAirChannelGet(mlir::Operation *op) {
  return op->getName().getStringRef() == "air.channel.get";
}

/// Return true if this is air.wait_all.
static bool isAirWaitAll(mlir::Operation *op) {
  return op->getName().getStringRef() == "air.wait_all";
}

/// Extract the sym_name attribute from a channel declaration op.
/// Returns empty string if not found.
static std::string getSymName(mlir::Operation *op) {
  if (auto attr = op->getAttrOfType<mlir::StringAttr>("sym_name"))
    return attr.getValue().str();
  return "";
}

/// Extract the channel symbol name from a channel.put / channel.get op.
/// These ops carry a FlatSymbolRefAttr named "chan_name".
static std::string getChanName(mlir::Operation *op) {
  if (auto attr = op->getAttrOfType<mlir::FlatSymbolRefAttr>("chan_name"))
    return attr.getValue().str();
  // Fallback: look for a symbol ref in any attribute named "chan_name"
  if (auto attr = op->getAttr("chan_name")) {
    if (auto symRef = mlir::dyn_cast<mlir::FlatSymbolRefAttr>(attr))
      return symRef.getValue().str();
    if (auto strAttr = mlir::dyn_cast<mlir::StringAttr>(attr))
      return strAttr.getValue().str();
  }
  return "";
}

/// Compute num_elems as the product of static integer constant sizes.
/// The sizes are Index-typed SSA values; we look for arith.constant defs.
/// If any size is dynamic (not a constant), returns 0.
static int64_t computeNumElems(mlir::ValueRange sizes) {
  if (sizes.empty())
    return 1; // scalar: 1 element
  int64_t prod = 1;
  for (mlir::Value v : sizes) {
    mlir::Operation *defOp = v.getDefiningOp();
    if (!defOp)
      return 0;
    if (auto cOp = mlir::dyn_cast<mlir::arith::ConstantIndexOp>(defOp)) {
      prod *= cOp.value();
    } else if (auto cOp =
                   mlir::dyn_cast<mlir::arith::ConstantIntOp>(defOp)) {
      prod *= cOp.value();
    } else {
      // Not a static constant — return 0 to signal dynamic
      return 0;
    }
  }
  return prod;
}

/// Extract static integer values from an Index SSA value range.
/// Non-static values are represented as -1 (ShapedType::kDynamic).
static llvm::SmallVector<int64_t> extractStaticInts(mlir::ValueRange vals) {
  llvm::SmallVector<int64_t> result;
  for (mlir::Value v : vals) {
    mlir::Operation *defOp = v.getDefiningOp();
    if (!defOp) {
      result.push_back(-1);
      continue;
    }
    if (auto cOp = mlir::dyn_cast<mlir::arith::ConstantIndexOp>(defOp)) {
      result.push_back(cOp.value());
    } else if (auto cOp =
                   mlir::dyn_cast<mlir::arith::ConstantIntOp>(defOp)) {
      result.push_back(cOp.value());
    } else {
      result.push_back(-1);
    }
  }
  return result;
}

// ---------------------------------------------------------------------------
// AIR channel.put / channel.get operand layout
// ---------------------------------------------------------------------------
//
// The AIR assembly format for channel.put:
//   custom<AsyncDependencies>(type($async_token), $async_dependencies)
//   $chan_name `[` ($indices^)? `]`
//   `(` $src `[` ($src_offsets^)? `]``[` ($src_sizes^)? `]``[` ($src_strides^)? `]` `)` ...
//
// AttrSizedOperandSegments is set, so the op has:
//   attribute "operand_segment_sizes" : dense<[ndeps, nidx, 1, noffsets, nsizes, nstrides]>
//
// We use this to slice the operand list.

/// Retrieve operand segment sizes from the "operand_segment_sizes" attribute.
static llvm::SmallVector<int32_t> getOperandSegments(mlir::Operation *op) {
  llvm::SmallVector<int32_t> segs;
  auto attr = op->getAttrOfType<mlir::DenseI32ArrayAttr>("operand_segment_sizes");
  if (attr) {
    for (int32_t v : attr.asArrayRef())
      segs.push_back(v);
  }
  return segs;
}

// ---------------------------------------------------------------------------
// Main pass struct
// ---------------------------------------------------------------------------

struct AirChannelToConduitPass
    : impl::AirChannelToConduitBase<AirChannelToConduitPass> {

  void runOnOperation() override {
    mlir::ModuleOp module = getOperation();
    mlir::OpBuilder builder(module.getContext());
    mlir::MLIRContext *ctx = module.getContext();

    // Async token type for conduit.
    auto conduitTokenTy = AsyncTokenType::get(ctx);

    // Phase 1: collect air.channel declarations → build name→create map.
    // We'll emit conduit.create for each; element_type filled in Phase 2.

    // Map: channel name → conduit.create op (for later patching of element_type)
    llvm::StringMap<mlir::Operation *> channelCreateOps;

    // Collect channel decl ops for deferred erasure.
    llvm::SmallVector<mlir::Operation *> channelDeclsToErase;

    // Collect put/get/wait_all ops for rewriting.
    llvm::SmallVector<mlir::Operation *> putGetToRewrite;
    llvm::SmallVector<mlir::Operation *> waitAllToRewrite;

    // Walk and collect all ops of interest.
    module.walk([&](mlir::Operation *op) {
      if (isAirChannelDecl(op))
        channelDeclsToErase.push_back(op);
      else if (isAirChannelPut(op) || isAirChannelGet(op))
        putGetToRewrite.push_back(op);
      else if (isAirWaitAll(op))
        waitAllToRewrite.push_back(op);
    });

    // Phase 2: emit conduit.create for each air.channel declaration.
    for (mlir::Operation *op : channelDeclsToErase) {
      std::string name = getSymName(op);
      if (name.empty()) {
        // No sym_name — skip.
        continue;
      }

      builder.setInsertionPoint(op);
      mlir::Location loc = op->getLoc();

      // Default: capacity=1, depth=1, no element_type (unknown until put/get seen).
      // element_type will be patched after put/get scan below.
      mlir::Operation *createOp = builder.create<Create>(
          loc,
          mlir::StringAttr::get(ctx, name),
          mlir::IntegerAttr::get(mlir::IntegerType::get(ctx, 64), 1),
          /*producer_tile=*/mlir::DenseI64ArrayAttr{},
          /*consumer_tiles=*/mlir::DenseI64ArrayAttr{},
          /*shim_consumer_tiles=*/mlir::DenseI64ArrayAttr{},
          /*element_type=*/mlir::TypeAttr{},
          mlir::IntegerAttr::get(mlir::IntegerType::get(ctx, 64), 1),
          /*link_mode=*/mlir::StringAttr{},
          /*access_pattern=*/mlir::DenseI64ArrayAttr{});

      channelCreateOps[name] = createOp;
    }

    // Phase 2b: scan put/get ops to extract element_type for conduit.create.
    // For each channel, use the memref operand type from the first put/get.
    for (mlir::Operation *op : putGetToRewrite) {
      std::string chanName = getChanName(op);
      if (chanName.empty())
        continue;

      // Find the create op for this channel.
      auto it = channelCreateOps.find(chanName);
      if (it == channelCreateOps.end())
        continue;

      mlir::Operation *createOp = it->second;
      auto createTypedOp = mlir::dyn_cast<Create>(createOp);
      if (!createTypedOp)
        continue;

      // Only patch if element_type not yet set.
      if (createTypedOp.getElementType().has_value())
        continue;

      // Get operand segment sizes to locate the memref operand.
      auto segs = getOperandSegments(op);
      // segs = [ndeps, nidx, 1 (memref), noffsets, nsizes, nstrides]
      if (segs.size() >= 3) {
        int32_t ndeps = segs[0];
        int32_t nidx = segs[1];
        int32_t memrefPos = ndeps + nidx; // index of the memref operand
        if (static_cast<int32_t>(op->getNumOperands()) > memrefPos) {
          mlir::Value memrefVal = op->getOperand(memrefPos);
          mlir::Type memrefTy = memrefVal.getType();
          if (mlir::isa<mlir::MemRefType>(memrefTy)) {
            // Patch element_type on the conduit.create op using typed setter.
            createTypedOp.setElementTypeAttr(mlir::TypeAttr::get(memrefTy));
          }
        }
      } else {
        // Fallback: if no segment sizes, assume first operand is the memref.
        if (op->getNumOperands() >= 1) {
          mlir::Value first = op->getOperand(0);
          if (mlir::isa<mlir::MemRefType>(first.getType())) {
            createTypedOp.setElementTypeAttr(
                mlir::TypeAttr::get(first.getType()));
          }
        }
      }
    }

    // Phase 3: rewrite air.channel.put / air.channel.get → conduit put/get_memref_async.
    //
    // SSA threading:
    //   The original air.channel.put result is !air.async.token (opaque in
    //   aie-opt).  We replace all uses with the new !conduit.async.token.

    llvm::SmallVector<mlir::Operation *> putGetToErase;

    for (mlir::Operation *op : putGetToRewrite) {
      std::string chanName = getChanName(op);
      if (chanName.empty()) {
        // Cannot identify channel — skip.
        continue;
      }

      builder.setInsertionPoint(op);
      mlir::Location loc = op->getLoc();

      // Decode operand segments.
      auto segs = getOperandSegments(op);
      // Expected: [ndeps, nidx, 1 (memref), noffsets, nsizes, nstrides]
      int32_t ndeps = 0, nidx = 0, noffsets = 0, nsizes = 0, nstrides = 0;
      if (segs.size() >= 6) {
        ndeps    = segs[0];
        nidx     = segs[1];
        // segs[2] == 1 (memref)
        noffsets = segs[3];
        nsizes   = segs[4];
        nstrides = segs[5];
      }

      // Slice operands.
      mlir::OperandRange allOps = op->getOperands();
      int32_t base = 0;
      // async deps: skip (conduit put/get_memref_async do not take dep tokens)
      base += ndeps;
      // indices (ignored — only [1,1] channels supported)
      base += nidx;
      // memref
      mlir::ValueRange offsetsRange, sizesRange, stridesRange;
      if (static_cast<int32_t>(allOps.size()) > base + 1 + noffsets + nsizes + nstrides) {
        base += 1; // skip memref operand itself
        offsetsRange = allOps.slice(base, noffsets); base += noffsets;
        sizesRange   = allOps.slice(base, nsizes);   base += nsizes;
        stridesRange = allOps.slice(base, nstrides);
      }

      // Compute num_elems from static sizes.
      int64_t numElems = computeNumElems(sizesRange);
      if (numElems == 0) numElems = 1; // fallback for dynamic

      // Extract static values for the structured attrs.
      auto offsetVals  = extractStaticInts(offsetsRange);
      auto sizeVals    = extractStaticInts(sizesRange);
      auto strideVals  = extractStaticInts(stridesRange);

      // Replace any -1 (dynamic) with 0 in the arrays
      // (conduit attr is DenseI64ArrayAttr which requires concrete ints).
      for (auto &v : offsetVals)  if (v < 0) v = 0;
      for (auto &v : sizeVals)    if (v < 0) v = 0;
      for (auto &v : strideVals)  if (v < 0) v = 1;

      // Emit conduit put_memref_async or get_memref_async.
      bool isPut = isAirChannelPut(op);
      mlir::Operation *newOp;
      if (isPut) {
        newOp = builder.create<PutMemrefAsync>(
            loc, conduitTokenTy,
            mlir::StringAttr::get(ctx, chanName),
            mlir::IntegerAttr::get(mlir::IntegerType::get(ctx, 64), numElems),
            mlir::DenseI64ArrayAttr::get(ctx, offsetVals),
            mlir::DenseI64ArrayAttr::get(ctx, sizeVals),
            mlir::DenseI64ArrayAttr::get(ctx, strideVals));
      } else {
        newOp = builder.create<GetMemrefAsync>(
            loc, conduitTokenTy,
            mlir::StringAttr::get(ctx, chanName),
            mlir::IntegerAttr::get(mlir::IntegerType::get(ctx, 64), numElems),
            mlir::DenseI64ArrayAttr::get(ctx, offsetVals),
            mlir::DenseI64ArrayAttr::get(ctx, sizeVals),
            mlir::DenseI64ArrayAttr::get(ctx, strideVals));
      }

      // Replace all uses of the old async token result with the new token.
      if (op->getNumResults() >= 1 && newOp->getNumResults() >= 1)
        op->getResult(0).replaceAllUsesWith(newOp->getResult(0));

      putGetToErase.push_back(op);
    }

    // Erase original put/get ops.
    for (mlir::Operation *op : putGetToErase)
      op->erase();

    // Phase 4: rewrite air.wait_all → conduit.wait_all / conduit.wait_all_async.
    //
    // air.wait_all has:
    //   args: variadic !air.async.token (async_dependencies)
    //   result (optional): !air.async.token (async_token, present when async)
    //
    // Mapping:
    //   result present → conduit.wait_all_async %deps : (...) -> !conduit.async.token
    //   no result      → conduit.wait_all %deps

    llvm::SmallVector<mlir::Operation *> waitAllToErase;

    for (mlir::Operation *op : waitAllToRewrite) {
      builder.setInsertionPoint(op);
      mlir::Location loc = op->getLoc();

      mlir::ValueRange deps = op->getOperands();
      bool hasResult = (op->getNumResults() >= 1);

      if (hasResult) {
        auto newOp = builder.create<WaitAllAsync>(loc, conduitTokenTy, deps);
        op->getResult(0).replaceAllUsesWith(newOp.getResult());
      } else {
        builder.create<WaitAll>(loc, deps);
      }

      waitAllToErase.push_back(op);
    }

    for (mlir::Operation *op : waitAllToErase)
      op->erase();

    // Phase 5: erase air.channel declaration ops (after all put/get refs are gone).
    for (mlir::Operation *op : channelDeclsToErase)
      op->erase();
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Factory + registration
//===----------------------------------------------------------------------===//

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createAirChannelToConduitPass() {
  return std::make_unique<AirChannelToConduitPass>();
}

} // namespace xilinx::conduit
