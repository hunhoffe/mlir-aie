//===- ConduitOps.cpp - Conduit dialect implementation ----------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2026 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//
//
// Implementation of the Conduit dialect.  All ops use the generated
// parser/printer from the TableGen assemblyFormat directives.
//
// Custom verifiers:
//   SubviewAccess::verify() — M2: index bounds against conduit depth
//   ObjectFifoLink::verify() — M3: mode structural invariants + offset counts
//   Create::verify() — M4: dynamic-dim warning; M5: routing_mode; M6: CSDF balance
//
//===----------------------------------------------------------------------===//

#include "aie/Dialect/Conduit/IR/ConduitDialect.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/Operation.h"

#include "llvm/ADT/TypeSwitch.h"

using namespace mlir;
using namespace xilinx;
using namespace xilinx::conduit;

//===----------------------------------------------------------------------===//
// Conduit dialect — generated type definitions
//===----------------------------------------------------------------------===//

#define GET_TYPEDEF_CLASSES
#include "aie/Dialect/Conduit/IR/ConduitTypes.cpp.inc"

//===----------------------------------------------------------------------===//
// Conduit dialect — initialize
//===----------------------------------------------------------------------===//

#include "aie/Dialect/Conduit/IR/ConduitOpsDialect.cpp.inc"

void ConduitDialect::initialize() {
  addTypes<
#define GET_TYPEDEF_LIST
#include "aie/Dialect/Conduit/IR/ConduitTypes.cpp.inc"
      >();
  addOperations<
#define GET_OP_LIST
#include "aie/Dialect/Conduit/IR/ConduitOps.cpp.inc"
      >();
}

//===----------------------------------------------------------------------===//
// Conduit ops — additional verifiers
//===----------------------------------------------------------------------===//

// SubviewAccess verifier: ensure result type matches the window's element type.
// M2: also check that the index attribute is within the conduit depth when
// the depth is statically known from the defining conduit.create op.
::mlir::LogicalResult SubviewAccess::verify() {
  auto winTy = mlir::dyn_cast<WindowType>(getWindow().getType());
  if (!winTy)
    return emitOpError("operand must be !conduit.window<T>");
  if (getResult().getType() != winTy.getElementType())
    return emitOpError("result type ")
           << getResult().getType()
           << " does not match window element type "
           << winTy.getElementType();

  // M2: index bounds check against depth from the defining conduit.create.
  // The index is an I64Attr (compile-time constant), so read it directly.
  uint64_t idx = getIndex();
  // Walk up the def-use chain: subview_access takes a !conduit.window<T> which
  // is produced by conduit.acquire.  The acquire carries a name attribute that
  // we can match against a conduit.create in the same block/region.
  // For simplicity, look for a conduit.create in the enclosing region that
  // has a depth attribute and whose name matches the acquire's name, if the
  // window operand is defined by a conduit.acquire op.
  mlir::Value win = getWindow();
  if (auto *defOp = win.getDefiningOp()) {
    if (auto acqOp = mlir::dyn_cast<Acquire>(defOp)) {
      llvm::StringRef conduitName = acqOp.getName();
      // Search sibling ops in the same block for a matching conduit.create.
      mlir::Block *block = getOperation()->getBlock();
      if (block) {
        for (mlir::Operation &op : *block) {
          if (auto createOp = mlir::dyn_cast<Create>(op)) {
            if (createOp.getName() == conduitName) {
              if (auto depthOpt = createOp.getDepth()) {
                uint64_t depth = *depthOpt;
                if (idx >= depth)
                  return emitOpError("index ")
                         << idx << " out of bounds for conduit of depth "
                         << depth;
              }
              break;
            }
          }
        }
      }
    }
  }

  return ::mlir::success();
}

//===----------------------------------------------------------------------===//
// Conduit ops — custom verifiers
//===----------------------------------------------------------------------===//

::mlir::LogicalResult ObjectFifoLink::verify() {
  auto modeStr = getMode();
  auto srcs = getSrcs();
  auto dsts = getDsts();
  auto offsets = getOffsets();

  // M3: mode validation — enforce structural invariants per mode.
  if (modeStr == "distribute") {
    if (srcs.size() != 1)
      return emitOpError("distribute mode requires exactly 1 src, got ")
             << srcs.size();
  } else if (modeStr == "join") {
    if (dsts.size() != 1)
      return emitOpError("join mode requires exactly 1 dst, got ")
             << dsts.size();
  } else if (modeStr == "forward") {
    if (srcs.size() != 1 || dsts.size() != 1)
      return emitOpError(
          "forward mode requires exactly 1 src and 1 dst, got ")
             << srcs.size() << " src(s) and " << dsts.size() << " dst(s)";
  } else {
    return emitOpError("unknown mode '")
           << modeStr << "'; expected distribute, join, or forward";
  }

  // Offset count consistency checks.
  if (offsets.has_value() && !offsets->empty()) {
    if (modeStr == "distribute") {
      if (offsets->size() != dsts.size())
        return emitOpError("distribute mode: offsets count (")
               << offsets->size() << ") must equal dsts count (" << dsts.size()
               << ")";
    } else if (modeStr == "join") {
      if (offsets->size() != srcs.size())
        return emitOpError("join mode: offsets count (")
               << offsets->size() << ") must equal srcs count (" << srcs.size()
               << ")";
    }
  }
  return ::mlir::success();
}

//===----------------------------------------------------------------------===//
// Conduit ops — Create verifier
//===----------------------------------------------------------------------===//

// M4: warn when element_type has dynamic dimensions (capacity is approximate).
// M5: validate routing_mode when present; must be "circuit" or "packet".
// M6: CSDF balance check — if producer_rates and/or consumer_rates are present,
//     verify both are present and the CSDF consistency equation holds:
//
//       sum(producer_rates) * len(consumer_rates)
//         == sum(consumer_rates) * len(producer_rates)
//
//     This is the Lee-Messerschmitt (1987) CSDF single-channel consistency
//     condition.  For a producer with phase-period q = len(P) and a consumer
//     with phase-period r = len(C), balance requires that over lcm(q,r)/q
//     producer firings and lcm(q,r)/r consumer firings the token counts match:
//
//       sum(P) * (lcm(q,r)/q) == sum(C) * (lcm(q,r)/r)
//       sum(P) * r == sum(C) * q                           [cancel lcm(q,r)]
//       sum(P) * len(C) == sum(C) * len(P)
//
//     Note: sum(P) == sum(C) is strictly weaker — it only handles the special
//     case q == r (same period).  Example of a sum-equal but CSDF-imbalanced
//     channel: P=[3] (sum=3,q=1), C=[1,2] (sum=3,r=2) — sum(P)*r=6 ≠
//     sum(C)*q=3, so no integer firing vector exists.
::mlir::LogicalResult Create::verify() {
  if (auto elemTypeOpt = getElementType()) {
    mlir::Type ty = *elemTypeOpt;
    if (auto shaped = mlir::dyn_cast<mlir::ShapedType>(ty)) {
      for (int64_t dim : shaped.getShape()) {
        if (mlir::ShapedType::isDynamic(dim)) {
          emitWarning("conduit.create: element_type has dynamic dimensions; "
                      "capacity is approximate");
          break;
        }
      }
    }
  }
  if (auto rmOpt = getRoutingMode()) {
    llvm::StringRef rm = *rmOpt;
    if (rm != "circuit" && rm != "packet")
      return emitOpError("routing_mode must be \"circuit\" or \"packet\", got \"")
             << rm << "\"";
  }

  // M6: CSDF balance check (necessary condition).
  // producer_rates and consumer_rates must appear together.  When both are
  // present, the Lee-Messerschmitt consistency equation must hold:
  //   sum(P) * len(C) == sum(C) * len(P)
  // This is NECESSARY for a periodic schedule.  The SUFFICIENT condition
  // (deadlock freedom: buffer never underflows or overflows) is also statically
  // computable by simulating one hyper-period and checking token occupancy
  // against the capacity attribute — this is TODO M7.
  bool hasPR = getProducerRates().has_value();
  bool hasCR = getConsumerRates().has_value();
  if (hasPR != hasCR)
    return emitOpError("CSDF requires both producer_rates and consumer_rates; "
                       "only one was provided");
  if (hasPR && hasCR) {
    auto pRates = *getProducerRates();
    auto cRates = *getConsumerRates();
    int64_t psum = 0;
    for (int64_t v : pRates)
      psum += v;
    int64_t csum = 0;
    for (int64_t v : cRates)
      csum += v;
    int64_t plen = static_cast<int64_t>(pRates.size());
    int64_t clen = static_cast<int64_t>(cRates.size());
    // CSDF balance: sum(P)*len(C) == sum(C)*len(P)
    if (psum * clen != csum * plen)
      return emitOpError("CSDF rate imbalance: sum(producer_rates)*len(consumer_rates)=")
             << (psum * clen)
             << " != sum(consumer_rates)*len(producer_rates)="
             << (csum * plen)
             << " (producer_rates has sum=" << psum << " period=" << plen
             << ", consumer_rates has sum=" << csum << " period=" << clen << ")";
  }

  return ::mlir::success();
}

//===----------------------------------------------------------------------===//
// Conduit ops — generated op definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "aie/Dialect/Conduit/IR/ConduitOps.cpp.inc"
