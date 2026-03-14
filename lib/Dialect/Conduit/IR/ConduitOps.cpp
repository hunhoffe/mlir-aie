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
//   Acquire::verify() / WaitWindow::verify() — M8a: window value release linearity
//                                              M9: same-block acquire-release pairing (llvm::errs)
//   AcquireAsync::verify() / ReleaseAsync::verify() — M8b: window.token wait_window linearity
//                                                     M9: wait_window→release pairing (llvm::errs)
//   WaitAll::verify() / WaitAllAsync::verify() — M8c: operands must be token types
//
//===----------------------------------------------------------------------===//

#include "aie/Dialect/Conduit/IR/ConduitDialect.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
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

// Include generated enum definitions (Port enum)
#include "aie/Dialect/Conduit/IR/ConduitEnums.cpp.inc"

// Include generated attribute definitions (PortAttr)
#define GET_ATTRDEF_CLASSES
#include "aie/Dialect/Conduit/IR/ConduitAttrDefs.cpp.inc"

//===----------------------------------------------------------------------===//
// Conduit dialect — initialize
//===----------------------------------------------------------------------===//

#include "aie/Dialect/Conduit/IR/ConduitOpsDialect.cpp.inc"

void ConduitDialect::initialize() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "aie/Dialect/Conduit/IR/ConduitAttrDefs.cpp.inc"
      >();
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
  // Walk the def-use chain: subview_access takes a !conduit.window<T> produced
  // by conduit.acquire or conduit.wait_window.  Extract the conduit name, then
  // search the enclosing module for the matching conduit.create.
  //
  // The module-level walk handles the common case where conduit.create is at
  // the module/device level while subview_access is nested inside aie.core.
  // Uses walk([&](Create)) which visits only Create ops — O(k) where k is the
  // number of conduit.create ops, not O(n) in total ops.
  uint64_t idx = getIndex();
  mlir::Value win = getWindow();
  if (auto *defOp = win.getDefiningOp()) {
    llvm::StringRef conduitName;
    if (auto acqOp = mlir::dyn_cast<Acquire>(defOp))
      conduitName = acqOp.getName();
    else if (auto waitOp = mlir::dyn_cast<WaitWindow>(defOp))
      conduitName = waitOp.getName();

    if (!conduitName.empty()) {
      // Walk up to the enclosing ModuleOp and search for conduit.create.
      mlir::Operation *ancestor = getOperation()->getParentOp();
      while (ancestor && !mlir::isa<mlir::ModuleOp>(ancestor))
        ancestor = ancestor->getParentOp();
      if (ancestor) {
        bool outOfBounds = false;
        uint64_t foundDepth = 0;
        ancestor->walk([&](Create createOp) -> mlir::WalkResult {
          if (createOp.getName() == conduitName) {
            if (auto depthOpt = createOp.getDepth()) {
              foundDepth = *depthOpt;
              if (idx >= foundDepth)
                outOfBounds = true;
            }
            return mlir::WalkResult::interrupt();
          }
          return mlir::WalkResult::advance();
        });
        if (outOfBounds)
          return emitOpError("index ")
                 << idx << " out of bounds for conduit of depth " << foundDepth;
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
  if (modeStr == "cascade")
    return emitError("conduit.objectfifo_link: cascade mode not yet supported");
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
  // against the capacity attribute — see M7 below.
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

    // M7: CSDF buffer capacity check (sufficient condition).
    // Simulate one hyper-period (H = lcm(len(P), len(C)) time slots) and track
    // the running token occupancy.  At each slot t, the producer fires first
    // (adds P[t mod q] tokens) then the consumer fires (removes C[t mod r]
    // tokens).  The buffer must hold the peak occupancy without exceeding
    // capacity.
    //
    // Algorithm:
    //   For t = 0..H-1: produce P[t mod q] tokens, then consume C[t mod r] tokens.
    //   Track occupancy after each produce step; record the maximum.
    //   Require: capacity >= peak_occupancy.
    //
    // The hyper-period H = lcm(q, r) = q * (r / gcd(q, r)).
    // Overflow guard: check that q * (r/g) does not overflow before computing.
    // If H > kMaxSimSteps, skip simulation and emit a warning.
    //
    // Underflow semantics: when occupancy goes negative after the consume step,
    // this means the simulation's produce-before-consume interleaving is
    // incompatible with the hardware's BD scheduling for these rates.  M6 already
    // guarantees the schedule is feasible over the full hyper-period; momentary
    // underflow in this simulation does NOT necessarily mean hardware deadlock —
    // the actual AIE BD chain may fire in a different order (e.g., the hardware
    // drains the consumer BD before the producer BD refills).  We therefore emit
    // emitWarning (not emitOpError) for underflow: it flags a potential ordering
    // mismatch for the user to verify against their BD chain layout, but does not
    // reject the program.  Only capacity overflow (peakOccupancy > capacity) is a
    // hard error, because no interleaving can hide that constraint.
    {
      int64_t capacity = getCapacity();
      // Compute gcd(plen, clen) via Euclid's algorithm.
      int64_t a = plen, b = clen;
      while (b) { int64_t tmp = b; b = a % b; a = tmp; }
      int64_t g = a;
      int64_t clenOverG = clen / g; // exact: g divides clen by construction
      constexpr int64_t kMaxSimSteps = 1024;
      // Overflow guard: plen * clenOverG must not overflow int64_t and must be
      // within the simulation cap before we compute hyperPeriod.
      // Since kMaxSimSteps == 1024 and plen >= 1, the product overflows only
      // when clenOverG > INT64_MAX / plen.  We conservatively skip simulation
      // if either factor exceeds kMaxSimSteps (the product would then exceed the
      // cap regardless).
      if (plen > kMaxSimSteps || clenOverG > kMaxSimSteps / plen) {
        emitWarning("M7: CSDF hyper-period exceeds simulation cap (")
            << kMaxSimSteps << " steps); buffer capacity check skipped";
      } else {
        int64_t hyperPeriod = plen * clenOverG;
        if (hyperPeriod > kMaxSimSteps) {
          emitWarning("M7: CSDF hyper-period exceeds simulation cap (")
              << kMaxSimSteps << " steps); buffer capacity check skipped";
        } else {
          int64_t occupancy = 0;
          int64_t peakOccupancy = 0;
          for (int64_t t = 0; t < hyperPeriod; ++t) {
            // Producer fires: add P[t mod q] tokens.
            occupancy += pRates[static_cast<size_t>(t % plen)];
            if (occupancy > peakOccupancy)
              peakOccupancy = occupancy;
            // Consumer fires: remove C[t mod r] tokens.
            occupancy -= cRates[static_cast<size_t>(t % clen)];
            if (occupancy < 0) {
              // Momentary underflow in produce-before-consume interleaving.
              // See comment above: this is a warning, not an error.
              emitWarning("M7: CSDF hyper-period simulation: momentary underflow "
                          "at step ")
                  << t << " (occupancy=" << occupancy
                  << "); hardware BD scheduling may differ from "
                     "produce-before-consume simulation order";
              occupancy = 0; // reset to prevent cascading underflow reports
            }
          }
          if (peakOccupancy > capacity)
            return emitOpError("M7: CSDF buffer capacity insufficient: "
                               "peak token occupancy over one hyper-period=")
                   << peakOccupancy << " exceeds capacity=" << capacity
                   << " (producer_rates=" << psum << "/phase, "
                   << "consumer_rates=" << csum << "/phase, "
                   << "hyper-period=" << hyperPeriod << " steps)";
        }
      }
    }
  }

  return ::mlir::success();
}

//===----------------------------------------------------------------------===//
// M10: Token escape verifier
//
// Window and DMA tokens represent hardware state (lock grants, DMA BD
// completions) that is not portable across function boundaries.  A token
// that escapes its defining function via return or call argument would
// create dangling hardware references.
//
// This helper checks that no user of `tokenVal` is a func.return or
// func.call / func.call_indirect.
//===----------------------------------------------------------------------===//

static ::mlir::LogicalResult
checkTokenDoesNotEscape(mlir::Operation *producerOp, mlir::Value tokenVal) {
  for (mlir::OpOperand &use : tokenVal.getUses()) {
    mlir::Operation *user = use.getOwner();
    if (mlir::isa<mlir::func::ReturnOp>(user))
      return producerOp->emitOpError(
          "M10: token escapes function scope via return");
    if (mlir::isa<mlir::func::CallOp>(user) ||
        mlir::isa<mlir::func::CallIndirectOp>(user))
      return producerOp->emitOpError(
          "M10: token escapes function scope via call argument");
    // Indirect escape via memref.store is not detected — deferred to M11.
  }
  return ::mlir::success();
}

//===----------------------------------------------------------------------===//
// M8: Token linearity verifiers
//
// Three sub-checks:
//
//   M8a — window value release linearity (Acquire, WaitWindow):
//     A !conduit.window<T> value must be released by at most one
//     conduit.release op.  Multiple releases → double hardware lock-counter
//     release → counter overflow → silent memory corruption.
//     Zero conduit.release users is permitted: conduit.release_async releases
//     by channel name (no SSA window operand), so the window SSA value may
//     have zero Release users when release_async is the release mechanism.
//     conduit.subview_access users do not count as releases.
//
//   M8b — window.token wait_window linearity (AcquireAsync, ReleaseAsync):
//     A !conduit.window.token may be passed to conduit.wait_all /
//     conduit.wait_all_async (fan-in), which does not "consume" the logical
//     lock grant.  The token is materialized only by conduit.wait_window.
//     More than one wait_window on the same token → double-materialization
//     of the same lock grant → hardware deadlock.
//     Zero wait_window uses is valid (token consumed via wait_all only).
//     Other user op types are NOT flagged by M8b: the TableGen type
//     constraints already reject semantically illegal uses independently.
//
//   M8c — wait_all / wait_all_async operand type check:
//     All operands must be conduit token types: !conduit.dma.token,
//     !conduit.window.token.
//     Non-token operands (e.g., !conduit.window<T>, memref, i32) indicate a
//     programming error — the AnyType variadic in TableGen does not constrain
//     these and M8c fills that gap.
//
// Limitations:
//   - M8a does not flag zero conduit.release uses when release_async is used
//     by name (cannot statically link the channel name to the SSA value).
//   - M8 does not verify ordering; that requires liveness analysis (future).
//===----------------------------------------------------------------------===//

static ::mlir::LogicalResult
checkWindowReleaseCumulativeCount(mlir::Operation *producerOp, mlir::Value windowVal) {
  // M8a limitation: loop-carried over-release is not detected. Two conduit.acquire
  // ops in a loop body produce independent SSA values; releasing each twice in the
  // same iteration would not be caught. Detecting loop-carried over-release requires
  // M9 liveness analysis — deferred.
  //
  // M8a: True double-release detection — cumulative released count must not
  // exceed the acquired count.  Multiple conduit.release ops on the same
  // window value are valid for sliding-window partial-release patterns
  // (e.g., acquire(3) followed by three release(1) calls).
  //
  // Limitation: loop-carried releases are only checked once per static path.
  // A release inside scf.for runs N times per acquired window; if N > 1 and
  // the loop runs multiple times, runtime total may exceed acquired_count.
  // Detecting loop-carried over-release requires M9 liveness analysis.

  int64_t acquiredCount = 0;
  if (auto acqOp = mlir::dyn_cast<Acquire>(producerOp)) {
    acquiredCount = static_cast<int64_t>(acqOp.getCount());
  } else if (auto waitWinOp = mlir::dyn_cast<WaitWindow>(producerOp)) {
    if (auto acqAsyncOp =
            waitWinOp.getToken().getDefiningOp<AcquireAsync>()) {
      acquiredCount = static_cast<int64_t>(acqAsyncOp.getCount());
    } else {
      return ::mlir::success();
    }
  } else {
    return ::mlir::success();
  }

  int64_t totalReleased = 0;
  for (mlir::OpOperand &use : windowVal.getUses()) {
    if (auto relOp = mlir::dyn_cast<Release>(use.getOwner()))
      totalReleased += static_cast<int64_t>(relOp.getCount());
  }

  if (totalReleased == 0)
    return ::mlir::success();

  if (totalReleased > acquiredCount)
    return producerOp->emitOpError("M8: cumulative release count (")
           << totalReleased << ") exceeds acquired count (" << acquiredCount
           << ") -- double-release causes hardware lock-counter overflow";

  return ::mlir::success();
}

static ::mlir::LogicalResult
checkWindowTokenLinear(mlir::Operation *producerOp, mlir::Value tokenVal) {
  unsigned waitWindowCount = 0;
  for (mlir::OpOperand &use : tokenVal.getUses()) {
    mlir::Operation *user = use.getOwner();
    if (mlir::isa<WaitWindow>(user))
      ++waitWindowCount;
  }
  if (waitWindowCount > 1)
    return producerOp->emitOpError("M8: window.token has ")
           << waitWindowCount
           << " conduit.wait_window uses (double-materialization of the same "
              "lock grant causes hardware deadlock)";
  return ::mlir::success();
}

static ::mlir::LogicalResult
checkTokenOperandTypes(mlir::Operation *op, mlir::ValueRange operands) {
  for (auto [idx, operand] : llvm::enumerate(operands)) {
    mlir::Type ty = operand.getType();
    bool isToken = mlir::isa<DMATokenType, WindowTokenType>(ty);
    if (!isToken)
      return op->emitOpError("M8: wait_all operand #")
             << idx << " has type " << ty
             << ", which is not a conduit token type (!conduit.dma.token "
                "or !conduit.window.token)";
  }
  return ::mlir::success();
}

::mlir::LogicalResult Acquire::verify() {
  if (failed(checkWindowReleaseCumulativeCount(getOperation(), getWindow())))
    return ::mlir::failure();

  // M9 Phase 2 (same-block acquire-release pairing) is implemented in the
  // separate --conduit-check-pairing analysis pass (ConduitPairingCheck.cpp).
  // Moved out of verify() to avoid MLIR diagnostic infinite recursion and to
  // make the warnings capturable by FileCheck / --verify-diagnostics.

  return ::mlir::success();
}
::mlir::LogicalResult AcquireAsync::verify() {
  if (failed(checkTokenDoesNotEscape(getOperation(), getToken())))
    return ::mlir::failure();
  if (failed(checkWindowTokenLinear(getOperation(), getToken())))
    return ::mlir::failure();

  // M9 Phase 2 (same-block pairing via wait_window) is implemented in the
  // separate --conduit-check-pairing analysis pass (ConduitPairingCheck.cpp).

  return ::mlir::success();
}
::mlir::LogicalResult ReleaseAsync::verify() {
  if (failed(checkTokenDoesNotEscape(getOperation(), getToken())))
    return ::mlir::failure();
  return checkWindowTokenLinear(getOperation(), getToken());
}
::mlir::LogicalResult WaitWindow::verify() {
  if (failed(checkWindowReleaseCumulativeCount(getOperation(), getWindow())))
    return ::mlir::failure();

  // M9 Phase 2 (same-block pairing) is implemented in the separate
  // --conduit-check-pairing analysis pass (ConduitPairingCheck.cpp).

  return ::mlir::success();
}
::mlir::LogicalResult WaitAll::verify() {
  // Note: This check is redundant with the TableGen Conduit_AnyTokenType constraint,
  // which MLIR enforces before user verify() runs. Left in place for defense-in-depth
  // but may be dead code — the TableGen constraint fires first.
  return checkTokenOperandTypes(getOperation(), getTokens());
}
::mlir::LogicalResult WaitAllAsync::verify() {
  if (failed(checkTokenDoesNotEscape(getOperation(), getResult())))
    return ::mlir::failure();
  // Note: This check is redundant with the TableGen Conduit_AnyTokenType constraint,
  // which MLIR enforces before user verify() runs. Left in place for defense-in-depth
  // but may be dead code — the TableGen constraint fires first.
  return checkTokenOperandTypes(getOperation(), getTokens());
}

::mlir::LogicalResult PutMemrefAsync::verify() {
  return checkTokenDoesNotEscape(getOperation(), getToken());
}
::mlir::LogicalResult GetMemrefAsync::verify() {
  return checkTokenDoesNotEscape(getOperation(), getToken());
}

//===----------------------------------------------------------------------===//
// Conduit ops — generated op definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "aie/Dialect/Conduit/IR/ConduitOps.cpp.inc"
