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
//   Create::verify() — depth>=0 check; element_type MemRefType check;
//                   sync_mode/disable_synchronization conflict;
//                   M4: dynamic-dim warning; M5: routing_mode; M6: CSDF balance
//   PutMemref::verify() / GetMemref::verify() — M3: num_elems>0,
//                       offsets/sizes/strides length match, sizes>0, product
//   Release::verify() — M2: count>0, port consistency, count<=acquire count
//   Acquire::verify() / WaitWindow::verify() — M8a: window value release
//   linearity
//                                              M9: same-block acquire-release
//                                              pairing (llvm::errs)
//   AcquireAsync::verify() / ReleaseAsync::verify() — M8b: window.token
//   wait_window linearity
//                                                     M9: wait_window→release
//                                                     pairing (llvm::errs)
//   WaitAll::verify() / WaitAllAsync::verify() — M8c: operands must be token
//   types ScatterOp::verify() — DMA budget, memtile format GatherOp::verify() —
//   DMA budget, memtile format TransposeOp::verify() — DMA budget, offsets,
//   packet ID budget, memtile format
//
//===----------------------------------------------------------------------===//

#include "aie/Dialect/Conduit/IR/ConduitDialect.h"

#include "aie/Dialect/AIE/IR/AIEDialect.h"

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
           << getResult().getType() << " does not match window element type "
           << winTy.getElementType();

  // M2: index bounds check against acquire count from the defining
  // conduit.acquire op.  The valid range for subview_access is [0, count)
  // where count is the number of slots acquired by the parent acquire op.
  // Using conduit.create depth as the bound was incorrect: a depth=2 conduit
  // can be acquired with count=3 (sliding-window pattern) making index=2 valid.
  //
  // If the defining op is conduit.acquire, use its count attribute directly.
  // If the defining op is conduit.wait_window, trace back to the
  // conduit.acquire_async to get the count.
  // If the bound cannot be determined statically, skip the check.
  uint64_t idx = getIndex();
  mlir::Value win = getWindow();
  if (auto *defOp = win.getDefiningOp()) {
    uint64_t acquireCount = 0;
    bool haveCount = false;
    if (auto acqOp = mlir::dyn_cast<Acquire>(defOp)) {
      acquireCount = acqOp.getCount();
      haveCount = true;
    } else if (auto waitOp = mlir::dyn_cast<WaitWindow>(defOp)) {
      if (auto acqAsyncOp = waitOp.getToken().getDefiningOp<AcquireAsync>()) {
        acquireCount = acqAsyncOp.getCount();
        haveCount = true;
      }
    }
    if (haveCount && idx >= acquireCount)
      return emitOpError("index ")
             << idx << " out of bounds for acquire count " << acquireCount;
  }

  return ::mlir::success();
}

//===----------------------------------------------------------------------===//
// Conduit ops — custom verifiers
//===----------------------------------------------------------------------===//

// ---------------------------------------------------------------------------
// Shared CSDF helper: find conduit.create by name in the enclosing DeviceOp.
// Used by Link::verify() for M6-join/M7-join/M6-dist/M7-dist checks.
// Returns nullptr when not found (conduit.create may be in a different
// translation unit or a test fragment; skip rather than error).
//
// Walks from the nearest AIE::DeviceOp ancestor (conduit.create has
// HasParent<"AIE::DeviceOp">), which is faster and more precise than
// walking from the enclosing ModuleOp.
// ---------------------------------------------------------------------------
static Create findConduitCreateByName(mlir::Operation *anchor,
                                      llvm::StringRef name) {
  mlir::Operation *parent = anchor;
  while (parent && !mlir::isa<AIE::DeviceOp>(parent))
    parent = parent->getParentOp();
  if (!parent)
    return {};
  Create result{};
  parent->walk([&](Create op) -> mlir::WalkResult {
    if (op.getName() == name) {
      result = op;
      return mlir::WalkResult::interrupt();
    }
    return mlir::WalkResult::advance();
  });
  return result;
}

//===----------------------------------------------------------------------===//
// Conduit ops — Create verifier
//===----------------------------------------------------------------------===//

// M4: warn when element_type has dynamic dimensions (capacity is approximate).
// M5: validate routing_mode when present; must be "circuit", "packet",
//     "cascade", or "any" (mode=any triggers Step 3.5 in Pass C).
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
  // depth < 0 rejection.
  // depth=0 is the sentinel for "unresolved" (set by Pass A/B, resolved by
  // --conduit-depth-promote). Positive values are explicit depths.
  // Negative values are always invalid.
  if (auto d = getDepth()) {
    if (static_cast<int64_t>(*d) < 0)
      return emitOpError("depth must be >= 0 (0 = unresolved sentinel, "
                         ">0 = explicit depth); got ")
             << static_cast<int64_t>(*d);
  }

  // element_type must be MemRefType (required attr, enforced by ODS type,
  // but verify the inner type for clarity).
  mlir::Type ty = getElementType();
  if (!mlir::isa<mlir::MemRefType>(ty))
    return emitOpError("element_type must be a MemRefType, got ") << ty;
  if (auto shaped = mlir::dyn_cast<mlir::ShapedType>(ty)) {
    for (int64_t dim : shaped.getShape()) {
      if (mlir::ShapedType::isDynamic(dim)) {
        emitWarning("conduit.create: element_type has dynamic dimensions; "
                    "buffer capacity is approximate");
        break;
      }
    }
  }

  // routing_mode and sync_mode are ODS enums — invalid values rejected at
  // parse time. No explicit string validation needed.

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
      return emitOpError("CSDF rate imbalance: "
                         "sum(producer_rates)*len(consumer_rates)=")
             << (psum * clen)
             << " != sum(consumer_rates)*len(producer_rates)=" << (csum * plen)
             << " (producer_rates has sum=" << psum << " period=" << plen
             << ", consumer_rates has sum=" << csum << " period=" << clen
             << ")";

    // M7: CSDF buffer capacity check (sufficient condition).
    // Simulate one hyper-period (H = lcm(len(P), len(C)) time slots) and track
    // the running token occupancy.  At each slot t, the producer fires first
    // (adds P[t mod q] tokens) then the consumer fires (removes C[t mod r]
    // tokens).  The buffer must hold the peak occupancy without exceeding
    // capacity.
    //
    // Algorithm:
    //   For t = 0..H-1: produce P[t mod q] tokens, then consume C[t mod r]
    //   tokens. Track occupancy after each produce step; record the maximum.
    //   Require: capacity >= peak_occupancy.
    //
    // The hyper-period H = lcm(q, r) = q * (r / gcd(q, r)).
    // Overflow guard: check that q * (r/g) does not overflow before computing.
    // If H > kMaxSimSteps, skip simulation and emit a warning.
    //
    // Underflow semantics: when occupancy goes negative after the consume step,
    // this means the simulation's produce-before-consume interleaving is
    // incompatible with the hardware's BD scheduling for these rates.  M6
    // already guarantees the schedule is feasible over the full hyper-period;
    // momentary underflow in this simulation does NOT necessarily mean hardware
    // deadlock — the actual AIE BD chain may fire in a different order (e.g.,
    // the hardware drains the consumer BD before the producer BD refills).  We
    // therefore emit emitWarning (not emitOpError) for underflow: it flags a
    // potential ordering mismatch for the user to verify against their BD chain
    // layout, but does not reject the program.  Only capacity overflow
    // (peakOccupancy > capacity) is a hard error, because no interleaving can
    // hide that constraint.
    {
      // Compute buffer capacity from depth * product(element_type.shape).
      // slot_elems is no longer stored as an attribute.
      // Skip M7 when depth = 0 (sentinel for pre-depth-promote pass); the
      // capacity check requires a resolved depth.
      int64_t slot_elems = 0;
      bool hasResolvedDepth = false;
      if (auto d = getDepth()) {
        if (*d > 0) {
          hasResolvedDepth = true;
          if (auto shaped = mlir::dyn_cast<mlir::ShapedType>(getElementType())) {
            slot_elems = static_cast<int64_t>(*d);
            for (int64_t dim : shaped.getShape()) {
              if (mlir::ShapedType::isDynamic(dim)) { slot_elems = 0; break; }
              slot_elems *= dim;
            }
          }
        }
      }
      if (hasResolvedDepth) {
      // Compute gcd(plen, clen) via Euclid's algorithm.
      int64_t a = plen, b = clen;
      while (b) {
        int64_t tmp = b;
        b = a % b;
        a = tmp;
      }
      int64_t g = a;
      int64_t clenOverG = clen / g; // exact: g divides clen by construction
      constexpr int64_t kMaxSimSteps = 1024;
      // Overflow guard: plen * clenOverG must not overflow int64_t and must be
      // within the simulation cap before we compute hyperPeriod.
      // Since kMaxSimSteps == 1024 and plen >= 1, the product overflows only
      // when clenOverG > INT64_MAX / plen.  We conservatively skip simulation
      // if either factor exceeds kMaxSimSteps (the product would then exceed
      // the cap regardless).
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
              emitWarning(
                  "M7: CSDF hyper-period simulation: momentary underflow "
                  "at step ")
                  << t << " (occupancy=" << occupancy
                  << "); hardware BD scheduling may differ from "
                     "produce-before-consume simulation order";
              occupancy = 0; // reset to prevent cascading underflow reports
            }
          }
          if (peakOccupancy > slot_elems)
            return emitOpError("M7: CSDF buffer capacity insufficient: "
                               "peak token occupancy over one hyper-period=")
                   << peakOccupancy << " exceeds slot_elems =" << slot_elems
                   << " (producer_rates=" << psum << "/phase, "
                   << "consumer_rates=" << csum << "/phase, "
                   << "hyper-period=" << hyperPeriod << " steps)";
        }
      }
      } // end if (hasResolvedDepth)
    }
  }

  // window_size was deleted from conduit.create in Sprint 6 (it was dead).
  // Sliding-window capacity is now checked by the pairing pass using
  // acquire count vs. depth directly.

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
checkWindowReleaseCumulativeCount(mlir::Operation *producerOp,
                                  mlir::Value windowVal) {
  // M8a: True double-release detection — cumulative released count at the same
  // nesting level must not exceed the acquired count.  Multiple conduit.release
  // ops on the same window value are valid for sliding-window partial-release
  // patterns (e.g., acquire(3) followed by three release(1) calls).
  //
  // Only releases in the SAME parent block as the acquire op are counted.
  // Releases inside nested regions (e.g., loop bodies) execute once per loop
  // iteration; their relationship to the acquire count is enforced at runtime
  // and depends on the loop trip count — static counting would yield false
  // positives.  M9 liveness analysis (separate pass) handles loop-carried
  // cases.
  //
  // Example valid pattern (cross-block):
  //   %win = conduit.acquire {count=1}    // in block B0
  //   conduit.release %win {count=1}      // in block B0 — counted
  //   scf.for ... {
  //     conduit.release %win {count=1}    // in nested block B1 — NOT counted
  //   }
  // The loop body's release fires once per iteration; Pass A emits this pattern
  // when the producer releases before re-acquiring each iteration.

  int64_t acquiredCount = 0;
  if (auto acqOp = mlir::dyn_cast<Acquire>(producerOp)) {
    acquiredCount = static_cast<int64_t>(acqOp.getCount());
  } else if (auto waitWinOp = mlir::dyn_cast<WaitWindow>(producerOp)) {
    if (auto acqAsyncOp = waitWinOp.getToken().getDefiningOp<AcquireAsync>()) {
      acquiredCount = static_cast<int64_t>(acqAsyncOp.getCount());
    } else {
      return ::mlir::success();
    }
  } else {
    return ::mlir::success();
  }

  // Count only releases at the same nesting level (same parent block).
  mlir::Block *producerBlock = producerOp->getBlock();
  int64_t totalReleased = 0;
  for (mlir::OpOperand &use : windowVal.getUses()) {
    if (auto relOp = mlir::dyn_cast<Release>(use.getOwner())) {
      if (relOp->getBlock() == producerBlock)
        totalReleased += static_cast<int64_t>(relOp.getCount());
    }
  }

  if (totalReleased == 0)
    return ::mlir::success();

  if (totalReleased > acquiredCount)
    return producerOp->emitOpError("M8: cumulative release count (")
           << totalReleased << ") exceeds acquired count (" << acquiredCount
           << ") -- double-release causes hardware lock-counter overflow";

  return ::mlir::success();
}

static ::mlir::LogicalResult checkWindowTokenLinear(mlir::Operation *producerOp,
                                                    mlir::Value tokenVal) {
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

static ::mlir::LogicalResult checkTokenOperandTypes(mlir::Operation *op,
                                                    mlir::ValueRange operands) {
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

  // M7-window: sliding-window buffer capacity check.
  // When window_size is set, the channel depth must be >= window_size to avoid
  // deadlock (the DMA ring must have enough physical slots to satisfy the
  // maximum concurrent hold).
  if (auto ws = getWindowSize()) {
    int64_t wsize = static_cast<int64_t>(*ws);
    Create createOp = findConduitCreateByName(getOperation(), getName());
    if (createOp) {
      if (auto d = createOp.getDepth()) {
        int64_t depth = static_cast<int64_t>(*d);
        if (depth < wsize)
          return emitOpError("depth must be >= window_size for sliding-window "
                             "channel (depth=")
                 << depth << ", window_size=" << wsize << ")";
      }
    }
  }

  // M9 Phase 2 (same-block acquire-release pairing) is implemented in the
  // separate --conduit-check-pairing analysis pass (ConduitPairingCheck.cpp).
  // Moved out of verify() to avoid MLIR diagnostic infinite recursion and to
  // make the warnings capturable by FileCheck / --verify-diagnostics.

  return ::mlir::success();
}
::mlir::LogicalResult AcquireAsync::verify() {
  if (getToken().use_empty())
    return emitOpError("(M8-drop) window.token has no uses: acquired lock "
                       "will never be released — deadlock");
  if (failed(checkTokenDoesNotEscape(getOperation(), getToken())))
    return ::mlir::failure();
  if (failed(checkWindowTokenLinear(getOperation(), getToken())))
    return ::mlir::failure();

  // M7-window: sliding-window buffer capacity check.
  if (auto ws = getWindowSize()) {
    int64_t wsize = static_cast<int64_t>(*ws);
    Create createOp = findConduitCreateByName(getOperation(), getName());
    if (createOp) {
      if (auto d = createOp.getDepth()) {
        int64_t depth = static_cast<int64_t>(*d);
        if (depth < wsize)
          return emitOpError("depth must be >= window_size for sliding-window "
                             "channel (depth=")
                 << depth << ", window_size=" << wsize << ")";
      }
    }
  }

  // M9 Phase 2 (same-block pairing via wait_window) is implemented in the
  // separate --conduit-check-pairing analysis pass (ConduitPairingCheck.cpp).

  return ::mlir::success();
}
::mlir::LogicalResult ReleaseAsync::verify() {
  if (failed(checkTokenDoesNotEscape(getOperation(), getToken())))
    return ::mlir::failure();
  // When the optional $window operand is present, verify:
  //   (1) The operand is a !conduit.window<T> type (enforced by ODS already,
  //       but belt-and-suspenders check for diagnostics clarity).
  //   (2) If the defining op is a conduit.acquire or conduit.wait_window,
  //       its channel name must match $name.
  if (mlir::Value win = getWindow()) {
    if (!mlir::isa<WindowType>(win.getType()))
      return emitOpError(
                 "$window operand must be of type !conduit.window<T>, got ")
             << win.getType();
    if (auto *defOp = win.getDefiningOp()) {
      llvm::StringRef defName;
      bool haveDefName = false;
      if (auto acqOp = mlir::dyn_cast<Acquire>(defOp)) {
        defName = acqOp.getName();
        haveDefName = true;
      } else if (auto waitOp = mlir::dyn_cast<WaitWindow>(defOp)) {
        defName = waitOp.getName();
        haveDefName = true;
      }
      if (haveDefName && defName != getName())
        return emitOpError("$window is from channel '")
               << defName << "' but $name is '" << getName()
               << "' — release_async must release the same channel it acquired";
    }
  }
  return checkWindowTokenLinear(getOperation(), getToken());
}
::mlir::LogicalResult WaitWindow::verify() {
  if (failed(checkWindowReleaseCumulativeCount(getOperation(), getWindow())))
    return ::mlir::failure();

  // M9 Phase 2 (same-block pairing) is implemented in the separate
  // --conduit-check-pairing analysis pass (ConduitPairingCheck.cpp).

  // Channel name consistency: the wait_window's name must match the name of
  // the acquire_async that produced the token operand.  A mismatch indicates
  // that a token from channel "foo" is being presented to wait_window for
  // channel "bar", which would cause Pass C to emit use_lock on the wrong
  // lock and silently corrupt the program.
  if (auto acqAsync = getToken().getDefiningOp<AcquireAsync>()) {
    if (acqAsync.getName() != this->getName()) {
      return emitOpError("wait_window channel name '")
             << this->getName() << "' does not match the channel name '"
             << acqAsync.getName() << "' of the acquire_async token operand";
    }
  }

  return ::mlir::success();
}
::mlir::LogicalResult WaitAll::verify() {
  // Note: This check is redundant with the TableGen Conduit_AnyTokenType
  // constraint, which MLIR enforces before user verify() runs. Left in place
  // for defense-in-depth but may be dead code — the TableGen constraint fires
  // first.
  return checkTokenOperandTypes(getOperation(), getTokens());
}
::mlir::LogicalResult WaitAllAsync::verify() {
  if (failed(checkTokenDoesNotEscape(getOperation(), getResult())))
    return ::mlir::failure();
  // Note: This check is redundant with the TableGen Conduit_AnyTokenType
  // constraint, which MLIR enforces before user verify() runs. Left in place
  // for defense-in-depth but may be dead code — the TableGen constraint fires
  // first.
  return checkTokenOperandTypes(getOperation(), getTokens());
}

//===----------------------------------------------------------------------===//
// M2: Release verifier
//
// Checks:
//   1. count > 0 — releasing zero elements is nonsensical.
//   2. Port consistency — the release port must match the port of the
//      defining acquire (or acquire_async via wait_window).  A mismatch
//      means the consumer releases the producer's lock (or vice versa),
//      causing silent hardware lock-counter corruption.
//   3. Single-op count bound — a single release cannot return more slots
//      than were acquired.  (Cumulative multi-release check is M8a,
//      enforced from the acquire side.)
//===----------------------------------------------------------------------===//

::mlir::LogicalResult Release::verify() {
  // 1. count > 0.
  if (getCount() == 0)
    return emitOpError("release count must be > 0");

  // 2–3. Port consistency and count bound via the defining op.
  mlir::Value win = getWindow();
  if (auto *defOp = win.getDefiningOp()) {
    Port releasePort = getPort();

    if (auto acqOp = mlir::dyn_cast<Acquire>(defOp)) {
      // Port consistency.
      if (acqOp.getPort() != releasePort)
        return emitOpError("release port (")
               << stringifyPort(releasePort)
               << ") does not match acquire port ("
               << stringifyPort(acqOp.getPort()) << ")";
      // Single release count <= acquire count.
      if (getCount() > acqOp.getCount())
        return emitOpError("release count (")
               << getCount() << ") exceeds acquire count ("
               << acqOp.getCount() << ")";
    } else if (auto waitOp = mlir::dyn_cast<WaitWindow>(defOp)) {
      // Trace through wait_window → acquire_async for port and count.
      if (auto acqAsyncOp = waitOp.getToken().getDefiningOp<AcquireAsync>()) {
        if (acqAsyncOp.getPort() != releasePort)
          return emitOpError("release port (")
                 << stringifyPort(releasePort)
                 << ") does not match acquire_async port ("
                 << stringifyPort(acqAsyncOp.getPort()) << ")";
        if (getCount() > acqAsyncOp.getCount())
          return emitOpError("release count (")
                 << getCount() << ") exceeds acquire_async count ("
                 << acqAsyncOp.getCount() << ")";
      }
    }
  }

  return ::mlir::success();
}

//===----------------------------------------------------------------------===//
// M3: PutMemref / GetMemref verifiers (blocking Tier 3 memref-DMA ops)
//
// Checks:
//   1. num_elems > 0 — transferring zero elements is nonsensical.
//   2. offsets, sizes, strides must have equal length (same N-D space).
//   3. Each sizes[i] > 0 — a zero-size dimension makes the transfer empty.
//   4. num_elems == product(sizes) — consistency check.
//===----------------------------------------------------------------------===//

/// Shared verifier logic for put_memref and get_memref.
/// Both ops have identical structural attributes: name, num_elems, offsets,
/// sizes, strides.
static ::mlir::LogicalResult
verifyMemrefDmaOp(mlir::Operation *op, int64_t numElems,
                  llvm::ArrayRef<int64_t> offsets,
                  llvm::ArrayRef<int64_t> sizes,
                  llvm::ArrayRef<int64_t> strides) {
  // 1. num_elems > 0.
  if (numElems <= 0)
    return op->emitOpError("num_elems must be > 0, got ") << numElems;

  // 2. offsets, sizes, strides must have equal length.
  if (offsets.size() != sizes.size())
    return op->emitOpError("offsets length (")
           << offsets.size() << ") does not match sizes length ("
           << sizes.size() << ")";
  if (strides.size() != sizes.size())
    return op->emitOpError("strides length (")
           << strides.size() << ") does not match sizes length ("
           << sizes.size() << ")";

  // 3. Each sizes[i] > 0.
  for (size_t i = 0; i < sizes.size(); ++i) {
    if (sizes[i] <= 0)
      return op->emitOpError("sizes[")
             << i << "] must be > 0, got " << sizes[i];
  }

  // 4. num_elems == product(sizes).
  int64_t product = 1;
  for (int64_t s : sizes)
    product *= s;
  if (numElems != product)
    return op->emitOpError("num_elems (")
           << numElems << ") does not match product of sizes (" << product
           << ")";

  return ::mlir::success();
}

::mlir::LogicalResult PutMemref::verify() {
  return verifyMemrefDmaOp(getOperation(), getNumElems(), getOffsets(),
                           getSizes(), getStrides());
}
::mlir::LogicalResult GetMemref::verify() {
  return verifyMemrefDmaOp(getOperation(), getNumElems(), getOffsets(),
                           getSizes(), getStrides());
}

::mlir::LogicalResult PutMemrefAsync::verify() {
  return checkTokenDoesNotEscape(getOperation(), getToken());
}
::mlir::LogicalResult GetMemrefAsync::verify() {
  return checkTokenDoesNotEscape(getOperation(), getToken());
}

//===----------------------------------------------------------------------===//
// Relay op memtile verifier helper
//
// memtile is a StrAttr "tile(col,row)". Validated here at verify time.
// TODO (post-Sprint 6): migrate to FlatSymbolRefAttr once aie.tile ops have
// sym_names, then this helper can be replaced by MLIR symbol resolution.
//===----------------------------------------------------------------------===//

static std::pair<int64_t, int64_t>
parseTileCoordForVerifier(llvm::StringRef s) {
  if (!s.starts_with("tile(") || !s.ends_with(")"))
    return {-1, -1};
  auto inner = s.drop_front(5).drop_back(1); // "COL,ROW"
  auto [colStr, rowStr] = inner.split(',');
  int64_t col, row;
  if (colStr.trim().getAsInteger(10, col) ||
      rowStr.trim().getAsInteger(10, row))
    return {-1, -1};
  return {col, row};
}

static ::mlir::LogicalResult
verifyMemtileStr(mlir::Operation *op, llvm::StringRef memtile) {
  if (parseTileCoordForVerifier(memtile).first == -1)
    return op->emitOpError(
               "memtile attribute must be of the form 'tile(col,row)', got '")
           << memtile << "'";
  return ::mlir::success();
}

//===----------------------------------------------------------------------===//
// ScatterOp
//===----------------------------------------------------------------------===//

::mlir::LogicalResult ScatterOp::verify() {
  auto dsts = getDsts();
  if (dsts.empty())
    return emitOpError("scatter requires at least 1 dst, got 0");
  // MemTile DMA budget: 1 S2MM + N MM2S <= 12.
  if (1 + dsts.size() > 12)
    return emitOpError("scatter DMA budget exceeded: 1 src + ")
           << dsts.size() << " dsts = " << (1 + dsts.size())
           << " channels, maximum is 12 (MemTile has 6 MM2S + 6 S2MM)";
  if (failed(verifyMemtileStr(getOperation(), getMemtile())))
    return ::mlir::failure();
  return ::mlir::success();
}

//===----------------------------------------------------------------------===//
// GatherOp
//===----------------------------------------------------------------------===//

::mlir::LogicalResult GatherOp::verify() {
  auto srcs = getSrcs();
  if (srcs.empty())
    return emitOpError("gather requires at least 1 src, got 0");
  // MemTile DMA budget: N S2MM + 1 MM2S <= 12.
  if (srcs.size() + 1 > 12)
    return emitOpError("gather DMA budget exceeded: ")
           << srcs.size() << " srcs + 1 dst = " << (srcs.size() + 1)
           << " channels, maximum is 12 (MemTile has 6 MM2S + 6 S2MM)";
  if (failed(verifyMemtileStr(getOperation(), getMemtile())))
    return ::mlir::failure();
  return ::mlir::success();
}

//===----------------------------------------------------------------------===//
// TransposeOp
//===----------------------------------------------------------------------===//

::mlir::LogicalResult TransposeOp::verify() {
  auto srcs = getSrcs();
  auto dsts = getDsts();
  auto offsets = getOffsets();
  if (srcs.empty())
    return emitOpError("transpose requires at least 1 src, got 0");
  if (dsts.empty())
    return emitOpError("transpose requires at least 1 dst, got 0");
  // MemTile DMA budget: N S2MM + M MM2S <= 12.
  if (srcs.size() + dsts.size() > 12)
    return emitOpError("transpose DMA budget exceeded: ")
           << srcs.size() << " srcs + " << dsts.size()
           << " dsts = " << (srcs.size() + dsts.size())
           << " channels, maximum is 12 (MemTile has 6 MM2S + 6 S2MM)";
  // offsets must have exactly srcs.size() * dsts.size() entries.
  size_t expectedOffsets = srcs.size() * dsts.size();
  if (offsets.size() != expectedOffsets)
    return emitOpError("offsets size must equal srcs.size() * dsts.size() = ")
           << expectedOffsets << ", got " << offsets.size();
  // Packet ID budget: N*M <= 32.
  if (expectedOffsets > 32)
    return emitOpError("transpose packet ID budget exceeded: ")
           << srcs.size() << " * " << dsts.size() << " = " << expectedOffsets
           << ", maximum is 32 (AIE2 packet ID space)";
  if (failed(verifyMemtileStr(getOperation(), getMemtile())))
    return ::mlir::failure();
  return ::mlir::success();
}


//===----------------------------------------------------------------------===//
// Conduit ops — generated op definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "aie/Dialect/Conduit/IR/ConduitOps.cpp.inc"
