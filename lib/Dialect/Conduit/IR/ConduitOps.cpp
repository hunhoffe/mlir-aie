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
//   packet ID budget, memtile format RegisterBuffersOp::verify() — provenance
//   (aie.buffer / aie.external_buffer)
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
  // Addition 1 — depth < 0 rejection.
  // depth=0 is the sentinel for "unresolved" (set by Pass A/B, resolved by
  // --conduit-depth-promote).  Positive values are explicit depths.
  // Negative values are always invalid.
  if (auto d = getDepth()) {
    if (static_cast<int64_t>(*d) < 0)
      return emitOpError("depth must be >= 0 (0 = unresolved sentinel, "
                         ">0 = explicit depth); got ")
             << static_cast<int64_t>(*d);
  }

  // Addition 3 — sync_mode + disable_synchronization conflict.
  // sync_mode specifies an active synchronization protocol;
  // disable_synchronization suppresses all lock emission.  The two are mutually
  // exclusive.
  if (getSyncMode().has_value() && getDisableSynchronization().value_or(false))
    return emitOpError(
        "sync_mode and disable_synchronization=true are mutually exclusive");

  if (auto elemTypeOpt = getElementType()) {
    mlir::Type ty = *elemTypeOpt;
    // Addition 2 — element_type must be MemRefType.
    if (!mlir::isa<mlir::MemRefType>(ty))
      return emitOpError("element_type must be a MemRefType when present, got ")
             << ty;
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
  // M5: routing_mode is now an ODS enum (RoutingModeAttr) — invalid values are
  // rejected by the parser before the verifier runs. No explicit check needed.

  // B-5: producer_dimensions / consumer_dimensions type checking is now
  // enforced by ODS (BDDimLayoutArrayAttr / BDDimLayoutArrayArrayAttr
  // constraints in Conduit.td). Invalid types are rejected at parse time.

  // plio verifier: plio=true requires a shim-row (row == 0) endpoint — either
  // the producer tile or at least one consumer tile must be on the shim row.
  //
  // Caveat: inside a DeviceOp, Pass A records shim consumers via
  // aie.shim_dma_allocation ops rather than consumer_tiles.  We cannot do a
  // module-level walk in the verifier, so skip the consumer-tiles check when
  // inside a DeviceOp (trust Pass A correctness).  For hand-written IR
  // outside a DeviceOp, consumer_tiles is the authoritative list.
  //
  // NOTE: Since Phase 9 enforces HasParent<DeviceOp> on conduit.create, the
  // !insideDevice branch below is effectively dead code for valid IR — all
  // conduit.create ops are now required to be inside a DeviceOp.  The branch
  // is retained as a safety net for test contexts that may construct ops
  // outside DeviceOp programmatically.
  if (auto plioAttr = getPlio()) {
    if (*plioAttr) {
      bool producerIsShim = false;
      if (auto tileArr = getProducerTile()) {
        auto arr = *tileArr;
        if (arr.size() >= 2 && arr[1] == 0)
          producerIsShim = true;
      }
      if (!producerIsShim) {
        bool insideDevice =
            getOperation()->getParentOfType<xilinx::AIE::DeviceOp>() != nullptr;
        if (!insideDevice) {
          // Hand-written IR: consumer_tiles is authoritative.
          bool consumerHasShim = false;
          if (auto consArr = getConsumerTiles()) {
            auto arr = *consArr;
            for (size_t i = 0; i + 1 < arr.size(); i += 2) {
              if (arr[i + 1] == 0) {
                consumerHasShim = true;
                break;
              }
            }
          }
          if (!consumerHasShim)
            return emitOpError("plio=true requires a shim tile (row 0) as "
                               "producer_tile or consumer_tiles");
        }
        // Inside a DeviceOp: shim consumers may exist via
        // aie.shim_dma_allocation — skip the check.
      }
    }
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
      int64_t slot_elems = getSlotElems();
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
    }
  }

  // M7 extension: window_size must not exceed depth.
  // Enforces that the buffer pool is large enough for the sliding window.
  if (auto ws = getWindowSize()) {
    if (auto d = getDepth()) {
      int64_t depth = static_cast<int64_t>(*d);
      if (depth > 0 && static_cast<int64_t>(*ws) > depth) {
        return emitOpError("window_size (")
               << *ws << ") exceeds depth (" << depth
               << "); buffer pool too small for sliding window";
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

::mlir::LogicalResult PutMemrefAsync::verify() {
  return checkTokenDoesNotEscape(getOperation(), getToken());
}
::mlir::LogicalResult GetMemrefAsync::verify() {
  return checkTokenDoesNotEscape(getOperation(), getToken());
}

//===----------------------------------------------------------------------===//
// parseTileCoordForVerifier — helper for relay op memtile format validation
//===----------------------------------------------------------------------===//

/// Parse a tile coordinate string of the form "tile(col,row)" and return
/// {col, row}. Returns {-1, -1} on any parse failure.
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

//===----------------------------------------------------------------------===//
// ScatterOp
//===----------------------------------------------------------------------===//

::mlir::LogicalResult ScatterOp::verify() {
  // ScatterOp has a singular $src (FlatSymbolRefAttr) — no size-1 check needed.
  auto dsts = getDsts();

  // dsts must contain at least 1 entry.
  if (dsts.empty())
    return emitOpError("scatter requires at least 1 dst, got 0");

  // MemTile DMA budget: 1 S2MM (source) + N MM2S (destinations) <= 12
  // (MemTile has 6 MM2S + 6 S2MM channels; using 1 S2MM leaves 11 MM2S max).
  if (1 + dsts.size() > 12)
    return emitOpError("scatter DMA budget exceeded: 1 src + ")
           << dsts.size() << " dsts = " << (1 + dsts.size())
           << " channels, maximum is 12 (MemTile has 6 MM2S + 6 S2MM)";

  // memtile attribute must be of the form "tile(col,row)".
  if (parseTileCoordForVerifier(getMemtile()).first == -1)
    return emitOpError(
               "memtile attribute must be of the form 'tile(col,row)', got '")
           << getMemtile() << "'";

  return ::mlir::success();
}

//===----------------------------------------------------------------------===//
// GatherOp
//===----------------------------------------------------------------------===//

::mlir::LogicalResult GatherOp::verify() {
  auto srcs = getSrcs();

  // srcs must contain at least 1 entry.
  if (srcs.empty())
    return emitOpError("gather requires at least 1 src, got 0");

  // GatherOp has a singular $dst (FlatSymbolRefAttr) — no size-1 check needed.
  // MemTile DMA budget: N S2MM (sources) + 1 MM2S (destination) <= 12.
  if (srcs.size() + 1 > 12)
    return emitOpError("gather DMA budget exceeded: ")
           << srcs.size() << " srcs + 1 dst = " << (srcs.size() + 1)
           << " channels, maximum is 12 (MemTile has 6 MM2S + 6 S2MM)";

  // memtile attribute must be of the form "tile(col,row)".
  if (parseTileCoordForVerifier(getMemtile()).first == -1)
    return emitOpError(
               "memtile attribute must be of the form 'tile(col,row)', got '")
           << getMemtile() << "'";

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

  // MemTile DMA budget: N S2MM (sources) + M MM2S (destinations) <= 12.
  if (srcs.size() + dsts.size() > 12)
    return emitOpError("transpose DMA budget exceeded: ")
           << srcs.size() << " srcs + " << dsts.size()
           << " dsts = " << (srcs.size() + dsts.size())
           << " channels, maximum is 12 (MemTile has 6 MM2S + 6 S2MM)";

  // offsets.size() must equal srcs.size() * dsts.size().
  size_t expectedOffsets = srcs.size() * dsts.size();
  if (offsets.size() != expectedOffsets)
    return emitOpError("offsets size must equal srcs.size() * dsts.size() = ")
           << expectedOffsets << ", got " << offsets.size();

  // Packet ID budget: N*M <= 32.
  if (expectedOffsets > 32)
    return emitOpError("transpose packet ID budget exceeded: srcs.size() * "
                       "dsts.size() = ")
           << srcs.size() << " * " << dsts.size() << " = " << expectedOffsets
           << ", maximum is 32 (AIE2 packet ID space)";

  // memtile attribute must be of the form "tile(col,row)".
  if (parseTileCoordForVerifier(getMemtile()).first == -1)
    return emitOpError(
               "memtile attribute must be of the form 'tile(col,row)', got '")
           << getMemtile() << "'";

  return ::mlir::success();
}

//===----------------------------------------------------------------------===//
// RegisterBuffersOp
//===----------------------------------------------------------------------===//

::mlir::LogicalResult RegisterBuffersOp::verify() {
  for (Value buf : getBuffers()) {
    Operation *defOp = buf.getDefiningOp();
    if (!defOp || (!mlir::isa<AIE::BufferOp>(defOp) &&
                   !mlir::isa<AIE::ExternalBufferOp>(defOp)))
      return emitOpError("buffer operand must be defined by aie.buffer or "
                         "aie.external_buffer, got ")
             << (defOp ? defOp->getName().getStringRef() : "block argument");
  }
  return ::mlir::success();
}

//===----------------------------------------------------------------------===//
// Conduit ops — generated op definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "aie/Dialect/Conduit/IR/ConduitOps.cpp.inc"
