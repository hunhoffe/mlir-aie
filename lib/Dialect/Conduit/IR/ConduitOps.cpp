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
//   Link::verify() — M3: mode structural invariants + offset counts
//                    M6-join / M7-join: CSDF balance + buffer capacity for N:1 join
//                    M6-dist / M7-dist: CSDF balance + buffer capacity for 1:N distribute
//   Create::verify() — M4: dynamic-dim warning; M5: routing_mode; M6: CSDF balance
//   Acquire::verify() / WaitWindow::verify() — M8a: window value release linearity
//                                              M9: same-block acquire-release pairing (llvm::errs)
//   AcquireAsync::verify() / ReleaseAsync::verify() — M8b: window.token wait_window linearity
//                                                     M9: wait_window→release pairing (llvm::errs)
//   WaitAll::verify() / WaitAllAsync::verify() — M8c: operands must be token types
//
// Denolf 2007 channel type mapping (DOI: 10.1155/2007/84078):
//   conduit.link mode="distribute" (1:N) — Denolf §3.3.3 multi-consumer / nondestructive-read.
//     Level 1: Bilsen 1:1 equation applied per-edge (Eq. 45).
//     Level 2: Composed consume buffer capacity (Eq. 46/48) — cross-conduit check on the
//     source buffer, accounting for the slowest consumer gating buffer reuse.
//   conduit.link mode="join" (N:1) — Denolf §3.3.4 multi-producer / shared-buffer pattern.
//     Per-edge Bilsen 1:1 check (conservative structural approximation).
//     NOTE: Denolf §3.3.4 proves that N:1 join has NO equivalent standard CSDF channel
//     because token arrival order depends on runtime response time.  No exact
//     composed-produce formula exists; the per-edge check is a conservative bound.
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
      if (auto acqAsyncOp =
              waitOp.getToken().getDefiningOp<AcquireAsync>()) {
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
// Shared CSDF helper: find conduit.create by name in the enclosing module.
// Used by Link::verify() for M6-join/M7-join/M6-dist/M7-dist checks.
// Returns nullptr when not found (conduit.create may be in a different
// translation unit or a test fragment; skip rather than error).
// ---------------------------------------------------------------------------
static Create findConduitCreateByName(mlir::Operation *anchor,
                                      llvm::StringRef name) {
  mlir::Operation *mod = anchor;
  while (mod && !mlir::isa<mlir::ModuleOp>(mod))
    mod = mod->getParentOp();
  if (!mod)
    return {};
  Create result{};
  mod->walk([&](Create op) -> mlir::WalkResult {
    if (op.getName() == name) {
      result = op;
      return mlir::WalkResult::interrupt();
    }
    return mlir::WalkResult::advance();
  });
  return result;
}

// ---------------------------------------------------------------------------
// Shared CSDF helper: apply the Bilsen 1:1 balance check (M6) and
// hyper-period buffer capacity check (M7) to a single channel edge.
//
// Theory basis: Bilsen et al. 1996 (IEEE Transactions on Signal Processing,
// DOI: 10.1109/78.485935) defines the CSDF consistency equation for a
// single channel:  sum(P) * len(C) == sum(C) * len(P)
//
// Application to link topologies: this function applies the 1:1 equation
// per-edge to each conduit in a join or distribute link.  For distribute,
// this is supplemented by checkDistributeComposedConsume (Denolf Eq. 46/48)
// which performs the cross-conduit composed consume analysis.  For join,
// Denolf §3.3.4 proves no exact CSDF equivalent exists (token arrival
// order depends on runtime response time); the per-edge check is a
// conservative structural approximation.
//
// This function implements the per-edge check.
//
// Parameters:
//   diagnosticOp — the op to attach error messages to (conduit.link)
//   edgeLabel    — human-readable label for error messages (e.g., "join source 'foo'")
//   pRates       — producer rate sequence P for this edge
//   cRates       — consumer rate sequence C for this edge
//   capacity     — declared buffer capacity for this conduit
//
// Returns failure() if M6 or M7 is violated; success() otherwise.
// ---------------------------------------------------------------------------
static ::mlir::LogicalResult checkCSDF1x1(mlir::Operation *diagnosticOp,
                                          llvm::StringRef edgeLabel,
                                          llvm::ArrayRef<int64_t> pRates,
                                          llvm::ArrayRef<int64_t> cRates,
                                          int64_t capacity) {
  int64_t psum = 0;
  for (int64_t v : pRates)
    psum += v;
  int64_t csum = 0;
  for (int64_t v : cRates)
    csum += v;
  int64_t plen = static_cast<int64_t>(pRates.size());
  int64_t clen = static_cast<int64_t>(cRates.size());

  // M6: Bilsen 1996 balance equation.
  if (psum * clen != csum * plen)
    return diagnosticOp->emitOpError("M6-")
           << edgeLabel << ": CSDF rate imbalance: "
           << "sum(producer_rates)*len(consumer_rates)=" << (psum * clen)
           << " != sum(consumer_rates)*len(producer_rates)=" << (csum * plen)
           << " (producer_rates sum=" << psum << " period=" << plen
           << ", consumer_rates sum=" << csum << " period=" << clen << ")";

  // M7: hyper-period buffer capacity simulation (same algorithm as Create::verify()).
  // Compute gcd(plen, clen) via Euclid's algorithm.
  int64_t a = plen, b = clen;
  while (b) { int64_t tmp = b; b = a % b; a = tmp; }
  int64_t g = a;
  int64_t clenOverG = clen / g;
  constexpr int64_t kMaxSimSteps = 1024;
  if (plen > kMaxSimSteps || clenOverG > kMaxSimSteps / plen) {
    diagnosticOp->emitWarning("M7-")
        << edgeLabel << ": CSDF hyper-period exceeds simulation cap ("
        << kMaxSimSteps << " steps); buffer capacity check skipped";
    return ::mlir::success();
  }
  int64_t hyperPeriod = plen * clenOverG;
  if (hyperPeriod > kMaxSimSteps) {
    diagnosticOp->emitWarning("M7-")
        << edgeLabel << ": CSDF hyper-period exceeds simulation cap ("
        << kMaxSimSteps << " steps); buffer capacity check skipped";
    return ::mlir::success();
  }

  int64_t occupancy = 0;
  int64_t peakOccupancy = 0;
  for (int64_t t = 0; t < hyperPeriod; ++t) {
    occupancy += pRates[static_cast<size_t>(t % plen)];
    if (occupancy > peakOccupancy)
      peakOccupancy = occupancy;
    occupancy -= cRates[static_cast<size_t>(t % clen)];
    if (occupancy < 0) {
      diagnosticOp->emitWarning("M7-")
          << edgeLabel << ": CSDF hyper-period simulation: "
             "momentary underflow at step " << t
          << " (occupancy=" << occupancy
          << "); hardware BD scheduling may differ from "
             "produce-before-consume simulation order";
      occupancy = 0;
    }
  }
  if (peakOccupancy > capacity)
    return diagnosticOp->emitOpError("M7-")
           << edgeLabel
           << ": CSDF buffer capacity insufficient: "
              "peak token occupancy over one hyper-period="
           << peakOccupancy << " exceeds capacity=" << capacity
           << " (producer_rates=" << psum << "/phase"
           << ", consumer_rates=" << csum << "/phase"
           << ", hyper-period=" << hyperPeriod << " steps)";

  return ::mlir::success();
}

// ---------------------------------------------------------------------------
// Denolf Eq. 46/48: composed consume + buffer capacity for 1:N distribute.
//
// Denolf et al. 2007 (DOI: 10.1155/2007/84078) §3.3.3 Equations 45-48.
//
// In a 1:N distribute (multi-consumer / nondestructive-read), N consumers
// share a single source buffer on the MemTile relay.  A buffer container
// can only be freed once ALL consumers have consumed from it.
//
// Eq. 46 — composed consume: cc(j) = min_{1<=y<=N} cumCons_y(t)
//   The composed (aggregate) consumption at step t is the minimum of
//   the cumulative consumption across all N consumers.  This reflects the
//   hardware constraint that the slowest consumer gates buffer reuse.
//
// Eq. 48 — buffer capacity:
//   d >= max over hyper-period of (cumProd(t) - min_{y} cumCons_y(t))
//   The source buffer depth must be at least the peak occupancy computed
//   using the composed consume, not just the per-edge consume.
//
// This check is CROSS-CONDUIT: it combines the source conduit's
// producer_rates with each destination conduit's consumer_rates.
// The per-edge checkCSDF1x1 checks each conduit independently and
// cannot detect bottlenecks caused by a slow consumer in the distribute.
//
// Parameters:
//   diagnosticOp — the conduit.link op for error attachment
//   srcProdRates — the source conduit's producer_rates (P)
//   srcCapacity  — the source conduit's declared buffer capacity
//   dstConsRates — each destination conduit's consumer_rates (C_y)
//   dstNames     — destination conduit names (for error messages)
//
// Returns failure() if the source buffer is undersized; success() otherwise.
// ---------------------------------------------------------------------------
static ::mlir::LogicalResult checkDistributeComposedConsume(
    mlir::Operation *diagnosticOp,
    llvm::ArrayRef<int64_t> srcProdRates,
    int64_t srcCapacity,
    llvm::SmallVectorImpl<llvm::SmallVector<int64_t>> &dstConsRates,
    llvm::SmallVectorImpl<std::string> &dstNames) {
  if (dstConsRates.size() < 2)
    return ::mlir::success(); // single consumer: per-edge check suffices

  // Compute hyper-period H = lcm of all periods.
  auto gcd = [](int64_t a, int64_t b) -> int64_t {
    while (b) { int64_t tmp = b; b = a % b; a = tmp; }
    return a;
  };
  auto lcm = [&gcd](int64_t a, int64_t b) -> int64_t {
    if (a == 0 || b == 0) return 0;
    return (a / gcd(a, b)) * b;
  };

  int64_t H = static_cast<int64_t>(srcProdRates.size());
  for (auto &cRates : dstConsRates)
    H = lcm(H, static_cast<int64_t>(cRates.size()));

  constexpr int64_t kMaxSimSteps = 1024;
  if (H <= 0 || H > kMaxSimSteps) {
    diagnosticOp->emitWarning(
        "M7-dist composed-consume: hyper-period exceeds simulation cap (")
        << kMaxSimSteps
        << " steps); Denolf Eq. 48 buffer capacity check skipped";
    return ::mlir::success();
  }

  // Simulate the hyper-period.
  int64_t plen = static_cast<int64_t>(srcProdRates.size());
  unsigned N = dstConsRates.size();
  int64_t cumProd = 0;
  llvm::SmallVector<int64_t> cumCons(N, 0);
  int64_t peakOccupancy = 0;

  for (int64_t t = 0; t < H; ++t) {
    // Producer fires: add tokens to source buffer.
    cumProd += srcProdRates[static_cast<size_t>(t % plen)];

    // Each consumer fires: track cumulative consumption.
    for (unsigned y = 0; y < N; ++y) {
      int64_t clen = static_cast<int64_t>(dstConsRates[y].size());
      cumCons[y] += dstConsRates[y][static_cast<size_t>(t % clen)];
    }

    // Composed consume (Eq. 46): min over all consumers.
    int64_t composedConsume = cumCons[0];
    for (unsigned y = 1; y < N; ++y) {
      if (cumCons[y] < composedConsume)
        composedConsume = cumCons[y];
    }

    // Occupied containers = produced - composed consume.
    int64_t occupied = cumProd - composedConsume;
    if (occupied > peakOccupancy)
      peakOccupancy = occupied;
  }

  if (peakOccupancy > srcCapacity) {
    // Identify the bottleneck consumer (min cumulative at end).
    unsigned bottleneck = 0;
    for (unsigned y = 1; y < N; ++y) {
      if (cumCons[y] < cumCons[bottleneck])
        bottleneck = y;
    }
    return diagnosticOp->emitOpError(
               "M7-dist composed-consume (Denolf Eq. 48): "
               "source buffer capacity insufficient for multi-consumer "
               "distribute: peak occupancy=")
           << peakOccupancy << " exceeds source capacity=" << srcCapacity
           << " (bottleneck consumer: '" << dstNames[bottleneck]
           << "', hyper-period=" << H << " steps"
           << "; a container can only be freed after ALL "
           << N << " consumers have consumed it)";
  }

  return ::mlir::success();
}

::mlir::LogicalResult Link::verify() {
  auto modeStr = getMode();
  auto srcs = getSrcs();
  auto dsts = getDsts();
  auto offsets = getOffsets();

  // M3: mode validation — enforce structural invariants per mode.
  if (modeStr == "cascade")
    return emitError("conduit.link: cascade mode not yet supported");
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

  // -------------------------------------------------------------------------
  // M6-join / M7-join: CSDF balance and buffer capacity for N:1 join.
  //
  // Theory: Denolf et al. 2007 (DOI: 10.1155/2007/84078) §3.3.4 shows
  // that the N:1 multi-producer (join) pattern CANNOT be reformulated as
  // an equivalent standard CSDF channel.  The paper states:
  //   "there does not exist an equivalent standard channel for a channel
  //    with multiple producers.  The reason is that token order depends on
  //    runtime response time."
  //
  // Consequence: there is no exact Denolf-style composed-produce formula
  // for join.  The per-edge Bilsen 1:1 check applied to each source and
  // the destination is a CONSERVATIVE STRUCTURAL APPROXIMATION: it verifies
  // that each individual conduit is internally balanced, but cannot verify
  // the cross-conduit token arrival ordering that is inherently runtime-
  // dependent.
  //
  // Implementation: for each source conduit and the destination conduit,
  // apply the Bilsen 1:1 balance equation and M7 buffer capacity simulation
  // independently.  This catches rate imbalances and undersized buffers on
  // individual conduits but does not attempt cross-conduit analysis.
  //
  // If any conduit lacks rate annotations, the check is skipped for that edge
  // (not an error: rates are optional; M6 only fires when explicitly provided
  // or inferred by --conduit-infer-rates).
  // -------------------------------------------------------------------------
  if (modeStr == "join") {
    // Check each source conduit independently.
    for (auto srcAttr : srcs) {
      llvm::StringRef srcName =
          mlir::cast<mlir::StringAttr>(srcAttr).getValue();
      Create srcCreate = findConduitCreateByName(getOperation(), srcName);
      if (!srcCreate)
        continue; // conduit.create not in scope — skip
      if (!srcCreate.getProducerRates().has_value() ||
          !srcCreate.getConsumerRates().has_value())
        continue; // no rate annotations — skip
      auto pRates = *srcCreate.getProducerRates();
      auto cRates = *srcCreate.getConsumerRates();
      int64_t cap = srcCreate.getCapacity();
      std::string label = "join source '" + srcName.str() + "'";
      if (failed(checkCSDF1x1(getOperation(), label, pRates, cRates, cap)))
        return ::mlir::failure();
    }
    // Check destination conduit.
    llvm::StringRef dstName =
        mlir::cast<mlir::StringAttr>(dsts[0]).getValue();
    Create dstCreate = findConduitCreateByName(getOperation(), dstName);
    if (dstCreate && dstCreate.getProducerRates().has_value() &&
        dstCreate.getConsumerRates().has_value()) {
      auto pRates = *dstCreate.getProducerRates();
      auto cRates = *dstCreate.getConsumerRates();
      int64_t cap = dstCreate.getCapacity();
      std::string label = "join destination '" + dstName.str() + "'";
      if (failed(checkCSDF1x1(getOperation(), label, pRates, cRates, cap)))
        return ::mlir::failure();
    }
  }

  // -------------------------------------------------------------------------
  // M6-dist / M7-dist: CSDF balance and buffer capacity for 1:N distribute.
  //
  // Denolf et al. 2007 (DOI: 10.1155/2007/84078) §3.3.3 Equations 45-48.
  //
  // Two levels of verification:
  //
  // Level 1 — Per-edge balance (Eq. 45): each conduit's own producer_rates
  //   and consumer_rates must satisfy the Bilsen 1:1 balance equation.
  //   This is the same check applied to individual conduit.create ops and
  //   catches imbalanced rate sequences per edge.
  //
  // Level 2 — Composed consume buffer capacity (Eq. 46/48): the source
  //   conduit's buffer is shared by all N destination consumers.  A buffer
  //   container can only be freed after ALL consumers have consumed from it
  //   (Eq. 46: composed consume = min over all consumers).  The source
  //   buffer capacity must be >= the peak occupancy under this constraint
  //   (Eq. 48).  This is a cross-conduit check that per-edge analysis
  //   cannot detect: the source's own M7 check passes (its relay consumer
  //   drains at the declared rate), but a slow destination consumer gates
  //   buffer reuse and can cause overflow.
  // -------------------------------------------------------------------------
  if (modeStr == "distribute") {
    // Level 1: Per-edge balance checks (Eq. 45).
    // Check source conduit.
    llvm::StringRef srcName =
        mlir::cast<mlir::StringAttr>(srcs[0]).getValue();
    Create srcCreate = findConduitCreateByName(getOperation(), srcName);
    if (srcCreate && srcCreate.getProducerRates().has_value() &&
        srcCreate.getConsumerRates().has_value()) {
      auto pRates = *srcCreate.getProducerRates();
      auto cRates = *srcCreate.getConsumerRates();
      int64_t cap = srcCreate.getCapacity();
      std::string label = "distribute source '" + srcName.str() + "'";
      if (failed(checkCSDF1x1(getOperation(), label, pRates, cRates, cap)))
        return ::mlir::failure();
    }
    // Check each destination conduit independently.
    for (auto dstAttr : dsts) {
      llvm::StringRef dstName =
          mlir::cast<mlir::StringAttr>(dstAttr).getValue();
      Create dstCreate = findConduitCreateByName(getOperation(), dstName);
      if (!dstCreate)
        continue;
      if (!dstCreate.getProducerRates().has_value() ||
          !dstCreate.getConsumerRates().has_value())
        continue;
      auto pRates = *dstCreate.getProducerRates();
      auto cRates = *dstCreate.getConsumerRates();
      int64_t cap = dstCreate.getCapacity();
      std::string label = "distribute destination '" + dstName.str() + "'";
      if (failed(checkCSDF1x1(getOperation(), label, pRates, cRates, cap)))
        return ::mlir::failure();
    }

    // Level 2: Composed consume buffer capacity (Denolf Eq. 46/48).
    // Collect destination consumer_rates and check the source buffer
    // capacity against the multi-consumer composed consume.
    if (srcCreate && srcCreate.getProducerRates().has_value()) {
      llvm::SmallVector<llvm::SmallVector<int64_t>> allDstConsRates;
      llvm::SmallVector<std::string> allDstNames;
      bool allDstsHaveRates = true;
      for (auto dstAttr : dsts) {
        llvm::StringRef dstName =
            mlir::cast<mlir::StringAttr>(dstAttr).getValue();
        Create dstCreate = findConduitCreateByName(getOperation(), dstName);
        if (!dstCreate || !dstCreate.getConsumerRates().has_value()) {
          allDstsHaveRates = false;
          break;
        }
        auto cRates = *dstCreate.getConsumerRates();
        llvm::SmallVector<int64_t> rates(cRates.begin(), cRates.end());
        allDstConsRates.push_back(std::move(rates));
        allDstNames.push_back(dstName.str());
      }
      if (allDstsHaveRates && allDstConsRates.size() >= 2) {
        auto srcPRates = *srcCreate.getProducerRates();
        llvm::SmallVector<int64_t> srcPR(srcPRates.begin(), srcPRates.end());
        if (failed(checkDistributeComposedConsume(
                getOperation(), srcPR, srcCreate.getCapacity(),
                allDstConsRates, allDstNames)))
          return ::mlir::failure();
      }
    }
  }

  // -------------------------------------------------------------------------
  // A-10: cascade channels cannot be used in distribute or join links.
  //
  // Cascade is a register-level rendezvous with no FIFO buffering and no DMA
  // channels — it is structurally incompatible with the multi-producer /
  // multi-consumer split/merge semantics of distribute and join.  Attempting
  // to route a cascade conduit through a link would silently produce incorrect
  // hardware code (no actual flow is emitted for cascade, so the non-cascade
  // consumers/producers would deadlock).
  // -------------------------------------------------------------------------
  if (modeStr == "distribute" || modeStr == "join") {
    // Check all src and dst channel names against their conduit.create
    // routing_mode.  Only the "cascade" value is illegal here.
    auto checkCascade = [&](mlir::ArrayAttr names) -> mlir::LogicalResult {
      for (auto attr : names) {
        llvm::StringRef name = mlir::cast<mlir::StringAttr>(attr).getValue();
        Create chanCreate = findConduitCreateByName(getOperation(), name);
        if (!chanCreate)
          continue; // not in scope — skip
        auto routingModeOpt = chanCreate.getRoutingMode();
        if (routingModeOpt && *routingModeOpt == "cascade") {
          return emitOpError("cascade channel '")
                 << name << "' cannot be used in a '" << modeStr << "' link";
        }
      }
      return ::mlir::success();
    };
    if (failed(checkCascade(srcs)) || failed(checkCascade(dsts)))
      return ::mlir::failure();
  }

  return ::mlir::success();
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
    if (rm != "circuit" && rm != "packet" && rm != "cascade" && rm != "any" &&
        rm != "stream")
      return emitOpError(
                 "routing_mode must be \"circuit\", \"packet\", \"cascade\", "
                 "\"stream\", or \"any\", got \"")
             << rm << "\"";
  }

  // B-5: producer_dimensions / consumer_dimensions type guard.
  //
  // Both attributes are stored as AnyAttr to avoid a cross-dialect TableGen
  // dependency on AIE::BDDimLayoutArrayAttr / AIE::BDDimLayoutArrayArrayAttr.
  //
  // AIE::BDDimLayoutArrayAttr is NOT a subclass of mlir::ArrayAttr — it is a
  // custom attribute defined with ArrayOfAttr<> in TableGen, which produces
  // its own C++ class.  Therefore mlir::isa<mlir::ArrayAttr>() cannot be used
  // to validate it from this file (which does not include the AIE dialect).
  //
  // The minimal safe check: reject obviously wrong scalar attribute types
  // (StringAttr, IntegerAttr) that can NEVER be valid BDDimLayout descriptors
  // and would cause a crash when Pass C attempts to cast the attribute.
  // Valid BDDimLayoutArrayAttr attributes will always pass this check.
  if (auto prodDimsAttr = getProducerDimensions()) {
    if (mlir::isa<mlir::StringAttr, mlir::IntegerAttr>(*prodDimsAttr))
      return emitOpError(
          "producer_dimensions must be an AIE::BDDimLayoutArrayAttr; "
          "got a scalar attribute — was this conduit.create round-tripped "
          "without the AIE dialect loaded?");
  }
  if (auto consDimsAttr = getConsumerDimensions()) {
    if (mlir::isa<mlir::StringAttr, mlir::IntegerAttr>(*consDimsAttr))
      return emitOpError(
          "consumer_dimensions must be an AIE::BDDimLayoutArrayArrayAttr; "
          "got a scalar attribute — was this conduit.create round-tripped "
          "without the AIE dialect loaded?");
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
  // M8a: True double-release detection — cumulative released count at the same
  // nesting level must not exceed the acquired count.  Multiple conduit.release
  // ops on the same window value are valid for sliding-window partial-release
  // patterns (e.g., acquire(3) followed by three release(1) calls).
  //
  // Only releases in the SAME parent block as the acquire op are counted.
  // Releases inside nested regions (e.g., loop bodies) execute once per loop
  // iteration; their relationship to the acquire count is enforced at runtime
  // and depends on the loop trip count — static counting would yield false
  // positives.  M9 liveness analysis (separate pass) handles loop-carried cases.
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
    if (auto acqAsyncOp =
            waitWinOp.getToken().getDefiningOp<AcquireAsync>()) {
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

  // Channel name consistency: the wait_window's name must match the name of
  // the acquire_async that produced the token operand.  A mismatch indicates
  // that a token from channel "foo" is being presented to wait_window for
  // channel "bar", which would cause Pass C to emit use_lock on the wrong
  // lock and silently corrupt the program.
  if (auto acqAsync = getToken().getDefiningOp<AcquireAsync>()) {
    if (acqAsync.getName() != this->getName()) {
      return emitOpError("wait_window channel name '")
             << this->getName()
             << "' does not match the channel name '"
             << acqAsync.getName()
             << "' of the acquire_async token operand";
    }
  }

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
// Cascade op verifiers
//
// Verify that the referenced conduit.create has routing_mode = "cascade".
// Walk up to the enclosing ModuleOp and search for the matching create op.
//===----------------------------------------------------------------------===//

static ::mlir::LogicalResult
checkCascadeConduit(mlir::Operation *op, llvm::StringRef name) {
  mlir::Operation *ancestor = op->getParentOp();
  while (ancestor && !mlir::isa<mlir::ModuleOp>(ancestor))
    ancestor = ancestor->getParentOp();
  if (!ancestor)
    return ::mlir::success(); // No module found — skip (test fragment).

  bool found = false;
  bool wrongMode = false;
  ancestor->walk([&](Create createOp) -> mlir::WalkResult {
    if (createOp.getName() != name)
      return mlir::WalkResult::advance();
    found = true;
    auto rmOpt = createOp.getRoutingMode();
    if (!rmOpt || *rmOpt != "cascade")
      wrongMode = true;
    return mlir::WalkResult::interrupt();
  });

  if (found && wrongMode)
    return op->emitOpError("references conduit '")
           << name << "' which does not have routing_mode = \"cascade\"";
  // If not found: conduit.create may not be in scope yet (test fragment).
  return ::mlir::success();
}

/// Return the bit-width of an integer or vector-of-integer type, or 0.
static unsigned cascadeTypeBitWidth(mlir::Type ty) {
  if (auto intTy = mlir::dyn_cast<mlir::IntegerType>(ty))
    return intTy.getWidth();
  if (auto vecTy = mlir::dyn_cast<mlir::VectorType>(ty)) {
    if (auto eltInt = mlir::dyn_cast<mlir::IntegerType>(vecTy.getElementType()))
      return static_cast<unsigned>(vecTy.getNumElements()) * eltInt.getWidth();
  }
  return 0;
}

/// Validate cascade value type: must be an integer or integer vector.
/// Architecture-correct widths: AIE1=384 bits, AIE2=512 bits.
/// Wrong basic type → error. Non-standard width → error (prevents silent
/// mismatch; use vector<16xi32> for AIE2 or vector<8xi48>/i384 for AIE1).
static ::mlir::LogicalResult
checkCascadeValueType(mlir::Operation *op, mlir::Type ty) {
  unsigned bits = cascadeTypeBitWidth(ty);
  if (bits == 0) {
    return op->emitOpError("cascade value type ")
           << ty << " is not an integer or integer vector type; cascade "
              "requires a fixed-width integer or integer vector matching the "
              "architecture cascade width (AIE1: i384 or vector<8xi48>, "
              "AIE2: i512 or vector<16xi32>)";
  }
  if (bits != 384 && bits != 512) {
    return op->emitOpError("cascade value type ")
           << ty << " has width " << bits
           << " bits; must be 384 bits (AIE1: i384 or vector<8xi48>) "
              "or 512 bits (AIE2: i512 or vector<16xi32>)";
  }
  return ::mlir::success();
}

::mlir::LogicalResult PutCascade::verify() {
  if (failed(checkCascadeConduit(getOperation(), getName())))
    return ::mlir::failure();
  return checkCascadeValueType(getOperation(), getValue().getType());
}

::mlir::LogicalResult GetCascade::verify() {
  if (failed(checkCascadeConduit(getOperation(), getName())))
    return ::mlir::failure();
  return checkCascadeValueType(getOperation(), getValue().getType());
}

//===----------------------------------------------------------------------===//
// Conduit ops — generated op definitions
//===----------------------------------------------------------------------===//

#define GET_OP_CLASSES
#include "aie/Dialect/Conduit/IR/ConduitOps.cpp.inc"
