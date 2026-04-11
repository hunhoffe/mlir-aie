//===- ConduitInferRates.cpp - conduit-infer-rates pass ----------*-C++-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2026 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//
//
// --conduit-infer-rates: infer CSDF producer_rates / consumer_rates from
// conduit ops in IR.  Origin-agnostic: works after Pass A (Tier 2 acquire/
// release) or Pass B (Tier 3 put/get_memref_async).
//
// Background
// ----------
// Pass B (--air-channel-to-conduit) produces conduit.put_memref_async and
// conduit.get_memref_async ops with a num_elems attribute recording how many
// elements each DMA transfer moves.  When different puts (or gets) on the same
// channel move different numbers of elements, the channel is CSDF (Cyclostatic
// Data Flow) with period > 1.
//
// The M6/M7 verifiers in Create::verify() (ConduitOps.cpp) check CSDF balance
// and buffer capacity, but only fire when producer_rates and consumer_rates are
// explicitly set on the conduit.create op.  Pass B does not set them because it
// lacks the full put/get sequence at channel-declaration time.
//
// This pass fills that gap: it walks the IR per aie.device, collects num_elems
// sequences for each channel, and attaches producer_rates / consumer_rates to
// the matching conduit.create.  On the next mlir verify step, M6/M7 fire
// automatically.
//
// Algorithm (device-scoped)
// -------------------------
// For each aie.device op in the module, for each conduit.create inside it
// that has no existing producer_rates / consumer_rates:
//
//   1. Collect all conduit.put_memref_async ops within the device body that
//      reference this channel name (name == conduit sym_name).  Extract their
//      num_elems attribute values in program order.
//
//   2. Collect all conduit.get_memref_async ops similarly.
//
//   3. If any num_elems value is missing or non-integer: emit a remark and
//      skip this conduit (not an error — dynamic shapes are valid).
//
//   4. Apply three correction cases (see below) before attaching rates.
//
//   5. Attach producer_rates = DenseI64ArrayAttr(put_elems) and
//      consumer_rates = DenseI64ArrayAttr(get_elems) to the conduit.create.
//      M6/M7 now fire automatically at the next verify step.
//
// Rate sequences
// --------------
// The order of put/get ops in program order defines the CSDF phase sequence.
// For uniform channels (all puts move P elements, all gets move C elements),
// producer_rates = [P] and consumer_rates = [C] (period-1 SDF).
// For true CSDF, the full sequence is attached: [p0, p1, ..., p_{q-1}].
//
// Three correction cases (applied before attaching rates)
// -------------------------------------------------------
// Per DIALECT_REDESIGN.md §4:
//
// (a) Sliding-window channels: If any conduit.acquire for this channel has a
//     count attribute greater than any conduit.release count, skip rate
//     attachment for this channel.  Emitting rates would cause false M6
//     rejection (the sliding window holds more tokens than it releases per
//     step, appearing imbalanced to M6).
//
// (b) Time-multiplexed channels (dma_repeat set): If the channel's
//     conduit.create has dma_repeat set (non-zero), skip ALL rate attachment.
//     Pass C infers BD chain length from putCount directly; it does not need
//     producer_rates.  Emitting only producer_rates without consumer_rates
//     would violate M6 on any subsequent verify pass.
//
// Interaction with M6/M7
// ----------------------
// M6 balance check: sum(producer_rates)*len(consumer_rates)
//                     == sum(consumer_rates)*len(producer_rates)
// M7 hyper-period simulation: checks peak token occupancy against capacity.
// Both fire inside Create::verify() as soon as the attributes are present.
//
// Limitations
// -----------
// - Only conduit.put_memref_async / conduit.get_memref_async are inspected
//   for Tier 3 rate inference (produced by Pass B).
// - Tier 2 acquire/release ops (produced by Pass A) are inspected for the
//   sliding-window correction case (a) only.
// - Only the num_elems attribute is used.  Offsets/sizes/strides are ignored.
// - Conduit ops that appear outside any aie.device are not processed.
//   conduit.create without an enclosing aie.device is invalid IR; the pass
//   correctly does nothing for such programs.
// - If zero put ops are found but gets exist (or vice versa), the rates are
//   not attached (asymmetric coverage — emit a remark instead).
//
// This pass runs AFTER Pass A or Pass B and BEFORE Pass C.
// It is OPT-IN and NOT part of the default pipeline.
//
// Run with:  aie-opt --conduit-infer-rates <input.mlir>
//
//===----------------------------------------------------------------------===//

#include "aie/Dialect/Conduit/Transforms/ConduitPasses.h"

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/Conduit/IR/ConduitDialect.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"

namespace xilinx::conduit {

#define GEN_PASS_DEF_CONDUITINFERRATES
#include "aie/Dialect/Conduit/Transforms/ConduitPasses.h.inc"

namespace {

// ---------------------------------------------------------------------------
// Per-device rate inference helper
// ---------------------------------------------------------------------------
//
// Performs all collection and attachment steps within a single scope
// (either an aie.device body or the entire module for device-less tests).
// `scope` is the op whose regions are walked.

static void inferRatesInScope(mlir::Operation *scope, mlir::MLIRContext *ctx) {

  // -----------------------------------------------------------------------
  // Step 1: Collect Tier 3 num_elems sequences per channel name.
  //
  // Walk in program order (top-down, left-to-right within blocks) so the
  // sequence matches the CSDF phase order the programmer intended.
  // -----------------------------------------------------------------------
  llvm::StringMap<llvm::SmallVector<int64_t>> putElems; // name → [p0,p1,...]
  llvm::StringMap<llvm::SmallVector<int64_t>> getElems; // name → [c0,c1,...]
  // Tracks names that have a non-constant num_elems (skip entirely).
  llvm::StringMap<bool> hasDynamicElems;

  scope->walk([&](PutMemrefAsync op) {
    auto nameAttr = op->getAttrOfType<mlir::FlatSymbolRefAttr>("name");
    if (!nameAttr)
      return;
    llvm::StringRef name = nameAttr.getValue();
    if (hasDynamicElems.count(name))
      return; // already flagged dynamic — skip further collection

    auto numElemsAttr = op->getAttrOfType<mlir::IntegerAttr>("num_elems");
    if (!numElemsAttr) {
      hasDynamicElems[name] = true;
      return;
    }
    putElems[name].push_back(numElemsAttr.getInt());
  });

  scope->walk([&](GetMemrefAsync op) {
    auto nameAttr = op->getAttrOfType<mlir::FlatSymbolRefAttr>("name");
    if (!nameAttr)
      return;
    llvm::StringRef name = nameAttr.getValue();
    if (hasDynamicElems.count(name))
      return;

    auto numElemsAttr = op->getAttrOfType<mlir::IntegerAttr>("num_elems");
    if (!numElemsAttr) {
      hasDynamicElems[name] = true;
      return;
    }
    getElems[name].push_back(numElemsAttr.getInt());
  });

  // -----------------------------------------------------------------------
  // Step 2: Collect Tier 2 acquire/release counts per channel name.
  //
  // Used only for the sliding-window correction (case a): if any acquire
  // count exceeds the minimum release count for a channel, that channel
  // uses a sliding window and M6 balance would fire spuriously.
  // -----------------------------------------------------------------------
  llvm::StringMap<int64_t> maxAcquireCount; // name → max count seen
  llvm::StringMap<int64_t> minReleaseCount; // name → min count seen

  scope->walk([&](mlir::Operation *op) {
    if (mlir::isa<Acquire, AcquireAsync>(op)) {
      auto nameAttr = op->getAttrOfType<mlir::FlatSymbolRefAttr>("name");
      auto countAttr = op->getAttrOfType<mlir::IntegerAttr>("count");
      if (!nameAttr || !countAttr)
        return;
      llvm::StringRef name = nameAttr.getValue();
      int64_t count = countAttr.getInt();
      auto it = maxAcquireCount.find(name);
      if (it == maxAcquireCount.end())
        maxAcquireCount[name] = count;
      else
        it->second = std::max(it->second, count);
    }
    if (auto rel = mlir::dyn_cast<Release>(op)) {
      // Release takes a window SSA value; trace to the defining acquire to
      // recover the channel name (Release has no name attr of its own).
      auto countAttr = rel->getAttrOfType<mlir::IntegerAttr>("count");
      if (!countAttr)
        return;
      // Walk the def chain: acquire → (possibly) wait_window → release.
      mlir::Value win = rel.getWindow();
      mlir::FlatSymbolRefAttr nameAttr;
      if (auto acq = win.getDefiningOp<Acquire>())
        nameAttr = acq.getNameAttr();
      else if (auto acq = win.getDefiningOp<AcquireAsync>())
        nameAttr = acq.getNameAttr();
      if (!nameAttr)
        return;
      llvm::StringRef name = nameAttr.getValue();
      int64_t count = countAttr.getInt();
      auto it = minReleaseCount.find(name);
      if (it == minReleaseCount.end())
        minReleaseCount[name] = count;
      else
        it->second = std::min(it->second, count);
    }
    if (auto rel = mlir::dyn_cast<ReleaseAsync>(op)) {
      // ReleaseAsync has a $name attr directly.
      auto nameAttr = rel->getAttrOfType<mlir::FlatSymbolRefAttr>("name");
      auto countAttr = rel->getAttrOfType<mlir::IntegerAttr>("count");
      if (!nameAttr || !countAttr)
        return;
      llvm::StringRef name = nameAttr.getValue();
      int64_t count = countAttr.getInt();
      auto it = minReleaseCount.find(name);
      if (it == minReleaseCount.end())
        minReleaseCount[name] = count;
      else
        it->second = std::min(it->second, count);
    }
  });

  // -----------------------------------------------------------------------
  // Step 3: For each conduit.create with no existing rates, apply correction
  // cases then attach inferred producer_rates and consumer_rates.
  // -----------------------------------------------------------------------
  scope->walk([&](Create op) {
    // Skip if rates are already explicitly set.
    if (op.getProducerRates().has_value() || op.getConsumerRates().has_value())
      return;

    llvm::StringRef name = op.getSymName();
    if (name.empty())
      return;

    // Skip conduits with dynamic num_elems.
    if (hasDynamicElems.count(name)) {
      op->emitRemark("conduit-infer-rates: skipping '")
          << name << "': num_elems is non-constant on at least one put/get op";
      return;
    }

    // -------------------------------------------------------------------
    // Correction case (a): Sliding-window channels.
    //
    // If any conduit.acquire for this channel has count > any release count,
    // the channel uses a sliding window.  Attaching rates would cause M6 to
    // report a false imbalance (acquire moves K tokens, release moves 1).
    // Skip rate attachment entirely for such channels.
    // -------------------------------------------------------------------
    auto acqIt = maxAcquireCount.find(name);
    auto relIt = minReleaseCount.find(name);
    if (acqIt != maxAcquireCount.end() && relIt != minReleaseCount.end()) {
      if (acqIt->second > relIt->second) {
        op->emitRemark("conduit-infer-rates: skipping '")
            << name << "': sliding-window channel (acquire count "
            << acqIt->second << " > release count " << relIt->second << ")";
        return;
      }
    }

    // -------------------------------------------------------------------
    // Correction case (b): Time-multiplexed channels (dma_repeat set).
    //
    // dma_repeat set means the BD chain fires a finite number of times.
    // Pass C infers BD chain length from putCount independently — it does
    // not need producer_rates on dma_repeat channels.  Emitting only
    // producer_rates without consumer_rates would violate M6 ("CSDF
    // requires both; only one was provided") on any subsequent verify
    // pass.  Skip all rate attachment for such channels.
    // -------------------------------------------------------------------
    if (auto dmaRepeatAttr = op.getDmaRepeat()) {
      if (*dmaRepeatAttr > 0) {
        op->emitRemark("conduit-infer-rates: skipping '")
            << name
            << "': dma_repeat set; Pass C infers BD chain length from "
               "putCount independently";
        return;
      }
    }

    // No dma_repeat — normal Tier 3 rate inference.
    auto putIt = putElems.find(name);
    auto getIt = getElems.find(name);
    bool hasPuts = putIt != putElems.end();
    bool hasGets = getIt != getElems.end();

    // Skip if only one side is present (asymmetric coverage).
    if (!hasPuts && !hasGets)
      return; // no memref ops for this conduit — nothing to infer

    if (!hasPuts || !hasGets) {
      op->emitRemark("conduit-infer-rates: skipping '")
          << name << "': found " << (hasPuts ? "put" : "get")
          << " ops but no matching " << (hasPuts ? "get" : "put")
          << " ops; rates not inferred";
      return;
    }

    // Attach rates.  M6/M7 will fire on the next verify step.
    llvm::ArrayRef<int64_t> pRates = putIt->second;
    llvm::ArrayRef<int64_t> cRates = getIt->second;

    op->setAttr("producer_rates", mlir::DenseI64ArrayAttr::get(ctx, pRates));
    op->setAttr("consumer_rates", mlir::DenseI64ArrayAttr::get(ctx, cRates));

    op->emitRemark("conduit-infer-rates: attached producer_rates=[")
        << pRates[0] << (pRates.size() > 1 ? ",..." : "")
        << "] consumer_rates=[" << cRates[0]
        << (cRates.size() > 1 ? ",..." : "") << "] to conduit '" << name << "'";
  });
}

// ---------------------------------------------------------------------------
// Pass
// ---------------------------------------------------------------------------

struct ConduitInferRatesPass
    : public impl::ConduitInferRatesBase<ConduitInferRatesPass> {

  void runOnOperation() override {
    mlir::ModuleOp module = getOperation();
    mlir::MLIRContext *ctx = module.getContext();

    // -----------------------------------------------------------------------
    // Device-scoped walk: one inference pass per aie.device.
    //
    // CSDF rate patterns are meaningful only within the context of a single
    // hardware device.  Running per-device prevents spurious cross-device
    // name collisions when a module contains multiple aie.device ops (e.g.,
    // after --conduit-fuse-operators or in multi-device tests).
    //
    // conduit.create without an enclosing aie.device is invalid IR; the pass
    // does nothing for such programs (correct behavior — no fallback).
    // -----------------------------------------------------------------------
    module.walk([&](xilinx::AIE::DeviceOp deviceOp) {
      inferRatesInScope(deviceOp.getOperation(), ctx);
    });
  }
};

} // namespace

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createConduitInferRatesPass() {
  return std::make_unique<ConduitInferRatesPass>();
}

} // namespace xilinx::conduit
