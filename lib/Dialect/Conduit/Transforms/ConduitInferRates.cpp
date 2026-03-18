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
// conduit.put_memref_async and conduit.get_memref_async num_elems attributes.
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
// This pass fills that gap: it walks the IR, collects num_elems sequences for
// each channel, and attaches producer_rates / consumer_rates to the matching
// conduit.create.  On the next mlir verify step, M6/M7 fire automatically.
//
// Algorithm
// ---------
// For each conduit.create with no existing producer_rates / consumer_rates:
//
//   1. Collect all conduit.put_memref_async ops in the module that have
//      name == conduit name.  Extract their num_elems attribute values in
//      program order (walk order = source order for determinism).
//
//   2. Collect all conduit.get_memref_async ops similarly.
//
//   3. If any num_elems value is missing or non-integer: emit a remark and
//      skip this conduit (not an error — dynamic shapes are valid).
//
//   4. Attach producer_rates = DenseI64ArrayAttr(put_elems) and
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
//   (Tier 3 ops; produced by Pass B).  Tier 2 acquire/release ops (produced by
//   Pass A) do not carry num_elems; use access_pattern for those.
// - Only the num_elems attribute is used.  Offsets/sizes/strides are ignored.
// - Puts and gets with the same channel name but in different functions are
//   collected together.  This is correct for the SPSC static programs Pass B
//   currently handles (single function per channel endpoint).
// - If zero put ops are found but gets exist (or vice versa), the rates are
//   not attached (asymmetric coverage — emit a remark instead).
//
// This pass runs AFTER Pass B and BEFORE Pass C.
// It is OPT-IN and NOT part of the default pipeline.
//
// Run with:  aie-opt --conduit-infer-rates <input.mlir>
//
//===----------------------------------------------------------------------===//

#include "aie/Dialect/Conduit/Transforms/ConduitPasses.h"

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

struct ConduitInferRatesPass
    : public impl::ConduitInferRatesBase<ConduitInferRatesPass> {

  void runOnOperation() override {
    mlir::ModuleOp module = getOperation();
    mlir::MLIRContext *ctx = module.getContext();

    // -----------------------------------------------------------------------
    // Step 1: Collect num_elems sequences for each channel name.
    //
    // Walk in program order (module walk = top-down, left-to-right within
    // blocks) so the sequence matches the CSDF phase order the programmer
    // intended.
    // -----------------------------------------------------------------------
    llvm::StringMap<llvm::SmallVector<int64_t>> putElems;  // name → [p0,p1,...]
    llvm::StringMap<llvm::SmallVector<int64_t>> getElems;  // name → [c0,c1,...]
    // Tracks names that have a non-constant num_elems (skip entirely).
    llvm::StringMap<bool> hasDynamicElems;

    module.walk([&](PutMemrefAsync op) {
      auto nameAttr = op->getAttrOfType<mlir::StringAttr>("name");
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

    module.walk([&](GetMemrefAsync op) {
      auto nameAttr = op->getAttrOfType<mlir::StringAttr>("name");
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
    // Step 2: For each conduit.create with no existing rates, attach inferred
    // producer_rates and consumer_rates from the collected sequences.
    // -----------------------------------------------------------------------
    module.walk([&](Create op) {
      // Skip if rates are already explicitly set.
      if (op.getProducerRates().has_value() ||
          op.getConsumerRates().has_value())
        return;

      auto nameAttr = op->getAttrOfType<mlir::StringAttr>("name");
      if (!nameAttr)
        return;
      llvm::StringRef name = nameAttr.getValue();

      // Skip conduits with dynamic num_elems.
      if (hasDynamicElems.count(name)) {
        op->emitRemark("conduit-infer-rates: skipping '")
            << name
            << "': num_elems is non-constant on at least one put/get op";
        return;
      }

      auto putIt = putElems.find(name);
      auto getIt = getElems.find(name);
      bool hasPuts = (putIt != putElems.end() && !putIt->second.empty());
      bool hasGets = (getIt != getElems.end() && !getIt->second.empty());

      // Skip if only one side is present (asymmetric coverage).
      if (!hasPuts && !hasGets)
        return; // no memref ops for this conduit — nothing to infer

      if (!hasPuts || !hasGets) {
        op->emitRemark("conduit-infer-rates: skipping '")
            << name << "': found "
            << (hasPuts ? "put" : "get")
            << " ops but no matching "
            << (hasPuts ? "get" : "put")
            << " ops; rates not inferred";
        return;
      }

      // Attach rates.  M6/M7 will fire on the next verify step.
      llvm::ArrayRef<int64_t> pRates = putIt->second;
      llvm::ArrayRef<int64_t> cRates = getIt->second;

      op->setAttr("producer_rates",
                  mlir::DenseI64ArrayAttr::get(ctx, pRates));
      op->setAttr("consumer_rates",
                  mlir::DenseI64ArrayAttr::get(ctx, cRates));

      op->emitRemark("conduit-infer-rates: attached producer_rates=[")
          << pRates[0]
          << (pRates.size() > 1 ? ",..." : "")
          << "] consumer_rates=["
          << cRates[0]
          << (cRates.size() > 1 ? ",..." : "")
          << "] to conduit '"
          << name << "'";
    });
  }
};

} // namespace

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createConduitInferRatesPass() {
  return std::make_unique<ConduitInferRatesPass>();
}

} // namespace xilinx::conduit
