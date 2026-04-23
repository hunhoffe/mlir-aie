//===- DeviceMergeUtils.cpp - Device-merge host-orchestrator helpers ------===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2026 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//

#include "DeviceMergeUtils.h"

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIEX/IR/AIEXDialect.h"

#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Operation.h"

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"

namespace xilinx::conduit::detail {

// ---------------------------------------------------------------------------
// Device-body merge primitives.  See DeviceMergeUtils.h for design notes.
// ---------------------------------------------------------------------------

mlir::Operation *findRuntimeSequence(mlir::Block &body) {
  for (mlir::Operation &op : body) {
    if (op.getName().getStringRef() == "aie.runtime_sequence")
      return &op;
  }
  return nullptr;
}

llvm::SmallVector<mlir::Operation *>
movePhase1NonSequenceOps(mlir::Block &bodyA, mlir::Block &bodyB) {
  // Set up an OpBuilder positioned just before bodyA's terminator (or at the
  // end of bodyA if no terminator is present).  The original per-pass code
  // used `OpBuilder(ctx)` with the same insertion-point setup; matching it
  // exactly preserves the resulting op order.
  mlir::OpBuilder b(bodyA.getParent()->getContext());
  if (bodyA.mightHaveTerminator()) {
    if (mlir::Operation *term = bodyA.getTerminator())
      b.setInsertionPoint(term);
    else
      b.setInsertionPointToEnd(&bodyA);
  } else {
    b.setInsertionPointToEnd(&bodyA);
  }

  llvm::SmallVector<mlir::Operation *> seqOps;
  llvm::SmallVector<mlir::Operation *> nonSeq;
  for (mlir::Operation &op : bodyB) {
    if (op.hasTrait<mlir::OpTrait::IsTerminator>())
      continue;
    if (op.getName().getStringRef() == "aie.runtime_sequence")
      seqOps.push_back(&op);
    else
      nonSeq.push_back(&op);
  }
  for (mlir::Operation *op : nonSeq) {
    op->remove();
    b.insert(op);
  }
  return seqOps;
}

void moveToEndOfDeviceBody(mlir::Operation *op, mlir::Block &bodyA) {
  // Use moveBefore (a list-splice) for byte-identical behavior with the
  // original `op->remove(); builder.insert(op);` pattern when the builder
  // was positioned before bodyA's terminator.
  if (mlir::Operation *termA =
          bodyA.mightHaveTerminator() ? bodyA.getTerminator() : nullptr)
    op->moveBefore(termA);
  else
    op->moveBefore(&bodyA, bodyA.end());
}

void mergeRuntimeSequencesSimple(mlir::Operation *&seqA,
                                 llvm::ArrayRef<mlir::Operation *> seqOpsB,
                                 mlir::Block &bodyA) {
  for (mlir::Operation *seqB : seqOpsB) {
    if (seqA && seqA->getNumRegions() > 0 && seqB->getNumRegions() > 0) {
      mlir::Block &seqBodyA = seqA->getRegion(0).front();
      mlir::Block &seqBodyB = seqB->getRegion(0).front();

      // Append seqB's block args to seqA, building the IRMapping.
      mlir::IRMapping argMapping;
      for (mlir::BlockArgument arg : seqBodyB.getArguments()) {
        mlir::BlockArgument newArg =
            seqBodyA.addArgument(arg.getType(), arg.getLoc());
        argMapping.map(arg, newArg);
      }

      // Clone seqB's body into seqA before seqA's terminator.
      mlir::OpBuilder seqBuilder(seqA->getContext());
      if (seqBodyA.mightHaveTerminator()) {
        if (mlir::Operation *term = seqBodyA.getTerminator())
          seqBuilder.setInsertionPoint(term);
        else
          seqBuilder.setInsertionPointToEnd(&seqBodyA);
      } else {
        seqBuilder.setInsertionPointToEnd(&seqBodyA);
      }
      for (mlir::Operation &inner : seqBodyB) {
        if (inner.hasTrait<mlir::OpTrait::IsTerminator>())
          continue;
        seqBuilder.clone(inner, argMapping);
      }
    } else if (!seqA) {
      // devA has no sequence yet — promote devB's as-is.
      moveToEndOfDeviceBody(seqB, bodyA);
      seqA = seqB;
    }
  }
}

void sinkCoresMemsAndSequences(mlir::Block &bodyA) {
  llvm::SmallVector<mlir::Operation *> toSink;
  for (mlir::Operation &op : bodyA) {
    llvm::StringRef name = op.getName().getStringRef();
    if (name == "aie.core" || name == "aie.mem" ||
        name == "aie.runtime_sequence")
      toSink.push_back(&op);
  }
  mlir::Operation *termA =
      bodyA.mightHaveTerminator() ? bodyA.getTerminator() : nullptr;
  for (mlir::Operation *op : toSink) {
    if (termA)
      op->moveBefore(termA);
    else
      op->moveBefore(&bodyA, bodyA.end());
  }
}

// ---------------------------------------------------------------------------
// Host-orchestrator rewrite (FS3 fix).
// ---------------------------------------------------------------------------

namespace {

// Find the (typically single) aiex.run op inside a ConfigureOp's body.
// Returns nullptr if none is present.  If multiple are present (atypical for
// IRON-generated host code), the FIRST is returned.
static AIEX::RunOp findRunOpInConfigure(AIEX::ConfigureOp conf) {
  AIEX::RunOp result;
  for (mlir::Operation &op : conf.getBody().front()) {
    if (auto run = mlir::dyn_cast<AIEX::RunOp>(op)) {
      result = run;
      break;
    }
  }
  return result;
}

// Count aiex.run ops in a ConfigureOp body.
static unsigned countRunOpsInConfigure(AIEX::ConfigureOp conf) {
  unsigned n = 0;
  for (mlir::Operation &op : conf.getBody().front()) {
    if (mlir::isa<AIEX::RunOp>(op))
      ++n;
  }
  return n;
}

// Find the closest preceding sibling aiex.configure with the given symbol
// name in the same parent block as `conf`.  Returns null if no such sibling
// exists.
static AIEX::ConfigureOp findSiblingConfigureBefore(AIEX::ConfigureOp conf,
                                                    llvm::StringRef symName) {
  mlir::Block *parent = conf->getBlock();
  if (!parent)
    return {};
  AIEX::ConfigureOp closest;
  for (mlir::Operation &op : *parent) {
    if (&op == conf.getOperation())
      break;
    if (auto sib = mlir::dyn_cast<AIEX::ConfigureOp>(op)) {
      if (sib.getSymbol() == symName)
        closest = sib;
    }
  }
  return closest;
}

// Trim a specific set of block args from `orchestrator` (an outer host-side
// aie.runtime_sequence) — the args identified by the caller as having become
// dead due to THIS merge invocation (i.e. their corresponding aiex.run
// callsite references were dropped in Phase 2).
//
// Symmetric to Step 8c on the merged device-side runtime_sequence.  Erasing
// the merge-introduced dead args shrinks the host-ABI signature so the
// runtime no longer allocates a dead L3 buffer per invocation (Task #99).
//
// IMPORTANT: this trim consumes an EXPLICIT set of indices supplied by the
// caller — it does NOT do post-hoc liveness analysis on the orchestrator
// body.  Reason: an arg that is dead at IR level for reasons other than
// merge (e.g. caller convention reserves a slot — IRON's
// `FusedFullELFCallable` always passes 3 parent buffers (input/output/scratch)
// even when scratch is unused inside the body, raw user-written runtime_seqs
// with their own conventions, MHA's custom orchestrator, etc.) MUST NOT be
// trimmed here.  Doing so silently corrupts the host ABI: the host runtime
// continues to call `set_arg(i, bo)` against a now-nonexistent kernel slot,
// producing all-zero outputs with no exception (the original #99
// regression).
static void trimOrchestratorMergeIntroducedArgs(
    AIE::RuntimeSequenceOp orchestrator,
    const llvm::DenseSet<unsigned> &deadIndices) {
  if (deadIndices.empty())
    return;
  if (orchestrator->getNumRegions() == 0)
    return;
  mlir::Region &region = orchestrator.getBody();
  if (region.empty())
    return;
  mlir::Block &body = region.front();
  unsigned n = body.getNumArguments();
  if (n == 0)
    return;

  // Erase in reverse order so surviving indices stay valid during erasure.
  for (int idx = static_cast<int>(n) - 1; idx >= 0; --idx) {
    if (deadIndices.contains(static_cast<unsigned>(idx)))
      body.eraseArgument(static_cast<unsigned>(idx));
  }
}

// Helper: count the number of times each block-arg of the orchestrator is
// referenced as an operand of any aiex.run callsite NESTED inside the
// orchestrator body.  The aiex.run callsite is the channel by which fusion-
// induced argument death propagates from device-side to host-side; tracking
// just these references (rather than full liveness) is what lets us
// distinguish "dead because the merge made it so" from "dead from the start
// by caller convention".
static llvm::DenseMap<unsigned, unsigned>
collectAiexRunArgUsage(AIE::RuntimeSequenceOp orchestrator) {
  llvm::DenseMap<unsigned, unsigned> usage;
  if (orchestrator->getNumRegions() == 0)
    return usage;
  mlir::Region &region = orchestrator.getBody();
  if (region.empty())
    return usage;
  mlir::Block &body = region.front();
  orchestrator.walk([&](AIEX::RunOp run) {
    for (mlir::Value v : run.getArgs()) {
      if (auto ba = mlir::dyn_cast<mlir::BlockArgument>(v)) {
        if (ba.getOwner() == &body)
          usage[ba.getArgNumber()] += 1;
      }
    }
  });
  return usage;
}

// Find the FIRST following sibling aiex.configure with the given symbol
// name in the same parent block as `conf`.  Returns null if no such sibling
// exists.  Used as a fallback when no preceding sibling exists, so that the
// fold-into-confA collapse fires regardless of confA/confB textual order
// in the host orchestrator's parent block (FS-audit-2.2 fix).
static AIEX::ConfigureOp findSiblingConfigureAfter(AIEX::ConfigureOp conf,
                                                   llvm::StringRef symName) {
  mlir::Block *parent = conf->getBlock();
  if (!parent)
    return {};
  bool seenConf = false;
  for (mlir::Operation &op : *parent) {
    if (&op == conf.getOperation()) {
      seenConf = true;
      continue;
    }
    if (!seenConf)
      continue;
    if (auto sib = mlir::dyn_cast<AIEX::ConfigureOp>(op)) {
      if (sib.getSymbol() == symName)
        return sib;
    }
  }
  return {};
}

} // namespace

mlir::LogicalResult rewriteHostConfigureOnDeviceMerge(mlir::ModuleOp module,
                                                      AIE::DeviceOp devA,
                                                      AIE::DeviceOp devB,
                                                      mlir::Operation *seqA) {
  mlir::MLIRContext *ctx = module.getContext();

  llvm::StringRef devAName = devA.getSymName();
  llvm::StringRef devBName = devB.getSymName();
  if (devAName == devBName) {
    // Same symbol → same device; nothing to rewrite.
    return mlir::success();
  }

  // The surviving runtime_sequence's sym_name (if any).  After the body
  // merge, devB's sequence has either been spliced into seqA (seqA's name
  // preserved) or, if devA had no sequence pre-merge, seqB was promoted
  // (its name preserved).  Either way `seqA` here is the surviving op.
  llvm::StringRef survivingSeqName;
  if (auto rs = mlir::dyn_cast_or_null<AIE::RuntimeSequenceOp>(seqA))
    survivingSeqName = rs.getSymName();

  mlir::FlatSymbolRefAttr devARef = mlir::FlatSymbolRefAttr::get(ctx, devAName);
  mlir::FlatSymbolRefAttr survSeqRef;
  if (!survivingSeqName.empty())
    survSeqRef = mlir::FlatSymbolRefAttr::get(ctx, survivingSeqName);

  // Collect every aiex.configure in the module whose symbol matches devB.
  // Snapshot first; we'll mutate during the loop.
  llvm::SmallVector<AIEX::ConfigureOp> bRefs;
  module.walk([&](AIEX::ConfigureOp conf) {
    if (conf.getSymbol() == devBName)
      bRefs.push_back(conf);
  });

  if (bRefs.empty())
    return mlir::success();

  for (AIEX::ConfigureOp confB : bRefs) {
    // Look for a sibling confA in the same parent block.  Prefer a BEFORE
    // sibling (preserves byte-identical behavior with prior versions); fall
    // back to an AFTER sibling so the fold fires regardless of textual
    // configure order in the host orchestrator (FS-audit-2.2 fix).  If both
    // exist, BEFORE wins.
    AIEX::ConfigureOp confA = findSiblingConfigureBefore(confB, devAName);
    bool siblingIsAfter = false;
    if (!confA) {
      confA = findSiblingConfigureAfter(confB, devAName);
      siblingIsAfter = (bool)confA;
    }
    AIEX::RunOp runB = findRunOpInConfigure(confB);

    if (confA) {
      // --- Fold confB's body into confA (collapse two LoadPDI → one). ---
      AIEX::RunOp runA = findRunOpInConfigure(confA);

      // Refuse the fold if either side has multiple RunOps — IRON emits
      // exactly one per configure; multiple is a sign that the structure
      // doesn't match our spec, and the safe action is to leave the host
      // code alone and surface the issue to the caller.
      if (countRunOpsInConfigure(confA) > 1 ||
          countRunOpsInConfigure(confB) > 1) {
        confB.emitError(
            "device-merge: cannot fold aiex.configure with multiple aiex.run "
            "ops; expected at most one aiex.run per configure (IRON "
            "convention)");
        return mlir::failure();
      }

      mlir::Block &bodyA = confA.getBody().front();
      mlir::Block &bodyB = confB.getBody().front();

      // Move all non-RunOp ops in confB into bodyA.
      //
      //   - BEFORE case (confA precedes confB textually): insert before
      //     confA's RunOp (or push_back if confA has no run).  This places
      //     confB's setup ops AFTER confA's existing setup ops but still
      //     before the merged RunOp — preserving the original
      //     A-setup-then-B-setup-then-run order.
      //   - AFTER  case (confA follows confB textually): insert at the FRONT
      //     of bodyA so confB's setup ops appear before any of confA's body
      //     ops — preserving the original B-setup-then-A-setup ordering as
      //     observed in the host orchestrator before the collapse.  When
      //     bodyA is empty, push_back is equivalent.
      //
      // NOTE: do NOT call op->remove() before moveBefore(): moveBefore is
      // implemented as a list-splice and asserts the op is currently in a
      // block (Operation.cpp:558).  remove() detaches it (block becomes null),
      // which trips the assertion.  moveBefore handles the detach+reinsert
      // atomically.  For the push_back fallback we DO need to detach first
      // because Block::push_back expects a free-standing op.
      mlir::Operation *insertBefore;
      if (!siblingIsAfter) {
        insertBefore = runA ? runA.getOperation() : nullptr;
      } else {
        insertBefore = bodyA.empty() ? nullptr : &bodyA.front();
      }
      llvm::SmallVector<mlir::Operation *> toMove;
      for (mlir::Operation &op : bodyB) {
        if (mlir::isa<AIEX::RunOp>(op))
          continue;
        toMove.push_back(&op);
      }
      for (mlir::Operation *op : toMove) {
        if (insertBefore) {
          op->moveBefore(insertBefore);
        } else {
          op->remove();
          bodyA.push_back(op);
        }
      }

      // Build the folded RunOp's argument vector as the NAIVE concatenation
      // `runA.getArgs() ++ runB.getArgs()`.  This deliberately matches the
      // current (pre-Step-8c) sequence arity, which is the post-Phase-2 sum
      // `origArgCountA + origArgCountB`.
      //
      // Callers (e.g. --conduit-fuse-operators) that subsequently TRIM the
      // sequence's block args MUST follow up with
      // `reconcileHostRunArgsAfterTrim` to project the same drops into this
      // run's arg vector.  See DeviceMergeUtils.h for the split-phase
      // contract.
      if (runA && runB) {
        llvm::SmallVector<mlir::Value> newOperands;
        newOperands.reserve(runA.getArgs().size() + runB.getArgs().size());
        for (mlir::Value v : runA.getArgs())
          newOperands.push_back(v);
        for (mlir::Value v : runB.getArgs())
          newOperands.push_back(v);

        mlir::FlatSymbolRefAttr seqRef =
            survSeqRef ? survSeqRef : runA.getRuntimeSequenceSymbolAttr();

        mlir::OpBuilder builder(runA);
        auto newRun =
            builder.create<AIEX::RunOp>(runA.getLoc(), seqRef, newOperands);
        runA.erase();
        (void)newRun;
      } else if (runB && !runA) {
        // confA had no run (atypical) — move runB into confA.  insertBefore
        // is null here (it's runA), so we detach + push_back.
        runB->remove();
        bodyA.push_back(runB);
        if (survSeqRef)
          runB.setRuntimeSequenceSymbolAttr(survSeqRef);
      }
      // (runA && !runB): nothing to concatenate; leave runA alone.

      confB.erase();
    } else {
      // --- No sibling confA: rewrite confB in place. ---
      confB.setSymbolAttr(devARef);
      if (runB && survSeqRef)
        runB.setRuntimeSequenceSymbolAttr(survSeqRef);
    }
  }

  // NOTE: arity validation is deliberately deferred.  At this point, callers
  // that trim the merged sequence's block args (e.g. Step 8c in
  // --conduit-fuse-operators) have NOT yet run, so the freshly-folded
  // `aiex.run` carries the pre-trim concat arity that intentionally exceeds
  // the post-trim callee.  Callers must invoke `reconcileHostRunArgsAfterTrim`
  // after their trim to project the drops into the run-op arg vectors and
  // validate the final arity.  Callers that DO NOT trim (--aie-combine-device,
  // --conduit-fuse-core-bodies) need no follow-up: at this point the run's
  // arity already matches the merged sequence.

  return mlir::success();
}

mlir::LogicalResult
reconcileHostRunArgsAfterTrim(mlir::ModuleOp module, AIE::DeviceOp devA,
                              mlir::Operation *seqA, unsigned origArgCountA,
                              unsigned origArgCountB,
                              const llvm::DenseSet<unsigned> &deadA,
                              const llvm::DenseSet<unsigned> &deadB) {
  auto rs = mlir::dyn_cast_or_null<AIE::RuntimeSequenceOp>(seqA);
  if (!rs)
    return mlir::success();

  llvm::StringRef devAName = devA.getSymName();
  llvm::StringRef survivingSeqName = rs.getSymName();
  unsigned expected = rs.getBody().front().getNumArguments();

  // Snapshot run ops to mutate; we replace them in place.
  llvm::SmallVector<AIEX::RunOp> runs;
  module.walk([&](AIEX::RunOp run) {
    auto parentConf = run->getParentOfType<AIEX::ConfigureOp>();
    if (!parentConf)
      return;
    if (parentConf.getSymbol() != devAName)
      return;
    if (run.getRuntimeSequenceSymbol() != survivingSeqName)
      return;
    runs.push_back(run);
  });

  // Snapshot the orchestrator (outer aie.runtime_sequence) for each touched
  // run — these are the rtSeq ops whose host-ABI block-arg signatures may
  // have stale dead args after the run-callsite trim below.  Collect them
  // BEFORE the loop because each `run.erase()` drops the parent-of-type
  // anchor we'd need afterwards.
  llvm::DenseSet<mlir::Operation *> orchestrators;
  for (AIEX::RunOp run : runs) {
    if (auto rtSeq = run->getParentOfType<AIE::RuntimeSequenceOp>())
      orchestrators.insert(rtSeq.getOperation());
  }

  // Phase 3 prep: snapshot the per-orchestrator aiex.run arg-usage map BEFORE
  // the run-callsite rewrite below.  Phase 3 needs to compare BEFORE vs AFTER
  // to identify which orchestrator-arg slots became orphaned strictly because
  // of THIS merge (i.e. all aiex.run callsites that referenced them were
  // rewritten to drop that position).  An arg dead-from-start by caller
  // convention (e.g. IRON FusedFullELFCallable's reserved 3 parent-buffer
  // slots, raw user-written runtime_seqs with their own conventions, MHA's
  // custom orchestrator) MUST NOT be trimmed: doing so silently corrupts the
  // host ABI (the runtime keeps calling set_arg(i, bo) against a now-
  // nonexistent kernel slot, producing all-zero outputs with no exception
  // — the original Task #99 regression).
  llvm::DenseMap<mlir::Operation *, llvm::DenseMap<unsigned, unsigned>>
      prevAiexRunUsage;
  for (mlir::Operation *op : orchestrators) {
    auto rtSeq = mlir::dyn_cast<AIE::RuntimeSequenceOp>(op);
    if (!rtSeq)
      continue;
    prevAiexRunUsage[op] = collectAiexRunArgUsage(rtSeq);
  }

  bool failed = false;
  for (AIEX::RunOp run : runs) {
    unsigned argCount = run.getArgs().size();

    // Two layouts are valid at this point:
    //   - Folded:           argCount == origArgCountA + origArgCountB
    //                       (segment A then segment B, both projected).
    //   - Rewritten-in-place: argCount == origArgCountB
    //                       (only segment B; symbol was rewritten by
    //                       rewriteHostConfigureOnDeviceMerge but no fold).
    bool isFolded = (argCount == origArgCountA + origArgCountB);
    bool isInPlace = (argCount == origArgCountB && !isFolded);

    if (!isFolded && !isInPlace) {
      run->emitError("device-merge Phase 2: aiex.run arg count ")
          << argCount << " is neither the folded total ("
          << (origArgCountA + origArgCountB)
          << ") nor the rewritten-in-place B-only count (" << origArgCountB
          << "); cannot reconcile after Step 8c trim";
      failed = true;
      continue;
    }

    llvm::SmallVector<mlir::Value> newOperands;
    newOperands.reserve(argCount);

    if (isFolded) {
      // Segment A: indices [0, origArgCountA) — drop positions in deadA.
      for (unsigned i = 0; i < origArgCountA; ++i) {
        if (!deadA.contains(i))
          newOperands.push_back(run.getArgs()[i]);
      }
      // Segment B: indices [origArgCountA, origArgCountA + origArgCountB) —
      // drop positions in deadB (re-based to 0).
      for (unsigned i = 0; i < origArgCountB; ++i) {
        if (!deadB.contains(i))
          newOperands.push_back(run.getArgs()[origArgCountA + i]);
      }
    } else { // isInPlace
      for (unsigned i = 0; i < origArgCountB; ++i) {
        if (!deadB.contains(i))
          newOperands.push_back(run.getArgs()[i]);
      }
    }

    if (newOperands.size() != expected) {
      run->emitError("device-merge Phase 2: reconciled aiex.run arg count ")
          << newOperands.size()
          << " disagrees with merged aie.runtime_sequence @" << survivingSeqName
          << " arity " << expected;
      failed = true;
      continue;
    }

    mlir::OpBuilder builder(run);
    auto newRun = builder.create<AIEX::RunOp>(
        run.getLoc(), run.getRuntimeSequenceSymbolAttr(), newOperands);
    run.erase();
    (void)newRun;
  }

  // Phase 3: trim merge-introduced dead host-orchestrator block args.  After
  // the run-callsite rewrite above, fused-internal-channel intermediates
  // that the host previously staged through L3 have lost all aiex.run
  // references inside the orchestrator body.  Erasing those slots shrinks
  // the host ABI so the runtime no longer allocates a dead L3 buffer per
  // invocation (Task #99).
  //
  // Provenance rule (vs the original liveness-based trim that broke IRON
  // FusedFullELFCallable's host ABI in #122): a slot qualifies as
  // "merge-introduced dead" iff it was referenced by at least one aiex.run
  // callsite BEFORE the rewrite above and by ZERO aiex.run callsites AFTER.
  // Args dead-from-start by caller convention (reserved ABI slots, custom
  // orchestrator patterns) had prevUsage==0 and are therefore never erased
  // by this trim.  Only run on success — if any run failed validation we
  // leave the IR alone for the diagnostic to surface.
  if (!failed) {
    for (mlir::Operation *op : orchestrators) {
      auto rs = mlir::dyn_cast<AIE::RuntimeSequenceOp>(op);
      if (!rs)
        continue;
      auto postIt = prevAiexRunUsage.find(op);
      if (postIt == prevAiexRunUsage.end())
        continue; // defensive: should not happen given orchestrators-set seed
      const llvm::DenseMap<unsigned, unsigned> &prevUsage = postIt->second;
      llvm::DenseMap<unsigned, unsigned> postUsage = collectAiexRunArgUsage(rs);
      llvm::DenseSet<unsigned> mergeIntroducedDead;
      for (auto &kv : prevUsage) {
        unsigned argIdx = kv.first;
        unsigned prevCount = kv.second;
        unsigned postCount = 0;
        auto pit = postUsage.find(argIdx);
        if (pit != postUsage.end())
          postCount = pit->second;
        if (prevCount > 0 && postCount == 0)
          mergeIntroducedDead.insert(argIdx);
      }
      trimOrchestratorMergeIntroducedArgs(rs, mergeIntroducedDead);
    }
  }

  return failed ? mlir::failure() : mlir::success();
}

} // namespace xilinx::conduit::detail
