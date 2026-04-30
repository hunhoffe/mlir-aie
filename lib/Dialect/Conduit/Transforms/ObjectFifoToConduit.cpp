//===- ObjectFifoToConduit.cpp - ObjectFIFO → Conduit IR (Pass A) -*-C++-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2026 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//
//
// Pass A of the Conduit lowering pipeline: lift aie.objectfifo.* ops into
// Conduit IR.
//
// Architecture:
//
//   aie.objectfifo.*  ──┐
//                       ├──► Conduit IR ──► aie.dma_bd / aie.lock / aie.buffer
//   air.channel.*     ──┘
//
//   (this file)              (ConduitToDMA.cpp)
//
// What this pass does
// -------------------
// 1. Scans the aie.device body for aie.objectfifo ops and builds a name→tiles
//    map (producer tile, consumer tiles, element type, depth).
//
// 2. For each aie.objectfifo:
//      emits  conduit.create {name, element_type=<memref type>, depth=<depth>,
//                             routing_mode=<circuit|cascade|stream|absent>,
//                             sync_mode=<none|absent>}
//      (typed attributes; no conduit.annotate ops are emitted)
//
// 3. For each aie.objectfifo.link:
//      determines mode (distribute: 1 src, N dsts; join: N srcs, 1 dst)
//      picks memtile as the relay tile (heuristic: consumer of first src)
//      emits conduit.scatter or conduit.gather as appropriate
//
// 4. For each aie.objectfifo.acquire (inside core bodies):
//      emits conduit.acquire {name, count, port="Produce"|"Consume"}
//              : !conduit.window<elemType>
//      The window SSA value is threaded into each conduit.subview_access.
//
// 5. For each aie.objectfifo.subview.access:
//      emits conduit.subview_access %window {index}
//              : !conduit.window<T> -> T
//      Uses of the aie subview result are replaced with the conduit result.
//      No memref.alloc placeholder is generated.
//
// 6. For each aie.objectfifo.release:
//      emits conduit.release {name, count, port="Produce"|"Consume"}
//
// All original ObjectFIFO ops are erased after rewriting.
//
// Limitations (documented honestly)
// ----------------------------------
// - The memtile heuristic in link rewriting is approximate.
// - producer_dimensions/consumer_dimensions from the source objectfifo are
//   propagated to the forceCircuit check but not emitted on conduit.create.
// - The pass processes each aie.device independently with fresh maps,
//   preventing name collisions across devices in multi-device modules.
//
//===----------------------------------------------------------------------===//

#include "aie/Dialect/Conduit/Transforms/ConduitPasses.h"

#include "LoopAnalysisUtils.h"
#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIEX/IR/AIEXDialect.h"
#include "aie/Dialect/Conduit/IR/ConduitDialect.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

#include <string>

namespace xilinx::conduit {

#define GEN_PASS_DEF_OBJECTFIFOTOCONDUIT
#include "aie/Dialect/Conduit/Transforms/ConduitPasses.h.inc"

namespace {

// ---------------------------------------------------------------------------
// Helper: count static elements in a MemRefType (returns 1 if unknown)
// ---------------------------------------------------------------------------

static int64_t numElemsInMemref(mlir::Type ty) {
  auto mref = mlir::dyn_cast<mlir::MemRefType>(ty);
  if (!mref)
    return 1;
  int64_t count = 1;
  for (int64_t d : mref.getShape()) {
    if (mlir::ShapedType::isDynamic(d))
      return 1;
    count *= d;
  }
  return count;
}

// ---------------------------------------------------------------------------
// Struct: information gathered from aie.objectfifo ops
// ---------------------------------------------------------------------------

struct FifoInfo {
  llvm::SmallVector<int64_t> producerTileArr;  // [col, row]
  llvm::SmallVector<int64_t> consumerTilesArr; // non-shim: [col0,row0,...]
  llvm::SmallVector<int64_t>
      shimConsumerTilesArr; // shim (row==0): [col0,row0,...]
  int64_t depth = 1;
  int64_t numElems = 1;
  mlir::MemRefType elemType; // the actual element memref type
  // Cyclostatic (CSDF) access pattern.
  // Populated in Phase 1.5 by scanning acquire counts for each consumer of this
  // fifo.  If all acquires use the same count the pattern is absent (uniform
  // SDF). If acquires vary, this holds the sequence of counts in program order.
  llvm::SmallVector<int64_t> accessPattern;
  // CSDF rates inferred from Phase 1.5 acquire/release scans
  // (infer-rates=true). Only populated for single-consumer fifos.  For
  // multi-consumer fifos, rate annotation is skipped (per-consumer rates
  // differ; no single merged rate sequence is correct for the conduit.create's
  // M6 check).
  llvm::SmallVector<int64_t> inferredProducerRates;
  llvm::SmallVector<int64_t> inferredConsumerRates;
  // MVE-2: set when Phase 1.5 detected a sliding-window pattern
  // (max acquire count > min release count on the Consume port).  Phase 2
  // uses this to emit a remark and skip rate annotation instead of silently
  // leaving rates empty.
  bool slidingWindowSkip = false;
};

// ---------------------------------------------------------------------------
// dma_repeat inference (auto-fill when objectfifo carries no iter_count).
//
// Background:
//   IRON's `set_iter_count(N)` translates to objectfifo.iter_count = N, which
//   Phase 2 propagates to conduit.create's `dma_repeat`.  When the harness
//   does not call set_iter_count, the host runtime sequence emits ONE BD per
//   shim channel — but the producer/consumer cores' outer loops execute N
//   times.  Without dma_repeat, the DMA fires once and the cores stall
//   waiting for further data (the all-zero output Add → Mul fusion bug).
//
//   This helper walks the producer + consumer cores' acquire ops for a given
//   channel, computes the static trip count of each side's enclosing loop
//   nest, then divides by both the runtime-emission count and the BD's
//   acquires-per-fire ratio.  Three-factor formula:
//
//     dma_repeat = (total_core_acquires / rt_emissions_per_channel)
//                  / acquires_per_BD
//
//     total_core_acquires      = product of all enclosing scf.for /
//     scf.parallel
//                                trip counts at the acquire site
//     rt_emissions_per_channel = count of aiex.dma_configure_task_for ops in
//                                the runtime_sequence whose `alloc` symbol
//                                matches this channel
//     acquires_per_BD          = max(1, bd_len / fifo_elem_count)
//                                (proxy for product(TAP sizes) / product(elem
//                                shape) under linear/contiguous TAPs)
//
//   The middle factor matters for IRON design fns that emit multiple
//   rt.fill/rt.drain per channel via Python looping in `rt.sequence` — e.g.,
//   gemv with `num_batches > 1`.  For single-emission channels (most ops in
//   Llama Row #1) it reduces to total/per_BD as before.
//
//   The numerator is the *product* of trip counts at the acquire site, so
//   patterns that flatten loops (e.g., softmax's `range_(N * M)`) are handled
//   naturally — the single loop's trip count IS the total acquire count.
//
// Skip cases:
//   - silent (no remark):
//       * no core-side acquires at all
//       * derived dma_repeat <= 1
//       * trip exceeds kTripCountUnboundedSentinel (legacy `cmax = i64::MAX`)
//   - remark on the conduit.create (so the user can see why we backed off):
//       * dynamic loop bounds anywhere in the enclosing nest
//       * producer trip != consumer trip
//       * multiple aiex.dma_configure_task_for ops match with differing BD lens
//       * trip not divisible by rt_emissions_per_channel
//       * per-emission acquires not divisible by acquires_per_BD
//       * shim channel with exactly one aiex.dma_configure_task_for emission
//         (host-side num_invocations is invisible to Pass A; see emit.count==1
//         block in inferDmaRepeatForChannel below — Bug C falsification,
//         2026-04-24)
// ---------------------------------------------------------------------------

// Anything at or above this threshold is treated as the legacy infinite-loop
// sentinel (`cmax = i64::MAX` followed by `step = 1`) and skipped silently
// to preserve the behaviour of the existing while-true test corpus.
static constexpr int64_t kTripCountUnboundedSentinel = int64_t{1} << 30;

// Match either arith.constant_index or arith.constant of integer type.
static std::optional<int64_t> getConstIndexOrInt(mlir::Value v) {
  llvm::APInt val;
  if (mlir::matchPattern(v, mlir::m_ConstantInt(&val)))
    return val.getSExtValue();
  return std::nullopt;
}

enum class TripStatus { Static, Dynamic, NotALoop };

struct TripResult {
  TripStatus status = TripStatus::NotALoop;
  int64_t trip = 1;
};

// ---------------------------------------------------------------------------
// Pattern D — RTP-driven `dma_repeat` constant fold
// ---------------------------------------------------------------------------
//
// IRON's prefill GEMM (and softmax) drives outer-loop trip counts from a
// runtime-parameter buffer rather than from a literal scf.for upper bound:
//
//     %rtp = aie.buffer ... {sym_name = "rtp0_0", use_write_rtp = true}
//                     : memref<2xi32>
//     ...
//     aie.core(...) {
//       aie.use_lock(%barrier, Acquire, 1)
//       %i = arith.constant 1 : index
//       %v = memref.load %rtp[%i] : memref<2xi32>
//       %ub = arith.index_cast %v : i32 to index
//       scf.for %k = %c0 to %ub step %c1 { ...acquire @chan; release... }
//     }
//
//     aie.runtime_sequence() {
//       aiex.npu.rtp_write(@rtp0_0, 1, 4)
//       aiex.set_lock(%barrier, 1)
//     }
//
// The host writes a compile-time integer literal into the RTP slot before
// releasing the worker barrier — by the time the core's scf.for evaluates
// its upper bound, the value is fixed.  `aiex.npu.rtp_write` carries this
// value as an I32Attr directly, so Pass A can fold the load by walking the
// runtime_sequence once and building a `(buffer_sym, index) -> value` map.
//
// The map keys on the RTP buffer's sym_name + index (both compile-time
// constants).  Conflicting writes to the same key (e.g. two
// `aiex.npu.rtp_write(@rtp, 0, X)` ops with different X values) mark the
// entry ambiguous; subsequent lookups for that key fail and the bound
// resolves to Dynamic with a remark explaining why.
//
// Lookups are gated to avoid false positives:
//   * memref.load operand chain must be `arith.index_cast %load` where
//     `%load = memref.load %rtp_buf[%idx_const]` (the canonical IRON
//     shape).
//   * `%rtp_buf` must trace to an `aie.buffer` with a sym_name attribute.
//   * `%idx_const` must be a literal `arith.constant` of index/int type.
//   * For scf.for, the lb is restricted to constant 0 and the step to
//     constant 1 (matching IRON's `range_(N)` lowering); anything else
//     falls through to Dynamic.
// ---------------------------------------------------------------------------

struct RtpConstantInfo {
  int64_t value = 0;
  bool ambiguous = false; // multiple writes with conflicting values
};

using RtpConstantMap =
    llvm::DenseMap<std::pair<mlir::StringAttr, uint32_t>, RtpConstantInfo>;

// Walk every `aiex.npu.rtp_write` directly nested in the device's
// `aie.runtime_sequence` regions and record `((buffer_sym, index) -> value)`.
// Writes inside nested control flow (scf.for / scf.if in the runtime
// sequence) are conservatively ignored — IRON does not emit dynamic RTP
// writes.  Conflicting-value writes for the same key flip the entry to
// `ambiguous`, after which lookups for that key always miss.
static RtpConstantMap collectRtpConstants(AIE::DeviceOp device) {
  RtpConstantMap map;
  device.walk([&](AIE::RuntimeSequenceOp rt) {
    if (rt.getBody().empty())
      return;
    for (mlir::Operation &raw : rt.getBody().front()) {
      auto rtpOp = mlir::dyn_cast<AIEX::NpuWriteRTPOp>(&raw);
      if (!rtpOp)
        continue;
      auto bufRef = rtpOp.getBufferAttr();
      if (!bufRef)
        continue;
      mlir::StringAttr key =
          mlir::StringAttr::get(rtpOp.getContext(), bufRef.getValue());
      uint32_t idx = rtpOp.getIndex();
      // I32Attr stored as ui32 by the generated accessor; reinterpret as
      // signed to preserve the user-visible value.
      int64_t val =
          static_cast<int64_t>(static_cast<int32_t>(rtpOp.getValue()));
      auto it = map.find({key, idx});
      if (it == map.end()) {
        map[{key, idx}] = RtpConstantInfo{val, /*ambiguous=*/false};
      } else if (!it->second.ambiguous && it->second.value != val) {
        it->second.ambiguous = true;
      }
    }
  });
  return map;
}

// Try to resolve an scf.for/scf.parallel upper-bound SSA value as an RTP
// constant.  Returns the folded integer when the chain matches the IRON
// shape `arith.index_cast (memref.load %rtp_buf[%const_idx])` AND the
// (buffer_sym, index) key is in `rtpMap` AND the entry is not ambiguous.
static std::optional<int64_t> tryFoldRtpBound(mlir::Value bound,
                                              const RtpConstantMap &rtpMap) {
  mlir::Value cur = bound;
  if (auto castOp = cur.getDefiningOp<mlir::arith::IndexCastOp>())
    cur = castOp.getIn();
  auto loadOp = cur.getDefiningOp<mlir::memref::LoadOp>();
  if (!loadOp)
    return std::nullopt;
  if (loadOp.getIndices().size() != 1)
    return std::nullopt;
  auto bufOp = loadOp.getMemRef().getDefiningOp<AIE::BufferOp>();
  if (!bufOp)
    return std::nullopt;
  mlir::StringAttr symAttr = bufOp.getSymNameAttr();
  if (!symAttr)
    return std::nullopt;
  auto idxOpt = getConstIndexOrInt(loadOp.getIndices().front());
  if (!idxOpt || *idxOpt < 0 ||
      *idxOpt > static_cast<int64_t>(std::numeric_limits<uint32_t>::max()))
    return std::nullopt;
  auto it = rtpMap.find({symAttr, static_cast<uint32_t>(*idxOpt)});
  if (it == rtpMap.end())
    return std::nullopt;
  if (it->second.ambiguous)
    return std::nullopt;
  return it->second.value;
}

// Static trip count of an scf.for / scf.parallel.  Anything else returns
// NotALoop (caller treats as "skip this op while walking").
static TripResult tripCountOfLoop(mlir::Operation *op,
                                  const RtpConstantMap &rtpMap) {
  TripResult res;
  if (auto forOp = mlir::dyn_cast<mlir::scf::ForOp>(op)) {
    auto tc = forOp.getStaticTripCount();
    if (tc) {
      res.status = TripStatus::Static;
      res.trip = tc->getSExtValue();
      return res;
    }
    // Pattern D: try RTP-constant fold on the upper bound.  Restricted to
    // the canonical IRON shape: lb == 0, step == 1, ub from
    // `arith.index_cast (memref.load %rtp_buf[%const])` where the buffer +
    // index pair was written with a constant value in the runtime_sequence.
    auto lbConst = getConstIndexOrInt(forOp.getLowerBound());
    auto stepConst = getConstIndexOrInt(forOp.getStep());
    if (lbConst && *lbConst == 0 && stepConst && *stepConst == 1) {
      if (auto folded = tryFoldRtpBound(forOp.getUpperBound(), rtpMap)) {
        if (*folded < 0)
          *folded = 0;
        res.status = TripStatus::Static;
        res.trip = *folded;
        return res;
      }
    }
    res.status = TripStatus::Dynamic;
    return res;
  }
  if (auto parOp = mlir::dyn_cast<mlir::scf::ParallelOp>(op)) {
    auto lbs = parOp.getLowerBound();
    auto ubs = parOp.getUpperBound();
    auto sts = parOp.getStep();
    int64_t total = 1;
    for (size_t i = 0; i < lbs.size(); ++i) {
      auto lb = getConstIndexOrInt(lbs[i]);
      auto ub = getConstIndexOrInt(ubs[i]);
      auto st = getConstIndexOrInt(sts[i]);
      // Pattern D fold for parallel-dim bounds: same restrictions
      // (lb==0, step==1) and same RTP map lookup.  Falls through to the
      // Dynamic exit if the fold misses on this dim.
      if (lb && ub && st && *st != 0) {
        // Fully-constant dim — proceed as before.
      } else if (lb && *lb == 0 && st && *st == 1 && !ub) {
        if (auto folded = tryFoldRtpBound(ubs[i], rtpMap)) {
          int64_t span = *folded;
          if (span <= 0) {
            res.status = TripStatus::Static;
            res.trip = 0;
            return res;
          }
          total *= span;
          continue;
        }
        res.status = TripStatus::Dynamic;
        return res;
      } else {
        res.status = TripStatus::Dynamic;
        return res;
      }
      int64_t span = *ub - *lb;
      if (span <= 0) {
        res.status = TripStatus::Static;
        res.trip = 0;
        return res;
      }
      int64_t trip = (span + *st - 1) / *st;
      total *= trip;
    }
    res.status = TripStatus::Static;
    res.trip = total;
    return res;
  }
  return res;
}

// Attempt to constant-fold an scf.if condition into a boolean.  Returns the
// folded value when provably constant, std::nullopt for dynamic conditions.
//
//   * Direct `arith.constant` of i1 (or any integer that matches m_ConstantInt)
//     folds trivially — covers the canonical `arith.constant true` guard.
//   * `arith.cmpi <pred>, %a, %b` where both operands resolve to integer
//     constants — either as literal `arith.constant` or via the RTP-fold map
//     (`tryFoldRtpBound`) — is evaluated using
//     `xilinx::conduit::evaluateConstantsInMap` over the affine expression
//     `s0 - s1`, then the predicate sign is applied.  This catches the
//     IRON-style `if rtp_n_tiles_per_core > 1` shape once the RTP slot value
//     is observable.
//   * Anything else (e.g., `arith.cmpi` whose operands depend on a loop IV
//     such as `arith.remui %i, %c2`) returns std::nullopt → caller treats
//     the enclosing scf.if as introducing dynamic per-iteration acquire
//     counts and skips dma_repeat inference with a remark.
static std::optional<bool> foldIfCondition(mlir::Value cond,
                                           const RtpConstantMap &rtpMap) {
  llvm::APInt lit;
  if (mlir::matchPattern(cond, mlir::m_ConstantInt(&lit)))
    return lit.getBoolValue();

  auto cmp = cond.getDefiningOp<mlir::arith::CmpIOp>();
  if (!cmp)
    return std::nullopt;

  auto resolve = [&](mlir::Value v) -> std::optional<int64_t> {
    if (auto c = getConstIndexOrInt(v))
      return c;
    return tryFoldRtpBound(v, rtpMap);
  };
  auto lhs = resolve(cmp.getLhs());
  auto rhs = resolve(cmp.getRhs());
  if (!lhs || !rhs)
    return std::nullopt;

  mlir::MLIRContext *ctx = cmp.getContext();
  auto map = mlir::AffineMap::get(/*dimCount=*/0, /*symbolCount=*/2,
                                  mlir::getAffineSymbolExpr(0, ctx) -
                                      mlir::getAffineSymbolExpr(1, ctx),
                                  ctx);
  llvm::SmallVector<std::optional<int64_t>, 2> inputs{lhs, rhs};
  auto folded = xilinx::conduit::evaluateConstantsInMap(map, inputs, ctx);
  if (!folded)
    return std::nullopt;
  int64_t diff = *folded;
  switch (cmp.getPredicate()) {
  case mlir::arith::CmpIPredicate::eq:
    return diff == 0;
  case mlir::arith::CmpIPredicate::ne:
    return diff != 0;
  case mlir::arith::CmpIPredicate::slt:
  case mlir::arith::CmpIPredicate::ult:
    return diff < 0;
  case mlir::arith::CmpIPredicate::sle:
  case mlir::arith::CmpIPredicate::ule:
    return diff <= 0;
  case mlir::arith::CmpIPredicate::sgt:
  case mlir::arith::CmpIPredicate::ugt:
    return diff > 0;
  case mlir::arith::CmpIPredicate::sge:
  case mlir::arith::CmpIPredicate::uge:
    return diff >= 0;
  }
  return std::nullopt;
}

// Walk parent chain from `op` (exclusive) upward, stopping at `boundary`
// (exclusive).  Multiplies the static trip counts of every enclosing
// scf.for / scf.parallel.  Dynamic anywhere → status = Dynamic.  No loops
// in chain → status = Static, trip = 1.
//
// scf.if handling (Task #38, fixes the over-count bug pinned by
// `infer_iter_count_scf_if_conditional_acquire_overcount.mlir`): when
// stepping through an `scf::IfOp`, attempt to constant-fold its condition.
//   * Folded true  AND descending child sits in then-region → continue
//     walking (always-taken; preserves the `arith.constant true` shape
//     pinned by `infer_iter_count_inside_if_branch.mlir`).
//   * Folded false AND descending child sits in else-region → continue
//     walking (symmetric always-taken).
//   * Otherwise (unfoldable condition OR child sits in the not-taken
//     branch) → return Dynamic.  The caller emits a remark and skips
//     dma_repeat stamping rather than over-counting acquires through a
//     conditional that may not fire on every loop iteration.
static TripResult productOfEnclosingLoops(mlir::Operation *op,
                                          mlir::Operation *boundary,
                                          const RtpConstantMap &rtpMap) {
  TripResult acc;
  acc.status = TripStatus::Static;
  acc.trip = 1;
  // `prev` is always the most recent `cur` we stepped past — i.e., a direct
  // child of the new `cur`.  When `cur` is an scf::IfOp, `prev` therefore
  // sits directly in one of its regions, so `prev->getParentRegion()`
  // identifies the then- vs else-region cleanly.
  mlir::Operation *prev = op;
  mlir::Operation *cur = op->getParentOp();
  while (cur && cur != boundary) {
    if (auto ifOp = mlir::dyn_cast<mlir::scf::IfOp>(cur)) {
      auto folded = foldIfCondition(ifOp.getCondition(), rtpMap);
      if (!folded) {
        acc.status = TripStatus::Dynamic;
        return acc;
      }
      mlir::Region *prevRegion = prev->getParentRegion();
      bool inThen = (prevRegion == &ifOp.getThenRegion());
      bool inElse = !ifOp.getElseRegion().empty() &&
                    (prevRegion == &ifOp.getElseRegion());
      bool taken = (*folded && inThen) || (!*folded && inElse);
      if (!taken) {
        // Provably not taken: the acquire is unreachable.  Stamping
        // `dma_repeat = 0` would be wrong for a real workload, so signal
        // Dynamic and let the caller back off with a remark.
        acc.status = TripStatus::Dynamic;
        return acc;
      }
      // Always-taken: fall through and continue walking past the if.
    } else {
      TripResult t = tripCountOfLoop(cur, rtpMap);
      if (t.status == TripStatus::Dynamic) {
        acc.status = TripStatus::Dynamic;
        return acc;
      }
      if (t.status == TripStatus::Static)
        acc.trip *= t.trip;
    }
    prev = cur;
    cur = cur->getParentOp();
  }
  return acc;
}

// Walk every `aiex.dma_configure_task_for` op in `device` whose `alloc`
// FlatSymbolRefAttr matches `channelName` and report:
//   * count   — number of matching ops (rt_emissions_per_channel)
//   * bdLen   — the embedded aie.dma_bd's `len`, unified across all matches
//   * ambiguous — set when at least two matches carry differing lens; caller
//                 treats that as a skip-with-remark case
struct ChannelEmissionInfo {
  int64_t count = 0;
  std::optional<int64_t> bdLen;
  bool ambiguous = false;
};

static ChannelEmissionInfo
inspectChannelEmissions(AIE::DeviceOp device, llvm::StringRef channelName) {
  ChannelEmissionInfo info;
  device.walk([&](mlir::Operation *op) {
    if (op->getName().getStringRef() != "aiex.dma_configure_task_for")
      return;
    mlir::FlatSymbolRefAttr allocAttr;
    if (auto inh = op->getInherentAttr("alloc"))
      allocAttr = mlir::dyn_cast<mlir::FlatSymbolRefAttr>(*inh);
    if (!allocAttr)
      allocAttr = op->getAttrOfType<mlir::FlatSymbolRefAttr>("alloc");
    if (!allocAttr || allocAttr.getValue() != channelName)
      return;
    info.count++;
    AIE::DMABDOp bd;
    op->walk([&](AIE::DMABDOp b) { bd = b; });
    if (!bd)
      return;
    auto lenOpt = bd.getLen();
    if (!lenOpt)
      return;
    int64_t len = static_cast<int64_t>(*lenOpt);
    if (info.bdLen.has_value() && *info.bdLen != len) {
      info.ambiguous = true;
      return;
    }
    info.bdLen = len;
  });
  return info;
}

struct DmaRepeatInference {
  std::optional<int64_t> dmaRepeat;
  // Empty == silent skip; non-empty == emit a remark on the conduit.create.
  std::string reason;
};

// Per-port acquire bucket: all acquires for one channel within one core.
struct PortAcquireBucket {
  AIE::CoreOp core;
  llvm::SmallVector<mlir::Operation *> acquires;
};

static DmaRepeatInference
inferDmaRepeatForChannel(AIE::DeviceOp device, llvm::StringRef channelName,
                         int64_t fifoElemCount, const RtpConstantMap &rtpMap) {
  DmaRepeatInference out;

  llvm::SmallVector<PortAcquireBucket, 2> prodSides;
  llvm::SmallVector<PortAcquireBucket, 2> consSides;

  device.walk([&](AIE::ObjectFifoAcquireOp acq) {
    if (acq.getObjFifoName() != channelName)
      return;
    auto core = acq->getParentOfType<AIE::CoreOp>();
    if (!core)
      return;
    auto &bucket =
        (acq.getPort() == AIE::ObjectFifoPort::Produce) ? prodSides : consSides;
    PortAcquireBucket *match = nullptr;
    for (auto &p : bucket)
      if (p.core == core) {
        match = &p;
        break;
      }
    if (!match) {
      bucket.push_back({core, {}});
      match = &bucket.back();
    }
    match->acquires.push_back(acq.getOperation());
  });

  if (prodSides.empty() && consSides.empty())
    return out; // shim-on-shim or unused — silent skip.

  // For each side, compute the per-core trip count = product of enclosing
  // loops above each acquire (taken from the first acquire in the core).
  // The simple "first-acquire enclosing loops" is sufficient for the target
  // pattern (one acquire per innermost loop iter); preamble + body cores
  // would need true total-acquire counting and are out of scope here.
  enum class SideStatus { Empty, Static, Dynamic, Mismatch };
  auto sideTrip = [&](llvm::ArrayRef<PortAcquireBucket> sides,
                      int64_t &outTrip) -> SideStatus {
    if (sides.empty())
      return SideStatus::Empty;
    std::optional<int64_t> agreed;
    for (const auto &p : sides) {
      if (p.acquires.empty())
        continue;
      AIE::CoreOp coreCopy = p.core;
      TripResult tr = productOfEnclosingLoops(p.acquires.front(),
                                              coreCopy.getOperation(), rtpMap);
      if (tr.status == TripStatus::Dynamic)
        return SideStatus::Dynamic;
      int64_t trip = tr.trip;
      if (agreed && *agreed != trip)
        return SideStatus::Mismatch;
      agreed = trip;
    }
    if (!agreed)
      return SideStatus::Empty;
    outTrip = *agreed;
    return SideStatus::Static;
  };

  int64_t prodTrip = 0, consTrip = 0;
  SideStatus prodSt = sideTrip(prodSides, prodTrip);
  SideStatus consSt = sideTrip(consSides, consTrip);

  if (prodSt == SideStatus::Dynamic || consSt == SideStatus::Dynamic) {
    out.reason = "dynamic loop bounds in producer or consumer core";
    return out;
  }
  if (prodSt == SideStatus::Mismatch || consSt == SideStatus::Mismatch) {
    out.reason =
        "multiple cores on the same port have differing loop trip counts";
    return out;
  }

  std::optional<int64_t> trip;
  if (prodSt == SideStatus::Static && consSt == SideStatus::Static) {
    if (prodTrip != consTrip) {
      llvm::raw_string_ostream os(out.reason);
      os << "producer trip " << prodTrip << " differs from consumer trip "
         << consTrip;
      return out;
    }
    trip = prodTrip;
  } else if (prodSt == SideStatus::Static) {
    trip = prodTrip;
  } else if (consSt == SideStatus::Static) {
    trip = consTrip;
  }

  if (!trip)
    return out;

  // Sentinel for legacy infinite-loop test patterns — silent skip.
  if (*trip >= kTripCountUnboundedSentinel)
    return out;

  if (*trip <= 1)
    return out; // dma_repeat=1 is the implicit default; do not annotate.

  // Three-factor inference:
  //   dma_repeat = (total_core_acquires / rt_emissions_per_channel)
  //                / acquires_per_BD
  //
  // rt_emissions defaults to 1 when there is no shim BD (compute-to-compute
  // fifo), so the formula degrades cleanly to the simple total/per_BD case.
  ChannelEmissionInfo emit = inspectChannelEmissions(device, channelName);
  if (emit.ambiguous) {
    out.reason =
        "multiple aiex.dma_configure_task_for ops match channel with differing "
        "BD lengths — ambiguous TAP";
    return out;
  }

  // SHIM CHANNEL skip (bug_c falsification 2026-04-24, EXTENDED 2026-04-28
  // Task #40).
  //
  // When there is at least one `aiex.dma_configure_task_for` for this channel
  // in the runtime_sequence (i.e. shim-bearing channel), Pass A cannot
  // disambiguate between three IR-equivalent runtime shapes:
  //   (A) 1 host dispatch + N BD-replays per dispatch  → dma_repeat = N
  //   (B) K host dispatches + (N/K) BD-replays each    → dma_repeat = N/K
  //   (C) N host dispatches + 1 BD fire each, BD covers `acquiresPerBD`
  //       acquires per fire                              → dma_repeat = 1
  // IRON's `num_invocations = N` (host-side `run()` loop) lowers to (C) but
  // is INVISIBLE to the IR Pass A sees — the runtime_sequence only encodes
  // BD defs without any host fan-out marker.  Stamping
  // `dma_repeat = (trip / per_BD)` was a speculative guess that picked
  // (A)/(B) and over-fired the shim BD on the actual (C) hardware path
  // (Bug C: NPU stalls after the first ELF dispatch's worth of work).
  // Stateful lowering emits NO `repeat_count` for the same shape and Llama
  // works at 6.808 TPS via the host-loop path.
  //
  // 2026-04-28 (Task #40): the same host-num_invocations invisibility
  // argument applies when emit.count > 1.  IRON's gemv pattern with
  // num_batches > 1 unrolls `for batch in range(N)` into N distinct
  // rt.fill calls per channel (visible: emit.count = N), but the host
  // still calls run() num_invocations times (invisible).  The three-factor
  // formula uses `total_core_acquires / emit.count` which under-divides by
  // num_invocations, producing dma_repeat = num_invocations and over-firing
  // the shim BD by exactly that factor.  Llama op7_GEMV (gemv_attn_scores,
  // num_batches=32, num_invocations=16) and op11_GEMV (gemv_attn_context,
  // same) hit this and emit `repeat_count = 16` on every per-batch shim
  // configure; upstream-dynamic-stateful emits no repeat_count for the
  // same input.  See .claude/plans/repeat-count-overfire-rootcause.md.
  //
  // Compute-to-compute fifos (emit.count == 0, no shim BD anywhere) keep
  // the legacy "outer loop drives dma_repeat directly" behavior — that
  // path is sound because there is no host dispatch loop in play.
  if (emit.count >= 1) {
    out.reason = "host-side num_invocations not observable in IR "
                 "(shim-bearing channel); deferring dma_repeat to runtime";
    return out;
  }

  int64_t acquiresPerBD = 1;
  if (emit.bdLen && fifoElemCount > 0) {
    int64_t ratio = *emit.bdLen / fifoElemCount;
    if (ratio > 1)
      acquiresPerBD = ratio;
  }

  // Treat 0 emissions (compute-to-compute, no shim BD at all) as 1 — the
  // outer loop drives `dma_repeat` directly with no host fan-out.
  int64_t emissions = emit.count > 0 ? emit.count : 1;

  if (*trip % emissions != 0) {
    llvm::raw_string_ostream os(out.reason);
    os << "trip " << *trip << " not divisible by rt_emissions " << emissions;
    return out;
  }
  int64_t perEmission = *trip / emissions;

  if (perEmission % acquiresPerBD != 0) {
    llvm::raw_string_ostream os(out.reason);
    os << "per-emission acquires " << perEmission
       << " not divisible by acquires-per-BD " << acquiresPerBD;
    return out;
  }

  int64_t result = perEmission / acquiresPerBD;
  if (result <= 1)
    return out;

  out.dmaRepeat = result;
  return out;
}

// ---------------------------------------------------------------------------
// Main pass
// ---------------------------------------------------------------------------

struct ObjectFifoToConduitPass
    : impl::ObjectFifoToConduitBase<ObjectFifoToConduitPass> {

  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry.insert<ConduitDialect>();
    // AIE dialect needed: Pass A emits aie.put_cascade / aie.get_cascade
    // directly for cascade objectfifos (routing_mode="cascade").
    registry.insert<xilinx::AIE::AIEDialect>();
  }

  // -----------------------------------------------------------------------
  // Shared state across phases
  // -----------------------------------------------------------------------

  /// Set to true when an unrecoverable error is detected.  Checked at
  /// phase boundaries in runOnOperation() to prevent subsequent phases
  /// from executing on corrupted state.  (CL-2: same anti-pattern as
  /// the P0-A bug in ConduitToDMA.cpp — signalPassFailure() inside a
  /// walk lambda does NOT stop the pass; it only marks the result as
  /// failed after all phases complete.)
  bool passFailed = false;

  /// Name → fifo metadata, populated by collectFifoInfo().
  llvm::DenseMap<mlir::StringAttr, FifoInfo> fifoInfoMap;

  /// Names of objectfifos that use aie_stream routing.
  /// Mapped to the Core stream port index (from aie_stream_port attribute).
  llvm::DenseMap<mlir::StringAttr, int32_t> aieStreamFifoPort;

  /// Pattern D — RTP-constant map for the current device, populated by
  /// collectRtpConstants() at the top of transformFifos().  Walked by
  /// inferDmaRepeatForChannel → productOfEnclosingLoops → tripCountOfLoop
  /// to fold scf.for upper bounds whose value is loaded from an RTP buffer
  /// the runtime_sequence wrote with a literal value.
  RtpConstantMap rtpConstantMap;

  /// ObjectFifo create ops to erase after all rewrites complete.
  llvm::SmallVector<AIE::ObjectFifoCreateOp> fifosToErase;

  /// Subview, acquire, and release ops to erase after Phase 4.
  llvm::SmallVector<AIE::ObjectFifoSubviewAccessOp> subviewsToErase;
  llvm::SmallVector<AIE::ObjectFifoAcquireOp> acquiresToErase;
  llvm::SmallVector<AIE::ObjectFifoReleaseOp> releasesToErase;

  // -----------------------------------------------------------------------
  // Phase 1: collectFifoInfo
  // -----------------------------------------------------------------------
  //
  // Walk the module to build fifoInfoMap (name → tile/depth/type info) and
  // detect cyclostatic (CSDF) access patterns from acquire op counts.

  void collectFifoInfo(AIE::DeviceOp device, mlir::MLIRContext *ctx) {
    fifoInfoMap.clear();
    aieStreamFifoPort.clear();

    // Phase 1: collect FifoInfo for all aie.objectfifo ops.
    device.walk([&](AIE::ObjectFifoCreateOp op) {
      // aie_stream ObjectFIFOs route data through the Core AXI stream port
      // rather than DMA. Record the stream port for conduit.create emission;
      // Pass C uses routing_mode="stream" to emit aie.flow(Core:N, ...).
      if (op.getAieStream().has_value()) {
        int32_t streamPort = 0;
        if (op.getAieStreamPort().has_value())
          streamPort = static_cast<int32_t>(op.getAieStreamPort().value());
        aieStreamFifoPort[op.getSymNameAttr()] = streamPort;
        // Fall through to collect FifoInfo normally.
      }

      FifoInfo info;
      // Producer tile
      auto prodTile =
          mlir::cast<AIE::TileOp>(op.getProducerTile().getDefiningOp());
      info.producerTileArr = {prodTile.getCol(), prodTile.getRow()};
      // Consumer tiles: separate shim tiles (row==0) from compute tiles.
      // Shim tiles are DMA endpoints with no local memory; Pass C handles
      // them via aie.shim_dma_allocation rather than aie.buffer + aie.lock.
      for (mlir::Value cons : op.getConsumerTiles()) {
        auto consTile = mlir::cast<AIE::TileOp>(cons.getDefiningOp());
        int64_t col = consTile.getCol();
        int64_t row = consTile.getRow();
        if (row == 0) {
          // Shim tile (row==0): no local memory, handled separately.
          info.shimConsumerTilesArr.push_back(col);
          info.shimConsumerTilesArr.push_back(row);
        } else {
          info.consumerTilesArr.push_back(col);
          info.consumerTilesArr.push_back(row);
        }
      }
      // Depth — producer depth is always index 0.
      info.depth = op.size(0);

      // Element type — must be a MemRefType for window semantics.
      auto objfifoTy = mlir::cast<AIE::AIEObjectFifoType>(op.getElemType());
      mlir::Type elemTy = objfifoTy.getElementType();
      if (auto mrefTy = mlir::dyn_cast<mlir::MemRefType>(elemTy)) {
        info.elemType = mrefTy;
        info.numElems = numElemsInMemref(mrefTy);
      } else {
        // Fallback: treat as single-element i32 memref
        info.elemType =
            mlir::MemRefType::get({1}, mlir::IntegerType::get(ctx, 32));
        info.numElems = 1;
      }

      fifoInfoMap[op.getSymNameAttr()] = std::move(info);
    });

    // Phase 1.5: detect cyclostatic (CSDF) access patterns and (when
    // infer-rates=true) infer CSDF producer_rates/consumer_rates.
    //
    // Bug fix: previously consumeAcquireCounts accumulated counts from ALL
    // consumer cores for each fifo name, merging them in DFS walk order.
    // For multi-consumer fifos (core A: always acquire 1, core B: always
    // acquire 2), this produced [1, 2] — a spurious CSDF pattern.
    //
    // Fix: track per-(fifoName, CoreOp) using a pair key.  Each core's
    // sequence is collected independently.  The merged accessPattern is
    // derived only from a single consumer core (the first one found), and
    // only when it is the sole consumer.
    //
    // For infer-rates=true:
    //   - Single-consumer fifos: derive producer_rates from Produce-port
    //     release counts and consumer_rates from the single consumer core's
    //     Consume-port acquire counts.
    //   - Multi-consumer fifos: skip with a remark.

    // Per-(fifoName, CoreOp): ordered list of (Consume) acquire counts in
    // program order within that core.
    using CoreKey = std::pair<mlir::StringAttr, mlir::Operation *>;
    llvm::DenseMap<CoreKey, llvm::SmallVector<int64_t>> perCoreConsumeCounts;
    // Per-(fifoName, CoreOp): ordered list of (Produce) release counts.
    llvm::DenseMap<CoreKey, llvm::SmallVector<int64_t>> perCoreProduceCounts;
    // Number of distinct consumer cores per fifo name.
    llvm::DenseMap<mlir::StringAttr, llvm::SmallVector<mlir::Operation *>>
        fifoConsumerCores;
    // Number of distinct producer cores per fifo name.
    llvm::DenseMap<mlir::StringAttr, mlir::Operation *> fifoProducerCore;

    device.walk([&](AIE::ObjectFifoAcquireOp op) {
      auto nameAttr = mlir::StringAttr::get(ctx, op.getObjFifoName().str());
      // Find the enclosing aie.core op.
      mlir::Operation *coreOp = op->getParentOp();
      while (coreOp && !mlir::isa<AIE::CoreOp>(coreOp))
        coreOp = coreOp->getParentOp();
      if (!coreOp)
        return;
      if (op.getPort() == AIE::ObjectFifoPort::Consume) {
        CoreKey key = {nameAttr, coreOp};
        perCoreConsumeCounts[key].push_back(op.acqNumber());
        // Track distinct consumer cores.
        auto &consumerList = fifoConsumerCores[nameAttr];
        if (llvm::find(consumerList, coreOp) == consumerList.end())
          consumerList.push_back(coreOp);
      }
    });

    // Collect Produce-port release counts per (fifoName, CoreOp).
    // Also collect Consume-port release counts for sliding-window detection.
    // Sliding-window: consumer acquire count > consumer release count per step.
    llvm::DenseMap<CoreKey, llvm::SmallVector<int64_t>> perCoreConsumeRelCounts;
    device.walk([&](AIE::ObjectFifoReleaseOp op) {
      auto nameAttr = mlir::StringAttr::get(ctx, op.getObjFifoName().str());
      mlir::Operation *coreOp = op->getParentOp();
      while (coreOp && !mlir::isa<AIE::CoreOp>(coreOp))
        coreOp = coreOp->getParentOp();
      if (!coreOp)
        return;
      CoreKey key = {nameAttr, coreOp};
      if (op.getPort() == AIE::ObjectFifoPort::Produce) {
        perCoreProduceCounts[key].push_back(static_cast<int64_t>(op.getSize()));
        fifoProducerCore[nameAttr] = coreOp;
      } else if (op.getPort() == AIE::ObjectFifoPort::Consume) {
        perCoreConsumeRelCounts[key].push_back(
            static_cast<int64_t>(op.getSize()));
      }
    });

    // For each fifo, derive accessPattern from the single consumer core's
    // acquire sequence (only when there is exactly one consumer core).
    // For multi-consumer fifos, do NOT merge across cores.
    for (auto &[nameAttr, consumerCores] : fifoConsumerCores) {
      auto fifoIt = fifoInfoMap.find(nameAttr);
      if (fifoIt == fifoInfoMap.end())
        continue;
      FifoInfo &info = fifoIt->second;

      if (consumerCores.size() == 1) {
        // Single consumer core: use its acquire sequence as the access pattern.
        CoreKey key = {nameAttr, consumerCores[0]};
        auto countIt = perCoreConsumeCounts.find(key);
        if (countIt == perCoreConsumeCounts.end())
          continue;
        const auto &counts = countIt->second;
        if (counts.empty())
          continue;
        bool uniform =
            llvm::all_of(counts, [&](int64_t c) { return c == counts[0]; });
        if (!uniform)
          info.accessPattern = counts;

        // infer-rates: for single-consumer fifos, derive producer/consumer
        // rates from the release/acquire sequences.
        if (inferRates) {
          // MVE-2: sliding-window guard.
          // A sliding-window fifo has acquire_count > release_count per step
          // (consumer keeps K elements in the window, releases M < K).
          // M6 checks CSDF balance on net token rates; attaching rates derived
          // from acquire counts would report a false balance violation because
          // the acquire count is not the net rate — the release count is.
          // Guard: if max(acquireCounts) > min(Consume-side releaseCounts),
          // this is a sliding-window — skip rate annotation and emit a remark.
          CoreKey consKey = {nameAttr, consumerCores[0]};
          auto relIt = perCoreConsumeRelCounts.find(consKey);
          bool isSlidingWindow = false;
          if (relIt != perCoreConsumeRelCounts.end() &&
              !relIt->second.empty()) {
            int64_t maxAcq = *llvm::max_element(counts);
            int64_t minRel = *llvm::min_element(relIt->second);
            if (maxAcq > minRel)
              isSlidingWindow = true;
          }
          if (isSlidingWindow) {
            // Leave inferredProducerRates/inferredConsumerRates empty.
            // Mark for remark emission in Phase 2.
            info.slidingWindowSkip = true;
          } else {
            // Consumer rates = the acquire count sequence for this core.
            info.inferredConsumerRates.assign(counts.begin(), counts.end());

            // Producer rates = the release count sequence from the producer
            // core.
            auto prodIt = fifoProducerCore.find(nameAttr);
            if (prodIt != fifoProducerCore.end()) {
              CoreKey prodKey = {nameAttr, prodIt->second};
              auto prodCountIt = perCoreProduceCounts.find(prodKey);
              if (prodCountIt != perCoreProduceCounts.end() &&
                  !prodCountIt->second.empty()) {
                info.inferredProducerRates.assign(prodCountIt->second.begin(),
                                                  prodCountIt->second.end());
              }
            }
          }
        }
      } else {
        // Multi-consumer fifo: do NOT merge acquire counts from multiple cores.
        // Each core may have a different (but internally uniform or CSDF)
        // pattern; merging them into one sequence produces a spurious CSDF
        // pattern that does not correspond to any single actor's rate.
        if (inferRates) {
          // Emit a remark on the conduit.create location (not available here;
          // the remark is emitted in Phase 2 where the op is built).
          // Just leave inferredProducerRates/inferredConsumerRates empty.
          // Phase 2 will detect the empty rates and emit the remark.
        }
        // accessPattern: keep empty (no merged multi-consumer pattern).
      }
    }

    // P2-D: erase aie.objectfifo.register_process ops.
    // register_process is a code-generation macro that the dedicated pre-pass
    // (--aie-register-objectFifos) expands into standard acquire/release loops
    // before the stateful transform runs. When that pre-pass has already run,
    // these ops are already gone. When it has not run, silently erasing them
    // is safe: the op has zero presence in the stateful-transform corpus (all
    // 130 files in test/objectFifo-stateful-transform/ use raw acquire/release,
    // not register_process) and the stateful transform itself ignores them.
    // Users who need register_process expansion must run
    // --aie-register-objectFifos before --objectfifo-to-conduit.
    llvm::SmallVector<AIE::ObjectFifoRegisterProcessOp> regProcOps;
    device.walk(
        [&](AIE::ObjectFifoRegisterProcessOp op) { regProcOps.push_back(op); });
    for (auto op : regProcOps)
      op.erase();
  }

  // -----------------------------------------------------------------------
  // Phase 2–4: transformFifos
  // -----------------------------------------------------------------------
  //
  // Emit conduit.create / conduit.scatter / conduit.gather / conduit.acquire /
  // conduit.subview_access / conduit.release, replacing all ObjectFIFO ops.
  // Original ops are collected in erasure vectors for later cleanup.

  void transformFifos(AIE::DeviceOp device, mlir::OpBuilder &builder,
                      mlir::MLIRContext *ctx) {
    fifosToErase.clear();
    subviewsToErase.clear();
    acquiresToErase.clear();
    releasesToErase.clear();

    // Pattern D: build the device-scoped (rtp_buf_sym, index) → constant
    // map once, before walking conduit.create candidates.  Consulted by
    // inferDmaRepeatForChannel via tripCountOfLoop to fold scf.for upper
    // bounds whose value is loaded from an RTP buffer the runtime_sequence
    // wrote with a literal value (IRON gemm/softmax shape).
    rtpConstantMap = collectRtpConstants(device);

    // Phase 2: rewrite each aie.objectfifo → conduit.create with typed attrs.
    // NOTE: do NOT erase the objectfifo op here — the AIE verifier requires
    // aie.objectfifo.acquire to reference a live objectfifo symbol.  Collect
    // for deferred erasure after Phase 4 completes.

    device.walk([&](AIE::ObjectFifoCreateOp op) {
      builder.setInsertionPoint(op);
      mlir::Location loc = op.getLoc();

      auto &info = fifoInfoMap[op.getSymNameAttr()];
      std::string name = op.getSymName().str();

      // conduit.create with typed attributes — no conduit.annotate ops.

      // infer-rates: attach CSDF producer_rates/consumer_rates when inferred.
      // For multi-consumer fifos, inferred rates are empty — emit a remark.
      mlir::DenseI64ArrayAttr inferredPRAttr;
      mlir::DenseI64ArrayAttr inferredCRAttr;
      if (inferRates) {
        // Count the number of non-shim consumer tiles to detect multi-consumer.
        int64_t numConsumers =
            static_cast<int64_t>(info.consumerTilesArr.size() / 2 +
                                 info.shimConsumerTilesArr.size() / 2);
        if (numConsumers > 1) {
          // Multi-consumer fifo: skip rate annotation with a remark.
          op.emitRemark("conduit-objectfifo: skipping CSDF rate annotation for "
                        "multi-consumer fifo '")
              << name
              << "' — per-consumer acquire sequences differ; use explicit "
                 "annotations or --conduit-infer-rates for Pass B programs";
        } else if (info.slidingWindowSkip) {
          // MVE-2: sliding-window fifo: skip rate annotation with a remark.
          // Attaching rates from acquire counts would give M6 an unbalanced
          // rate (acquire > release per step), causing a false rejection.
          op.emitRemark("conduit-objectfifo: skipping CSDF rate annotation for "
                        "sliding-window fifo '")
              << name
              << "' (acquire_count > release_count); use explicit "
                 "producer_rates/consumer_rates with window_size for M7 check";
        } else if (!info.inferredProducerRates.empty() &&
                   !info.inferredConsumerRates.empty()) {
          inferredPRAttr =
              mlir::DenseI64ArrayAttr::get(ctx, info.inferredProducerRates);
          inferredCRAttr =
              mlir::DenseI64ArrayAttr::get(ctx, info.inferredConsumerRates);
        }
      }

      // Extract repeat_count from the source objectfifo, if present.
      // Propagated unconditionally (any value including >1) so Pass C can set
      // effectiveBDs = depth * bd_repeat accordingly.
      mlir::IntegerAttr repeatCountAttr;
      if (op.getRepeatCount().has_value()) {
        repeatCountAttr = mlir::IntegerAttr::get(
            mlir::IntegerType::get(ctx, 64),
            static_cast<int64_t>(op.getRepeatCount().value()));
      }

      // Propagate disable_synchronization → sync_mode = None.
      SyncModeAttr disableSyncModeAttr;
      if (op.getDisableSynchronization())
        disableSyncModeAttr = SyncModeAttr::get(ctx, SyncMode::None);

      // Propagate dma_repeat (from objectfifo iter_count).
      mlir::IntegerAttr iterCountAttr;
      if (op.getIterCount().has_value()) {
        iterCountAttr = mlir::IntegerAttr::get(
            mlir::IntegerType::get(ctx, 64),
            static_cast<int64_t>(op.getIterCount().value()));
      }

      // dma_repeat inference fallback: when the source aie.objectfifo carries
      // no explicit iter_count, infer it from the producer/consumer cores'
      // outer-loop trip counts.  See the helper's doc-comment for the formula
      // and skip cases.  Without this, harnesses that omit set_iter_count emit
      // a single host BD per channel while the cores loop N times — the cores
      // stall after one fire (the all-zero-output Add → Mul fusion bug).
      //
      // Streaming-mode hardening (task #29): dma_repeat is the shim/memtile
      // BD-chain replay count; it is meaningful ONLY for routing modes that
      // lower to DMA (Circuit, Packet, DMA, SharedMemory, or absent →
      // heuristic).  Stream and Cascade routing emit no shim DMA BDs, so a
      // stamped dma_repeat is wrong-by-construction (Pass C currently
      // ignores it on Stream channels via skip-branches in
      // ConduitToDMAAlloc.cpp, but we should not stamp the attr in the first
      // place).  Detect Stream/Cascade routing intent on the source
      // aie.objectfifo (explicit `routing_mode = "stream"|"cascade"` attr,
      // explicit `via_cascade=true`, or participation in the aie_stream
      // routing map) and skip inference in those cases.
      bool routingSkipsDma = false;
      if (auto rmStrAttr =
              op->getAttrOfType<mlir::StringAttr>("routing_mode")) {
        llvm::StringRef rmStr = rmStrAttr.getValue();
        if (rmStr == "stream" || rmStr == "cascade")
          routingSkipsDma = true;
      }
      if (!routingSkipsDma && op.getViaCascade())
        routingSkipsDma = true;
      if (!routingSkipsDma && aieStreamFifoPort.find(op.getSymNameAttr()) !=
                                  aieStreamFifoPort.end())
        routingSkipsDma = true;

      if (!iterCountAttr && !routingSkipsDma) {
        auto inferred = inferDmaRepeatForChannel(device, op.getSymName(),
                                                 info.numElems, rtpConstantMap);
        if (inferred.dmaRepeat) {
          iterCountAttr = mlir::IntegerAttr::get(
              mlir::IntegerType::get(ctx, 64), *inferred.dmaRepeat);
        } else if (!inferred.reason.empty()) {
          op.emitRemark("conduit-objectfifo: dma_repeat inference skipped: ")
              << inferred.reason;
        }
      }

      // Propagate dimensionsToStream (producer side).
      // Stored as generic mlir::Attribute to avoid cross-dialect tablegen dep.
      mlir::Attribute prodDimsAttr;
      {
        auto dims = op.getDimensionsToStreamAttr();
        if (dims && !dims.getValue().empty())
          prodDimsAttr = dims;
      }

      // Propagate dimensionsFromStreamPerConsumer (per-consumer).
      // Check that at least one consumer has non-empty dims (the outer array
      // is always non-empty even for fifos with no dims — it has one [] per
      // consumer — so we must check the inner arrays).
      mlir::Attribute consDimsAttr;
      {
        auto dims = op.getDimensionsFromStreamPerConsumerAttr();
        bool hasNonEmptyDims = false;
        if (dims) {
          for (auto consArr : dims.getValue()) {
            if (!consArr.getValue().empty()) {
              hasNonEmptyDims = true;
              break;
            }
          }
        }
        if (hasNonEmptyDims)
          consDimsAttr = dims;
      }

      // Propagate via_DMA → routing_mode = Circuit.
      // Auto-set routing_mode=Circuit when:
      //   (a) dimensionsToStream or dimensionsFromStream are non-empty: the
      //       shared-memory path skips DMA BDs entirely, silently dropping N-D
      //       transforms. Forcing DMA (circuit routing) ensures BDDimLayout
      //       attributes are applied at the hardware level.
      //   (b) bd_repeat > 1: the BD chain is replayed N times by the DMA
      //       engine. Shared-memory has no BD replay mechanism — the hardware
      //       lock protocol would need the core to re-acquire N times, but with
      //       no consumer core body (the common bd_repeat pattern) no one
      //       drives the lock. Forcing DMA ensures the BD chain is emitted and
      //       the bd_repeat is applied via DMAStartOp.
      bool hasRepeat =
          op.getRepeatCount().has_value() && op.getRepeatCount().value() > 1;
      bool forceCircuit =
          op.getVia_DMA() || prodDimsAttr || consDimsAttr || hasRepeat;

      // Propagate via_cascade → routing_mode = Cascade.
      // Cascade has no hardware FIFO; depth must be 1.
      RoutingModeAttr routingModeAttr;

      // Explicit routing_mode override (Step 2): if the source aie.objectfifo
      // carries a discardable "routing_mode" StringAttr (set by the harness
      // via set_conduit_attrs), parse it into the RoutingMode enum and use
      // it directly. The explicit attr ALWAYS wins over via_cascade /
      // aie_stream / via_DMA / dims / repeat derivations — the user has
      // taken explicit control of routing.
      if (auto rmStrAttr =
              op->getAttrOfType<mlir::StringAttr>("routing_mode")) {
        llvm::StringRef rmStr = rmStrAttr.getValue();
        std::optional<RoutingMode> rmEnum;
        if (rmStr == "circuit")
          rmEnum = RoutingMode::Circuit;
        else if (rmStr == "packet")
          rmEnum = RoutingMode::Packet;
        else if (rmStr == "cascade")
          rmEnum = RoutingMode::Cascade;
        else if (rmStr == "stream")
          rmEnum = RoutingMode::Stream;
        else if (rmStr == "shared_memory")
          rmEnum = RoutingMode::SharedMemory;
        else if (rmStr == "dma")
          rmEnum = RoutingMode::DMA;
        if (!rmEnum) {
          op.emitError(
              "objectfifo-to-conduit: invalid 'routing_mode' attr value '")
              << rmStr
              << "'; expected one of: circuit, packet, cascade, stream, "
                 "shared_memory, dma";
          signalPassFailure();
          passFailed = true;
          return; // skip conduit.create for this fifo
        }
        routingModeAttr = RoutingModeAttr::get(ctx, *rmEnum);
      }

      if (!routingModeAttr && op.getViaCascade()) {
        if (info.depth != 1) {
          op.emitError(
              "objectfifo-to-conduit: via_cascade=true requires depth=1 "
              "(cascade has no hardware FIFO buffering), got depth=")
              << info.depth;
          signalPassFailure();
          passFailed = true;
          return; // skip conduit.create for this fifo
        }
        // CSDF patterns require buffering that the cascade stream cannot
        // provide. Cascade is a single-register pass-through with no queue;
        // any rate other than 1:1 produces deadlock or data corruption.
        // Detect CSDF by checking whether Phase 1.5 observed varying acquire
        // counts (non-empty accessPattern means at least two distinct counts).
        if (!info.accessPattern.empty()) {
          op.emitError("cascade conduit requires SDF rate (1,1); CSDF patterns "
                       "require buffering which cascade cannot provide");
          signalPassFailure();
          passFailed = true;
          return; // skip conduit.create for this fifo
        }
        routingModeAttr = RoutingModeAttr::get(ctx, RoutingMode::Cascade);
      }

      // Propagate aie_stream → routing_mode = Stream.
      // aie_stream ObjectFIFOs route data from the producer core's AXI
      // stream port directly into the consumer tile's DMA — no DMA engine
      // or buffers on the producer side. Pass C emits aie.flow(Core:N, ...)
      // instead of aie.flow(DMA:N, ...) and skips producer-side allocation.
      auto streamPortIt = aieStreamFifoPort.find(op.getSymNameAttr());

      // B-8: via_cascade=true and aie_stream are mutually exclusive.
      // Both would set routingModeAttr: cascade would set "cascade" and
      // aie_stream would overwrite it with "stream" silently.  The result
      // is an aie_stream conduit with cascade semantics applied — wrong on
      // both counts.  Reject this combination explicitly.
      // Skipped when an explicit routing_mode override is in effect — the
      // user has taken explicit control and is bypassing both derivations
      // (see followup #117).  Predicate keys off the raw input attrs
      // (`routing_mode` presence + `via_cascade` + `aie_stream`) rather than
      // routingModeAttr, so the check fires regardless of whether the
      // via_cascade derivation block above already set routingModeAttr.
      if (!op->hasAttr("routing_mode") && op.getViaCascade() &&
          streamPortIt != aieStreamFifoPort.end()) {
        op.emitError("objectfifo-to-conduit: objectfifo '")
            << op.getSymName()
            << "' has both via_cascade=true and aie_stream routing — "
               "these are mutually exclusive";
        signalPassFailure();
        passFailed = true;
        return; // skip conduit.create for this fifo
      }

      if (!routingModeAttr && streamPortIt != aieStreamFifoPort.end())
        routingModeAttr = RoutingModeAttr::get(ctx, RoutingMode::Stream);

      // If no cascade/stream routing, apply circuit override when via_DMA or
      // dims/repeat force DMA routing (replaces the old viaDMA bool attr).
      if (!routingModeAttr && forceCircuit)
        routingModeAttr = RoutingModeAttr::get(ctx, RoutingMode::Circuit);

      auto createOp = builder.create<Create>(
          loc, mlir::StringAttr::get(ctx, name),
          mlir::TypeAttr::get(info.elemType),
          mlir::IntegerAttr::get(mlir::IntegerType::get(ctx, 64), info.depth),
          routingModeAttr,
          /*sync_mode=*/disableSyncModeAttr,
          /*producer_rates=*/inferredPRAttr,
          /*consumer_rates=*/inferredCRAttr,
          /*fusion_group=*/op->getAttrOfType<mlir::StringAttr>("fusion_group"),
          /*bd_repeat=*/repeatCountAttr,
          /*dma_repeat=*/iterCountAttr,
          /*producer_dimensions=*/prodDimsAttr,
          /*consumer_dimensions=*/consDimsAttr);

      // Emit producer_tile / consumer_tiles as generic attrs so that
      // downstream passes (check, infer, fuse, Pass C) can determine tile
      // placement for conduits without aie.core-resident acquire ops.
      // These attrs are a transitional mechanism until tile inference
      // (inferAllTiles) handles all tile-association sources natively.
      if (!info.producerTileArr.empty()) {
        createOp->setAttr("producer_tile",
                          builder.getDenseI64ArrayAttr(info.producerTileArr));
      }
      if (!info.consumerTilesArr.empty()) {
        createOp->setAttr("consumer_tiles",
                          builder.getDenseI64ArrayAttr(info.consumerTilesArr));
      }

      // Set aie_stream_port as a generic attribute for stream conduits.
      if (streamPortIt != aieStreamFifoPort.end()) {
        createOp->setAttr(
            "aie_stream_port",
            mlir::IntegerAttr::get(mlir::IntegerType::get(ctx, 32),
                                   streamPortIt->second));
      }

      fifosToErase.push_back(op);
    });

    // Phase 3: rewrite aie.objectfifo.link → conduit.distribute or
    // conduit.join.
    device.walk([&](AIE::ObjectFifoLinkOp op) {
      builder.setInsertionPoint(op);
      mlir::Location loc = op.getLoc();

      auto fifoIns = op.getFifoIns();
      auto fifoOuts = op.getFifoOuts();

      // Determine mode: 1:N → distribute, N:1 → join, N:M → error.
      bool isDistribute;
      if (fifoIns.size() == 1 && fifoOuts.size() >= 1)
        isDistribute = true;
      else if (fifoIns.size() >= 1 && fifoOuts.size() == 1)
        isDistribute = false;
      else {
        // Fix 4h: N→M link (N>1 sources AND N>1 destinations) is not
        // supported. Emit an error rather than silently using "distribute".
        op.emitError("objectfifo-to-conduit: N→M link (N>1 sources AND N>1 "
                     "destinations) is not supported; use cascade mode when "
                     "implemented");
        signalPassFailure();
        passFailed = true;
        return;
      }

      // Build src/dst symbol ref arrays for conduit.distribute / conduit.join.
      llvm::SmallVector<mlir::Attribute> srcAttrs, dstAttrs;
      for (auto sym : fifoIns) {
        auto flat = mlir::cast<mlir::FlatSymbolRefAttr>(sym);
        srcAttrs.push_back(mlir::FlatSymbolRefAttr::get(ctx, flat.getValue()));
      }
      for (auto sym : fifoOuts) {
        auto flat = mlir::cast<mlir::FlatSymbolRefAttr>(sym);
        dstAttrs.push_back(mlir::FlatSymbolRefAttr::get(ctx, flat.getValue()));
      }

      // Relay tile detection: find the tile that sits between the source and
      // destination fifos in the link.  By definition, the relay tile is the
      // PRODUCER of the destination fifo (it receives data from upstream and
      // forwards it downstream).
      //
      // The old heuristic ("consumer of first src fifo") fails for broadcast
      // links where the source fifo has multiple consumers — it picks the
      // first consumer in array order, which may be a compute tile rather
      // than the actual MemTile relay.
      //
      // New logic: use the producer tile of the first dst fifo.  This is
      // always correct because in an objectfifo.link, each dst fifo's
      // producer IS the relay tile.  Falls back to consumer of first src
      // if no dst fifo info is available.
      int64_t memtileCol = -1, memtileRow = -1;
      bool found = false;

      // Primary: producer of the first dst fifo.
      if (!fifoOuts.empty()) {
        auto firstDstSym = mlir::cast<mlir::FlatSymbolRefAttr>(fifoOuts[0]);
        auto nameAttr = mlir::StringAttr::get(ctx, firstDstSym.getValue());
        auto it = fifoInfoMap.find(nameAttr);
        if (it != fifoInfoMap.end() && it->second.producerTileArr.size() >= 2) {
          memtileCol = it->second.producerTileArr[0];
          memtileRow = it->second.producerTileArr[1];
          found = true;
        }
      }

      // Fallback: consumer of first src fifo (original heuristic).
      if (!found && !fifoIns.empty()) {
        auto firstSym = mlir::cast<mlir::FlatSymbolRefAttr>(fifoIns[0]);
        auto nameAttr = mlir::StringAttr::get(ctx, firstSym.getValue());
        auto it = fifoInfoMap.find(nameAttr);
        if (it != fifoInfoMap.end() &&
            it->second.consumerTilesArr.size() >= 2) {
          memtileCol = it->second.consumerTilesArr[0];
          memtileRow = it->second.consumerTilesArr[1];
          found = true;
        }
      }

      if (!found) {
        op.emitError("objectfifo-to-conduit: cannot infer relay tile for "
                     "link op");
        signalPassFailure();
        passFailed = true;
        return;
      }

      // Extract offsets from the link op.
      mlir::DenseI64ArrayAttr offsetsAttr;
      auto joinOffsets = op.getSrcOffsets();
      auto distOffsets = op.getDstOffsets();

      llvm::SmallVector<int64_t> offVec;
      if (!isDistribute && joinOffsets && !joinOffsets.empty()) {
        for (auto attr : joinOffsets)
          offVec.push_back(mlir::cast<mlir::IntegerAttr>(attr).getInt());
      } else if (isDistribute && distOffsets && !distOffsets.empty()) {
        for (auto attr : distOffsets)
          offVec.push_back(mlir::cast<mlir::IntegerAttr>(attr).getInt());
      }
      if (!offVec.empty())
        offsetsAttr = mlir::DenseI64ArrayAttr::get(ctx, offVec);

      mlir::ArrayAttr srcsArr = mlir::ArrayAttr::get(ctx, srcAttrs);
      mlir::ArrayAttr dstsArr = mlir::ArrayAttr::get(ctx, dstAttrs);
      // Builder overload calls AIE::TileOp::getOrCreate internally — keeps the
      // relay tile alive past Pass A DCE (F1b invariant).
      if (isDistribute) {
        auto srcRef = mlir::cast<mlir::FlatSymbolRefAttr>(srcAttrs[0]);
        builder.create<ScatterOp>(loc, srcRef, dstsArr,
                                  static_cast<int>(memtileCol),
                                  static_cast<int>(memtileRow), offsetsAttr);
      } else {
        auto dstRef = mlir::cast<mlir::FlatSymbolRefAttr>(dstAttrs[0]);
        builder.create<GatherOp>(loc, srcsArr, dstRef,
                                 static_cast<int>(memtileCol),
                                 static_cast<int>(memtileRow), offsetsAttr);
      }

      op.erase();
    });

    // Phase 4: rewrite acquire/release/subview.access inside core bodies.
    //
    // SSA connectivity fix (Fix 1):
    //   conduit.acquire returns !conduit.window<T>.  This SSA value is passed
    //   directly to conduit.subview_access as the window operand.  All uses of
    //   the aie.objectfifo.subview.access result are replaced with the
    //   conduit.subview_access result.  No memref.alloc placeholder is needed.
    //
    // Port propagation fix (Fix 2):
    //   port="Produce"|"Consume" is read from the AIE op and forwarded.
    //
    // Use-after-erase fix:
    //   All subview.access ops are collected for deferred erasure BEFORE the
    //   acquire op is erased.  This prevents the acquire result SSA value from
    //   being invalidated while we still need it for the subview rewrite.

    // Collect cascade fifo names for Phase 4 dispatch.
    llvm::DenseSet<mlir::StringAttr> cascadeFifoNames;
    device.walk([&](AIE::ObjectFifoCreateOp op) {
      if (op.getViaCascade())
        cascadeFifoNames.insert(op.getSymNameAttr());
    });

    // Per-block window maps: block → (fifo name → window SSA value).
    // Populated as each block is visited; used for cross-block lookups.
    //
    // Domination invariant: conduit.release takes a !conduit.window<T> SSA
    // operand that must dominate the release site.  A flat module.walk over
    // all acquires followed by a flat walk over all releases breaks this
    // because the global `lastWindowForName` map gets clobbered by acquires
    // from later blocks.
    //
    // Fix: process every basic block independently.  Within each block the ops
    // are visited in program order, so the acquire always precedes its release
    // and the window SSA value is live at the release site.
    //
    // Cross-block dominance fix (C1):
    //   When a release is in block B but its acquire is in a dominating parent
    //   block A (e.g., release inside scf.if true region, acquire in the
    //   enclosing core entry block), the local blockWindowMap for B will be
    //   empty.  Before synthesizing a phantom acquire and emitting the C1
    //   warning, we walk up the region/block parent chain to see if any
    //   enclosing block already has a window value for this conduit name.
    //   If found, we reuse that dominating window SSA value directly — no
    //   phantom, no warning.  If not found (truly unreachable acquire), we
    //   fall back to the phantom + C1 warning as before.
    // Per-block window maps: block → (fifo name → list of conduit.acquire SSA
    // values emitted in that block, in program order).
    //
    // A SmallVector is used instead of a single Value so that
    // findWindowInDominatingBlock can find the earliest window that precedes
    // a given fence op.  The bug this fixes: when a block has multiple
    // acquire groups for the same fifo separated by a nested scf.for/scf.if
    // (e.g., preamble acquire(2) before scf.for, tail acquire(2) after
    // scf.for), a single-entry map stores only the tail window.  Queries from
    // inside the scf.for body fail the SSA dominance check against the tail
    // window and find nothing, causing the scf.for body to emit a full acquire
    // instead of a delta.
    llvm::DenseMap<
        mlir::Block *,
        llvm::DenseMap<mlir::StringAttr, llvm::SmallVector<mlir::Value, 4>>>
        allBlockWindowMaps;

    // Track conduit.window SSA values that have been released.
    // findWindowInDominatingBlock skips released windows to prevent a
    // released preamble acquire from being subsumed as a live window
    // by an inner loop body that needs its own fresh acquire.
    // Bug: without this, preamble acquire @outRows released before a loop
    // was reused inside the loop → no AcquireGreaterEqual in loop → deadlock.
    llvm::DenseSet<mlir::Value> releasedWindows;

    // Helper: walk the region/block parent chain from `startBlock` upward,
    // returning the first window value found for `nameAttr`, or null if none.
    // This handles the common case where the release is inside a nested region
    // (scf.if, scf.for body) and the acquire is in an enclosing block.
    //
    // Fix 4j: Note: if acquire and release are in the same block but release
    // appears BEFORE acquire in program order, the local blockWindowMap lookup
    // returns null and the dominating-block walk finds nothing, emitting a
    // spurious C1 warning. Forward-declared-then-released patterns trigger
    // false positives.
    auto findWindowInDominatingBlock =
        [&](mlir::Block *startBlock, mlir::Operation *beforeOp,
            mlir::StringAttr nameAttr) -> mlir::Value {
      mlir::Block *cursor = startBlock;
      // fence tracks the op in `cursor` that the window must precede
      // for SSA dominance.  Initially it is the parent op of the
      // requesting block; as we walk up the chain it becomes the
      // parent op of each intermediate block.
      mlir::Operation *fence = beforeOp;
      while (cursor) {
        // Check if this block's window map has an entry for the conduit.
        auto mapIt = allBlockWindowMaps.find(cursor);
        if (mapIt != allBlockWindowMaps.end()) {
          auto vecIt = mapIt->second.find(nameAttr);
          if (vecIt != mapIt->second.end()) {
            // Iterate all windows emitted in this block for `nameAttr`,
            // in reverse program order (last inserted = latest in block),
            // returning the latest one that still dominates the fence.
            // Reverse order ensures we pick the closest dominating acquire.
            const auto &wins = vecIt->second;
            for (auto it = wins.rbegin(); it != wins.rend(); ++it) {
              mlir::Value v = *it;
              if (!v)
                continue;
              // Skip windows that have been released: a released preamble
              // acquire must not be reused by an inner loop body that needs
              // its own fresh acquire and lock grant.
              if (releasedWindows.count(v))
                continue;
              // SSA dominance check: the window's defining op must appear
              // before `fence` in the block (or be in a different block,
              // in which case it trivially dominates via region nesting).
              mlir::Operation *defOp = v.getDefiningOp();
              bool dominates = true;
              if (fence && defOp && defOp->getBlock() == cursor)
                dominates = defOp->isBeforeInBlock(fence);
              if (dominates)
                return v;
              // This window doesn't dominate the fence — try the next earlier
              // one.
            }
            // No window in this block dominates the fence — continue up.
          }
        }
        // Walk up: the enclosing block is the block that contains
        // cursor's parent region's parent op.
        mlir::Operation *parentOp = cursor->getParentOp();
        if (!parentOp)
          break;
        fence = parentOp;
        cursor = parentOp->getBlock();
      }
      return {};
    };

    // Use PreOrder so parent blocks are visited before their nested child
    // blocks. This ensures that when the scf.if body block is visited, the
    // enclosing core entry block's window map is already populated — enabling
    // the parent-block walk in findWindowInDominatingBlock to succeed.
    device.walk<mlir::WalkOrder::PreOrder>([&](mlir::Block *block) {
      // Per-block window map: fifo name → all conduit.acquire SSA values
      // emitted in this block, in program order.  Multiple entries arise when
      // the same fifo has separate acquire groups (e.g., preamble before a
      // scf.for and tail after it).  findWindowInDominatingBlock searches the
      // vector in reverse to find the latest dominating window.
      llvm::DenseMap<mlir::StringAttr, llvm::SmallVector<mlir::Value, 4>>
          &blockWindowMap = allBlockWindowMaps[block];

      // Sequential acquire pattern (P2-D: AIE2_delayed_release):
      //
      // ObjectFIFO acquire semantics: acquire(N) means "I need N total
      // elements right now", not "give me N more".  If you call acquire(2)
      // then acquire(1) then acquire(3) then release(3), the stateful
      // transform emits: AcquireGreaterEqual(2), [nothing],
      // AcquireGreaterEqual(1), [nothing], Release(3) — only the incremental
      // delta is acquired each step.
      //
      // The Conduit IR window model is acquire-one-window-at-a-time.  To match
      // the oracle, Pass A collapses each release-group of consecutive acquires
      // into a SINGLE conduit.acquire with the MAXIMUM count in the group, then
      // reuses that one window for all subview_access rewrites within the
      // group.
      //
      // Pre-scan: for each block, walk its ops in program order and group
      // consecutive acquires on the same (fifo, port) pair between releases.
      // Record (acquire_op → effective_max_count_for_group).  When an acquire's
      // count does not exceed the current group max, it is "subsumed" by the
      // first acquire in the group and will not emit a new conduit.acquire.
      //
      // Only applies to non-cascade fifos (cascade has no window semantics).
      //
      // Example:  acquire(2), acquire(1), acquire(3), acquire(1), release(3)
      //   group max = 3 (from acquire(3))
      //   first acquire in group → conduit.acquire{count=3}
      //   remaining acquires → suppressed; reuse group window
      //   release(3) → conduit.release %groupWin {count=3}  [M8: 3≤3 OK]

      // Key: (fifo-name-attr, port-enum-as-int64)
      using GroupKey = std::pair<mlir::StringAttr, int64_t>;
      // Map from acquire-op ptr to the effective max count for its group.
      llvm::DenseMap<mlir::Operation *, int64_t> acqGroupMax;
      // Map from acquire-op ptr to whether it is the first (non-suppressed)
      // acquire in its group — only the first emits conduit.acquire.
      llvm::DenseMap<mlir::Operation *, bool> acqIsGroupLeader;

      {
        // current max held and the group leader op for each (name, port).
        llvm::DenseMap<GroupKey, int64_t> heldMax;
        llvm::DenseMap<GroupKey, mlir::Operation *> groupLeader;

        for (mlir::Operation &rawOp : *block) {
          if (auto acqOp = mlir::dyn_cast<AIE::ObjectFifoAcquireOp>(rawOp)) {
            auto nameAttr =
                mlir::StringAttr::get(ctx, acqOp.getObjFifoName().str());
            // Skip cascade fifos — they have no window semantics.
            if (cascadeFifoNames.count(nameAttr))
              continue;
            int64_t portInt =
                (acqOp.getPort() == AIE::ObjectFifoPort::Produce) ? 0 : 1;
            GroupKey key = {nameAttr, portInt};
            int64_t newCount = acqOp.acqNumber();
            int64_t &held = heldMax[key];
            if (newCount > held) {
              // This acquire extends the group (or starts a new one if
              // held==0).
              if (held == 0) {
                // New group: this op is the leader.
                groupLeader[key] = &rawOp;
                acqIsGroupLeader[&rawOp] = true;
              } else {
                // Extends existing group: update the current leader's effective
                // max so Pass C generates the right count.
                mlir::Operation *leader = groupLeader[key];
                acqGroupMax[leader] =
                    std::max(acqGroupMax.count(leader) ? acqGroupMax[leader]
                                                       : heldMax[key],
                             newCount);
                acqIsGroupLeader[&rawOp] = false;
              }
              held = newCount;
            } else {
              // Sub-max acquire: suppressed; reuse the current group leader.
              acqIsGroupLeader[&rawOp] = false;
            }
          } else if (auto relOp =
                         mlir::dyn_cast<AIE::ObjectFifoReleaseOp>(rawOp)) {
            auto nameAttr =
                mlir::StringAttr::get(ctx, relOp.getObjFifoName().str());
            if (cascadeFifoNames.count(nameAttr))
              continue;
            int64_t portInt =
                (relOp.getPort() == AIE::ObjectFifoPort::Produce) ? 0 : 1;
            GroupKey key = {nameAttr, portInt};
            // Finalise group leader's effective max (covers the case where the
            // leader was never updated by an extending acquire).
            if (groupLeader.count(key)) {
              mlir::Operation *leader = groupLeader[key];
              if (!acqGroupMax.count(leader))
                acqGroupMax[leader] = heldMax[key];
            }
            // Reset group tracking.
            heldMax[key] = 0;
            groupLeader.erase(key);
          }
        }
        // Finalise any open groups at end of block (no trailing release).
        for (auto &[key, leader] : groupLeader) {
          if (!acqGroupMax.count(leader))
            acqGroupMax[leader] = heldMax[key];
        }
      } // end pre-scan

      // Per-block group-window map: fifo name → current group leader's
      // conduit.acquire SSA value.  Used to reuse the group window for
      // suppressed (sub-max) acquires and for releases.
      llvm::DenseMap<mlir::StringAttr, mlir::Value> blockGroupWindow;
      for (mlir::Operation &rawOp : llvm::make_early_inc_range(*block)) {
        if (auto op = mlir::dyn_cast<AIE::ObjectFifoAcquireOp>(rawOp)) {
          builder.setInsertionPoint(op);
          mlir::Location loc = op.getLoc();

          std::string name = op.getObjFifoName().str();
          int64_t count = op.acqNumber();
          Port port = (op.getPort() == AIE::ObjectFifoPort::Produce)
                          ? Port::Produce
                          : Port::Consume;

          auto nameAttr = mlir::StringAttr::get(ctx, name);

          mlir::MemRefType elemType;
          auto it = fifoInfoMap.find(nameAttr);
          if (it != fifoInfoMap.end())
            elemType = it->second.elemType;
          if (!elemType)
            elemType =
                mlir::MemRefType::get({1}, mlir::IntegerType::get(ctx, 32));

          // Cascade path: two sub-cases.
          //
          // Consume: emit get_cascade with the memref's element type (scalar),
          //   find all memref.load users of %elem0 and replace their results
          //   with the get value, then erase the loads.  The subview and
          //   acquire are collected for deferred erasure in the normal order.
          //
          // Produce: do NOT process at acquire time — the stored value isn't
          //   available until the user's memref.store runs.  The release
          //   handler below walks the produce acquire, finds the subview, finds
          //   the store, extracts the stored value, emits put_cascade, then
          //   erases the store, subview, and acquire in dependency order (store
          //   → subview → acquire). The acquire is NOT added to acquiresToErase
          //   here; the release handler takes ownership of erasure.
          if (cascadeFifoNames.count(nameAttr)) {
            if (port == Port::Consume) {
              // Scalar element type (e.g., i32 from memref<1xi32>).
              mlir::Type elemTy = elemType.getElementType();
              auto getCascOp = builder.create<AIE::GetCascadeOp>(
                  loc, elemTy, mlir::FlatSymbolRefAttr::get(ctx, name));
              mlir::Value cascVal =
                  getCascOp.getCascadeValue(); // scalar i32/vector

              mlir::Value subviewResult = op.getResult();
              for (mlir::Operation *user :
                   llvm::make_early_inc_range(subviewResult.getUsers())) {
                if (auto accessOp =
                        mlir::dyn_cast<AIE::ObjectFifoSubviewAccessOp>(user)) {
                  // Replace memref.load users of %elem0 with the cascade value.
                  mlir::Value elem0 = accessOp.getResult(); // memref<1xi32>
                  for (mlir::Operation *loadUser :
                       llvm::make_early_inc_range(elem0.getUsers())) {
                    if (auto loadOp =
                            mlir::dyn_cast<mlir::memref::LoadOp>(loadUser)) {
                      loadOp.getResult().replaceAllUsesWith(cascVal);
                      loadOp.erase();
                    }
                  }
                  subviewsToErase.push_back(accessOp);
                }
              }
              acquiresToErase.push_back(op);
            }
            // Produce: skip acquire; release handler does everything.
            continue;
          }

          // Sequential acquire pattern (P2-D):
          // The pre-scan determined whether this acquire is a group leader
          // (first/max in its release-group) or is subsumed by the leader.
          //
          // - Group leader: emit conduit.acquire{count=effective_max}, record
          //   as blockGroupWindow.  The count used is the pre-scan max (not the
          //   raw op count) so that sub-max acquires in the same group are
          //   covered by this single window — M8 sees release_count ≤ max.
          // - Subsumed acquire: no conduit.acquire emitted; subview_access ops
          //   are rewritten to use the current blockGroupWindow instead.
          //
          // Cross-block subsumption (AIE2_dynamic_locks pattern):
          //   When an acquire in a nested block (e.g., scf.for body) requests
          //   the same or fewer elements than an open acquire in a dominating
          //   parent block, the nested acquire is subsumed — no new use_lock
          //   is needed because the elements are already held.  Matches the
          //   stateful transform which tracks held counts across block
          //   boundaries.

          mlir::Value winVal;
          bool isLeader = acqIsGroupLeader.lookup(&rawOp);

          // Compute effective count up-front (needed for both parent-window
          // count comparison and for the conduit.acquire emission below).
          int64_t effectiveCount =
              acqGroupMax.count(&rawOp) ? acqGroupMax[&rawOp] : count;

          if (isLeader || !blockGroupWindow.count(nameAttr)) {
            // Before emitting a new conduit.acquire, check if a dominating
            // parent block already holds a window for this fifo.  If so, the
            // acquire in the enclosing scope already covers the needed count
            // and no additional use_lock is required (cross-block subsumption).
            mlir::Value parentWin = findWindowInDominatingBlock(
                block->getParentOp() ? block->getParentOp()->getBlock()
                                     : nullptr,
                block->getParentOp(), nameAttr);

            // Non-uniform acquire count fix: if the dominating window was
            // acquired with a count less than the requested effective count,
            // do not reuse it.  The inner block needs more elements than the
            // parent acquired (e.g., preamble acquire(2) followed by loop
            // body acquire(3)), so a new conduit.acquire must be emitted with
            // the correct count to satisfy M2 bounds and ensure subview_access
            // indices [0, effectiveCount) are valid.
            if (parentWin) {
              if (auto parentAcq = parentWin.getDefiningOp<Acquire>()) {
                int64_t parentCount =
                    static_cast<int64_t>(parentAcq.getCount());
                if (parentCount < effectiveCount) {
                  parentWin = {}; // insufficient count — don't reuse
                }
              }
            }

            if (parentWin) {
              // Reuse the dominating block's window — cross-block subsumed.
              winVal = parentWin;
              blockGroupWindow[nameAttr] = winVal;
            } else {
              // Emit one conduit.acquire for the group.
              // Use the pre-scanned group max as the effective count so the
              // single window covers all elements that will be released.
              auto winTy = WindowType::get(ctx, elemType);
              winVal = builder.create<Acquire>(
                  loc, winTy, mlir::FlatSymbolRefAttr::get(ctx, name),
                  mlir::IntegerAttr::get(mlir::IntegerType::get(ctx, 64),
                                         effectiveCount),
                  PortAttr::get(ctx, port),
                  /*window_size=*/mlir::IntegerAttr{});
              // Record as the group leader window.
              blockGroupWindow[nameAttr] = winVal;
            }
          } else {
            // Subsumed acquire: reuse the existing group window.
            winVal = blockGroupWindow[nameAttr];
          }

          // Record the window for subsequent releases in this block and for
          // cross-block lookups in dominated nested blocks.
          // Use push_back (not assignment) so that multiple windows for the
          // same fifo in one block (e.g., preamble and tail groups separated
          // by a scf.for) are all stored.  findWindowInDominatingBlock iterates
          // the vector and picks the latest one that dominates the fence.
          blockWindowMap[nameAttr].push_back(winVal);

          // Rewrite subview.access users immediately.
          mlir::Value subviewResult = op.getResult();
          for (mlir::Operation *user :
               llvm::make_early_inc_range(subviewResult.getUsers())) {
            if (auto accessOp =
                    mlir::dyn_cast<AIE::ObjectFifoSubviewAccessOp>(user)) {
              builder.setInsertionPoint(accessOp);
              int64_t idx = accessOp.getIndex();
              mlir::Type resultTy = accessOp.getResult().getType();
              auto condAccess = builder.create<SubviewAccess>(
                  accessOp.getLoc(), resultTy, winVal,
                  mlir::IntegerAttr::get(mlir::IntegerType::get(ctx, 64), idx));
              accessOp.getResult().replaceAllUsesWith(condAccess.getResult());
              subviewsToErase.push_back(accessOp);
            } else {
              // Fix B2: unexpected user of objectfifo.acquire result.
              // Only subview_access users are lowered; other user types will
              // have dangling references after the acquire op is erased.
              op.emitWarning(
                  "objectfifo-to-conduit: unexpected user of "
                  "objectfifo.acquire result — only subview_access users are "
                  "lowered; other users will have dangling references");
            }
          }
          acquiresToErase.push_back(op);

        } else if (auto op = mlir::dyn_cast<AIE::ObjectFifoReleaseOp>(rawOp)) {
          builder.setInsertionPoint(op);
          mlir::Location loc = op.getLoc();

          std::string name = op.getObjFifoName().str();
          int64_t count = op.getSize();
          Port port = (op.getPort() == AIE::ObjectFifoPort::Produce)
                          ? Port::Produce
                          : Port::Consume;

          auto nameAttr = mlir::StringAttr::get(ctx, name);

          // Cascade release handling.
          if (cascadeFifoNames.count(nameAttr)) {
            if (port == Port::Produce) {
              // Find the corresponding acquire for this fifo in this block
              // (scanning forward is safe since we process in PreOrder).
              // The acquire was NOT added to acquiresToErase; we own it here.
              // Find the acquire immediately preceding this release in block
              // order. Scan forward and stop at the release op so we pick the
              // last matching acquire BEFORE this release, not the last one in
              // the whole block (which would be wrong when two acquire/release
              // pairs for the same cascade fifo appear in the same block).
              AIE::ObjectFifoAcquireOp acqOp;
              for (mlir::Operation &scan : *block) {
                if (&scan == op)
                  break; // stop at this release — don't look past it
                if (auto a = mlir::dyn_cast<AIE::ObjectFifoAcquireOp>(scan)) {
                  if (a.getObjFifoName() == name &&
                      a.getPort() == AIE::ObjectFifoPort::Produce)
                    acqOp = a;
                }
              }

              if (!acqOp) {
                op->emitWarning(
                    "objectfifo-to-conduit: cascade Produce release for '")
                    << name
                    << "' has no matching acquire — put_cascade skipped";
                releasesToErase.push_back(op);
                continue;
              }

              // Find the subview.access op (user of the acquire result).
              AIE::ObjectFifoSubviewAccessOp accessOp;
              for (mlir::Operation *user : acqOp.getResult().getUsers()) {
                if (auto a =
                        mlir::dyn_cast<AIE::ObjectFifoSubviewAccessOp>(user))
                  accessOp = a;
              }

              if (!accessOp) {
                op->emitWarning(
                    "objectfifo-to-conduit: cascade Produce acquire for '")
                    << name
                    << "' has no subview.access user — put_cascade skipped";
                releasesToErase.push_back(op);
                acqOp->erase();
                continue;
              }

              // Find the memref.store into elem0 and extract the stored value.
              // The stored value becomes the cascade stream value.
              mlir::Value elem0 = accessOp.getResult(); // memref<T>
              mlir::Value storedVal;
              llvm::SmallVector<mlir::memref::StoreOp> storesToErase;
              for (mlir::Operation *storeUser :
                   llvm::make_early_inc_range(elem0.getUsers())) {
                if (auto storeOp =
                        mlir::dyn_cast<mlir::memref::StoreOp>(storeUser)) {
                  storedVal = storeOp.getValueToStore();
                  storesToErase.push_back(storeOp);
                }
              }

              if (storedVal) {
                builder.setInsertionPoint(op);
                builder.create<AIE::PutCascadeOp>(
                    loc, storedVal, mlir::FlatSymbolRefAttr::get(ctx, name));
              } else {
                op->emitWarning("objectfifo-to-conduit: cascade Produce '")
                    << name
                    << "' has no memref.store into elem0 — "
                       "put_cascade skipped (no value to send)";
              }

              // Erase in dependency order: stores → subview → acquire.
              for (auto s : storesToErase)
                s.erase();
              if (accessOp.getResult().use_empty())
                accessOp.erase();
              if (acqOp.getResult().use_empty())
                acqOp.erase();

              releasesToErase.push_back(op);
              continue;
            }

            // Cascade Consume releases are no-ops (no hardware lock to
            // release).
            releasesToErase.push_back(op);
            continue;
          }

          // Sequential acquire pattern (P2-D): for releases in the same block,
          // prefer the group leader window (blockGroupWindow) over the most
          // recently seen acquire window (blockWindowMap).  The group leader
          // window was emitted with count=max_in_group, so M8's invariant
          // release_count ≤ acquired_count is satisfied.
          //
          // After the release, clear the group window so the next acquire in
          // this block starts a new group.
          mlir::Value winVal = blockGroupWindow.lookup(nameAttr);
          if (winVal) {
            // Same-block sequential acquire group: use group leader window.
            blockGroupWindow.erase(nameAttr);
          } else {
            // No group window in this block — try blockWindowMap (same block).
            // Use the last (most recently emitted) window for this fifo.
            const auto &vec = blockWindowMap[nameAttr];
            if (!vec.empty())
              winVal = vec.back();
          }

          if (!winVal) {
            // Cross-block case: look for a window value in a dominating
            // enclosing block (e.g., acquire in core entry block, release
            // inside scf.if true region).
            winVal = findWindowInDominatingBlock(
                block->getParentOp() ? block->getParentOp()->getBlock()
                                     : nullptr,
                block->getParentOp(), nameAttr);
          }

          if (!winVal) {
            // C1 diagnostic: no dominating acquire found anywhere in the
            // parent block chain.  Must synthesize a phantom window.
            // On Phoenix hardware this injects an extra AcquireGreaterEqual
            // use_lock that stalls the producer.  Manual review required.
            op->emitWarning(
                "ObjectFifoToConduit: cross-block acquire/release pattern "
                "detected for conduit '")
                << name
                << "'; phantom AcquireGreaterEqual synthesized. Hardware stall "
                   "risk on Phoenix. Manual review required.";

            mlir::MemRefType elemType;
            auto it = fifoInfoMap.find(nameAttr);
            if (it != fifoInfoMap.end())
              elemType = it->second.elemType;
            if (!elemType)
              elemType =
                  mlir::MemRefType::get({1}, mlir::IntegerType::get(ctx, 32));
            auto winTy = WindowType::get(ctx, elemType);
            winVal = builder.create<Acquire>(
                loc, winTy, mlir::FlatSymbolRefAttr::get(ctx, name),
                mlir::IntegerAttr::get(mlir::IntegerType::get(ctx, 64), count),
                PortAttr::get(ctx, port), /*window_size=*/mlir::IntegerAttr{});
            blockWindowMap[nameAttr].push_back(winVal);
          }

          // Mark this window as released so findWindowInDominatingBlock
          // will not reuse it for subsequent acquires in dominated blocks.
          if (winVal)
            releasedWindows.insert(winVal);

          builder.create<Release>(
              loc, winVal,
              mlir::IntegerAttr::get(mlir::IntegerType::get(ctx, 64), count),
              PortAttr::get(ctx, port));

          releasesToErase.push_back(op);
        }
      }
    });
  }

  // -----------------------------------------------------------------------
  // Phase 4.5+: eraseOriginalOps
  // -----------------------------------------------------------------------
  //
  // Erase all original ObjectFIFO ops (subview, acquire, release, create)
  // and emit shim DMA allocation symbols + allocate delegate tile transfer.

  void eraseOriginalOps(AIE::DeviceOp device, mlir::OpBuilder &builder,
                        mlir::MLIRContext *ctx) {
    // Deferred erasure: erase subviews before acquires (subview uses acquire
    // result).
    for (auto op : subviewsToErase)
      op.erase();
    for (auto op : acquiresToErase)
      op.erase();
    for (auto op : releasesToErase)
      op.erase();

    // Erase aie.objectfifo.register_external_buffers ops (no conduit
    // equivalent). Collect first to avoid walk-while-erase.
    llvm::SmallVector<AIE::ObjectFifoRegisterExternalBuffersOp> extBufOps;
    device.walk([&](AIE::ObjectFifoRegisterExternalBuffersOp op) {
      extBufOps.push_back(op);
    });
    for (auto extBufOp : extBufOps)
      extBufOp.erase();

    // Phase 4.5: preserve shim DMA symbols for runtime_sequence.
    //
    // When an objectfifo connects to a shim tile (row==0), the
    // runtime_sequence contains aiex.npu.dma_wait / aiex.npu.dma_memcpy_nd
    // ops that reference the objectfifo symbol.  Erasing the objectfifo
    // without providing a replacement symbol causes the verifier to fail.
    // Fix: emit aie.shim_dma_allocation @<name>_shim_alloc and rewrite all
    // symbol uses before erasing the objectfifo.
    //
    // Per-shim-tile channel counters: when multiple objectfifos use the same
    // shim tile in the same direction, each needs a distinct channel index.
    llvm::DenseMap<mlir::Value, int> shimMM2SCounter;
    llvm::DenseMap<mlir::Value, int> shimS2MMCounter;
    for (AIE::ObjectFifoCreateOp op : fifosToErase) {
      auto prodTile =
          mlir::cast<AIE::TileOp>(op.getProducerTile().getDefiningOp());
      int64_t prodRow = prodTile.getRow();

      // Collect all shim tiles involved in this objectfifo.
      // Producer shim: emit MM2S allocation.
      // Consumer shim(s): emit one S2MM allocation per shim consumer.
      llvm::SmallVector<std::pair<AIE::TileOp, AIE::DMAChannelDir>> shimEntries;
      if (prodRow == 0) {
        shimEntries.push_back({prodTile, AIE::DMAChannelDir::MM2S});
      } else {
        for (mlir::Value cons : op.getConsumerTiles()) {
          auto consTile = mlir::cast<AIE::TileOp>(cons.getDefiningOp());
          if (consTile.getRow() == 0)
            shimEntries.push_back({consTile, AIE::DMAChannelDir::S2MM});
        }
      }

      if (shimEntries.empty())
        continue;

      auto deviceOp = op->getParentOfType<AIE::DeviceOp>();
      if (!deviceOp)
        continue;

      // For single-consumer (common case) keep the original _shim_alloc name so
      // existing tests and runtime_sequence symbol references continue to work.
      // For multiple shim consumers, suffix with _0, _1, … to distinguish them.
      bool multiShim = shimEntries.size() > 1;
      std::string baseAllocSym = op.getSymName().str() + "_shim_alloc";

      for (unsigned shimIdx = 0; shimIdx < shimEntries.size(); ++shimIdx) {
        auto [shimTile, channelDir] = shimEntries[shimIdx];
        std::string allocSym =
            multiShim ? (baseAllocSym + "_" + std::to_string(shimIdx))
                      : baseAllocSym;

        // Allocate per-shim-tile per-direction channel index.
        int channelIdx;
        if (channelDir == AIE::DMAChannelDir::MM2S)
          channelIdx = shimMM2SCounter[shimTile.getResult()]++;
        else
          channelIdx = shimS2MMCounter[shimTile.getResult()]++;

        builder.setInsertionPoint(deviceOp.getBody()->getTerminator());
        auto shimAllocOp = builder.create<AIE::ShimDMAAllocationOp>(
            op.getLoc(), allocSym, shimTile.getResult(), channelDir,
            /*channel_index=*/static_cast<int64_t>(channelIdx),
            /*plio=*/op.getPlio(),
            /*packet=*/nullptr);
        shimAllocOp->setAttr(
            "conduit_channel",
            mlir::FlatSymbolRefAttr::get(ctx, op.getSymName().str()));

        // Only rewrite symbol uses for the first (or only) allocation so that
        // a single symbol name continues to refer to the objectfifo.
        if (shimIdx == 0) {
          // Save the original name so we can revert conduit op name attrs.
          std::string origName = op.getSymName().str();
          if (mlir::failed(mlir::SymbolTable::replaceAllSymbolUses(
                  op.getSymNameAttr(), builder.getStringAttr(allocSym),
                  deviceOp))) {
            op.emitWarning("ObjectFifoToConduit: failed to rewrite symbol uses "
                           "for shim-connected objectfifo '")
                << op.getSymName() << "'";
          }
          // replaceAllSymbolUses renames ALL FlatSymbolRefAttr occurrences
          // (including conduit.* op name attrs) from @origName to @allocSym.
          // However conduit.* name attrs reference the conduit.create symbol,
          // not the aie.objectfifo — they must stay as @origName so Pass C's
          // conduitMap lookup (keyed on conduit.create sym_name) succeeds.
          // Walk all conduit use-site ops and revert their name attr.
          mlir::FlatSymbolRefAttr allocRef =
              mlir::FlatSymbolRefAttr::get(ctx, allocSym);
          mlir::FlatSymbolRefAttr origRef =
              mlir::FlatSymbolRefAttr::get(ctx, origName);
          auto revertIfRenamed = [&](auto &walkOp) {
            if (walkOp.getNameAttr() == allocRef)
              walkOp.setNameAttr(origRef);
          };
          deviceOp.walk([&](Acquire op) { revertIfRenamed(op); });
          deviceOp.walk([&](AcquireAsync op) { revertIfRenamed(op); });
          deviceOp.walk([&](ReleaseAsync op) { revertIfRenamed(op); });
          deviceOp.walk([&](PutMemref op) { revertIfRenamed(op); });
          deviceOp.walk([&](GetMemref op) { revertIfRenamed(op); });
          deviceOp.walk([&](PutMemrefAsync op) { revertIfRenamed(op); });
          deviceOp.walk([&](GetMemrefAsync op) { revertIfRenamed(op); });
          deviceOp.walk([&](WaitWindow op) { revertIfRenamed(op); });
          // AIE::ObjectFifoAllocateOp uses a different attr field
          // ($objFifo_name → getObjFifoNameAttr); revert inline since the
          // shared helper assumes the conduit-style getNameAttr accessor.
          // Without this revert, Phase 4.6 alloc_tile lowering looks up the
          // post-rename symbol (@chan_shim_alloc) and fails to find the
          // matching conduit.create (which still uses @chan), warning + erasing
          // the allocate and silently dropping the user's MemTile placement.
          deviceOp.walk([&](AIE::ObjectFifoAllocateOp op) {
            if (op.getObjFifoNameAttr() == allocRef)
              op.setObjFifoNameAttr(origRef);
          });
          // Also revert srcs/dsts arrays on distribute/join/forward ops.
          // replaceAllSymbolUses renames FlatSymbolRefAttr elements inside
          // SymbolRefArrayAttr arrays as well.
          auto revertArray = [&](mlir::ArrayAttr arr) -> mlir::ArrayAttr {
            if (!arr)
              return arr;
            llvm::SmallVector<mlir::Attribute> newAttrs;
            bool changed = false;
            for (auto attr : arr) {
              if (attr == allocRef) {
                newAttrs.push_back(origRef);
                changed = true;
              } else {
                newAttrs.push_back(attr);
              }
            }
            if (!changed)
              return arr;
            return mlir::ArrayAttr::get(ctx, newAttrs);
          };
          deviceOp.walk([&](ScatterOp op) {
            if (op.getSrcAttr() == allocRef)
              op.setSrcAttr(origRef);
            auto newDsts = revertArray(op.getDsts());
            if (newDsts != op.getDsts())
              op.setDstsAttr(newDsts);
          });
          deviceOp.walk([&](GatherOp op) {
            auto newSrcs = revertArray(op.getSrcs());
            if (newSrcs != op.getSrcs())
              op.setSrcsAttr(newSrcs);
            if (op.getDstAttr() == allocRef)
              op.setDstAttr(origRef);
          });
          // DEFERRED-13: replaceAllSymbolUses also renamed the conduit_channel
          // attr on shimAllocOp from @origName to @allocSym.  Reset it to the
          // original objectfifo name so Phase 5a conduitChannelMap lookup
          // (keyed on conduit.create sym_name = @origName) succeeds.
          shimAllocOp->setAttr("conduit_channel", origRef);
        }
      }
    }

    // Phase 4.6: alloc_tile → scatter{N=1} lowering.
    //
    // An aie.objectfifo.allocate @fifo (%memtile) directive means "allocate
    // @fifo's buffers on %memtile's memory module instead of the default
    // tile."  When the delegate is a MemTile (row==1), this is semantically
    // equivalent to a 1:1 relay through the MemTile — the producer writes
    // into the MemTile, and the MemTile forwards to the original consumers.
    //
    // Lower this to:
    //   1. Update @fifo's consumer_tiles to point at the MemTile.
    //   2. Create @fifo_relay with producer_tile=MemTile, consumer_tiles=orig.
    //   3. Emit conduit.scatter { src=@fifo, dsts=[@fifo_relay] }.
    //   4. Rewrite consumer-side Acquire/AcquireAsync/ReleaseAsync ops from
    //      @fifo to @fifo_relay.
    //
    // For non-MemTile delegates (row>1): emit a warning and erase — the
    // existing test corpus only uses compute-tile delegates for buffer
    // coalescing, which the oracle already handles without relay.
    llvm::SmallVector<AIE::ObjectFifoAllocateOp> allocatesToErase;
    device.walk(
        [&](AIE::ObjectFifoAllocateOp op) { allocatesToErase.push_back(op); });
    for (AIE::ObjectFifoAllocateOp allocOp : allocatesToErase) {
      auto delegateTile = allocOp.getDelegateTileOp();
      int64_t delegateCol = delegateTile.getCol();
      int64_t delegateRow = delegateTile.getRow();

      if (delegateRow != 1) {
        // Non-MemTile delegate: emit warning and erase (current behavior).
        allocOp.emitWarning(
            "objectfifo-to-conduit: ignoring objectfifo.allocate with "
            "non-MemTile delegate tile (row=")
            << delegateRow
            << "); only MemTile (row=1) delegates are "
               "lowered to scatter{N=1} relays";
        allocOp.erase();
        continue;
      }

      // MemTile delegate (row==1): emit scatter{N=1} relay.
      std::string fifoName = allocOp.getObjFifoName().str();
      std::string relayName = fifoName + "_relay";

      // Find the conduit.create for this objectfifo.
      Create srcCreateOp = nullptr;
      device.walk([&](Create createOp) {
        if (createOp.getSymName() == fifoName)
          srcCreateOp = createOp;
      });
      if (!srcCreateOp) {
        allocOp.emitWarning(
            "objectfifo-to-conduit: cannot find conduit.create for '")
            << fifoName << "'; skipping alloc_tile lowering";
        allocOp.erase();
        continue;
      }

      // Note: producer_tile/consumer_tiles attrs are no longer emitted —
      // tile coordinates are inferred from IR structure via inferAllTiles().
      // The conduit.scatter op (Step 3) establishes the MemTile relay
      // relationship, and consumer-side acquire/release ops (Step 4) associate
      // the relay channel with consumer tiles.

      // Step 2: Create @fifo_relay conduit.create with same characteristics.
      builder.setInsertionPointAfter(srcCreateOp);
      builder.create<Create>(
          srcCreateOp.getLoc(), mlir::StringAttr::get(ctx, relayName),
          srcCreateOp.getElementTypeAttr(), srcCreateOp.getDepthAttr(),
          /*routing_mode=*/RoutingModeAttr{},
          /*sync_mode=*/SyncModeAttr{},
          /*producer_rates=*/nullptr,
          /*consumer_rates=*/nullptr,
          /*fusion_group=*/mlir::StringAttr{},
          /*bd_repeat=*/nullptr,
          /*dma_repeat=*/nullptr,
          /*producer_dimensions=*/nullptr,
          /*consumer_dimensions=*/nullptr);

      // Step 3: Emit conduit.scatter { src=@fifo, dsts=[@fifo_relay] }.
      // Builder overload uses TileOp::getOrCreate internally so the relay tile
      // cannot be DCE'd after Pass A when its only SSA users are the lowered
      // objectfifos (F1b invariant).
      mlir::FlatSymbolRefAttr srcRef =
          mlir::FlatSymbolRefAttr::get(ctx, fifoName);
      mlir::ArrayAttr dstsArr = mlir::ArrayAttr::get(
          ctx, {mlir::FlatSymbolRefAttr::get(ctx, relayName)});
      builder.create<ScatterOp>(srcCreateOp.getLoc(), srcRef, dstsArr,
                                static_cast<int>(delegateCol),
                                static_cast<int>(delegateRow),
                                /*offsets=*/nullptr);

      // Step 4: Rewrite consumer-side ops from @fifo to @fifo_relay.
      mlir::FlatSymbolRefAttr origNameRef =
          mlir::FlatSymbolRefAttr::get(ctx, fifoName);
      mlir::FlatSymbolRefAttr relayRef =
          mlir::FlatSymbolRefAttr::get(ctx, relayName);
      auto rewriteConsumer = [&](auto walkOp) {
        if (walkOp.getNameAttr() == origNameRef &&
            walkOp.getPort() == Port::Consume)
          walkOp.setNameAttr(relayRef);
      };
      device.walk([&](Acquire op) { rewriteConsumer(op); });
      device.walk([&](AcquireAsync op) { rewriteConsumer(op); });
      device.walk([&](ReleaseAsync op) { rewriteConsumer(op); });

      allocOp.erase();
    }

    for (AIE::ObjectFifoCreateOp op : fifosToErase)
      op.erase();
  }

  // -----------------------------------------------------------------------
  // runOnOperation: orchestrate the three phases
  // -----------------------------------------------------------------------

  void runOnOperation() override {
    mlir::ModuleOp module = getOperation();
    mlir::OpBuilder builder(module.getContext());
    mlir::MLIRContext *ctx = module.getContext();

    // Process each aie.device independently with fresh maps to prevent
    // name collisions across devices in multi-device modules (e.g.,
    // FusedMLIROperator IR with 19 device blocks all using @in_0, @out_0).
    module.walk([&](AIE::DeviceOp device) {
      passFailed = false;

      collectFifoInfo(device, ctx);
      if (passFailed)
        return;
      transformFifos(device, builder, ctx);
      if (passFailed)
        return;
      eraseOriginalOps(device, builder, ctx);
    });
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Factory + registration
//===----------------------------------------------------------------------===//

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createObjectFifoToConduitPass() {
  return std::make_unique<ObjectFifoToConduitPass>();
}

} // namespace xilinx::conduit
