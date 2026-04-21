//===- ConduitCombineDevicePass.cpp - aie-combine-device pass
//------*-C++-*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2026 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//
//
// --aie-combine-device: General-purpose device merge pass.
//
// IRON operators emit separate aie.device blocks per operator. Downstream
// fusion passes (--conduit-fuse-operators, --conduit-fuse-core-bodies) need
// all fusable tiles to reside in a single aie.device block. This pass
// merges two devices connected by a conduit channel with matching
// fusion_group attributes.
//
// Two modes:
//   - tile-offset (default): Offsets devB tile coordinates by max_col(devA)+1.
//     Used by spatial fusion (--conduit-fuse-operators).
//   - same-tile: Keeps original tile coordinates. Used as pre-merge for
//     loop-body fusion (--conduit-fuse-core-bodies).
//
// This pass does NOT erase or rewrite any conduit channels — it only
// merges the device bodies. Channel rewriting is left to the downstream
// fusion pass.
//
//===----------------------------------------------------------------------===//

#include "aie/Dialect/Conduit/Transforms/ConduitPasses.h"

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/Conduit/IR/ConduitDialect.h"

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/Pass/Pass.h"

#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/Support/raw_ostream.h"

namespace xilinx::conduit {

#define GEN_PASS_DEF_CONDUITCOMBINEDEVICE
#include "aie/Dialect/Conduit/Transforms/ConduitPasses.h.inc"

namespace {

// ---------------------------------------------------------------------------
// Helper: collect the maximum column index used by any tile in a DeviceOp.
// ---------------------------------------------------------------------------
static int64_t maxColInDevice(AIE::DeviceOp device) {
  int64_t maxCol = -1;
  device.walk([&](AIE::TileOp tile) {
    if (tile.getCol() > maxCol)
      maxCol = tile.getCol();
  });
  return maxCol;
}

// ---------------------------------------------------------------------------
// Helper: offset all tile coordinates in a DeviceOp by colOffset.
// Updates aie.tile(col, row) -> aie.tile(col + colOffset, row).
// ---------------------------------------------------------------------------
static void offsetDeviceTiles(AIE::DeviceOp device, int64_t colOffset) {
  if (colOffset == 0)
    return;
  llvm::SmallVector<AIE::TileOp> tiles;
  device.walk([&](AIE::TileOp t) { tiles.push_back(t); });
  mlir::OpBuilder builder(device.getContext());
  for (AIE::TileOp tile : tiles) {
    int64_t newCol = tile.getCol() + colOffset;
    tile->setAttr("col",
                  builder.getI32IntegerAttr(static_cast<int32_t>(newCol)));
  }
}

// ---------------------------------------------------------------------------
// Helper: check if two devices are connected by a conduit channel pair
// with matching fusion_group attributes. Returns true if at least one
// matching fusion_group pair exists across devA and devB.
// ---------------------------------------------------------------------------
static bool devicesConnectedByFusionGroup(AIE::DeviceOp devA,
                                          AIE::DeviceOp devB) {
  llvm::SmallVector<llvm::StringRef> groupsA;
  devA.walk([&](Create op) {
    auto fg = op.getFusionGroup();
    if (fg && !fg->empty())
      groupsA.push_back(*fg);
  });

  bool found = false;
  devB.walk([&](Create op) {
    if (found)
      return;
    auto fg = op.getFusionGroup();
    if (!fg || fg->empty())
      return;
    for (llvm::StringRef ga : groupsA) {
      if (ga == *fg) {
        found = true;
        return;
      }
    }
  });
  return found;
}

// ---------------------------------------------------------------------------
// Main pass struct.
// ---------------------------------------------------------------------------
struct ConduitCombineDevicePass
    : public impl::ConduitCombineDeviceBase<ConduitCombineDevicePass> {

  using ConduitCombineDeviceBase::ConduitCombineDeviceBase;

  void runOnOperation() override {
    mlir::ModuleOp module = getOperation();
    mlir::MLIRContext *ctx = module.getContext();

    // Collect all DeviceOps in module order.
    llvm::SmallVector<AIE::DeviceOp> devices;
    module.walk([&](AIE::DeviceOp dev) { devices.push_back(dev); });

    if (devices.size() < 2)
      return; // Nothing to merge.

    // Process consecutive device pairs (A, B).
    for (size_t i = 0; i + 1 < devices.size(); ++i) {
      AIE::DeviceOp devA = devices[i];
      AIE::DeviceOp devB = devices[i + 1];

      // Only merge devices connected by matching fusion_group.
      if (!devicesConnectedByFusionGroup(devA, devB))
        continue;

      // --- Tile offset (tile-offset mode only). ---
      if (!sameTile) {
        int64_t colMaxA = maxColInDevice(devA);
        int64_t colOffset = colMaxA + 1;
        offsetDeviceTiles(devB, colOffset);
      }

      // --- Merge device B into device A by physically moving ops. ---
      //
      // aie.device has IsolatedFromAbove semantics. We use op->remove() +
      // builder.insert(op) to physically move each op from bodyB into
      // bodyA (before bodyA's terminator). Moving preserves all SSA values.
      {
        mlir::Block &bodyA = devA.getBodyRegion().front();
        mlir::Block &bodyB = devB.getBodyRegion().front();

        // Find devA's runtime_sequence (may not exist).
        mlir::Operation *seqA = nullptr;
        for (mlir::Operation &op : bodyA) {
          if (op.getName().getStringRef() == "aie.runtime_sequence") {
            seqA = &op;
            break;
          }
        }

        // Phase 1: move all non-sequence, non-terminator ops from bodyB
        // into bodyA.
        mlir::OpBuilder b(ctx);
        if (bodyA.mightHaveTerminator()) {
          if (mlir::Operation *term = bodyA.getTerminator())
            b.setInsertionPoint(term);
          else
            b.setInsertionPointToEnd(&bodyA);
        } else {
          b.setInsertionPointToEnd(&bodyA);
        }
        llvm::SmallVector<mlir::Operation *> seqOps;
        {
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
        }

        // Phase 2: merge devB's runtime_sequence(s) into devA's sequence.
        //
        // aie.runtime_sequence is a Symbol with block-argument-typed host
        // memref parameters. When devA and devB have different signatures,
        // we merge them into a single combined sequence whose signature is
        // the concatenation of devA's args plus devB's args.
        {
          for (mlir::Operation *seqB : seqOps) {
            if (seqA && seqA->getNumRegions() > 0 &&
                seqB->getNumRegions() > 0) {
              mlir::Block &seqBodyA = seqA->getRegion(0).front();
              mlir::Block &seqBodyB = seqB->getRegion(0).front();

              // Append seqB's block args to seqA, building the IRMapping.
              mlir::IRMapping argMapping;
              for (mlir::BlockArgument arg : seqBodyB.getArguments()) {
                mlir::BlockArgument newArg =
                    seqBodyA.addArgument(arg.getType(), arg.getLoc());
                argMapping.map(arg, newArg);
              }

              // Clone each op from seqBodyB into seqBodyA.
              mlir::OpBuilder seqBuilder(ctx);
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
              // devA has no sequence yet — move devB's as-is.
              seqB->remove();
              b.insert(seqB);
              seqA = seqB;
            }
          }
        }

        // devB body is now empty except its aie.end terminator. Erase it.
        devB->erase();

        // --- Sink cores and runtime sequences to end of device body. ---
        //
        // Pass C emits aie.lock / aie.buffer ops after all aie.tile ops.
        // After merging, devA's original aie.core ops appear BEFORE devB's
        // tiles, so locks emitted by Pass C would come after the cores,
        // violating MLIR dominance. Fix: move all aie.core, aie.mem, and
        // aie.runtime_sequence ops to just before the aie.end terminator.
        {
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
      }
    }
  }
};

} // namespace

std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createConduitCombineDevicePass() {
  return std::make_unique<ConduitCombineDevicePass>();
}

} // namespace xilinx::conduit
