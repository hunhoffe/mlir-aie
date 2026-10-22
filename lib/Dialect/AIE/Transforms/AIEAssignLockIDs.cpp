//===- AIEAssignLockIDs.cpp -------------------------------------*- C++ -*-===//
//
// Copyright (C) 2022 Xilinx, Inc.
// Copyright (C) 2022-2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// This pass aims to assign lockIDs to AIE.lock operations. The lockID is
// numbered from the most recent AIE.lock within the same tile. If the lockID
// exceeds the number of locks on the tile, the pass generates an error and
// terminates. AIE.lock operations for different tiles are numbered
// independently. If there are existing lock IDs, this pass is idempotent
// and only assigns lock IDs to locks without an ID.

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIE/Transforms/AIECompileTrace.h"
#include "aie/Dialect/AIE/Transforms/AIEPasses.h"

#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"

namespace xilinx::AIE {
#define GEN_PASS_DEF_AIEASSIGNLOCKIDS
#include "aie/Dialect/AIE/Transforms/AIEPasses.h.inc"
} // namespace xilinx::AIE

#define DEBUG_TYPE "aie-assign-lock-ids"

using namespace mlir;
using namespace xilinx;
using namespace xilinx::AIE;

struct AIEAssignLockIDsPass
    : xilinx::AIE::impl::AIEAssignLockIDsBase<AIEAssignLockIDsPass> {
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<func::FuncDialect>();
    registry.insert<AIEDialect>();
  }

  void runOnOperation() override {
    DeviceOp device = getOperation();
    OpBuilder rewriter = OpBuilder::atBlockTerminator(device.getBody());

    // All of the lock ops on a tile, separated into ops which have been
    // assigned to a lock, and ops which have not.
    struct TileLockOps {
      DenseSet<int> assigned;
      SmallVector<LockOp> unassigned;
    };

    DenseMap<TileOp, TileLockOps> tileToLocks;

    // Construct data structure storing locks by tile.
    device.walk<WalkOrder::PreOrder>([&](LockOp lockOp) {
      TileOp tileOp = lockOp.getTileOp();
      if (lockOp.getLockID().has_value()) {
        auto lockID = lockOp.getLockID().value();
        auto iter = tileToLocks.find(tileOp);
        if (iter == tileToLocks.end())
          tileToLocks.insert({tileOp, {{lockID}, /* unassigned = */ {}}});
        else {
          if (iter->second.assigned.find(lockID) !=
              iter->second.assigned.end()) {
            auto diag = lockOp->emitOpError("is assigned to the same lock (")
                        << lockID << ") as another op.";
            diag.attachNote(tileOp.getLoc())
                << "tile has lock ops assigned to same lock.";
            return signalPassFailure();
          }
          iter->second.assigned.insert(lockID);
        }
      } else {
        auto iter = tileToLocks.find(tileOp);
        if (iter == tileToLocks.end())
          tileToLocks.insert({tileOp, {/* assigned = */ {}, {lockOp}}});
        else
          iter->second.unassigned.push_back(lockOp);
      }
    });

    // IR mutation: assign locks to all unassigned lock ops.
    for (auto [tileOp, locks] : tileToLocks) {
      const auto locksPerTile =
          getTargetModel(tileOp).getNumLocks(tileOp.getCol(), tileOp.getRow());
      uint32_t nextID = 0;
      for (auto lockOp : locks.unassigned) {
        while (nextID < locksPerTile &&
               (locks.assigned.find(nextID) != locks.assigned.end())) {
          ++nextID;
        }
        if (nextID == locksPerTile) {
          mlir::InFlightDiagnostic diag =
              lockOp->emitOpError("not allocated a lock.");
          diag.attachNote(tileOp.getLoc()) << "because only " << locksPerTile
                                           << " locks available in this tile.";
          return signalPassFailure();
        }
        lockOp.setLockIDAttr(rewriter.getI32IntegerAttr(nextID));
        ++nextID;
      }
    }

    // Each tile's locks in id order: [id, name, assigned here].
    AIE_TRACE("alloc.locks", [&](llvm::json::OStream &j) {
      j.attributeArray("tiles", [&] {
        for (auto &[tileOp, locks] : tileToLocks) {
          SmallVector<std::tuple<int, std::string, bool>> rows;
          device.walk([&](LockOp lockOp) {
            if (lockOp.getTileOp() != tileOp || !lockOp.getLockID())
              return;
            std::string name;
            if (auto sym = lockOp.getSymName())
              name = sym->str();
            rows.push_back({*lockOp.getLockID(), name,
                            !locks.assigned.contains(*lockOp.getLockID())});
          });
          llvm::sort(rows);
          j.object([&] {
            j.attribute("tile",
                        llvm::json::Array{tileOp.getCol(), tileOp.getRow()});
            j.attribute("limit", getTargetModel(tileOp).getNumLocks(
                                     tileOp.getCol(), tileOp.getRow()));
            j.attributeArray("locks", [&] {
              for (auto &[id, name, fresh] : rows)
                j.value(llvm::json::Array{id, name, fresh});
            });
          });
        }
      });
    });
  }
};

std::unique_ptr<OperationPass<DeviceOp>> AIE::createAIEAssignLockIDsPass() {
  return std::make_unique<AIEAssignLockIDsPass>();
}
