//===- AIETargetCDODirect.cpp -----------------------------------*- C++ -*-===//
//
// Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aie/Targets/AIERT.h"
#include "aie/Targets/AIETargets.h"
extern "C" {
#include "cdo-driver/cdo_driver.h"
}

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIE/IR/AIEEnums.h"
#include "aie/Dialect/AIEX/IR/AIEXDialect.h"

#include "mlir/IR/Block.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Operation.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Support/LogicalResult.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Debug.h"

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <functional>
#include <string>
#include <vector>

// Forward-declare cdo_MaskWrite32 from
// third_party/bootgen/cdo-driver/cdo_driver.c. It is intentionally not exported
// via cdo_driver.h, but the symbol is linked in. Used below to inject a
// per-process-unique nonce into empty-device PDI bytes so that firmware-side
// PDI content caching is defeated (Bug C/D in CLAUDE.md: firmware caches PDIs
// by content and silently elides re-execution when the content is
// byte-identical, so the device-reset that the empty PDI is supposed to trigger
// never actually fires on the second LoadPDI).
extern "C" {
void cdo_MaskWrite32(uint64_t Addr, uint32_t Mask, uint32_t Data);
}

#ifndef NDEBUG
#define XAIE_DEBUG
#endif

extern "C" {
#include "xaiengine/xaie_elfloader.h"
#include "xaiengine/xaie_interrupt.h"
#include "xaiengine/xaiegbl.h"
}

#define DEBUG_TYPE "aie-generate-cdo"

using namespace mlir;
using namespace xilinx;
using namespace xilinx::AIE;

static void initializeCDOGenerator(byte_ordering endianness, bool cdoDebug) {
  // Enables AXI-MM prints for configs being added in CDO
  if (cdoDebug)
    EnAXIdebug();
  setEndianness(endianness);
};

static LogicalResult
generateCDOBinary(const StringRef outputPath,
                  const std::function<LogicalResult()> &cb) {
  startCDOFileStream(outputPath.str().c_str());
  FileHeader();
  // Never generate a completely empty CDO file.  If the file only contains a
  // header, then bootgen flags it as invalid.
  insertNoOpCommand(4);
  if (failed(cb()))
    return failure();
  configureHeader();
  endCurrentCDOFileStream();
  return success();
}

// Cache-bust nonce for the empty-device PDI.
//
// Background (CLAUDE.md Bug C/D): firmware content-caches PDIs.  The empty
// device's PDI is meant to trigger a full device reset on every LoadPDI, but
// because the empty PDI bytes are identical across compiles (just header +
// NOP + footer), firmware sees a cache hit and elides re-execution -- the
// reset never fires on subsequent LoadPDIs, so multi-invocation state leaks
// across runs (Llama decode 2nd-token timeout; matrix 4col_med non-DMA
// numerical regressions).
//
// Fix: emit one extra cdo_MaskWrite32 with mask=0 into the empty PDI's
// bytestream.  The data field carries a per-process steady_clock nonce, so
// bytes-on-the-wire differ across compiles -> firmware cache miss -> reset
// actually fires.  Mask=0 makes the write functionally inert at the
// register level (no bits change), so no AIE state is touched.
//
// Address: 0x000340F0 = PL_MODULE_TIMER_TRIG_EVENT_LOW_VALUE on AIE2P/NPU2
// (per third_party/aie-rt/.../xaie2pgbl_params.h).  Read-side-effect-free
// scratch-equivalent at the shim tile (col 0 row 0) base.
//
// Per-process seed pattern from Q3 cache-bust commit bba6046899, plus a
// per-emit counter so that distinct empty devices within ONE compile (e.g.
// empty_0 and empty_1, alternated by AIEExpandLoadPdi) also differ in
// content -- not just across compiles.  Within-compile differentiation is
// what defeats the firmware cache for back-to-back LoadPdis in the same
// process (Bug C: Llama 2nd-decode timeout caused by empty_0 / empty_1
// alternation being name-only with byte-identical content).
static void emitEmptyDevicePdiCacheBustNonce() {
  static const uint32_t kProcessSeed = static_cast<uint32_t>(
      std::chrono::steady_clock::now().time_since_epoch().count());
  static uint32_t emitIndex = 0;
  // Mix seed and counter so neither across-compile (seed) nor within-compile
  // (counter) collisions can produce byte-identical PDIs.
  uint32_t nonce = kProcessSeed ^ (0x9E3779B9u * (++emitIndex));
  constexpr uint64_t kBenignAddr = 0x000340F0ULL;
  cdo_MaskWrite32(/*Addr=*/kBenignAddr, /*Mask=*/0, /*Data=*/nonce);
}

static LogicalResult
generateCDOBinariesSeparately(AIERTControl &ctl, const StringRef workDirPath,
                              DeviceOp &targetOp, bool aieSim, bool enableCores,
                              bool isEmptyDevice) {
  auto ps = std::filesystem::path::preferred_separator;

  LLVM_DEBUG(llvm::dbgs() << "Generating aie_cdo_elfs.bin");
  if (failed(generateCDOBinary((llvm::Twine(workDirPath) + std::string(1, ps) +
                                targetOp.getSymName() + "_aie_cdo_elfs.bin")
                                   .str(),
                               [&ctl, &targetOp, &workDirPath, &aieSim] {
                                 return ctl.addAieElfs(targetOp, workDirPath,
                                                       aieSim);
                               })))
    return failure();

  LLVM_DEBUG(llvm::dbgs() << "Generating aie_cdo_init.bin");
  // For empty devices, attach the cache-bust nonce to the init CDO (which is
  // the dominant byte-content section for non-empty devices too, so empty
  // devices' "init" file is the natural carrier).  See
  // emitEmptyDevicePdiCacheBustNonce comment for rationale.
  if (failed(generateCDOBinary((llvm::Twine(workDirPath) + std::string(1, ps) +
                                targetOp.getSymName() + "_aie_cdo_init.bin")
                                   .str(),
                               [&ctl, &targetOp, isEmptyDevice] {
                                 if (failed(ctl.addInitConfig(targetOp)))
                                   return failure();
                                 if (isEmptyDevice)
                                   emitEmptyDevicePdiCacheBustNonce();
                                 return success();
                               })))
    return failure();

  LLVM_DEBUG(llvm::dbgs() << "Generating aie_cdo_enable.bin");
  if (enableCores &&
      failed(generateCDOBinary(
          (llvm::Twine(workDirPath) + std::string(1, ps) +
           targetOp.getSymName() + "_aie_cdo_enable.bin")
              .str(),
          [&ctl, &targetOp] { return ctl.addCoreEnable(targetOp); })))
    return failure();

  return success();
}

static LogicalResult generateCDOUnified(AIERTControl &ctl,
                                        const StringRef workDirPath,
                                        DeviceOp &targetOp, bool aieSim,
                                        bool enableCores, bool isEmptyDevice) {
  auto ps = std::filesystem::path::preferred_separator;

  return generateCDOBinary(
      (llvm::Twine(workDirPath) + std::string(1, ps) + targetOp.getSymName() +
       "_aie_cdo.bin")
          .str(),
      [&ctl, &targetOp, &workDirPath, &aieSim, &enableCores, isEmptyDevice] {
        if (!targetOp.getOps<CoreOp>().empty() &&
            failed(ctl.addAieElfs(targetOp, workDirPath, aieSim)))
          return failure();
        if (failed(ctl.addInitConfig(targetOp)))
          return failure();
        if (enableCores && !targetOp.getOps<CoreOp>().empty() &&
            failed(ctl.addCoreEnable(targetOp)))
          return failure();
        if (isEmptyDevice)
          emitEmptyDevicePdiCacheBustNonce();
        return success();
      });
}

static LogicalResult
translateToCDODirect(ModuleOp m, llvm::StringRef workDirPath,
                     llvm::StringRef deviceName, byte_ordering endianness,
                     bool emitUnified, bool cdoDebug, bool aieSim,
                     bool xaieDebug, bool enableCores) {

  DeviceOp targetOp = AIE::DeviceOp::getForSymbolInModuleOrError(m, deviceName);
  if (!targetOp) {
    return failure();
  }
  const AIETargetModel &targetModel =
      (const AIETargetModel &)targetOp.getTargetModel();

  // things like XAIE_MEM_TILE_ROW_START and the missing
  // shim dma on tile (0,0) are hard-coded assumptions about NPU...
  assert(targetModel.hasProperty(AIETargetModel::IsNPU) &&
         "Only NPU currently supported");

  AIERTControl ctl(targetModel);
  if (failed(ctl.setIOBackend(aieSim, xaieDebug)))
    return failure();
  initializeCDOGenerator(endianness, cdoDebug);

  // Devices created by AIEExpandLoadPdi (named "empty_*") exist solely to
  // trigger a firmware-mediated device reset on LoadPDI.  Their PDI bytes are
  // otherwise byte-identical across compiles, which causes firmware PDI
  // content-caching to elide the reset.  See emitEmptyDevicePdiCacheBustNonce.
  bool isEmptyDevice = targetOp.getSymName().starts_with("empty_");

  auto result = [&]() {
    if (emitUnified) {
      return generateCDOUnified(ctl, workDirPath, targetOp, aieSim, enableCores,
                                isEmptyDevice);
    }
    return generateCDOBinariesSeparately(ctl, workDirPath, targetOp, aieSim,
                                         enableCores, isEmptyDevice);
  }();
  return result;
}

LogicalResult xilinx::AIE::AIETranslateToCDODirect(
    ModuleOp m, llvm::StringRef workDirPath, llvm::StringRef deviceName,
    bool bigEndian, bool emitUnified, bool cdoDebug, bool aieSim,
    bool xaieDebug, bool enableCores) {
  byte_ordering endianness =
      bigEndian ? byte_ordering::Big_Endian : byte_ordering::Little_Endian;
  return translateToCDODirect(m, workDirPath, deviceName, endianness,
                              emitUnified, cdoDebug, aieSim, xaieDebug,
                              enableCores);
}
