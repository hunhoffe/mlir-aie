//===- TraceElfs.h - Each core's ELF in the compile trace ----*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Copyright (C) 2026, Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//
//
// `core.elf` events: what went into each core's program and data memory, read
// from the linked ELF, with each symbol traced back to the object that defined
// it (a kernel's .o, or the core's own compiled code).
//
//===----------------------------------------------------------------------===//

#ifndef AIECC_TRACE_ELFS_H
#define AIECC_TRACE_ELFS_H

#include "IRTransforms.h"
#include "StackSizeAnalysis.h"

#include "aie/Dialect/AIE/IR/AIEDialect.h"
#include "aie/Dialect/AIE/Transforms/AIECompileTrace.h"

#include "llvm/ADT/StringMap.h"
#include "llvm/Object/ELFObjectFile.h"
#include "llvm/Object/ObjectFile.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/Path.h"

#include <string>
#include <vector>

namespace xilinx::aiecc {

// The symbols an object defines, by name, for attributing a linked symbol.
inline void collectDefinedSymbols(llvm::StringRef path, int index,
                                  llvm::StringMap<int> &definedBy) {
  auto obj = llvm::object::ObjectFile::createObjectFile(path);
  if (!obj) {
    llvm::consumeError(obj.takeError());
    return;
  }
  for (const llvm::object::SymbolRef &sym : obj->getBinary()->symbols()) {
    auto flags = sym.getFlags();
    if (!flags || (*flags & llvm::object::SymbolRef::SF_Undefined))
      continue;
    auto name = sym.getName();
    if (!name) {
      llvm::consumeError(name.takeError());
      continue;
    }
    definedBy.try_emplace(*name, index);
  }
}

inline void traceCoreElfs(
    mlir::ModuleOp module,
    llvm::function_ref<std::string(xilinx::AIE::CoreOp)> elfForCore,
    llvm::function_ref<std::string(xilinx::AIE::CoreOp)> objectForCore,
    llvm::function_ref<std::string(llvm::StringRef)> resolve) {
  namespace json = llvm::json;
  using namespace llvm::object;
  if (!xilinx::AIE::trace::enabled())
    return;
  for (auto device : module.getOps<xilinx::AIE::DeviceOp>()) {
    llvm::StringMap<int64_t> overrides = collectStackSizeOverrides(device);
    device.walk([&](xilinx::AIE::CoreOp core) {
      std::string elf = elfForCore(core);
      if (elf.empty())
        return;
      auto bin = ObjectFile::createObjectFile(elf);
      if (!bin) {
        llvm::consumeError(bin.takeError());
        return;
      }
      auto *obj = llvm::dyn_cast<ELFObjectFileBase>(bin->getBinary());
      if (!obj)
        return;
      // What was linked: the core's own object first, then its link files.
      std::vector<std::string> links;
      if (std::string own = objectForCore(core); !own.empty())
        links.push_back(own);
      if (auto files = core.getLinkFiles())
        for (auto f : files->getAsValueRange<mlir::StringAttr>())
          links.push_back(resolve(f));
      else if (auto with = core.getLinkWith())
        links.push_back(resolve(*with));
      llvm::StringMap<int> definedBy;
      for (auto [i, path] : llvm::enumerate(links))
        collectDefinedSymbols(path, i, definedBy);

      auto tile =
          mlir::cast<xilinx::AIE::TileOp>(core.getTile().getDefiningOp());
      auto stack = computeStackRequirement(elf, overrides);
      if (!stack.bytes)
        stack.bytes = std::nullopt;
      AIE_TRACE("core.elf", [&](json::OStream &j) {
        j.attribute("dev", device.getSymName());
        j.attribute("core", json::Array{tile.getCol(), tile.getRow()});
        j.attribute("key", coreKey(core));
        j.attribute("elf", llvm::sys::path::filename(elf));
        j.attribute("pm",
                    xilinx::AIE::getTargetModel(core).getProgramMemorySize());
        j.attribute("stack", core.getEffectiveStackSize());
        if (stack.bytes)
          j.attribute("stackNeeded", *stack.bytes);
        j.attributeArray("links", [&] {
          for (auto &l : links)
            j.value(l);
        });
        // Allocated sections: [name, address, size, executable].
        j.attributeArray("sections", [&] {
          for (const ELFSectionRef &sec : obj->sections()) {
            if (!(sec.getFlags() & llvm::ELF::SHF_ALLOC) || !sec.getSize())
              continue;
            auto name = sec.getName();
            j.value(json::Array{name ? name->str() : std::string(),
                                static_cast<int64_t>(sec.getAddress()),
                                static_cast<int64_t>(sec.getSize()),
                                sec.isText()});
            if (!name)
              llvm::consumeError(name.takeError());
          }
        });
        // Sized functions and data: [name, address, size, function, the
        // index in `links` of the object that defined it, or -1].
        j.attributeArray("symbols", [&] {
          for (const ELFSymbolRef &sym : obj->symbols()) {
            uint8_t type = sym.getELFType();
            if ((type != llvm::ELF::STT_FUNC &&
                 type != llvm::ELF::STT_OBJECT) ||
                !sym.getSize())
              continue;
            auto name = sym.getName();
            auto addr = sym.getAddress();
            if (!name || !addr) {
              if (!name)
                llvm::consumeError(name.takeError());
              if (!addr)
                llvm::consumeError(addr.takeError());
              continue;
            }
            auto from = definedBy.find(*name);
            j.value(json::Array{name->str(), static_cast<int64_t>(*addr),
                                static_cast<int64_t>(sym.getSize()),
                                type == llvm::ELF::STT_FUNC,
                                from == definedBy.end() ? -1 : from->second});
          }
        });
      });
    });
  }
}

} // namespace xilinx::aiecc

#endif // AIECC_TRACE_ELFS_H
