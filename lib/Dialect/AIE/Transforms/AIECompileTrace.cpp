//===- AIECompileTrace.cpp --------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "aie/Dialect/AIE/Transforms/AIECompileTrace.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"

#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <mutex>

using namespace xilinx::AIE;

namespace {
struct Sink {
  std::mutex mu;
  std::unique_ptr<llvm::raw_fd_ostream> os;
  std::chrono::steady_clock::time_point start;
  std::atomic<bool> on{false};

  void openPath(llvm::StringRef path) {
    std::lock_guard<std::mutex> lock(mu);
    on = false;
    os.reset();
    if (path.empty())
      return;
    std::error_code ec;
    os = std::make_unique<llvm::raw_fd_ostream>(path, ec,
                                                llvm::sys::fs::OF_Text);
    if (ec) {
      llvm::errs() << "aie: cannot write compile trace '" << path
                   << "': " << ec.message() << "\n";
      os.reset();
      return;
    }
    start = std::chrono::steady_clock::now();
    on = true;
  }
};

Sink &sink() {
  static Sink *s = [] {
    auto *s = new Sink;
    if (const char *path = std::getenv("AIE_COMPILE_TRACE"))
      s->openPath(path);
    return s;
  }();
  return *s;
}
} // namespace

static thread_local std::string currentStage;

void trace::setStage(llvm::StringRef stage) { currentStage = stage.str(); }

bool trace::enabled() { return sink().on.load(std::memory_order_relaxed); }

void trace::open(llvm::StringRef path) { sink().openPath(path); }

void trace::emit(llvm::StringRef ev, const char *file, int line,
                 llvm::function_ref<void(llvm::json::OStream &)> fields) {
  Sink &s = sink();
  // Format outside the lock; events from parallel passes interleave by line.
  std::string buf;
  llvm::raw_string_ostream bufOs(buf);
  {
    llvm::json::OStream j(bufOs);
    j.object([&] {
      double ms = std::chrono::duration<double, std::milli>(
                      std::chrono::steady_clock::now() - s.start)
                      .count();
      j.attributeBegin("t");
      j.rawValue(llvm::formatv("{0:f3}", ms).str());
      j.attributeEnd();
      j.attribute("ev", ev);
      j.attribute(
          "src",
          (llvm::sys::path::filename(file) + ":" + llvm::Twine(line)).str());
      if (!currentStage.empty())
        j.attribute("stage", currentStage);
      fields(j);
    });
  }
  buf.push_back('\n');
  std::lock_guard<std::mutex> lock(s.mu);
  if (s.os) {
    *s.os << buf;
    s.os->flush();
  }
}

void trace::attributeNumber(llvm::json::OStream &j, llvm::StringRef key,
                            double value, int digits) {
  if (!std::isfinite(value)) {
    j.attribute(key, nullptr);
    return;
  }
  char buf[32];
  std::snprintf(buf, sizeof(buf), "%.*g", digits, value);
  j.attributeBegin(key);
  j.rawValue(buf);
  j.attributeEnd();
}

std::string trace::locString(mlir::Location loc) {
  std::string out;
  loc->walk([&](mlir::Location l) {
    if (auto flc = llvm::dyn_cast<mlir::FileLineColLoc>(l)) {
      out = (flc.getFilename().getValue() + ":" + llvm::Twine(flc.getLine()) +
             ":" + llvm::Twine(flc.getColumn()))
                .str();
      return mlir::WalkResult::interrupt();
    }
    return mlir::WalkResult::advance();
  });
  return out;
}
