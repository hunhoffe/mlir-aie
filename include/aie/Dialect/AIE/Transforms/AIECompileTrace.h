//===- AIECompileTrace.h ----------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A structured record of what the compiler decided and why, as JSON lines,
// for tools that replay a compile (placement moves, routing iterations,
// allocation decisions). Off unless a path is set, by aiecc --compile-trace or
// the AIE_COMPILE_TRACE environment variable; while off, an event costs one
// branch. Unlike LLVM_DEBUG it is in release builds and is meant to
// be parsed, so an event's fields are a stable interface: add fields freely,
// rename or drop them only with a version bump (kVersion).
//
// Each line is one object: {"t": ms since the trace opened, "ev": name,
// "src": "file:line" of the emitting code, "stage": the aiecc stage running
// it, if any, ...the event's own fields}.
//
//===----------------------------------------------------------------------===//

#ifndef AIE_COMPILE_TRACE_H
#define AIE_COMPILE_TRACE_H

#include "mlir/IR/Location.h"
#include "llvm/ADT/STLFunctionalExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/JSON.h"

#include <string>

namespace xilinx::AIE::trace {

constexpr int kVersion = 1;

/// Whether events are being recorded. Cheap: check it before building one.
bool enabled();

/// Record to `path` from now on (truncating it), or stop if empty. Takes
/// precedence over AIE_COMPILE_TRACE.
void open(llvm::StringRef path);

/// The aiecc stage this thread is running, which each event it writes names
/// (as "stage") until it is cleared with "".
void setStage(llvm::StringRef stage);

/// Write one event; `fields` adds the event's own members.
void emit(llvm::StringRef ev, const char *file, int line,
          llvm::function_ref<void(llvm::json::OStream &)> fields);

/// Write `key: value` with `digits` significant digits rather than the 17 a
/// double otherwise gets, or null if `value` is not finite.
void attributeNumber(llvm::json::OStream &j, llvm::StringRef key, double value,
                     int digits = 4);

/// The location a viewer can show for an op: the first file:line:col in
/// `loc`, looking through names, call sites and fusions, or "" if none.
std::string locString(mlir::Location loc);

} // namespace xilinx::AIE::trace

/// AIE_TRACE("place.iter", [&](llvm::json::OStream &j) { j.attribute(...); });
#define AIE_TRACE(EV, FIELDS)                                                  \
  do {                                                                         \
    if (::xilinx::AIE::trace::enabled())                                       \
      ::xilinx::AIE::trace::emit(EV, __FILE__, __LINE__, FIELDS);              \
  } while (false)

#endif // AIE_COMPILE_TRACE_H
