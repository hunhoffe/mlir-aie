//===- ConduitDialect.h -----------------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2026 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//
//
// This file declares the Conduit dialect, a portable bridge between mlir-aie
// objectfifo semantics and mlir-air channel semantics.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_CONDUIT_DIALECT_H
#define MLIR_CONDUIT_DIALECT_H

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/Types.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

// Include generated dialect declarations
#include "aie/Dialect/Conduit/IR/ConduitOpsDialect.h.inc"

// Include generated type class declarations
#define GET_TYPEDEF_CLASSES
#include "aie/Dialect/Conduit/IR/ConduitTypes.h.inc"

// Include generated op class declarations
#define GET_OP_CLASSES
#include "aie/Dialect/Conduit/IR/ConduitOps.h.inc"

#endif // MLIR_CONDUIT_DIALECT_H
