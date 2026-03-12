#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2026 Advanced Micro Devices, Inc.

add_mlir_dialect(Conduit conduit)
add_mlir_doc(Conduit ConduitDialect ./ -gen-dialect-doc -dialect=conduit)
set(LLVM_TARGET_DEFINITIONS Conduit.td)
mlir_tablegen(ConduitOps.cpp.inc -gen-op-defs)
mlir_tablegen(ConduitOps.h.inc -gen-op-decls)
add_public_tablegen_target(MLIRConduitIncGen)
