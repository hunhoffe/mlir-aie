#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2026 Advanced Micro Devices, Inc.

add_mlir_doc(Conduit ConduitDialect ./ -gen-dialect-doc -dialect=conduit)

# Dialect declaration and definition
set(LLVM_TARGET_DEFINITIONS Conduit.td)
mlir_tablegen(ConduitOpsDialect.h.inc -gen-dialect-decls -dialect=conduit)
mlir_tablegen(ConduitOpsDialect.cpp.inc -gen-dialect-defs -dialect=conduit)
add_public_tablegen_target(MLIRConduitDialectIncGen)
add_dependencies(aie-headers MLIRConduitDialectIncGen)

# Op declarations and definitions
set(LLVM_TARGET_DEFINITIONS Conduit.td)
mlir_tablegen(ConduitOps.h.inc -gen-op-decls)
mlir_tablegen(ConduitOps.cpp.inc -gen-op-defs)
add_public_tablegen_target(MLIRConduitIncGen)
add_dependencies(aie-headers MLIRConduitIncGen)

# Type declarations and definitions
set(LLVM_TARGET_DEFINITIONS Conduit.td)
mlir_tablegen(ConduitTypes.h.inc -gen-typedef-decls -typedefs-dialect=conduit)
mlir_tablegen(ConduitTypes.cpp.inc -gen-typedef-defs -typedefs-dialect=conduit)
add_public_tablegen_target(MLIRConduitTypesIncGen)
add_dependencies(aie-headers MLIRConduitTypesIncGen)

# Enum declarations and definitions (Port enum)
set(LLVM_TARGET_DEFINITIONS Conduit.td)
mlir_tablegen(ConduitEnums.h.inc -gen-enum-decls)
mlir_tablegen(ConduitEnums.cpp.inc -gen-enum-defs)
add_public_tablegen_target(MLIRConduitEnumsIncGen)
add_dependencies(aie-headers MLIRConduitEnumsIncGen)

# Attribute declarations and definitions (PortAttr)
set(LLVM_TARGET_DEFINITIONS Conduit.td)
mlir_tablegen(ConduitAttrDefs.h.inc -gen-attrdef-decls -attrdefs-dialect=conduit)
mlir_tablegen(ConduitAttrDefs.cpp.inc -gen-attrdef-defs -attrdefs-dialect=conduit)
add_public_tablegen_target(MLIRConduitAttrDefsIncGen)
add_dependencies(aie-headers MLIRConduitAttrDefsIncGen)
