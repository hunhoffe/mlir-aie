#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2026 Advanced Micro Devices, Inc.

set(LLVM_TARGET_DEFINITIONS Passes.td)
mlir_tablegen(ConduitPasses.h.inc -gen-pass-decls -name Conduit)
add_public_tablegen_target(MLIRConduitPassIncGen)

add_mlir_doc(Passes ConduitPasses ./ -gen-pass-doc)
