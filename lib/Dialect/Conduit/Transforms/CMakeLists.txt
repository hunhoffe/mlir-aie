#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2026 Advanced Micro Devices, Inc.

add_mlir_dialect_library(ConduitTransforms
  AirChannelToConduit.cpp
  ConduitDepthPromotion.cpp
  ConduitPairingCheck.cpp
  ObjectFifoToConduit.cpp
  ConduitToDMA.cpp

  ADDITIONAL_HEADER_DIRS
  ${AIE_BINARY_DIR}/include

  DEPENDS
  MLIRConduitPassIncGen
  MLIRConduitDialectIncGen
  MLIRConduitIncGen
  MLIRConduitTypesIncGen
  MLIRAIEAttrDefsIncGen
  MLIRAIEEnumsIncGen
  MLIRAIEIncGen
  MLIRAIETypesIncGen
  MLIRAIEInterfacesIncGen
  MLIRAIEOpsIncGen
  MLIRAIEPassIncGen

  LINK_LIBS PUBLIC
  Conduit
  AIE
  MLIRIR
  MLIRPass
  MLIRSupport
  MLIRTransformUtils
  MLIRFuncDialect
)
