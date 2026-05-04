#
# This file is licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# (c) Copyright 2026 Advanced Micro Devices, Inc.

add_mlir_dialect_library(ConduitTransforms
  PARTIAL_SOURCES_INTENDED
  ConduitAppendCoreSpin.cpp
  ConduitCombineDevicePass.cpp
  ConduitCheckChannels.cpp
  ConduitCheckDeps.cpp
  CanonicalizeChannelPuts.cpp
  CanonicalizeChannelPutsUtils.cpp
  ChannelPutsExpand.cpp
  patterns/HomogeneousRepeatPattern.cpp
  patterns/ArithProgressionPattern.cpp
  ConduitCheckLoopBalance.cpp
  ConduitCheckOrdering.cpp
  ConduitCheckTiers.cpp
  ConduitDmaTaskToConduit.cpp
  ConduitDepthPromotion.cpp
  ConduitFuseChannels.cpp
  ConduitFuseCoreBodyPass.cpp
  ConduitFuseRelay.cpp
  ConduitFuseOperators.cpp
  DeviceMergeUtils.cpp
  ConduitInferModes.cpp
  ConduitInferRates.cpp
  ConduitLivenessCheck.cpp
  LoopAnalysisUtils.cpp
  ConduitPairingCheck.cpp
  ConduitPruneRuntimeSeqArgs.cpp
  ConduitResourceModel.cpp
  ConduitTileInference.cpp
  ConduitToDMAAlloc.cpp
  ConduitToDMACommon.cpp
  ConduitToDMACollect.cpp
  ConduitToDMALink.cpp
  ConduitToDMALower.cpp
  ConduitToDMAPass.cpp
  ConduitToDMARoute.cpp
  ObjectFifoToConduit.cpp

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
  MLIRAIEXIncGen

  LINK_LIBS PUBLIC
  Conduit
  AIE
  AIEX
  MLIRIR
  MLIRPass
  MLIRSupport
  MLIRTransformUtils
  MLIRFuncDialect
  MLIRSCFDialect
)
