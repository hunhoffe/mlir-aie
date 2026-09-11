// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

// Set the core's saturation mode register. An `srs` shift or a narrowing
// store clamps to the output range only when the register says so, and a
// fresh core boots with saturation off; a design calls this once, before the
// first kernel whose contract names the mode it needs
// (-DSATURATION_MODE=saturate binds `set_saturation_saturate`). See
// KernelContract.saturation_mode.

#include <aie_api/aie.hpp>

#ifndef SATURATION_MODE
#error Please specify the mode at compile time, e.g. -DSATURATION_MODE=saturate.
#endif

#define SET_SATURATION_CAT(a, b) a##b
#define SET_SATURATION_NAME(mode) SET_SATURATION_CAT(set_saturation_, mode)

extern "C" {
void SET_SATURATION_NAME(SATURATION_MODE)() {
  ::aie::set_saturation(aie::saturation_mode::SATURATION_MODE);
}
}
