// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

// Read the core's rounding and saturation mode registers into a tile, so a
// host can see what state a kernel ran in and what it left behind (the
// dirty-state sweep in aie.utils.kernel_harness calls this before and after
// the kernel under test). The codes are positions in aie.iron.kernels'
// `ROUNDING_MODES[2:]` / `SATURATION_MODES[2:]`, not the enums' numeric
// values, so the host never needs the AIE API's numbering. Word 2 is a marker
// that proves the probe ran into a poisoned tile; word 3 is reserved.

#include <aie_api/aie.hpp>
#include <stdint.h>

static inline int32_t rounding_code(aie::rounding_mode m) {
  switch (m) {
  case aie::rounding_mode::floor:
    return 0;
  case aie::rounding_mode::ceil:
    return 1;
  case aie::rounding_mode::positive_inf:
    return 2;
  case aie::rounding_mode::negative_inf:
    return 3;
  case aie::rounding_mode::symmetric_inf:
    return 4;
  case aie::rounding_mode::symmetric_zero:
    return 5;
  case aie::rounding_mode::conv_even:
    return 6;
  case aie::rounding_mode::conv_odd:
    return 7;
  default:
    return -1;
  }
}

static inline int32_t saturation_code(aie::saturation_mode m) {
  switch (m) {
  case aie::saturation_mode::none:
    return 0;
  case aie::saturation_mode::saturate:
    return 1;
  case aie::saturation_mode::symmetric:
    return 2;
  default:
    return -1;
  }
}

extern "C" {
void read_core_state(int32_t *out) {
  out[0] = rounding_code(::aie::get_rounding());
  out[1] = saturation_code(::aie::get_saturation());
  out[2] = 0x50524F42; // "PROB": the probe wrote this tile
  out[3] = 0;
}
}
