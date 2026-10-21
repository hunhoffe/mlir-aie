/*
Copyright (C) 2014-2022 Xilinx, Inc.
Copyright (C) 2022-2026 Advanced Micro Devices, Inc.
    SPDX-License-Identifier: MIT
*/

#ifndef AIE_KERNELS_COMMON_NARROW_H
#define AIE_KERNELS_COMMON_NARROW_H

#include <aie_api/aie.hpp>

// Narrows N floats to bf16 through an accfloat accumulator. The attention
// kernels narrow in place, with the same buffer as source and destination: the
// bf16 pointer advances at half the byte rate of the float one and so trails
// it. The pointers must not be `restrict`, because they may alias.
template <int N>
void narrow_to_bf16(bfloat16 *out, const float *in) {
  constexpr int vec_factor = 16;
  static_assert(N % vec_factor == 0);
  for (int i = 0; i < N / vec_factor; i++) {
    aie::accum<accfloat, vec_factor> acc;
    acc.from_vector(aie::load_v<vec_factor>(in));
    aie::store_v(out, acc.template to_vector<bfloat16>());
    in += vec_factor;
    out += vec_factor;
  }
}

#endif // AIE_KERNELS_COMMON_NARROW_H
