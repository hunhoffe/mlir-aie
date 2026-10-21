//===- ternary.cc -----------------------------------------------*- C++ -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// Ternary weights: each -1, 0 or +1 times its row's bf16 scale for every 128
// columns. A block is TERNARY_M_TILE rows by TERNARY_K_TILE columns: each
// row's 2-bit codes (0 is 0, 1 is +1, 2 is -1), 32 bytes per 128 columns, byte
// i holding columns i, i + 32, i + 64 and i + 96 at bits 0, 2, 4 and 6; then
// the scales, row by row. Products are signed copies of x, so a GEMV adds
// without multiplying until the group's scale.
#include "../aie_kernel_utils.h"
#include "../common/narrow.h"
#include "../common/zero.h"

#include <aie_api/aie.hpp>
#include <stdint.h>

#if !defined(TERNARY_M_TILE) || !defined(TERNARY_K_TILE)
#error "ternary.cc needs its geometry -D defined"
#endif

namespace {

constexpr int M_TILE = TERNARY_M_TILE;
constexpr int K_TILE = TERNARY_K_TILE;
constexpr int GROUP = 128;
constexpr int GROUPS = K_TILE / GROUP;
constexpr int CODE_BYTES = M_TILE * K_TILE / 4;

static_assert(K_TILE % GROUP == 0, "a k tile holds whole groups");
static_assert(M_TILE % 16 == 0, "the store narrows 16 rows at a time");

// The 32 values of v whose codes in byte lane j of packed are +1 and -1,
// signed, the rest zero.
inline aie::vector<bfloat16, 32> signed_by(aie::vector<uint8, 32> packed, int j,
                                           aie::vector<bfloat16, 32> v) {
  aie::vector<uint8, 32> code =
      aie::bit_and((uint8)3, aie::downshift(packed, 2 * j));
  aie::vector<bfloat16, 32> out =
      aie::select(aie::zeros<bfloat16, 32>(), v, aie::eq(code, (uint8)1));
  // -v by its sign bit: Peano has no bf16 vector negate.
  aie::vector<bfloat16, 32> neg = aie::vector_cast<bfloat16>(
      aie::bit_xor((uint16)0x8000, aie::vector_cast<uint16>(v)));
  return aie::select(out, neg, aie::eq(code, (uint8)2));
}

} // namespace

extern "C" {

/// Once per output tile, before the k loop.
void ternary_gemv_zero(float *y_acc) {
  event0();
  zero_vectorized<float, M_TILE, 1, false>(y_acc);
  event1();
}

/// One block: y_acc += w x[k]. k selects the slice of x.
void ternary_gemv_block(const uint8_t *w, const bfloat16 *x, float *y_acc,
                        int k) {
  event0();
  const bfloat16 *scales = (const bfloat16 *)(w + CODE_BYTES);
  x += k * K_TILE;
  for (int m = 0; m < M_TILE; m++) {
    float row = 0;
    for (int g = 0; g < GROUPS; g++) {
      aie::vector<uint8, 32> packed =
          aie::load_v<32>(w + (m * GROUPS + g) * 32);
      aie::accum<accfloat, 32> acc = aie::zeros<accfloat, 32>();
      for (int j = 0; j < 4; j++)
        acc = aie::add(
            acc, signed_by(packed, j, aie::load_v<32>(x + g * GROUP + j * 32)));
      row += (float)scales[m * GROUPS + g] *
             aie::reduce_add(acc.to_vector<float>());
    }
    y_acc[m] += row;
  }
  event1();
}

/// Once per output tile, after the k loop: y = y_acc in bf16.
void ternary_gemv_store(bfloat16 *y, const float *y_acc) {
  event0();
  narrow_to_bf16<M_TILE>(y, y_acc);
  event1();
}

/// One block as bf16 weights, (M_TILE, K_TILE) row-major: each a signed copy
/// of its group's scale, exact.
void ternary_dequant(const uint8_t *w, bfloat16 *out) {
  event0();
  const bfloat16 *scales = (const bfloat16 *)(w + CODE_BYTES);
  for (int m = 0; m < M_TILE; m++) {
    for (int g = 0; g < GROUPS; g++) {
      aie::vector<uint8, 32> packed =
          aie::load_v<32>(w + (m * GROUPS + g) * 32);
      aie::vector<bfloat16, 32> scale =
          aie::broadcast<bfloat16, 32>(scales[m * GROUPS + g]);
      for (int j = 0; j < 4; j++)
        aie::store_v(out + m * K_TILE + g * GROUP + j * 32,
                     signed_by(packed, j, scale));
    }
  }
  event1();
}
}
