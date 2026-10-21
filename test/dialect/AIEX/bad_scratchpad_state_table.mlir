//===- bad_scratchpad_state_table.mlir --------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Slots go in declaration order, each the lowest no other parameter of a
// command reading it holds. Every command here reads 32, but @c's 31 take
// 0-30, so @y, read by @b and @c, takes 31; @b's 30 take 0-29 and @a's 31 take
// 0-30. @x, read by @a and @b, then finds all 32 held.
//
//===----------------------------------------------------------------------===//

// RUN: not aie-opt --aie-lower-scratchpad-parameters %s 2>&1 | FileCheck %s

// CHECK: error: parameter 'x' finds no free state-table entry among 32
module {
  aiex.scratchpad_parameter @c0 : i32
  aiex.scratchpad_parameter @c1 : i32
  aiex.scratchpad_parameter @c2 : i32
  aiex.scratchpad_parameter @c3 : i32
  aiex.scratchpad_parameter @c4 : i32
  aiex.scratchpad_parameter @c5 : i32
  aiex.scratchpad_parameter @c6 : i32
  aiex.scratchpad_parameter @c7 : i32
  aiex.scratchpad_parameter @c8 : i32
  aiex.scratchpad_parameter @c9 : i32
  aiex.scratchpad_parameter @c10 : i32
  aiex.scratchpad_parameter @c11 : i32
  aiex.scratchpad_parameter @c12 : i32
  aiex.scratchpad_parameter @c13 : i32
  aiex.scratchpad_parameter @c14 : i32
  aiex.scratchpad_parameter @c15 : i32
  aiex.scratchpad_parameter @c16 : i32
  aiex.scratchpad_parameter @c17 : i32
  aiex.scratchpad_parameter @c18 : i32
  aiex.scratchpad_parameter @c19 : i32
  aiex.scratchpad_parameter @c20 : i32
  aiex.scratchpad_parameter @c21 : i32
  aiex.scratchpad_parameter @c22 : i32
  aiex.scratchpad_parameter @c23 : i32
  aiex.scratchpad_parameter @c24 : i32
  aiex.scratchpad_parameter @c25 : i32
  aiex.scratchpad_parameter @c26 : i32
  aiex.scratchpad_parameter @c27 : i32
  aiex.scratchpad_parameter @c28 : i32
  aiex.scratchpad_parameter @c29 : i32
  aiex.scratchpad_parameter @c30 : i32
  aiex.scratchpad_parameter @y : i32
  aiex.scratchpad_parameter @b0 : i32
  aiex.scratchpad_parameter @b1 : i32
  aiex.scratchpad_parameter @b2 : i32
  aiex.scratchpad_parameter @b3 : i32
  aiex.scratchpad_parameter @b4 : i32
  aiex.scratchpad_parameter @b5 : i32
  aiex.scratchpad_parameter @b6 : i32
  aiex.scratchpad_parameter @b7 : i32
  aiex.scratchpad_parameter @b8 : i32
  aiex.scratchpad_parameter @b9 : i32
  aiex.scratchpad_parameter @b10 : i32
  aiex.scratchpad_parameter @b11 : i32
  aiex.scratchpad_parameter @b12 : i32
  aiex.scratchpad_parameter @b13 : i32
  aiex.scratchpad_parameter @b14 : i32
  aiex.scratchpad_parameter @b15 : i32
  aiex.scratchpad_parameter @b16 : i32
  aiex.scratchpad_parameter @b17 : i32
  aiex.scratchpad_parameter @b18 : i32
  aiex.scratchpad_parameter @b19 : i32
  aiex.scratchpad_parameter @b20 : i32
  aiex.scratchpad_parameter @b21 : i32
  aiex.scratchpad_parameter @b22 : i32
  aiex.scratchpad_parameter @b23 : i32
  aiex.scratchpad_parameter @b24 : i32
  aiex.scratchpad_parameter @b25 : i32
  aiex.scratchpad_parameter @b26 : i32
  aiex.scratchpad_parameter @b27 : i32
  aiex.scratchpad_parameter @b28 : i32
  aiex.scratchpad_parameter @b29 : i32
  aiex.scratchpad_parameter @a0 : i32
  aiex.scratchpad_parameter @a1 : i32
  aiex.scratchpad_parameter @a2 : i32
  aiex.scratchpad_parameter @a3 : i32
  aiex.scratchpad_parameter @a4 : i32
  aiex.scratchpad_parameter @a5 : i32
  aiex.scratchpad_parameter @a6 : i32
  aiex.scratchpad_parameter @a7 : i32
  aiex.scratchpad_parameter @a8 : i32
  aiex.scratchpad_parameter @a9 : i32
  aiex.scratchpad_parameter @a10 : i32
  aiex.scratchpad_parameter @a11 : i32
  aiex.scratchpad_parameter @a12 : i32
  aiex.scratchpad_parameter @a13 : i32
  aiex.scratchpad_parameter @a14 : i32
  aiex.scratchpad_parameter @a15 : i32
  aiex.scratchpad_parameter @a16 : i32
  aiex.scratchpad_parameter @a17 : i32
  aiex.scratchpad_parameter @a18 : i32
  aiex.scratchpad_parameter @a19 : i32
  aiex.scratchpad_parameter @a20 : i32
  aiex.scratchpad_parameter @a21 : i32
  aiex.scratchpad_parameter @a22 : i32
  aiex.scratchpad_parameter @a23 : i32
  aiex.scratchpad_parameter @a24 : i32
  aiex.scratchpad_parameter @a25 : i32
  aiex.scratchpad_parameter @a26 : i32
  aiex.scratchpad_parameter @a27 : i32
  aiex.scratchpad_parameter @a28 : i32
  aiex.scratchpad_parameter @a29 : i32
  aiex.scratchpad_parameter @a30 : i32
  aiex.scratchpad_parameter @x : i32
  aie.device(npu2) @a {
    %t02 = aie.tile(0, 2)
    aie.core(%t02) {
      %a0 = aiex.read_scratchpad_parameter @a0 : i32
      %a1 = aiex.read_scratchpad_parameter @a1 : i32
      %a2 = aiex.read_scratchpad_parameter @a2 : i32
      %a3 = aiex.read_scratchpad_parameter @a3 : i32
      %a4 = aiex.read_scratchpad_parameter @a4 : i32
      %a5 = aiex.read_scratchpad_parameter @a5 : i32
      %a6 = aiex.read_scratchpad_parameter @a6 : i32
      %a7 = aiex.read_scratchpad_parameter @a7 : i32
      %a8 = aiex.read_scratchpad_parameter @a8 : i32
      %a9 = aiex.read_scratchpad_parameter @a9 : i32
      %a10 = aiex.read_scratchpad_parameter @a10 : i32
      %a11 = aiex.read_scratchpad_parameter @a11 : i32
      %a12 = aiex.read_scratchpad_parameter @a12 : i32
      %a13 = aiex.read_scratchpad_parameter @a13 : i32
      %a14 = aiex.read_scratchpad_parameter @a14 : i32
      %a15 = aiex.read_scratchpad_parameter @a15 : i32
      %a16 = aiex.read_scratchpad_parameter @a16 : i32
      %a17 = aiex.read_scratchpad_parameter @a17 : i32
      %a18 = aiex.read_scratchpad_parameter @a18 : i32
      %a19 = aiex.read_scratchpad_parameter @a19 : i32
      %a20 = aiex.read_scratchpad_parameter @a20 : i32
      %a21 = aiex.read_scratchpad_parameter @a21 : i32
      %a22 = aiex.read_scratchpad_parameter @a22 : i32
      %a23 = aiex.read_scratchpad_parameter @a23 : i32
      %a24 = aiex.read_scratchpad_parameter @a24 : i32
      %a25 = aiex.read_scratchpad_parameter @a25 : i32
      %a26 = aiex.read_scratchpad_parameter @a26 : i32
      %a27 = aiex.read_scratchpad_parameter @a27 : i32
      %a28 = aiex.read_scratchpad_parameter @a28 : i32
      %a29 = aiex.read_scratchpad_parameter @a29 : i32
      %a30 = aiex.read_scratchpad_parameter @a30 : i32
      %x = aiex.read_scratchpad_parameter @x : i32
      aie.end
    }
    aie.runtime_sequence @seq() {
    }
  }
  aie.device(npu2) @b {
    %t02 = aie.tile(0, 2)
    aie.core(%t02) {
      %b0 = aiex.read_scratchpad_parameter @b0 : i32
      %b1 = aiex.read_scratchpad_parameter @b1 : i32
      %b2 = aiex.read_scratchpad_parameter @b2 : i32
      %b3 = aiex.read_scratchpad_parameter @b3 : i32
      %b4 = aiex.read_scratchpad_parameter @b4 : i32
      %b5 = aiex.read_scratchpad_parameter @b5 : i32
      %b6 = aiex.read_scratchpad_parameter @b6 : i32
      %b7 = aiex.read_scratchpad_parameter @b7 : i32
      %b8 = aiex.read_scratchpad_parameter @b8 : i32
      %b9 = aiex.read_scratchpad_parameter @b9 : i32
      %b10 = aiex.read_scratchpad_parameter @b10 : i32
      %b11 = aiex.read_scratchpad_parameter @b11 : i32
      %b12 = aiex.read_scratchpad_parameter @b12 : i32
      %b13 = aiex.read_scratchpad_parameter @b13 : i32
      %b14 = aiex.read_scratchpad_parameter @b14 : i32
      %b15 = aiex.read_scratchpad_parameter @b15 : i32
      %b16 = aiex.read_scratchpad_parameter @b16 : i32
      %b17 = aiex.read_scratchpad_parameter @b17 : i32
      %b18 = aiex.read_scratchpad_parameter @b18 : i32
      %b19 = aiex.read_scratchpad_parameter @b19 : i32
      %b20 = aiex.read_scratchpad_parameter @b20 : i32
      %b21 = aiex.read_scratchpad_parameter @b21 : i32
      %b22 = aiex.read_scratchpad_parameter @b22 : i32
      %b23 = aiex.read_scratchpad_parameter @b23 : i32
      %b24 = aiex.read_scratchpad_parameter @b24 : i32
      %b25 = aiex.read_scratchpad_parameter @b25 : i32
      %b26 = aiex.read_scratchpad_parameter @b26 : i32
      %b27 = aiex.read_scratchpad_parameter @b27 : i32
      %b28 = aiex.read_scratchpad_parameter @b28 : i32
      %b29 = aiex.read_scratchpad_parameter @b29 : i32
      %y = aiex.read_scratchpad_parameter @y : i32
      %x = aiex.read_scratchpad_parameter @x : i32
      aie.end
    }
    aie.runtime_sequence @seq() {
    }
  }
  aie.device(npu2) @c {
    %t02 = aie.tile(0, 2)
    aie.core(%t02) {
      %c0 = aiex.read_scratchpad_parameter @c0 : i32
      %c1 = aiex.read_scratchpad_parameter @c1 : i32
      %c2 = aiex.read_scratchpad_parameter @c2 : i32
      %c3 = aiex.read_scratchpad_parameter @c3 : i32
      %c4 = aiex.read_scratchpad_parameter @c4 : i32
      %c5 = aiex.read_scratchpad_parameter @c5 : i32
      %c6 = aiex.read_scratchpad_parameter @c6 : i32
      %c7 = aiex.read_scratchpad_parameter @c7 : i32
      %c8 = aiex.read_scratchpad_parameter @c8 : i32
      %c9 = aiex.read_scratchpad_parameter @c9 : i32
      %c10 = aiex.read_scratchpad_parameter @c10 : i32
      %c11 = aiex.read_scratchpad_parameter @c11 : i32
      %c12 = aiex.read_scratchpad_parameter @c12 : i32
      %c13 = aiex.read_scratchpad_parameter @c13 : i32
      %c14 = aiex.read_scratchpad_parameter @c14 : i32
      %c15 = aiex.read_scratchpad_parameter @c15 : i32
      %c16 = aiex.read_scratchpad_parameter @c16 : i32
      %c17 = aiex.read_scratchpad_parameter @c17 : i32
      %c18 = aiex.read_scratchpad_parameter @c18 : i32
      %c19 = aiex.read_scratchpad_parameter @c19 : i32
      %c20 = aiex.read_scratchpad_parameter @c20 : i32
      %c21 = aiex.read_scratchpad_parameter @c21 : i32
      %c22 = aiex.read_scratchpad_parameter @c22 : i32
      %c23 = aiex.read_scratchpad_parameter @c23 : i32
      %c24 = aiex.read_scratchpad_parameter @c24 : i32
      %c25 = aiex.read_scratchpad_parameter @c25 : i32
      %c26 = aiex.read_scratchpad_parameter @c26 : i32
      %c27 = aiex.read_scratchpad_parameter @c27 : i32
      %c28 = aiex.read_scratchpad_parameter @c28 : i32
      %c29 = aiex.read_scratchpad_parameter @c29 : i32
      %c30 = aiex.read_scratchpad_parameter @c30 : i32
      %y = aiex.read_scratchpad_parameter @y : i32
      aie.end
    }
    aie.runtime_sequence @seq() {
    }
  }
}
