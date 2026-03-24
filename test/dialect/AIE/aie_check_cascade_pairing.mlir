//===- aie_check_cascade_pairing.mlir --------------------------*- MLIR -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// (c) Copyright 2026 Advanced Micro Devices, Inc.
//
//===----------------------------------------------------------------------===//
//
// Tests for --aie-check-cascade-pairing.
//
// Four test cases (split by // -----):
//   1. Valid: cascade_flow + put_cascade in source + get_cascade in dest — no error.
//   2. Negative: cascade_flow present, put_cascade in source, NO get_cascade in dest.
//   3. Negative: cascade_flow present, get_cascade in dest, NO put_cascade in source.
//   4. Negative: put_cascade in a core body with NO corresponding cascade_flow.
//
//===----------------------------------------------------------------------===//

// RUN: aie-opt --aie-check-cascade-pairing --verify-diagnostics %s

// -----

// TEST 1: valid — cascade_flow + put_cascade in source core + get_cascade in
// dest core.  No errors expected.

module @valid_cascade_pairing {
  aie.device(xcve2802) {
    %t13 = aie.tile(1, 3)
    %t23 = aie.tile(2, 3)
    aie.cascade_flow(%t13, %t23)
    %c13 = aie.core(%t13) {
      %val = arith.constant dense<0> : vector<16xi32>
      aie.put_cascade(%val : vector<16xi32>)
      aie.end
    }
    %c23 = aie.core(%t23) {
      %val = aie.get_cascade() : vector<16xi32>
      aie.end
    }
  }
}

// -----

// TEST 2: negative — cascade_flow present, put_cascade in source core, but NO
// get_cascade in dest core.  Expected error on the cascade_flow op.

module @missing_get_cascade {
  aie.device(xcve2802) {
    %t13 = aie.tile(1, 3)
    %t23 = aie.tile(2, 3)
    // expected-error@+1 {{'aie.cascade_flow' dest tile has no 'aie.get_cascade' in its core body}}
    aie.cascade_flow(%t13, %t23)
    %c13 = aie.core(%t13) {
      %val = arith.constant dense<0> : vector<16xi32>
      aie.put_cascade(%val : vector<16xi32>)
      aie.end
    }
    // Dest core intentionally has NO get_cascade.
    %c23 = aie.core(%t23) {
      aie.end
    }
  }
}

// -----

// TEST 3: negative — cascade_flow present, get_cascade in dest core, but NO
// put_cascade in source core.  Expected error on the cascade_flow op.

module @missing_put_cascade {
  aie.device(xcve2802) {
    %t13 = aie.tile(1, 3)
    %t23 = aie.tile(2, 3)
    // expected-error@+1 {{'aie.cascade_flow' source tile has no 'aie.put_cascade' in its core body}}
    aie.cascade_flow(%t13, %t23)
    // Source core intentionally has NO put_cascade.
    %c13 = aie.core(%t13) {
      aie.end
    }
    %c23 = aie.core(%t23) {
      %val = aie.get_cascade() : vector<16xi32>
      aie.end
    }
  }
}

// -----

// TEST 4: negative — put_cascade in a core body with NO corresponding
// cascade_flow naming that tile as source.  Expected error on the put_cascade.

module @orphan_put_cascade {
  aie.device(xcve2802) {
    %t13 = aie.tile(1, 3)
    // No aie.cascade_flow references %t13 as source.
    %c13 = aie.core(%t13) {
      %val = arith.constant dense<0> : vector<16xi32>
      // expected-error@+1 {{'aie.put_cascade' in core body has no corresponding 'aie.cascade_flow' naming this tile as source}}
      aie.put_cascade(%val : vector<16xi32>)
      aie.end
    }
  }
}
