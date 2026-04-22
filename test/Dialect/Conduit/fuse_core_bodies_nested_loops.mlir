// RUN: aie-opt --conduit-fuse-core-bodies %s | FileCheck %s
//
// Test for --conduit-fuse-core-bodies with 2-level loop nesting.
//
// The bug: when both producer and consumer have 2-level nesting
// (outer num_invocations + inner tile loop), the consumer's inner for
// was cloned inside the producer's inner for, producing NxM iterations
// instead of the intended N.
//
// Positive test: matching outer (trip=2) + inner (trip=4) on both cores.
// After fusion: single core with outer (trip=2) x inner (trip=4) = 8 iters.
// Must NOT have a nested consumer for inside the producer's inner for
// (which would give 2 x 4 x 4 = 32 iterations).
//
// Negative test: mismatched outer trip counts -> no fusion.

// ===== Positive: matching 2-level nesting =====

// CHECK-LABEL: module @fuse_nested_matching

// The intermediate conduit should be erased:
// CHECK-NOT:   conduit.create @intermediate_nested

// Input and output conduits should survive:
// CHECK:       conduit.create @input_nested
// CHECK:       conduit.create @output_nested

// Only one aie.core should remain:
// CHECK:       aie.core
// The fused core must have a single outer scf.for with trip count 2
// (lb=0, ub=2, step=1):
// CHECK:       scf.for %{{.*}} = %c0{{.*}} to %c2{{.*}} step %c1
// Inside that, a single inner scf.for with trip count 4
// (lb=0, ub=4, step=1):
// CHECK:       scf.for %{{.*}} = %c0{{.*}} to %c4{{.*}} step %c1
// The consumer kernel call should be inside the inner for, not in a
// deeper nested for:
// CHECK:       func.call @consume_nested_kernel
// There must NOT be a second scf.for inside the inner for (the bug
// would clone the consumer's inner for here):
// CHECK-NOT:   scf.for
// CHECK:       aie.end
// No second core:
// CHECK-NOT:   aie.core

module @fuse_nested_matching {
  aie.device(npu2) {
    %tile_0_2 = aie.tile(0, 2)

    conduit.create @intermediate_nested {element_type = memref<128xbf16>, depth = 2 : i64}
    conduit.create @input_nested  {element_type = memref<128xbf16>, depth = 2 : i64}
    conduit.create @output_nested {element_type = memref<128xbf16>, depth = 2 : i64}

    func.func private @produce_nested_kernel(memref<128xbf16>, memref<128xbf16>)
    func.func private @consume_nested_kernel(memref<128xbf16>, memref<128xbf16>)

    // Core A: producer with 2-level nesting (outer=2, inner=4).
    aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %c4 = arith.constant 4 : index
      scf.for %i_outer = %c0 to %c2 step %c1 {
        scf.for %i_inner = %c0 to %c4 step %c1 {
          %in_win = conduit.acquire {name = @input_nested, count = 1 : i64,
                                     port = #conduit.port<Consume>}
                        : !conduit.window<memref<128xbf16>>
          %in_buf = conduit.subview_access %in_win {index = 0 : i64}
                        : !conduit.window<memref<128xbf16>> -> memref<128xbf16>

          %inter_win = conduit.acquire {name = @intermediate_nested, count = 1 : i64,
                                        port = #conduit.port<Produce>}
                           : !conduit.window<memref<128xbf16>>
          %inter_buf = conduit.subview_access %inter_win {index = 0 : i64}
                           : !conduit.window<memref<128xbf16>> -> memref<128xbf16>

          func.call @produce_nested_kernel(%in_buf, %inter_buf)
              : (memref<128xbf16>, memref<128xbf16>) -> ()

          conduit.release %inter_win {count = 1 : i64, port = #conduit.port<Produce>}
              : !conduit.window<memref<128xbf16>>
          conduit.release %in_win {count = 1 : i64, port = #conduit.port<Consume>}
              : !conduit.window<memref<128xbf16>>
        }
      }
      aie.end
    } {link_with = "producer_nested.o"}

    // Core B: consumer with matching 2-level nesting (outer=2, inner=4).
    aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      %c4 = arith.constant 4 : index
      scf.for %j_outer = %c0 to %c2 step %c1 {
        scf.for %j_inner = %c0 to %c4 step %c1 {
          %inter_win = conduit.acquire {name = @intermediate_nested, count = 1 : i64,
                                        port = #conduit.port<Consume>}
                           : !conduit.window<memref<128xbf16>>
          %inter_buf = conduit.subview_access %inter_win {index = 0 : i64}
                           : !conduit.window<memref<128xbf16>> -> memref<128xbf16>

          %out_win = conduit.acquire {name = @output_nested, count = 1 : i64,
                                      port = #conduit.port<Produce>}
                         : !conduit.window<memref<128xbf16>>
          %out_buf = conduit.subview_access %out_win {index = 0 : i64}
                         : !conduit.window<memref<128xbf16>> -> memref<128xbf16>

          func.call @consume_nested_kernel(%inter_buf, %out_buf)
              : (memref<128xbf16>, memref<128xbf16>) -> ()

          conduit.release %out_win {count = 1 : i64, port = #conduit.port<Produce>}
              : !conduit.window<memref<128xbf16>>
          conduit.release %inter_win {count = 1 : i64, port = #conduit.port<Consume>}
              : !conduit.window<memref<128xbf16>>
        }
      }
      aie.end
    } {link_with = "consumer_nested.o"}

    aie.end
  }
}

// ===== Negative: mismatched outer trip counts -> no fusion =====

// CHECK-LABEL: module @fuse_nested_mismatch

// Both conduits should survive (no fusion):
// CHECK:       conduit.create @intermediate_mm
// CHECK:       conduit.create @input_mm
// CHECK:       conduit.create @output_mm

// Both cores should survive:
// CHECK:       aie.core
// CHECK:       aie.core

module @fuse_nested_mismatch {
  aie.device(npu2) {
    %tile_0_2 = aie.tile(0, 2)

    conduit.create @intermediate_mm {element_type = memref<128xbf16>, depth = 2 : i64}
    conduit.create @input_mm  {element_type = memref<128xbf16>, depth = 2 : i64}
    conduit.create @output_mm {element_type = memref<128xbf16>, depth = 2 : i64}

    func.func private @produce_mm_kernel(memref<128xbf16>, memref<128xbf16>)
    func.func private @consume_mm_kernel(memref<128xbf16>, memref<128xbf16>)

    // Core A: producer with outer=2 trip count.
    aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c2 = arith.constant 2 : index
      scf.for %i = %c0 to %c2 step %c1 {
        %in_win = conduit.acquire {name = @input_mm, count = 1 : i64,
                                   port = #conduit.port<Consume>}
                      : !conduit.window<memref<128xbf16>>
        %in_buf = conduit.subview_access %in_win {index = 0 : i64}
                      : !conduit.window<memref<128xbf16>> -> memref<128xbf16>
        %inter_win = conduit.acquire {name = @intermediate_mm, count = 1 : i64,
                                      port = #conduit.port<Produce>}
                         : !conduit.window<memref<128xbf16>>
        %inter_buf = conduit.subview_access %inter_win {index = 0 : i64}
                         : !conduit.window<memref<128xbf16>> -> memref<128xbf16>
        func.call @produce_mm_kernel(%in_buf, %inter_buf)
            : (memref<128xbf16>, memref<128xbf16>) -> ()
        conduit.release %inter_win {count = 1 : i64, port = #conduit.port<Produce>}
            : !conduit.window<memref<128xbf16>>
        conduit.release %in_win {count = 1 : i64, port = #conduit.port<Consume>}
            : !conduit.window<memref<128xbf16>>
      }
      aie.end
    } {link_with = "producer_mm.o"}

    // Core B: consumer with outer=3 trip count (mismatch -> no fusion).
    aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c3 = arith.constant 3 : index
      scf.for %j = %c0 to %c3 step %c1 {
        %inter_win = conduit.acquire {name = @intermediate_mm, count = 1 : i64,
                                      port = #conduit.port<Consume>}
                         : !conduit.window<memref<128xbf16>>
        %inter_buf = conduit.subview_access %inter_win {index = 0 : i64}
                         : !conduit.window<memref<128xbf16>> -> memref<128xbf16>
        %out_win = conduit.acquire {name = @output_mm, count = 1 : i64,
                                    port = #conduit.port<Produce>}
                       : !conduit.window<memref<128xbf16>>
        %out_buf = conduit.subview_access %out_win {index = 0 : i64}
                       : !conduit.window<memref<128xbf16>> -> memref<128xbf16>
        func.call @consume_mm_kernel(%inter_buf, %out_buf)
            : (memref<128xbf16>, memref<128xbf16>) -> ()
        conduit.release %out_win {count = 1 : i64, port = #conduit.port<Produce>}
            : !conduit.window<memref<128xbf16>>
        conduit.release %inter_win {count = 1 : i64, port = #conduit.port<Consume>}
            : !conduit.window<memref<128xbf16>>
      }
      aie.end
    } {link_with = "consumer_mm.o"}

    aie.end
  }
}
