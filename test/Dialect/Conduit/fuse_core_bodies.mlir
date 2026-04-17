// RUN: aie-opt --conduit-fuse-core-bodies %s | FileCheck %s
//
// Basic test for --conduit-fuse-core-bodies: two aie.core ops on the same
// tile connected by an intermediate conduit.create are fused into one core.
//
// Topology (before fusion):
//   Core A (producer): acquires @input (Consume), acquires @intermediate (Produce),
//     computes, releases @intermediate, releases @input.
//   Core B (consumer): acquires @intermediate (Consume), acquires @output (Produce),
//     computes, releases @output, releases @intermediate.
//
// After fusion:
//   Single core: acquires @input, computes producer work into L1 alloc,
//     clones consumer work from L1 alloc to @output, releases @output,
//     releases @input.
//   @intermediate conduit.create is erased. Core B is erased.

// CHECK-LABEL: module @fuse_flat_body

// The intermediate conduit should be erased:
// CHECK-NOT:   conduit.create @intermediate

// Input and output conduits should survive:
// CHECK:       conduit.create @input
// CHECK:       conduit.create @output

// Only one aie.core should remain, with merged link_files:
// CHECK:       aie.core
// CHECK:       link_files = ["producer.o", "consumer.o"]
// CHECK-NOT:   aie.core

module @fuse_flat_body {
  aie.device(npu2) {
    %tile_0_2 = aie.tile(0, 2)

    conduit.create @intermediate {element_type = memref<128xbf16>, depth = 2 : i64}
    conduit.create @input  {element_type = memref<128xbf16>, depth = 2 : i64}
    conduit.create @output {element_type = memref<128xbf16>, depth = 2 : i64}

    func.func private @produce_kernel(memref<128xbf16>, memref<128xbf16>)
    func.func private @consume_kernel(memref<128xbf16>, memref<128xbf16>)

    // Core A: producer — reads @input, writes @intermediate.
    aie.core(%tile_0_2) {
      %in_win = conduit.acquire {name = @input, count = 1 : i64,
                                 port = #conduit.port<Consume>}
                    : !conduit.window<memref<128xbf16>>
      %in_buf = conduit.subview_access %in_win {index = 0 : i64}
                    : !conduit.window<memref<128xbf16>> -> memref<128xbf16>

      %inter_win = conduit.acquire {name = @intermediate, count = 1 : i64,
                                    port = #conduit.port<Produce>}
                       : !conduit.window<memref<128xbf16>>
      %inter_buf = conduit.subview_access %inter_win {index = 0 : i64}
                       : !conduit.window<memref<128xbf16>> -> memref<128xbf16>

      func.call @produce_kernel(%in_buf, %inter_buf)
          : (memref<128xbf16>, memref<128xbf16>) -> ()

      conduit.release %inter_win {count = 1 : i64, port = #conduit.port<Produce>}
          : !conduit.window<memref<128xbf16>>
      conduit.release %in_win {count = 1 : i64, port = #conduit.port<Consume>}
          : !conduit.window<memref<128xbf16>>
      aie.end
    } {link_with = "producer.o"}

    // Core B: consumer — reads @intermediate, writes @output.
    aie.core(%tile_0_2) {
      %inter_win = conduit.acquire {name = @intermediate, count = 1 : i64,
                                    port = #conduit.port<Consume>}
                       : !conduit.window<memref<128xbf16>>
      %inter_buf = conduit.subview_access %inter_win {index = 0 : i64}
                       : !conduit.window<memref<128xbf16>> -> memref<128xbf16>

      %out_win = conduit.acquire {name = @output, count = 1 : i64,
                                  port = #conduit.port<Produce>}
                     : !conduit.window<memref<128xbf16>>
      %out_buf = conduit.subview_access %out_win {index = 0 : i64}
                     : !conduit.window<memref<128xbf16>> -> memref<128xbf16>

      func.call @consume_kernel(%inter_buf, %out_buf)
          : (memref<128xbf16>, memref<128xbf16>) -> ()

      conduit.release %out_win {count = 1 : i64, port = #conduit.port<Produce>}
          : !conduit.window<memref<128xbf16>>
      conduit.release %inter_win {count = 1 : i64, port = #conduit.port<Consume>}
          : !conduit.window<memref<128xbf16>>
      aie.end
    } {link_with = "consumer.o"}

    aie.end
  }
}
