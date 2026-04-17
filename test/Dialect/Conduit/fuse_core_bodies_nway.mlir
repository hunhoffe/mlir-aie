// RUN: aie-opt --conduit-fuse-core-bodies %s | FileCheck %s
//
// Regression test: N-way chain fusion in --conduit-fuse-core-bodies.
//
// The pass must handle 3-way chains (A→B→C) by fusing one pair at a time
// and restarting until no more fusible pairs remain.  This was fixed in
// commit f8e7227b ("conduit: fix fuse-core-bodies N-way chain fusion via
// fuse-one-then-restart").
//
// Topology (before fusion):
//   Core A: reads @input, writes @inter1  (link_with = "a.o")
//   Core B: reads @inter1, writes @inter2 (link_with = "b.o")
//   Core C: reads @inter2, writes @output (link_with = "c.o")
//
// After fusion:
//   Both @inter1 and @inter2 should be erased.
//   Only @input and @output conduits should survive.
//   A single aie.core should remain with all three link_files merged.

// CHECK-LABEL: module @fuse_nway_chain

// Both intermediates erased:
// CHECK-NOT:   conduit.create @inter1
// CHECK-NOT:   conduit.create @inter2

// External conduits survive:
// CHECK:       conduit.create @input
// CHECK:       conduit.create @output

// Only one core remains, with all three link_files:
// CHECK:       aie.core
// CHECK:       link_files = ["a.o", "b.o", "c.o"]
// CHECK-NOT:   aie.core

module @fuse_nway_chain {
  aie.device(npu2) {
    %tile_0_2 = aie.tile(0, 2)

    conduit.create @inter1 {element_type = memref<128xbf16>, depth = 2 : i64}
    conduit.create @inter2 {element_type = memref<128xbf16>, depth = 2 : i64}
    conduit.create @input  {element_type = memref<128xbf16>, depth = 2 : i64}
    conduit.create @output {element_type = memref<128xbf16>, depth = 2 : i64}

    func.func private @kernel_a(memref<128xbf16>, memref<128xbf16>)
    func.func private @kernel_b(memref<128xbf16>, memref<128xbf16>)
    func.func private @kernel_c(memref<128xbf16>, memref<128xbf16>)

    // Core A: reads @input, writes @inter1.
    aie.core(%tile_0_2) {
      %in_win = conduit.acquire {name = @input, count = 1 : i64,
                                 port = #conduit.port<Consume>}
                    : !conduit.window<memref<128xbf16>>
      %in_buf = conduit.subview_access %in_win {index = 0 : i64}
                    : !conduit.window<memref<128xbf16>> -> memref<128xbf16>

      %i1_win = conduit.acquire {name = @inter1, count = 1 : i64,
                                 port = #conduit.port<Produce>}
                    : !conduit.window<memref<128xbf16>>
      %i1_buf = conduit.subview_access %i1_win {index = 0 : i64}
                    : !conduit.window<memref<128xbf16>> -> memref<128xbf16>

      func.call @kernel_a(%in_buf, %i1_buf)
          : (memref<128xbf16>, memref<128xbf16>) -> ()

      conduit.release %i1_win {count = 1 : i64, port = #conduit.port<Produce>}
          : !conduit.window<memref<128xbf16>>
      conduit.release %in_win {count = 1 : i64, port = #conduit.port<Consume>}
          : !conduit.window<memref<128xbf16>>
      aie.end
    } {link_with = "a.o"}

    // Core B: reads @inter1, writes @inter2.
    aie.core(%tile_0_2) {
      %i1_win = conduit.acquire {name = @inter1, count = 1 : i64,
                                 port = #conduit.port<Consume>}
                    : !conduit.window<memref<128xbf16>>
      %i1_buf = conduit.subview_access %i1_win {index = 0 : i64}
                    : !conduit.window<memref<128xbf16>> -> memref<128xbf16>

      %i2_win = conduit.acquire {name = @inter2, count = 1 : i64,
                                 port = #conduit.port<Produce>}
                    : !conduit.window<memref<128xbf16>>
      %i2_buf = conduit.subview_access %i2_win {index = 0 : i64}
                    : !conduit.window<memref<128xbf16>> -> memref<128xbf16>

      func.call @kernel_b(%i1_buf, %i2_buf)
          : (memref<128xbf16>, memref<128xbf16>) -> ()

      conduit.release %i2_win {count = 1 : i64, port = #conduit.port<Produce>}
          : !conduit.window<memref<128xbf16>>
      conduit.release %i1_win {count = 1 : i64, port = #conduit.port<Consume>}
          : !conduit.window<memref<128xbf16>>
      aie.end
    } {link_with = "b.o"}

    // Core C: reads @inter2, writes @output.
    aie.core(%tile_0_2) {
      %i2_win = conduit.acquire {name = @inter2, count = 1 : i64,
                                 port = #conduit.port<Consume>}
                    : !conduit.window<memref<128xbf16>>
      %i2_buf = conduit.subview_access %i2_win {index = 0 : i64}
                    : !conduit.window<memref<128xbf16>> -> memref<128xbf16>

      %out_win = conduit.acquire {name = @output, count = 1 : i64,
                                  port = #conduit.port<Produce>}
                     : !conduit.window<memref<128xbf16>>
      %out_buf = conduit.subview_access %out_win {index = 0 : i64}
                     : !conduit.window<memref<128xbf16>> -> memref<128xbf16>

      func.call @kernel_c(%i2_buf, %out_buf)
          : (memref<128xbf16>, memref<128xbf16>) -> ()

      conduit.release %out_win {count = 1 : i64, port = #conduit.port<Produce>}
          : !conduit.window<memref<128xbf16>>
      conduit.release %i2_win {count = 1 : i64, port = #conduit.port<Consume>}
          : !conduit.window<memref<128xbf16>>
      aie.end
    } {link_with = "c.o"}

    aie.end
  }
}
