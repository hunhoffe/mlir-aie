// RUN: aie-opt --verify-diagnostics --conduit-canonicalize-channel-puts %s | FileCheck %s
//
// Boundary-stress pin for ArithProgressionPattern: when the input puts
// already carry the maximum allowed BD data-layout-dim count for the tile
// (4 dims on shim/MemTile), prepending the canon's outer wrap+stride dim
// would push the post-collapse BD to 5 dims and trip:
//   * AIEDialect.cpp:2233-2236 dma_bd verifier
//     ('At most four data layout transformation dimensions may be provided.')
//     for compute (cap=3) / MemTile (cap=4) parents, and
//   * AIEDMATasksToNPU.cpp:347-350 runtime-sequence cap (4) for shim BDs.
//
// Canon must REFUSE the collapse, emit a diagnostic naming the channel +
// dim cap, and leave the IR un-collapsed (both puts survive).
//
// Geometry:
//   shim(0,0) MM2S → compute(0,2) consumer.  Channel @C_L2L3 has 2 sliding-
//   offset put_memref_async ops, EACH already stamped with a 4-dim
//   producer_dimensions list (3 padding `<size=1,stride=0>` + 1 data
//   `<size=64,stride=1>`, mirroring op7-shape input).  Offsets in arith
//   progression [0, 524288] (delta = 524288).  Without the dim-cap refuse
//   canon would collapse to 1 put + 5-dim producer_dimensions.

// CHECK-LABEL: aie.device(npu1)
// Channel must NOT gain an outer wrap dimension on producer_dimensions —
// the existing 4-dim attr should be preserved verbatim.
// CHECK: conduit.create @C_L2L3
// CHECK-NOT: <size = 2, stride = 524288>
//
// Both puts survive — pin via two CHECK lines.
// CHECK: conduit.put_memref_async
// CHECK-SAME: name = @C_L2L3
// CHECK: conduit.put_memref_async
// CHECK-SAME: name = @C_L2L3

module @canon_arith_progression_dim_cap_refuse {
  aie.device(npu1) {
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)

    // expected-warning@+1 {{canonicalize-loop-unroll-puts: refusing to collapse 2 puts on @C_L2L3}}
    conduit.create @C_L2L3 {
      element_type = memref<64xbf16>,
      depth = 2 : i64
    }

    aie.shim_dma_allocation @C_L2L3_shim_alloc(%tile_0_0, MM2S, 0) {conduit_channel = @C_L2L3}

    %core_0_2 = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c2 = arith.constant 2 : index
      %c1 = arith.constant 1 : index
      scf.for %i = %c0 to %c2 step %c1 {
        %g = conduit.get_memref_async {name = @C_L2L3,
                  num_elems = 64 : i64,
                  offsets = array<i64: 0>,
                  sizes = array<i64: 64>,
                  strides = array<i64: 1>} : !conduit.dma.token
        conduit.wait_all %g : !conduit.dma.token
      }
      aie.end
    } {dynamic_objfifo_lowering = true}

    func.func @sequence(%arg0: memref<1048576xbf16>) {
      // 2 sliding-offset puts (delta = 524288), each already at 4-dim
      // producer_dimensions (3 padding + 1 data) — op7-shape input.
      %t0 = conduit.put_memref_async {name = @C_L2L3,
            num_elems = 64 : i64,
            offsets = array<i64: 0>,
            sizes = array<i64: 64>,
            strides = array<i64: 1>,
            producer_dimensions = #aie<bd_dim_layout_array[
              <size = 1, stride = 0>,
              <size = 1, stride = 0>,
              <size = 1, stride = 0>,
              <size = 64, stride = 1>]>} : !conduit.dma.token
      conduit.wait_all %t0 {token = true} : !conduit.dma.token
      conduit.wait_all %t0 {token = false} : !conduit.dma.token

      %t1 = conduit.put_memref_async {name = @C_L2L3,
            num_elems = 64 : i64,
            offsets = array<i64: 524288>,
            sizes = array<i64: 64>,
            strides = array<i64: 1>,
            producer_dimensions = #aie<bd_dim_layout_array[
              <size = 1, stride = 0>,
              <size = 1, stride = 0>,
              <size = 1, stride = 0>,
              <size = 64, stride = 1>]>} : !conduit.dma.token
      conduit.wait_all %t1 {token = true} : !conduit.dma.token
      conduit.wait_all %t1 {token = false} : !conduit.dma.token
      return
    }
  }
}
