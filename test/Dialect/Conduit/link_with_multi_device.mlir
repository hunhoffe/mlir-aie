// RUN: aie-opt --conduit-to-dma %s | FileCheck %s
//
// Regression test for M7: link_with propagation to ALL aie.device blocks.
//
// Before the M7 fix, --conduit-to-dma only walked the first aie.device block
// to propagate link_with from func.func declarations to aie.core ops.
// Multi-device fused modules (from --conduit-fuse-operators or
// --aie-combine-device) would have link_with set on device 0 cores but
// missing from device 1+ cores, causing linker failures.
//
// This test verifies that BOTH devices get link_with propagated.

// CHECK-LABEL: module @link_with_multi_device

// Device 0 core gets link_with:
// CHECK:       aie.device(npu2) @dev0
// CHECK:       aie.core
// CHECK:       link_with = "kernels.a"

// Device 1 core also gets link_with (M7 fix):
// CHECK:       aie.device(npu2) @dev1
// CHECK:       aie.core
// CHECK:       link_with = "kernels.a"

module @link_with_multi_device {
  aie.device(npu2) @dev0 {
    %tile_0_2 = aie.tile(0, 2)
    func.func private @kernel() attributes {link_with = "kernels.a"}
    aie.core(%tile_0_2) {
      func.call @kernel() : () -> ()
      aie.end
    }
    aie.end
  }

  aie.device(npu2) @dev1 {
    %tile_0_2 = aie.tile(0, 2)
    func.func private @kernel() attributes {link_with = "kernels.a"}
    aie.core(%tile_0_2) {
      func.call @kernel() : () -> ()
      aie.end
    }
    aie.end
  }
}
