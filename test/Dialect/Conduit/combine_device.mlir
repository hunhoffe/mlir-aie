// RUN: aie-opt --aie-combine-device %s | FileCheck %s --check-prefix=OFFSET
// RUN: aie-opt --aie-combine-device="same-tile=true" %s | FileCheck %s --check-prefix=SAME
//
// Basic test for --aie-combine-device: two aie.device ops connected by a
// conduit channel with matching fusion_group are merged into one device.
//
// tile-offset mode: devB tiles get offset by max_col(devA)+1.
// same-tile mode:   devB tiles keep original coordinates.

// --- tile-offset mode ---

// One merged device, second device erased:
// OFFSET:       aie.device(npu2)
// OFFSET-NOT:   aie.device(npu2)

// Device A tile stays at (0,2); device B tile offset to (1,2):
// OFFSET:       aie.tile(0, 2)
// OFFSET:       conduit.create @chanA
// OFFSET:       aie.tile(1, 2)
// OFFSET:       conduit.create @chanB

// Both cores survive:
// OFFSET:       aie.core
// OFFSET:       aie.core

// --- same-tile mode ---

// One merged device:
// SAME:         aie.device(npu2)
// SAME-NOT:     aie.device(npu2)

// Both tiles stay at (0,2):
// SAME:         aie.tile(0, 2)
// SAME:         conduit.create @chanA
// SAME:         aie.tile(0, 2)
// SAME:         conduit.create @chanB

// Both cores survive:
// SAME:         aie.core
// SAME:         aie.core

module @combine_basic {
  aie.device(npu2) @devA {
    %tile_a = aie.tile(0, 2)

    conduit.create @chanA {element_type = memref<128xbf16>, depth = 2 : i64,
                           fusion_group = "group0"}

    func.func private @kernelA(memref<128xbf16>)

    aie.core(%tile_a) {
      %win = conduit.acquire {name = @chanA, count = 1 : i64,
                              port = #conduit.port<Produce>}
                 : !conduit.window<memref<128xbf16>>
      %buf = conduit.subview_access %win {index = 0 : i64}
                 : !conduit.window<memref<128xbf16>> -> memref<128xbf16>
      func.call @kernelA(%buf) : (memref<128xbf16>) -> ()
      conduit.release %win {count = 1 : i64, port = #conduit.port<Produce>}
          : !conduit.window<memref<128xbf16>>
      aie.end
    }
    aie.end
  }

  aie.device(npu2) @devB {
    %tile_b = aie.tile(0, 2)

    conduit.create @chanB {element_type = memref<128xbf16>, depth = 2 : i64,
                           fusion_group = "group0"}

    func.func private @kernelB(memref<128xbf16>)

    aie.core(%tile_b) {
      %win = conduit.acquire {name = @chanB, count = 1 : i64,
                              port = #conduit.port<Consume>}
                 : !conduit.window<memref<128xbf16>>
      %buf = conduit.subview_access %win {index = 0 : i64}
                 : !conduit.window<memref<128xbf16>> -> memref<128xbf16>
      func.call @kernelB(%buf) : (memref<128xbf16>) -> ()
      conduit.release %win {count = 1 : i64, port = #conduit.port<Consume>}
          : !conduit.window<memref<128xbf16>>
      aie.end
    }
    aie.end
  }
}
