// RUN: not aie-opt --objectfifo-to-conduit %s 2>&1 | FileCheck %s
//
// Risk #4 guard: cascade hardware is point-to-point (single producer →
// single consumer), so an aie.objectfifo with via_cascade=true and more
// than one consumer tile is semantically undefined.  Pass A must reject
// the combination explicitly rather than emit multiple aie.put_cascade
// users with no defined ordering.
//
// CHECK: error: {{.*}}objectfifo-to-conduit: via_cascade=true requires exactly one consumer tile
// CHECK-SAME: cascade hardware is point-to-point

module @objectfifo_cascade_broadcast_error {
  aie.device(npu1_1col) {
    %prod_tile = aie.tile(0, 2)
    %cons_tile_a = aie.tile(0, 3)
    %cons_tile_b = aie.tile(0, 4)

    aie.objectfifo @bad_cascade (%prod_tile, {%cons_tile_a, %cons_tile_b}, 1 : i32)
        {via_cascade = true} : !aie.objectfifo<memref<1xi32>>

    aie.core(%prod_tile) { aie.end }
    aie.core(%cons_tile_a) { aie.end }
    aie.core(%cons_tile_b) { aie.end }
  }
}
