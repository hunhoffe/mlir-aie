// RUN: not aie-opt --conduit-to-dma %s 2>&1 | FileCheck %s
//
// Negative test: S2MM DMA channel overflow on a compute tile.
//
// A compute tile on xcve2302 has 2 S2MM DMA channels. Three non-adjacent
// producers all route to the same consumer tile, exhausting the S2MM budget.
//
// CHECK: error:{{.*}}S2MM DMA channel exhausted on tile (2,2)

module @s2mm_overflow {
  aie.device(xcve2302) {
    %t02 = aie.tile(0, 2)
    %t03 = aie.tile(0, 3)
    %t42 = aie.tile(4, 2)
    %t22 = aie.tile(2, 2)

    // Three non-adjacent producers all targeting tile(2,2).
    conduit.create @a {slot_elems = 32 : i64, element_type = memref<16xi32>, depth = 2 : i64}
    conduit.create @b {slot_elems = 32 : i64, element_type = memref<16xi32>, depth = 2 : i64}
    conduit.create @c {slot_elems = 32 : i64, element_type = memref<16xi32>, depth = 2 : i64}

    // Producer cores — structural info for tile inference.
    %core_0_2 = aie.core(%t02) {
      %0 = conduit.acquire {count = 1 : i64, name = @a,
                            port = #conduit.port<Produce>} : <memref<16xi32>>
      conduit.release %0 {count = 1 : i64,
                          port = #conduit.port<Produce>} : <memref<16xi32>>
      aie.end
    }
    %core_0_3 = aie.core(%t03) {
      %0 = conduit.acquire {count = 1 : i64, name = @b,
                            port = #conduit.port<Produce>} : <memref<16xi32>>
      conduit.release %0 {count = 1 : i64,
                          port = #conduit.port<Produce>} : <memref<16xi32>>
      aie.end
    }
    %core_4_2 = aie.core(%t42) {
      %0 = conduit.acquire {count = 1 : i64, name = @c,
                            port = #conduit.port<Produce>} : <memref<16xi32>>
      conduit.release %0 {count = 1 : i64,
                          port = #conduit.port<Produce>} : <memref<16xi32>>
      aie.end
    }

    // Consumer core — tile(2,2) consumes all 3 channels.
    %core_2_2 = aie.core(%t22) {
      %0 = conduit.acquire {count = 1 : i64, name = @a,
                            port = #conduit.port<Consume>} : <memref<16xi32>>
      conduit.release %0 {count = 1 : i64,
                          port = #conduit.port<Consume>} : <memref<16xi32>>
      %1 = conduit.acquire {count = 1 : i64, name = @b,
                            port = #conduit.port<Consume>} : <memref<16xi32>>
      conduit.release %1 {count = 1 : i64,
                          port = #conduit.port<Consume>} : <memref<16xi32>>
      %2 = conduit.acquire {count = 1 : i64, name = @c,
                            port = #conduit.port<Consume>} : <memref<16xi32>>
      conduit.release %2 {count = 1 : i64,
                          port = #conduit.port<Consume>} : <memref<16xi32>>
      aie.end
    }
  }
}
