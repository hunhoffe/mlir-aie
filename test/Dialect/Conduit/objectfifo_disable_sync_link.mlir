// RUN: aie-opt --objectfifo-to-conduit --conduit-to-dma %s | FileCheck %s
//
// Tests disable_synchronization=true on an objectfifo.link destination:
// no aie.lock should appear for link3 (the join destination with
// disable_synchronization=true). link1 and link2 are regular fifos and
// must retain their locks and use_lock ops.

// CHECK-LABEL: module @disable_sync
// Locks for link2 and link1 (regular fifos) must be present.
// CHECK-DAG:   aie.lock({{.*}}) {init = 1 : i32, sym_name = "link2_prod_lock_0"}
// CHECK-DAG:   aie.lock({{.*}}) {init = 0 : i32, sym_name = "link2_cons_lock_0"}
// CHECK-DAG:   aie.lock({{.*}}) {init = 1 : i32, sym_name = "link1_prod_lock_0"}
// CHECK-DAG:   aie.lock({{.*}}) {init = 0 : i32, sym_name = "link1_cons_lock_0"}
// No locks for link3 (disable_synchronization=true): no join MemTile locks.
// CHECK-NOT: link3_prod_lock
// CHECK-NOT: link3_cons_lock
// Flows: compute→MemTile×2, MemTile→shim
// CHECK-DAG: aie.flow(%{{.*}}, DMA : 0, %{{.*}}, DMA : 0)
// CHECK-DAG: aie.flow(%{{.*}}, DMA : 0, %{{.*}}, DMA : 1)
// CHECK:     aie.memtile_dma
// CHECK:       aie.dma_start
// CHECK:       aie.dma_bd
// MemTile BD chains have no use_lock for the disable_sync path.
// CHECK-NOT:   aie.use_lock
// Producer mem BDs for link1 and link2 have use_lock.
// CHECK:     aie.mem
// CHECK:       aie.use_lock
// CHECK:     aie.mem
// CHECK:       aie.use_lock
// No residual Conduit ops
// CHECK-NOT: conduit.create
// CHECK-NOT: conduit.acquire
// CHECK-NOT: conduit.release

module @disable_sync {
 aie.device(xcve2302) {
    %tile20 = aie.tile(2, 0)
    %tile21 = aie.tile(2, 1)
    %tile22 = aie.tile(2, 2)
    %tile23 = aie.tile(2, 3)

    aie.objectfifo @link1 (%tile22, {%tile21}, 1 : i32) : !aie.objectfifo<memref<4x4xi32>>
    aie.objectfifo @link2 (%tile23, {%tile21}, 1 : i32) : !aie.objectfifo<memref<20xi32>>
    aie.objectfifo @link3 (%tile21, {%tile20}, 1 : i32) { disable_synchronization = true } : !aie.objectfifo<memref<36xi32>>

    aie.objectfifo.link [@link1, @link2] -> [@link3] ([0, 16][])

    %core22 = aie.core(%tile22) {
      %sv = aie.objectfifo.acquire @link1(Produce, 1) : !aie.objectfifosubview<memref<4x4xi32>>
      aie.objectfifo.release @link1(Produce, 1)
      aie.end
    }
    %core23 = aie.core(%tile23) {
      %sv = aie.objectfifo.acquire @link2(Produce, 1) : !aie.objectfifosubview<memref<20xi32>>
      aie.objectfifo.release @link2(Produce, 1)
      aie.end
    }
 }
}
