// RUN: aie-opt --objectfifo-to-conduit --conduit-to-dma %s | FileCheck %s
//
// Tests disable_synchronization=true with depth=2 on adjacent tiles (shared
// memory path — tile(0,2)→tile(0,3) on npu1_1col).  Core bodies on both
// tiles exercise acquire/release lowering.
//
// No aie.lock or aie.use_lock should appear.  No DMA (shared memory path).
// 2 aie.buffer on the producer tile (shared memory buffers live there).
// Consumer core should have rotation counter arith ops but no use_lock.

// CHECK-LABEL: module
// CHECK:   aie.device(npu1_1col) {
// CHECK-NOT: aie.lock
// CHECK-NOT: aie.use_lock
// 2 buffers on producer tile (shared memory)
// CHECK:     aie.buffer({{.*}}) {sym_name = "of_buff_0"} : memref<16xi32>
// CHECK:     aie.buffer({{.*}}) {sym_name = "of_buff_1"} : memref<16xi32>
// No DMA — shared memory path
// CHECK-NOT: aie.mem
// CHECK-NOT: aie.memtile_dma
// CHECK-NOT: aie.flow
// Consumer core: rotation counter management (arith ops, memref load/store)
// CHECK:     aie.core(%{{.*}}) {
// CHECK:       arith.constant 0 : i32
// CHECK:       memref.store
// CHECK:       memref.load
// CHECK:       arith.addi
// CHECK:       arith.remui
// CHECK:       memref.store
// CHECK:       aie.end
// No residual Conduit ops
// CHECK-NOT: conduit.create
// CHECK-NOT: conduit.acquire
// CHECK-NOT: conduit.release

module {
  aie.device(npu1_1col) {
    %tile_0_2 = aie.tile(0, 2)
    %tile_0_3 = aie.tile(0, 3)

    aie.objectfifo @of(%tile_0_2, {%tile_0_3}, 2 : i32) { disable_synchronization = true }
        : !aie.objectfifo<memref<16xi32>>

    %core_0_2 = aie.core(%tile_0_2) {
      %0 = aie.objectfifo.acquire @of(Produce, 1) : !aie.objectfifosubview<memref<16xi32>>
      aie.objectfifo.release @of(Produce, 1)
      aie.end
    }
    %core_0_3 = aie.core(%tile_0_3) {
      %0 = aie.objectfifo.acquire @of(Consume, 1) : !aie.objectfifosubview<memref<16xi32>>
      aie.objectfifo.release @of(Consume, 1)
      aie.end
    }
  }
}
