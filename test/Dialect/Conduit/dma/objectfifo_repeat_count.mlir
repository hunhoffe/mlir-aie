// RUN: aie-opt --objectfifo-to-conduit --conduit-to-dma %s | FileCheck %s
//
// Tests repeat_count=N: the producer lock init should be depth*N (here 1*3=3),
// the BD chain has N=3 blocks (unrolled), and the core acquires/releases N units.
//
// This matches the stateful transform semantics: the producer core must wait
// for N empty slots before writing (one per DMA repetition).

// CHECK-LABEL: module
// CHECK:   aie.device(npu1_1col) {
// Producer lock init = depth * repeat_count = 1 * 3 = 3
// CHECK:     aie.lock({{.*}}) {init = 3 : i32
// Core acquires 3 at once (waits for 3 empty slots)
// CHECK:     aie.use_lock({{.*}}, AcquireGreaterEqual, 3)
// Core releases 3 at once
// CHECK:     aie.use_lock({{.*}}, Release, 3)
// 3 BD blocks emitted (one per repeat)
// CHECK:     aie.mem
// CHECK:       aie.dma_start
// CHECK:       aie.dma_bd
// CHECK:       aie.next_bd
// CHECK:       aie.dma_bd
// CHECK:       aie.next_bd
// CHECK:       aie.dma_bd
// CHECK:       aie.next_bd

module {
  aie.device(npu1_1col) {
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)

    aie.objectfifo @of(%tile_0_2, {%tile_0_0}, 1 : i32) {repeat_count = 3 : i32}
        : !aie.objectfifo<memref<16xi32>>

    %core_0_2 = aie.core(%tile_0_2) {
      %0 = aie.objectfifo.acquire @of(Produce, 1) : !aie.objectfifosubview<memref<16xi32>>
      aie.objectfifo.release @of(Produce, 1)
      aie.end
    }
  }
}
