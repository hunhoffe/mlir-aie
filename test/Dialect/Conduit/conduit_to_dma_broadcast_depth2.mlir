// RUN: aie-opt --objectfifo-to-conduit --conduit-to-dma %s | FileCheck %s
//
// Pass A + Pass C end-to-end test: 1 compute producer → 2 consumers, depth=2.
//
// This test verifies the P0-B fix: broadcast conduits with depth>1 must emit
// aie.flow ops for ALL consumer tiles, including adjacent ones.  Before the
// fix, Phase 4.5a incorrectly skipped flows for adjacent consumers in broadcast
// conduits, resulting in 0 aie.flow ops (dead DMA, no data movement).
//
// Topology: producer tile(0,2) → consumer tiles (0,3) and (0,4), depth=2.
// On npu1_1col, tile(0,2) and tile(0,3) are adjacent, but broadcast always
// uses DMA (Phase 3c only handles single-consumer shared memory).
//
// Resources:
//   aie.buffer:  6  (2 per consumer tile × 2 consumers + 2 producer-side)
//   aie.flow:    2  (producer MM2S → each consumer S2MM, including adjacent)
//   aie.mem:     3  (1 producer MM2S + 2 consumer S2MM)

// CHECK-LABEL: module @broadcast_depth2_compute
// CHECK:   aie.device(npu1_1col) {
// CHECK:     aie.tile(0, 2)
// CHECK:     aie.tile(0, 3)
// CHECK:     aie.tile(0, 4)
// --- Producer-side buffers (depth=2, effectiveDepth=min(2, 1+1)=2) ---
// CHECK:     aie.buffer({{.*}}){{.*}}sym_name = "bcast_d2_buff_0"
// CHECK:     aie.buffer({{.*}}){{.*}}sym_name = "bcast_d2_buff_1"
// --- Consumer tile 1 buffers and locks ---
// CHECK:     aie.buffer({{.*}}){{.*}}sym_name = "bcast_d2_cons_1_buff_0"
// CHECK:     aie.buffer({{.*}}){{.*}}sym_name = "bcast_d2_cons_1_buff_1"
// CHECK:     aie.lock({{.*}}){{.*}}sym_name = "bcast_d2_cons_1_prod_lock_0"
// CHECK:     aie.lock({{.*}}){{.*}}sym_name = "bcast_d2_cons_1_cons_lock_0"
// --- Consumer tile 0 buffers and locks ---
// CHECK:     aie.buffer({{.*}}){{.*}}sym_name = "bcast_d2_cons_0_buff_0"
// CHECK:     aie.buffer({{.*}}){{.*}}sym_name = "bcast_d2_cons_0_buff_1"
// CHECK:     aie.lock({{.*}}){{.*}}sym_name = "bcast_d2_cons_0_prod_lock_0"
// CHECK:     aie.lock({{.*}}){{.*}}sym_name = "bcast_d2_cons_0_cons_lock_0"
// --- Cores: each uses its own tile's locks and buffers ---
// CHECK:     aie.core({{.*}}) {
// CHECK:       aie.use_lock({{.*}}bcast_d2_cons_0_cons{{.*}}, AcquireGreaterEqual, 1)
// CHECK:     aie.core({{.*}}) {
// CHECK:       aie.use_lock({{.*}}bcast_d2_cons_1_cons{{.*}}, AcquireGreaterEqual, 1)
// --- P0-B fix: flows emitted for ALL consumers, including adjacent tile(0,3) ---
// CHECK:     aie.flow({{.*}}, DMA : {{[0-9]+}}, {{.*}}, DMA : {{[0-9]+}})
// CHECK:     aie.flow({{.*}}, DMA : {{[0-9]+}}, {{.*}}, DMA : {{[0-9]+}})
// --- Producer MM2S aie.mem ---
// CHECK:     aie.mem({{.*}}) {
// CHECK:       aie.dma_start(MM2S
// CHECK:       aie.dma_bd({{.*}}bcast_d2_buff_0
// CHECK:       aie.dma_bd({{.*}}bcast_d2_buff_1
// --- Consumer S2MM aie.mem blocks ---
// CHECK:     aie.mem({{.*}}) {
// CHECK:       aie.dma_start(S2MM
// CHECK:       aie.dma_bd({{.*}}bcast_d2_cons_{{[0-9]+}}_buff_0
// CHECK:       aie.dma_bd({{.*}}bcast_d2_cons_{{[0-9]+}}_buff_1
// CHECK:     aie.mem({{.*}}) {
// CHECK:       aie.dma_start(S2MM
// CHECK:       aie.dma_bd({{.*}}bcast_d2_cons_{{[0-9]+}}_buff_0
// CHECK:       aie.dma_bd({{.*}}bcast_d2_cons_{{[0-9]+}}_buff_1
// CHECK-NOT: conduit.create
// CHECK-NOT: conduit.acquire
// CHECK-NOT: conduit.release

module @broadcast_depth2_compute {
  aie.device(npu1_1col) {
    func.func @consume_data(%buf: memref<16xi32>) -> () {
      return
    }
    func.func @produce_data(%buf: memref<16xi32>) -> () {
      return
    }

    %tile_0_2 = aie.tile(0, 2)
    %tile_0_3 = aie.tile(0, 3)
    %tile_0_4 = aie.tile(0, 4)

    // Compute producer → 2 consumers, depth=2.
    // tile(0,2) is adjacent to tile(0,3) on npu1 — P0-B fix ensures the
    // flow is still emitted (broadcast cannot use shared memory).
    aie.objectfifo @bcast_d2(%tile_0_2, {%tile_0_3, %tile_0_4}, 2 : i32)
        : !aie.objectfifo<memref<16xi32>>

    %core_0_2 = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c4 = arith.constant 4 : index
      scf.for %arg0 = %c0 to %c4 step %c1 {
        %0 = aie.objectfifo.acquire @bcast_d2(Produce, 1) : !aie.objectfifosubview<memref<16xi32>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>
        func.call @produce_data(%1) : (memref<16xi32>) -> ()
        aie.objectfifo.release @bcast_d2(Produce, 1)
      }
      aie.end
    } {dynamic_objfifo_lowering = true}

    %core_0_3 = aie.core(%tile_0_3) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c4 = arith.constant 4 : index
      scf.for %arg0 = %c0 to %c4 step %c1 {
        %0 = aie.objectfifo.acquire @bcast_d2(Consume, 1) : !aie.objectfifosubview<memref<16xi32>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>
        func.call @consume_data(%1) : (memref<16xi32>) -> ()
        aie.objectfifo.release @bcast_d2(Consume, 1)
      }
      aie.end
    } {dynamic_objfifo_lowering = true}

    %core_0_4 = aie.core(%tile_0_4) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c4 = arith.constant 4 : index
      scf.for %arg0 = %c0 to %c4 step %c1 {
        %0 = aie.objectfifo.acquire @bcast_d2(Consume, 1) : !aie.objectfifosubview<memref<16xi32>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<16xi32>> -> memref<16xi32>
        func.call @consume_data(%1) : (memref<16xi32>) -> ()
        aie.objectfifo.release @bcast_d2(Consume, 1)
      }
      aie.end
    } {dynamic_objfifo_lowering = true}
  }
}
