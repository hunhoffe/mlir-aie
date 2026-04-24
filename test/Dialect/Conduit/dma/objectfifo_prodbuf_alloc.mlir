// RUN: aie-opt --objectfifo-to-conduit --conduit-to-dma %s | FileCheck %s
//
// Regression test for ProdBuf-alloc: producer buffer count = min(depth, maxProdAcquire+1).
//
// objectfifo @of: depth=4, non-adjacent tiles (DMA path), producer acquires 1 at a time.
// effectiveDepth = min(4, 1+1) = 2: only 2 producer buffers should be allocated on
// the producer tile.
//
// Without the fix, all 4 depth buffers would be allocated on the producer tile,
// wasting tile SRAM in patterns where the producer never holds more than 1 element.
//
// Expected layout:
//   - 2 producer buffers on tile(1,2): of_buff_0, of_buff_1
//   - 4 consumer buffers on tile(3,3): of_cons_buff_0..3
//   - of_prod_lock_0 {init = 2}: reflects effectiveDepth, not raw depth

// CHECK-LABEL: module @prodBufAlloc
// CHECK:   aie.device(npu1) {
// Exactly 2 producer buffers (= min(depth=4, maxProdAcquire+1 = 2))
// CHECK-DAG:   aie.buffer({{.*tile_1_2.*}}) {sym_name = "of_buff_0"}
// CHECK-DAG:   aie.buffer({{.*tile_1_2.*}}) {sym_name = "of_buff_1"}
// Producer lock init = effectiveDepth = 2 (NOT depth = 4)
// CHECK-DAG:   aie.lock({{.*tile_1_2.*}}) {init = 2 : i32, sym_name = "of_prod_lock_0"}
// 4 consumer buffers (= depth, consumer-side is not reduced)
// CHECK-DAG:   aie.buffer({{.*tile_3_3.*}}) {sym_name = "of_cons_buff_0"}
// CHECK-DAG:   aie.buffer({{.*tile_3_3.*}}) {sym_name = "of_cons_buff_1"}
// CHECK-DAG:   aie.buffer({{.*tile_3_3.*}}) {sym_name = "of_cons_buff_2"}
// CHECK-DAG:   aie.buffer({{.*tile_3_3.*}}) {sym_name = "of_cons_buff_3"}
// Consumer lock init = depth = 4
// CHECK-DAG:   aie.lock({{.*tile_3_3.*}}) {init = 4 : i32, sym_name = "of_cons_prod_lock_0"}
// Flow emitted (non-adjacent tiles use DMA)
// CHECK:       aie.flow(%{{.*}}, DMA : 0, %{{.*}}, DMA : 0)
// Producer DMA: 2 BD blocks rotating through the 2 producer buffers
// CHECK:       aie.mem(%{{.*tile_1_2.*}})
// CHECK:         aie.dma_start(MM2S
// CHECK:         aie.dma_bd({{.*of_buff_0.*}})
// CHECK:         aie.next_bd
// CHECK:         aie.dma_bd({{.*of_buff_1.*}})
// CHECK:         aie.next_bd ^bb1
// No third or fourth producer BD (would indicate over-allocation)
// CHECK-NOT:   aie.buffer({{.*tile_1_2.*}}) {sym_name = "of_buff_2"}
// CHECK-NOT:   aie.buffer({{.*tile_1_2.*}}) {sym_name = "of_buff_3"}
// No residual Conduit ops
// CHECK-NOT: conduit.create
// CHECK-NOT: conduit.acquire
// CHECK-NOT: conduit.release

module @prodBufAlloc {
  aie.device(npu1) {
    %tile12 = aie.tile(1, 2)
    %tile33 = aie.tile(3, 3)
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c8 = arith.constant 8 : index
    %c4 = arith.constant 4 : index

    // depth=4: without effectiveDepth, all 4 buffers would be placed on producer tile.
    // With effectiveDepth = min(4, maxProdAcquire+1) = min(4, 2) = 2, only 2 are needed.
    aie.objectfifo @of (%tile12, {%tile33}, 4 : i32) : !aie.objectfifo<memref<16xi32>>

    // Producer acquires 1 at a time → maxProdAcquire = 1 → effectiveDepth = 2
    %core12 = aie.core(%tile12) {
      scf.for %i = %c0 to %c8 step %c1 {
        %sv = aie.objectfifo.acquire @of (Produce, 1) : !aie.objectfifosubview<memref<16xi32>>
        aie.objectfifo.release @of (Produce, 1)
      }
      aie.end
    }

    // Consumer acquires 2 at a time (allowed by depth=4, not limited by effectiveDepth)
    %core33 = aie.core(%tile33) {
      scf.for %i = %c0 to %c4 step %c1 {
        %sv = aie.objectfifo.acquire @of (Consume, 2) : !aie.objectfifosubview<memref<16xi32>>
        aie.objectfifo.release @of (Consume, 2)
      }
      aie.end
    }
  }
}
