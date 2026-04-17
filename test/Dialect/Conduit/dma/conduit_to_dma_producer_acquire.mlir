// RUN: aie-opt --objectfifo-to-conduit --conduit-to-dma %s | FileCheck %s
//
// Pass A + Pass C end-to-end test: producer-side acquire/release.
//
// Known gap (Finding 2): conduit.acquire/conduit.release do not carry a
// port=Produce|Consume attribute.  Pass C currently maps ALL conduit.acquire
// ops to the consLock (consumer lock), regardless of whether the core is
// the producer.  A producer core should acquire/release the prodLock instead.
//
// This test exercises a PRODUCER core that writes to an objectfifo:
//   - conduit.acquire in a Produce context → should lower to:
//       aie.use_lock(%prod_lock, AcquireGreaterEqual, 1)
//   - conduit.release in a Produce context → should lower to:
//       aie.use_lock(%cons_lock, Release, 1)
//
// The CHECK lines verify the CORRECT lock selection.  With the current bug,
// Pass C uses consLock for both producer and consumer cores, so the acquire
// in the producer core incorrectly acquires the consumer lock.
//
// Expected resources:
//   aie.buffer:  1  (out_fifo_buff_0 on tile_0_2)
//   aie.lock:    2  (out_fifo_prod_lock_0 init=1, out_fifo_cons_lock_0 init=0)
//   aie.flow:    1  (tile_0_2 -> shim_0,0)
//
// In the producer core (tile_0_2):
//   - acquire → AcquireGreaterEqual on PROD lock (waiting for a free slot)
//   - release → Release on CONS lock (signaling data is ready)

// CHECK-LABEL: module @producer_acquire
// CHECK:   aie.device(npu1_1col) {
// CHECK:     %{{.*}}tile_0_0 = aie.tile(0, 0)
// CHECK:     %{{.*}}tile_0_2 = aie.tile(0, 2)
// CHECK:     aie.buffer(%{{.*}}tile_0_2)
// CHECK-SAME:   sym_name = "out_fifo_buff_0"
// --- prod_lock has init=1 (one free slot for depth=1) ---
// CHECK:     %[[PROD_LOCK:.*]] = aie.lock(%{{.*}}tile_0_2
// CHECK-SAME:   init = 1
// CHECK-SAME:   sym_name = "out_fifo_prod_lock_0"
// --- cons_lock has init=0 (nothing filled yet) ---
// CHECK:     %[[CONS_LOCK:.*]] = aie.lock(%{{.*}}tile_0_2
// CHECK-SAME:   init = 0
// CHECK-SAME:   sym_name = "out_fifo_cons_lock_0"
// CHECK:     aie.core(%{{.*}}tile_0_2) {
// CHECK:       scf.for
// --- Producer acquire: waits for a FREE slot → uses PROD lock ---
// CHECK:         aie.use_lock(%[[PROD_LOCK]], AcquireGreaterEqual, 1)
// --- Producer release: signals data ready → uses CONS lock ---
// CHECK:         aie.use_lock(%[[CONS_LOCK]], Release, 1)
// CHECK:     }
// CHECK:     aie.flow(%{{.*}}tile_0_2, DMA : 0, %{{.*}}tile_0_0, DMA : 0)
// CHECK-NOT: conduit.create
// CHECK-NOT: conduit.acquire
// CHECK-NOT: conduit.release

module @producer_acquire {
  aie.device(npu1_1col) {
    func.func @generate_data(%buf: memref<10xi32>) -> () {
      return
    }

    %tile_0_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)

    // tile_0_2 is the PRODUCER; tile_0_0 (shim) is the consumer (DMA out)
    aie.objectfifo @out_fifo(%tile_0_2, {%tile_0_0}, 1 : i32) : !aie.objectfifo<memref<10xi32>>

    %core_0_2 = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c10 = arith.constant 10 : index

      scf.for %arg0 = %c0 to %c10 step %c1 {
        // Produce-side acquire: gets a free slot from prod_lock
        %0 = aie.objectfifo.acquire @out_fifo(Produce, 1) : !aie.objectfifosubview<memref<10xi32>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<10xi32>> -> memref<10xi32>
        func.call @generate_data(%1) : (memref<10xi32>) -> ()
        // Produce-side release: marks slot filled → signals cons_lock
        aie.objectfifo.release @out_fifo(Produce, 1)
      }

      aie.end
    } {dynamic_objfifo_lowering = true}
  }
}
