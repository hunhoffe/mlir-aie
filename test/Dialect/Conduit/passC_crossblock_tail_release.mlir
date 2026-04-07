// RUN: aie-opt --conduit-to-dma %s | FileCheck %s
//
// Regression: cross-block held-count propagation after scf.for.
//
// Sliding window: preamble acquire(2)/release(1) in outer block, then
// scf.for with acquire(3)/release(1) per iteration, then tail acquire(2).
//
// State trace:
//   [outer] acquire(2): held=0 → delta=2 → AGE(2). held=2, last=2.
//   [outer] release(1): held=1.
//   [loop]  child inherits lastAcquireCount=2 as heldCount=2.
//   [loop]  acquire(3): held=2 → delta=1 → AGE(1). held=3, last=3.
//   [loop]  release(1): held=2. (child exits with held=2)
//   [outer] after loop: with fix, parent held updated to 2 (child exit held).
//   [outer] acquire(2): held=2 → delta=2-2=0 → NO AcquireGreaterEqual.
//   [outer] release(2): Release(2). held=0.
//
// BUG (before fix): parent's heldCount was NOT updated after child region
// processing. Parent retained held=1 (pre-loop preamble state), causing
// tail acquire(2) → delta=2-1=1 → spurious AGE(1) → hardware deadlock
// (DMA tokens already exhausted).
//
// CHECK-LABEL: module @passC_crossblock_tail_release
// CHECK: aie.core(%tile_0_2)
//
// Preamble acquire(2): fresh → AGE(2).
// CHECK: aie.use_lock(%{{.*}}, AcquireGreaterEqual, 2)
// Preamble release(1).
// CHECK: aie.use_lock(%{{.*}}, Release, 1)
// Loop body acquire(3): inherited held=2 → delta=1 → AGE(1).
// CHECK: aie.use_lock(%{{.*}}, AcquireGreaterEqual, 1)
// Loop body release(1).
// CHECK: aie.use_lock(%{{.*}}, Release, 1)
// Tail: NO AcquireGreaterEqual (delta=0 after fix). Next lock op is Release(2).
// CHECK-NOT: aie.use_lock(%{{.*}}, AcquireGreaterEqual
// CHECK: aie.use_lock(%{{.*}}, Release, 2)

module @passC_crossblock_tail_release {
  aie.device(npu1_1col) {
    %shim = aie.tile(0, 0)
    %tile = aie.tile(0, 2)

    // depth=4 to accommodate acquire=3 + 1 spare slot.
    // slot_elems = 4 * 32 = 128; perBufLen = 32.
    conduit.create @fifo {slot_elems = 128 : i64, depth = 4 : i64,
                    element_type = memref<32xi32>,
                    producer_tile = array<i64: 0, 0>,
                    consumer_tiles = array<i64: 0, 2>}

    aie.shim_dma_allocation @fifo_shim_alloc(%shim, MM2S, 0)

    %core = aie.core(%tile) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c4 = arith.constant 4 : index
      %val = arith.constant 42 : i32

      // Preamble: acquire 2 (fresh), release 1 → outer held=1, last=2.
      %win_pre = conduit.acquire {name = @fifo, count = 2 : i64,
                                   port = #conduit.port<Consume>}
                     : !conduit.window<memref<32xi32>>
      %pre0 = conduit.subview_access %win_pre {index = 0 : i64}
                  : !conduit.window<memref<32xi32>> -> memref<32xi32>
      memref.store %val, %pre0[%c0] : memref<32xi32>
      conduit.release %win_pre {count = 1 : i64, port = #conduit.port<Consume>}
          : !conduit.window<memref<32xi32>>

      // Middle loop: acquire 3, release 1 per iteration.
      // Child inherits last=2 as held=2 → first iter: delta=3-2=1 → AGE(1).
      // Child exits each iter with held=2.
      scf.for %i = %c0 to %c4 step %c1 {
        %win_mid = conduit.acquire {name = @fifo, count = 3 : i64,
                                     port = #conduit.port<Consume>}
                       : !conduit.window<memref<32xi32>>
        %mid0 = conduit.subview_access %win_mid {index = 0 : i64}
                    : !conduit.window<memref<32xi32>> -> memref<32xi32>
        %mid1 = conduit.subview_access %win_mid {index = 1 : i64}
                    : !conduit.window<memref<32xi32>> -> memref<32xi32>
        %mid2 = conduit.subview_access %win_mid {index = 2 : i64}
                    : !conduit.window<memref<32xi32>> -> memref<32xi32>
        memref.store %val, %mid0[%c0] : memref<32xi32>
        conduit.release %win_mid {count = 1 : i64, port = #conduit.port<Consume>}
            : !conduit.window<memref<32xi32>>
      }

      // Tail: 2 slots still held after loop exits.
      // With fix: parent held updated to 2 → acquire(2) → delta=0 → NO AGE.
      // Without fix: parent held=1 (stale) → delta=1 → AGE(1). ← BUG.
      %win_tail = conduit.acquire {name = @fifo, count = 2 : i64,
                                    port = #conduit.port<Consume>}
                      : !conduit.window<memref<32xi32>>
      %tail0 = conduit.subview_access %win_tail {index = 0 : i64}
                   : !conduit.window<memref<32xi32>> -> memref<32xi32>
      %tail1 = conduit.subview_access %win_tail {index = 1 : i64}
                   : !conduit.window<memref<32xi32>> -> memref<32xi32>
      memref.store %val, %tail0[%c0] : memref<32xi32>
      conduit.release %win_tail {count = 2 : i64, port = #conduit.port<Consume>}
          : !conduit.window<memref<32xi32>>

      aie.end
    }

    aie.runtime_sequence(%in: memref<128xi32>) {
      aiex.npu.dma_memcpy_nd (%in[0,0,0,0][1,1,1,128][0,0,0,1])
          {metadata = @fifo_shim_alloc, id = 0 : i64} : memref<128xi32>
    }
  }
}
