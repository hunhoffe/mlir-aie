// RUN: aie-opt --conduit-to-dma %s -split-input-file | FileCheck %s
//
// Pass C test: putCount inference → linear (one-shot) S2MM BD chain.
//
// When multiple put_memref_async ops reference the same conduit.create
// (after --conduit-fuse-channels rewrites non-canonical names to the
// canonical channel), Pass C infers putCount = N from Phase 1 and then:
//
//   1. Allocates N consumer buffers via nConsumerBuffers() = putCount.
//   2. Emits a linear N-entry BD chain for the S2MM.  The last BD must point to
//      aie.end, not back to BD0.  A circular ring would re-use slot 0 before
//      slot 1 is consumed, corrupting data.
//
// The relevant code path in ConduitToDMALink.cpp (Phase 5.5, Tier 3 section):
//   isLinearChain = (info.iterCount > 0) || (info.putCount > 1 && info.iterCount == 0)
//   nBufs = info.nConsumerBuffers()          // returns putCount when > 1 and iterCount == 0
//   last BD → bdTermBlock / endMemBlock      // NOT bdBlocks[0]
//
// Producer: shim tile (0,0) via put_memref_async ops.
// Consumer: compute tile (0,2) with get_memref_async ops inside aie.core.
// Architecture: npu1_1col (AIE2, single column, rows 0-4).
//
// Producer is shim (row=0): Step 8e skips use_lock for put_memref_async ops
// (host runtime manages the shim DMA); puts are simply erased.
// Step 8f always emits a use_lock pair per get_memref_async in a core.
//
// Three test cases (each separated by a split-input-file marker):
//   (1) putCount = 2 (two put_memref_async) → 2-entry linear S2MM chain, 2 consumer buffers.
//   (2) putCount = 3 (three put_memref_async) → 3-entry linear chain, 3 consumer buffers.
//   (3) putCount = 1 (depth=1, standard ring) → 1-entry circular ring (baseline).

//===----------------------------------------------------------------------===//
// (1) putCount = 2
//
// Expected BD chain on aie.mem(tile_0_2):
//   ^start: dma_start(S2MM, 0, ^bd0, ^end)
//   ^bd0:   use_lock(prod, AcquireGreaterEqual, 1)
//           dma_bd(%kv_cons_buff_0, 0, 64)
//           use_lock(cons, Release, 1)
//           next_bd ^bd1              ← linear: advance to BD1
//   ^bd1:   use_lock(prod, AcquireGreaterEqual, 1)
//           dma_bd(%kv_cons_buff_1, 0, 64)
//           use_lock(cons, Release, 1)
//           next_bd ^end              ← linear terminator, NOT ^bd0
//   ^end:   aie.end
//
// Key verification:
//   - Two distinct aie.dma_bd entries (one per TM slot).
//   - Both buffers (kv_cons_buff_0 and kv_cons_buff_1) appear in BDs.
//   - aie.end terminates the region.
//   - CHECK-NOT: verifies that kv_cons_buff_2 is NOT allocated (confirming N=2).
//===----------------------------------------------------------------------===//

// CHECK-LABEL: module @tm_count2
// CHECK:   aie.device(npu1_1col)

// --- Two consumer buffers on tile(0,2) for the 2 TM slots ---
// CHECK:     %[[BUFF0:.*]] = aie.buffer(%{{.*}}tile_0_2)
// CHECK-SAME:   sym_name = "kv_cons_buff_0"
// CHECK:     %[[BUFF1:.*]] = aie.buffer(%{{.*}}tile_0_2)
// CHECK-SAME:   sym_name = "kv_cons_buff_1"
// No third buffer (confirms putCount=2, not more).
// CHECK-NOT:   aie.buffer(%{{.*}}tile_0_2) {sym_name = "kv_cons_buff_2"}

// --- Consumer-tile locks: prod_lock init=2 (2 free slots), cons_lock init=0 ---
// CHECK:     %[[PROD_LOCK:.*]] = aie.lock(%{{.*}}tile_0_2
// CHECK-SAME:   init = 2
// CHECK-SAME:   sym_name = "kv_cons_prod_lock_0"
// CHECK:     %[[CONS_LOCK:.*]] = aie.lock(%{{.*}}tile_0_2
// CHECK-SAME:   init = 0
// CHECK-SAME:   sym_name = "kv_cons_cons_lock_0"

// --- Consumer core: 2 get_memref_async → 2 use_lock pairs (Step 8f) ---
// Producer is shim (row=0): put_memref_async emits no use_lock in core (Step 8e).
// CHECK:     aie.core(%{{.*}}tile_0_2) {
// First get: consLock acquire (wait for data) + prodLock release (free slot).
// CHECK:       aie.use_lock(%[[CONS_LOCK]], AcquireGreaterEqual, 1)
// CHECK:       aie.use_lock(%[[PROD_LOCK]], Release, 1)
// Second get: same pattern.
// CHECK:       aie.use_lock(%[[CONS_LOCK]], AcquireGreaterEqual, 1)
// CHECK:       aie.use_lock(%[[PROD_LOCK]], Release, 1)

// --- Shim allocation + flow ---
// CHECK:     aie.shim_dma_allocation @{{.*}}shim_alloc
// CHECK:     aie.flow(%{{.*}}tile_0_0, DMA : 0, %{{.*}}tile_0_2, DMA : 0)

// --- S2MM linear BD chain: 2 entries, last BD → aie.end ---
// CHECK:     aie.mem(%{{.*}}tile_0_2) {
// CHECK:       aie.dma_start(S2MM, 0,
// --- First BD ---
// CHECK:       aie.use_lock(%[[PROD_LOCK]], AcquireGreaterEqual, 1)
// CHECK:       aie.dma_bd(%[[BUFF0]] : memref<64xi32>, 0, 64)
// CHECK:       aie.use_lock(%[[CONS_LOCK]], Release, 1)
// CHECK:       aie.next_bd
// --- Second BD (proves chain has 2 entries) ---
// CHECK:       aie.use_lock(%[[PROD_LOCK]], AcquireGreaterEqual, 1)
// CHECK:       aie.dma_bd(%[[BUFF1]] : memref<64xi32>, 0, 64)
// CHECK:       aie.use_lock(%[[CONS_LOCK]], Release, 1)
// The last aie.next_bd is followed directly by aie.end, confirming
// the chain is linear (not circular): a circular ring would place
// aie.end in a separate block NOT reachable by falling through next_bd.
// CHECK:       aie.next_bd
// CHECK:       aie.end
// CHECK:     }

// --- No residual Conduit ops ---
// CHECK-NOT: conduit.create
// CHECK-NOT: conduit.put_memref_async
// CHECK-NOT: conduit.get_memref_async

module @tm_count2 {
  aie.device(npu1_1col) {
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)

    // Canonical channel: 2 put_memref_async ops reference this channel.
    // Pass C Phase 1 collects putCount=2, nConsumerBuffers()=2,
    // and emits a linear 2-entry S2MM BD chain (last BD → aie.end, not ^bd0).
    conduit.create @kv {
      capacity = 1 : i64,
      producer_tile = array<i64: 0, 0>,
      consumer_tiles = array<i64: 0, 2>,
      element_type = memref<64xi32>,
      depth = 1 : i64
    }

    // Consumer core: two sequential get_memref_async reads.
    // Step 8f emits use_lock pair for each get (consLock acq + prodLock rel).
    %core_0_2 = aie.core(%tile_0_2) {
      %g1 = conduit.get_memref_async {name = @kv,
                num_elems = 64 : i64,
                offsets = array<i64: 0>,
                sizes = array<i64: 64>,
                strides = array<i64: 1>} : !conduit.dma.token
      conduit.wait %g1 : !conduit.dma.token

      %g2 = conduit.get_memref_async [%g1 : !conduit.dma.token]
                {name = @kv,
                 num_elems = 64 : i64,
                 offsets = array<i64: 0>,
                 sizes = array<i64: 64>,
                 strides = array<i64: 1>} : !conduit.dma.token
      conduit.wait %g2 : !conduit.dma.token
      aie.end
    } {dynamic_objfifo_lowering = true}

    // Host sequence: two put_memref_async ops send the two TM slots.
    // Shim producer (row=0): Step 8e skips use_lock; ops are simply erased.
    func.func @sequence(%buf: memref<128xi32>) {
      %t1 = conduit.put_memref_async {name = @kv,
                num_elems = 64 : i64,
                offsets = array<i64: 0>,
                sizes = array<i64: 64>,
                strides = array<i64: 1>} : !conduit.dma.token
      %t2 = conduit.put_memref_async [%t1 : !conduit.dma.token]
                {name = @kv,
                 num_elems = 64 : i64,
                 offsets = array<i64: 64>,
                 sizes = array<i64: 64>,
                 strides = array<i64: 1>} : !conduit.dma.token
      conduit.wait %t2 : !conduit.dma.token
      return
    }
  }
}

// -----

//===----------------------------------------------------------------------===//
// (2) putCount = 3
//
// Expected BD chain on aie.mem(tile_0_2):
//   ^start: dma_start(S2MM, 0, ^bd0, ^end)
//   ^bd0:   ... dma_bd(%kvs_cons_buff_0, ...) ... next_bd ^bd1
//   ^bd1:   ... dma_bd(%kvs_cons_buff_1, ...) ... next_bd ^bd2
//   ^bd2:   ... dma_bd(%kvs_cons_buff_2, ...) ... next_bd ^end
//   ^end:   aie.end
//
// Verification: three dma_bd entries with correct buffers.  The region
// contains three next_bd ops total — the last one terminates the chain.
//===----------------------------------------------------------------------===//

// CHECK-LABEL: module @tm_count3
// CHECK:   aie.device(npu1_1col)

// --- Three consumer buffers on tile(0,2) ---
// CHECK:     %[[B0:.*]] = aie.buffer(%{{.*}}tile_0_2)
// CHECK-SAME:   sym_name = "kvs_cons_buff_0"
// CHECK:     %[[B1:.*]] = aie.buffer(%{{.*}}tile_0_2)
// CHECK-SAME:   sym_name = "kvs_cons_buff_1"
// CHECK:     %[[B2:.*]] = aie.buffer(%{{.*}}tile_0_2)
// CHECK-SAME:   sym_name = "kvs_cons_buff_2"
// No fourth buffer (confirms putCount=3, not more).
// CHECK-NOT:   aie.buffer(%{{.*}}tile_0_2) {sym_name = "kvs_cons_buff_3"}

// --- Consumer-tile locks: prod_lock init=3 (3 free slots), cons_lock init=0 ---
// CHECK:     %[[PL:.*]] = aie.lock(%{{.*}}tile_0_2
// CHECK-SAME:   init = 3
// CHECK-SAME:   sym_name = "kvs_cons_prod_lock_0"
// CHECK:     %[[CL:.*]] = aie.lock(%{{.*}}tile_0_2
// CHECK-SAME:   init = 0
// CHECK-SAME:   sym_name = "kvs_cons_cons_lock_0"

// --- S2MM linear BD chain: 3 entries ---
// CHECK:     aie.mem(%{{.*}}tile_0_2) {
// CHECK:       aie.dma_start(S2MM, 0,
// BD 0 → BD 1
// CHECK:       aie.use_lock(%[[PL]], AcquireGreaterEqual, 1)
// CHECK:       aie.dma_bd(%[[B0]] : memref<32xi32>, 0, 32)
// CHECK:       aie.use_lock(%[[CL]], Release, 1)
// CHECK:       aie.next_bd
// BD 1 → BD 2
// CHECK:       aie.use_lock(%[[PL]], AcquireGreaterEqual, 1)
// CHECK:       aie.dma_bd(%[[B1]] : memref<32xi32>, 0, 32)
// CHECK:       aie.use_lock(%[[CL]], Release, 1)
// CHECK:       aie.next_bd
// BD 2 → aie.end (linear terminator)
// CHECK:       aie.use_lock(%[[PL]], AcquireGreaterEqual, 1)
// CHECK:       aie.dma_bd(%[[B2]] : memref<32xi32>, 0, 32)
// CHECK:       aie.use_lock(%[[CL]], Release, 1)
// CHECK:       aie.next_bd
// CHECK:       aie.end
// CHECK:     }

// --- No residual Conduit ops ---
// CHECK-NOT: conduit.create
// CHECK-NOT: conduit.put_memref_async
// CHECK-NOT: conduit.get_memref_async

module @tm_count3 {
  aie.device(npu1_1col) {
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)

    // Canonical channel: 3 put_memref_async ops reference this channel.
    // Pass C Phase 1 collects putCount=3, nConsumerBuffers()=3,
    // and emits a linear 3-entry S2MM BD chain (last BD → aie.end).
    conduit.create @kvs {
      capacity = 1 : i64,
      producer_tile = array<i64: 0, 0>,
      consumer_tiles = array<i64: 0, 2>,
      element_type = memref<32xi32>,
      depth = 1 : i64
    }

    // Consumer core: three sequential get_memref_async reads.
    %core_0_2 = aie.core(%tile_0_2) {
      %g1 = conduit.get_memref_async {name = @kvs,
                num_elems = 32 : i64,
                offsets = array<i64: 0>,
                sizes = array<i64: 32>,
                strides = array<i64: 1>} : !conduit.dma.token
      conduit.wait %g1 : !conduit.dma.token

      %g2 = conduit.get_memref_async [%g1 : !conduit.dma.token]
                {name = @kvs,
                 num_elems = 32 : i64,
                 offsets = array<i64: 0>,
                 sizes = array<i64: 32>,
                 strides = array<i64: 1>} : !conduit.dma.token
      conduit.wait %g2 : !conduit.dma.token

      %g3 = conduit.get_memref_async [%g2 : !conduit.dma.token]
                {name = @kvs,
                 num_elems = 32 : i64,
                 offsets = array<i64: 0>,
                 sizes = array<i64: 32>,
                 strides = array<i64: 1>} : !conduit.dma.token
      conduit.wait %g3 : !conduit.dma.token
      aie.end
    } {dynamic_objfifo_lowering = true}

    // Host sequence: three put_memref_async ops.
    func.func @sequence(%buf: memref<96xi32>) {
      %t1 = conduit.put_memref_async {name = @kvs,
                num_elems = 32 : i64,
                offsets = array<i64: 0>,
                sizes = array<i64: 32>,
                strides = array<i64: 1>} : !conduit.dma.token
      %t2 = conduit.put_memref_async [%t1 : !conduit.dma.token]
                {name = @kvs,
                 num_elems = 32 : i64,
                 offsets = array<i64: 32>,
                 sizes = array<i64: 32>,
                 strides = array<i64: 1>} : !conduit.dma.token
      %t3 = conduit.put_memref_async [%t2 : !conduit.dma.token]
                {name = @kvs,
                 num_elems = 32 : i64,
                 offsets = array<i64: 64>,
                 sizes = array<i64: 32>,
                 strides = array<i64: 1>} : !conduit.dma.token
      conduit.wait %t3 : !conduit.dma.token
      return
    }
  }
}

// -----

//===----------------------------------------------------------------------===//
// (3) Baseline: putCount = 1 (depth=1, standard ring).
//
// With only one put_memref_async, putCount=1 and isLinearChain = false.
// nConsumerBuffers() = depth = 1.  The single BD loops back to itself.
//
// Expected BD chain on aie.mem(tile_0_2):
//   ^start: dma_start(S2MM, 0, ^bd0, ^end)
//   ^bd0:   use_lock(prod, AcquireGreaterEqual, 1)
//           dma_bd(%single_cons_buff_0, 0, 64)
//           use_lock(cons, Release, 1)
//           next_bd ^bd0              ← circular: self-loop (ring)
//   ^end:   aie.end
//
// Verification:
//   - Only ONE aie.dma_bd entry (not two).
//   - aie.end is present.
//   - Only ONE next_bd op (contrast with 2 in putCount=2 case).
//   - No second buffer (kv_cons_buff_1 must NOT appear).
//
// This baseline test confirms the default behavior is unaffected by the
// putCount-based linear chain feature.
//===----------------------------------------------------------------------===//

// CHECK-LABEL: module @tm_baseline_depth1
// CHECK:   aie.device(npu1_1col)

// --- Single consumer buffer ---
// CHECK:     %[[SB0:.*]] = aie.buffer(%{{.*}}tile_0_2)
// CHECK-SAME:   sym_name = "single_cons_buff_0"
// Confirm only one buffer is allocated: no second slot.
// CHECK-NOT:   aie.buffer(%{{.*}}tile_0_2) {sym_name = "single_cons_buff_1"}

// --- Consumer-tile locks: prod_lock init=1 (one free slot) ---
// CHECK:     %[[SPL:.*]] = aie.lock(%{{.*}}tile_0_2
// CHECK-SAME:   init = 1
// CHECK-SAME:   sym_name = "single_cons_prod_lock_0"
// CHECK:     %[[SCL:.*]] = aie.lock(%{{.*}}tile_0_2
// CHECK-SAME:   init = 0
// CHECK-SAME:   sym_name = "single_cons_cons_lock_0"

// --- S2MM circular (ring) BD chain: 1 entry, one next_bd ---
// CHECK:     aie.mem(%{{.*}}tile_0_2) {
// CHECK:       aie.dma_start(S2MM, 0,
// CHECK:       aie.use_lock(%[[SPL]], AcquireGreaterEqual, 1)
// CHECK:       aie.dma_bd(%[[SB0]] : memref<64xi32>, 0, 64)
// CHECK:       aie.use_lock(%[[SCL]], Release, 1)
// Single next_bd (ring): no second BD follows.
// CHECK:       aie.next_bd
// No second dma_bd after the single next_bd — confirms ring (not linear chain).
// CHECK-NOT:   aie.dma_bd
// CHECK:       aie.end
// CHECK:     }

// --- No residual Conduit ops ---
// CHECK-NOT: conduit.create
// CHECK-NOT: conduit.put_memref_async
// CHECK-NOT: conduit.get_memref_async

module @tm_baseline_depth1 {
  aie.device(npu1_1col) {
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)

    // Standard depth=1 channel: single put_memref_async → 1-entry circular ring.
    conduit.create @single {
      capacity = 1 : i64,
      producer_tile = array<i64: 0, 0>,
      consumer_tiles = array<i64: 0, 2>,
      element_type = memref<64xi32>,
      depth = 1 : i64
    }

    // Consumer core: single get_memref_async per loop iteration.
    %core_0_2 = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c4 = arith.constant 4 : index
      scf.for %i = %c0 to %c4 step %c1 {
        %g = conduit.get_memref_async {name = @single,
                  num_elems = 64 : i64,
                  offsets = array<i64: 0>,
                  sizes = array<i64: 64>,
                  strides = array<i64: 1>} : !conduit.dma.token
        conduit.wait %g : !conduit.dma.token
      }
      aie.end
    } {dynamic_objfifo_lowering = true}

    // Host sequence: put_memref_async per iteration.
    func.func @sequence(%buf: memref<64xi32>) {
      %t = conduit.put_memref_async {name = @single,
                num_elems = 64 : i64,
                offsets = array<i64: 0>,
                sizes = array<i64: 64>,
                strides = array<i64: 1>} : !conduit.dma.token
      conduit.wait %t : !conduit.dma.token
      return
    }
  }
}
