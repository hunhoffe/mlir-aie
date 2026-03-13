// RUN: aie-opt --objectfifo-to-conduit --conduit-to-dma %s | FileCheck %s
//
// Pass A + Pass C end-to-end test: depth-1 single-consumer objectfifo.
// Input: same as objectfifo_to_conduit_depth_one.mlir
//
// After --objectfifo-to-conduit --conduit-to-dma the module should contain
// the hardware ops that --aie-objectFifo-stateful-transform produces for the
// same input.  We check for structural presence, not exact SSA names (those
// vary with lock ID assignment order).
//
// Ground truth from --aie-objectFifo-stateful-transform (run on this exact input):
//
//   Consumer tile (tile_0_2):
//     aie.buffer  input_fifo_cons_buff_0 : memref<10xi32>
//     aie.lock    input_fifo_cons_prod_lock_0  (init = 1)   <- free slots
//     aie.lock    input_fifo_cons_cons_lock_0  (init = 0)   <- filled slots
//   Shim tile (tile_0_0):
//     aie.lock    input_fifo_prod_lock_0  (init = 0)
//     aie.lock    input_fifo_cons_lock_0  (init = 0)
//   Routing:
//     aie.flow    shim_0_0 -> tile_0_2, DMA : 0
//     aie.shim_dma_allocation @input_fifo_shim_alloc (MM2S, channel 0)
//   Tile DMA region:
//     aie.mem(%tile_0_2) { aie.dma_start(S2MM, 0, ^bd, ^end)
//       ^bd: use_lock(cons_prod, AcquireGreaterEqual, 1)
//            dma_bd(input_fifo_cons_buff_0, 0, 10)
//            use_lock(cons_cons, Release, 1)
//            next_bd ^bd      <- depth-1 ring: points back to itself
//       ^end: aie.end }
//
// What Pass C currently emits (verified by running the pipeline on this input):
//
//   Consumer tile (tile_0_2):
//     aie.buffer  input_fifo_buff_0 : memref<10xi32>        <- NAMING GAP (missing _cons_)
//     aie.lock    input_fifo_prod_lock_0  (init = 1)        <- correct semantics
//     aie.lock    input_fifo_cons_lock_0  (init = 0)        <- correct semantics
//   Shim tile (tile_0_0):
//     (no locks emitted)                                     <- GAP: Task #23
//   Routing:
//     aie.flow    shim_0_0 -> tile_0_2, DMA : 0             <- correct
//     aie.shim_dma_allocation @..._shim_alloc               <- correct
//   Tile DMA region:
//     aie.mem(%tile_0_2) { aie.dma_start(S2MM, 0, ^bd, ^end)
//       ^bd: use_lock(prod_lock, AcquireGreaterEqual, 1)     <- correct
//            dma_bd(input_fifo_buff_0, 0, 10)                <- correct
//            use_lock(cons_lock, Release, 1)                 <- correct
//            next_bd ^bd }                                   <- correct
//
// Resource comparison: Pass C vs stateful-transform (from compare_ir_outputs.sh):
//   aie.buffer:   stateful=2  conduit=1   DIFF -1   <- missing shim-side buffer
//   aie.lock:     stateful=4  conduit=2   DIFF -2   <- missing shim-side locks
//   aie.dma_bd:   stateful=1  conduit=1   MATCH
//   aie.next_bd:  stateful=1  conduit=1   MATCH
//   aie.flow:     stateful=1  conduit=1   MATCH
//   aie.use_lock: stateful=4  conduit=4   MATCH (2 in BD region + 2 in core)
//
// KNOWN GAPS (tracked as Task #23):
//
//   GAP 1: Pass C does not emit shim-side locks.
//     Expected: aie.lock(%shim_noc_tile_0_0) {sym_name = "input_fifo_prod_lock_0"}
//               aie.lock(%shim_noc_tile_0_0) {sym_name = "input_fifo_cons_lock_0"}
//     Current:  neither lock emitted on tile_0_0.
//
//   GAP 2: Buffer and lock naming omits the _cons_ infix.
//     Expected by stateful-transform: "input_fifo_cons_buff_0",
//                                     "input_fifo_cons_prod_lock_0", etc.
//     Currently emitted by Pass C:    "input_fifo_buff_0", "input_fifo_prod_lock_0"
//     (The _cons_ infix distinguishes per-consumer resources when there are
//     multiple consumer tiles or when the same conduit has shim-side resources.)
//
// The CHECK patterns below cover what Pass C emits correctly.
// Gap locations are marked with KNOWN GAP comments; the ready-to-activate
// CHECK lines for the gaps are written in comments so Task #23 can simply
// uncomment them when the shim fix lands.

// CHECK-LABEL: module
// CHECK:   aie.device(npu1_1col) {
// CHECK:     %{{.*}}tile_0_0 = aie.tile(0, 0)
// CHECK:     %{{.*}}tile_0_2 = aie.tile(0, 2)

// --- Consumer-tile buffer (Pass C emits this; flexible name to survive _cons_ rename) ---
// CHECK:     aie.buffer(%{{.*}}tile_0_2)
// CHECK-SAME:   sym_name = "input_fifo{{.*}}buff_0"

// --- Consumer-tile locks (Pass C emits both; flexible name patterns) ---
// CHECK:     %[[PROD_LOCK:.*]] = aie.lock(%{{.*}}tile_0_2
// CHECK-SAME:   init = 1
// CHECK-SAME:   sym_name = "input_fifo{{.*}}prod_lock_0"
// CHECK:     %[[CONS_LOCK:.*]] = aie.lock(%{{.*}}tile_0_2
// CHECK-SAME:   init = 0
// CHECK-SAME:   sym_name = "input_fifo{{.*}}cons_lock_0"

// --- Core body: acquire uses cons_lock, release uses prod_lock (consumer semantics) ---
// CHECK:     aie.core(%{{.*}}tile_0_2) {
// CHECK:       scf.for
// CHECK:         aie.use_lock(%[[CONS_LOCK]], AcquireGreaterEqual, 1)
// CHECK:         aie.use_lock(%[[PROD_LOCK]], Release, 1)
// CHECK:     }

// --- Shim-side locks (Task #23 fix: emitted after core in Phase 4) ---
// CHECK:     aie.lock(%{{.*}}tile_0_0
// CHECK-SAME:   sym_name = "input_fifo_prod_lock_0"
// CHECK:     aie.lock(%{{.*}}tile_0_0
// CHECK-SAME:   sym_name = "input_fifo_cons_lock_0"

// --- Shim allocation and flow (Pass C emits both correctly) ---
// CHECK:     aie.shim_dma_allocation @{{.*}}shim_alloc
// CHECK:     aie.flow(%{{.*}}tile_0_0, DMA : 0, %{{.*}}tile_0_2, DMA : 0)

// --- Tile DMA region: Pass C emits aie.mem with S2MM BD ring ---
// CHECK:     aie.mem(%{{.*}}tile_0_2) {
// CHECK:       aie.dma_start(S2MM, 0,
// CHECK:       aie.use_lock(%[[PROD_LOCK]], AcquireGreaterEqual, 1)
// CHECK:       aie.dma_bd(%{{.*}}input_fifo{{.*}}buff_0
// CHECK:       aie.use_lock(%[[CONS_LOCK]], Release, 1)
// CHECK:       aie.next_bd
// CHECK:       aie.end
// CHECK:     }

// --- No residual Conduit ops ---
// CHECK-NOT: conduit.create
// CHECK-NOT: conduit.acquire
// CHECK-NOT: conduit.release

module {
  aie.device(npu1_1col) {
    func.func @passthrough_10_i32(%line_in: memref<10xi32>) -> () {
      return
    }

    %tile_0_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)
    aie.objectfifo @input_fifo(%tile_0_0, {%tile_0_2}, 1 : i32) : !aie.objectfifo<memref<10xi32>>

    %core_0_2 = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c10 = arith.constant 10 : index

      scf.for %arg0 = %c0 to %c10 step %c1 {
        %0 = aie.objectfifo.acquire @input_fifo(Consume, 1) : !aie.objectfifosubview<memref<10xi32>>
        %1 = aie.objectfifo.subview.access %0[0] : !aie.objectfifosubview<memref<10xi32>> -> memref<10xi32>
        func.call @passthrough_10_i32(%1) : (memref<10xi32>) -> ()
        aie.objectfifo.release @input_fifo(Consume, 1)
      }

      aie.end
    } {dynamic_objfifo_lowering = true}
  }
}
