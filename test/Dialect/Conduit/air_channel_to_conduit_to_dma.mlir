// RUN: aie-opt --allow-unregistered-dialect --air-channel-to-conduit --conduit-to-dma %s | FileCheck %s
//
// Pass B + Pass C end-to-end test: air.channel.put/get → conduit Tier 3 → aie hardware ops.
//
// Pipeline: --allow-unregistered-dialect --air-channel-to-conduit --conduit-to-dma
//
// Design note on tile placement:
//   Pass B (--air-channel-to-conduit) converts air.channel.put/get to
//   conduit.put_memref_async/get_memref_async, and emits conduit.create for each
//   air.channel declaration.  The emitted conduit.create has empty producer_tile /
//   consumer_tiles because AIR channel ops do not carry tile coordinates — those
//   come from a separate tile-placement step between Pass B and Pass C.
//
//   This test simulates that placement step by including a pre-specified conduit.create
//   with tile info alongside the air.channel declaration.  Pass B converts the
//   air.channel.put/get ops; Pass C lowers the conduit.create (with tile info)
//   to hardware ops (aie.buffer, aie.lock, aie.flow, aie.mem with DMA BDs).
//
// Input:
//   - aie.device(npu1_1col) with two tiles: shim(0,0) and compute(0,2)
//   - air.channel declaration @mychan [1, 1]
//   - conduit.create {name = "mychan"} with tile info (producer_tile=[0,0] shim,
//     consumer_tiles=[0,2] compute) and element_type=memref<64xi32>
//   - air.channel.put and air.channel.get with 1-D descriptor:
//       offsets=[0], sizes=[64], strides=[1] → num_elems=64
//
// Expected after --air-channel-to-conduit:
//   - conduit.put_memref_async {name="mychan", num_elems=64, offsets=[0], sizes=[64], strides=[1]}
//   - conduit.get_memref_async {same attrs}
//   - No residual air.channel / air.wait_all ops
//
// Expected after --conduit-to-dma:
//   - aie.buffer on tile(0,2) : memref<64xi32>
//   - aie.lock (prod_lock, init=1) and (cons_lock, init=0) on tile(0,2)
//   - aie.shim_dma_allocation on tile(0,0)
//   - aie.flow: tile(0,0) DMA:0 → tile(0,2) DMA:0
//   - aie.mem on tile(0,2) with aie.dma_start(S2MM, 0, ...) and aie.dma_bd
//
// Known gap (not yet in Pass C): conduit.put_memref_async / conduit.get_memref_async
// are not yet lowered to aie.dma_bd on the producer side.  They remain in the
// func body.  This is a known gap documented in ConduitToDMA.cpp.

// CHECK-LABEL: module
// CHECK:   aie.device(npu1_1col) {

// --- Consumer-tile buffer: conduit.create → aie.buffer ---
// CHECK:     aie.buffer(%{{.*}}tile_0_2)
// CHECK-SAME:   sym_name = "mychan_cons_buff_0"

// --- Consumer-tile locks ---
// CHECK:     %[[PROD_LOCK:.*]] = aie.lock(%{{.*}}tile_0_2
// CHECK-SAME:   init = 1
// CHECK-SAME:   sym_name = "mychan_cons_prod_lock_0"
// CHECK:     %[[CONS_LOCK:.*]] = aie.lock(%{{.*}}tile_0_2
// CHECK-SAME:   init = 0
// CHECK-SAME:   sym_name = "mychan_cons_cons_lock_0"

// --- conduit.put_memref_async remains in func body (Pass C has not yet lowered it) ---
// conduit.put_memref_async now has correct attrs from Pass B (num_elems=64, static descriptor).
// CHECK:     func.func @test(
// CHECK:       conduit.put_memref_async
// CHECK-SAME:   name = "mychan"
// CHECK-SAME:   num_elems = 64
// CHECK-SAME:   offsets = array<i64: 0>
// CHECK-SAME:   sizes = array<i64: 64>
// CHECK-SAME:   strides = array<i64: 1>
// CHECK-SAME:   : !conduit.dma.token
// CHECK:       conduit.get_memref_async
// CHECK-SAME:   name = "mychan"
// CHECK-SAME:   num_elems = 64

// --- Shim-side locks ---
// CHECK:     aie.lock(%{{.*}}tile_0_0
// CHECK-SAME:   sym_name = "mychan_prod_lock_0"
// CHECK:     aie.lock(%{{.*}}tile_0_0
// CHECK-SAME:   sym_name = "mychan_cons_lock_0"

// --- Shim DMA allocation and flow ---
// CHECK:     aie.shim_dma_allocation @mychan_shim_alloc
// CHECK:     aie.flow(%{{.*}}tile_0_0, DMA : 0, %{{.*}}tile_0_2, DMA : 0)

// --- Tile DMA region: conduit.create → aie.mem with S2MM BD ---
// CHECK:     aie.mem(%{{.*}}tile_0_2) {
// CHECK:       aie.dma_start(S2MM, 0,
// CHECK:       aie.use_lock(%[[PROD_LOCK]], AcquireGreaterEqual, 1)
// CHECK:       aie.dma_bd(%{{.*}}mychan_cons_buff_0
// CHECK:       aie.use_lock(%[[CONS_LOCK]], Release, 1)
// CHECK:       aie.next_bd
// CHECK:       aie.end
// CHECK:     }

// --- No residual air.channel or conduit.create ops ---
// CHECK-NOT: air.channel
// CHECK-NOT: conduit.create

module {
  aie.device(npu1_1col) {
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)

    // AIR channel declaration (Pass B converts this to conduit.create without tile info).
    "air.channel"() {sym_name = "mychan", size = [1, 1]} : () -> ()

    // conduit.create with tile placement already filled in (simulates the
    // tile-placement step that runs between Pass B and Pass C).
    // producer_tile=[0,0] = shim tile, consumer_tiles=[0,2] = compute tile.
    conduit.create {name = "mychan", capacity = 1 : i64, depth = 1 : i64,
                    element_type = memref<64xi32>,
                    producer_tile = array<i64: 0, 0>,
                    consumer_tiles = array<i64: 0, 2>}

    func.func @test(%src: memref<64xi32>, %dst: memref<64xi32>) {
      %c0 = arith.constant 0 : index
      %c64 = arith.constant 64 : index
      %c1 = arith.constant 1 : index

      // air.channel.put async with 1-D descriptor:
      //   offsets=[0], sizes=[64], strides=[1], num_elems=64
      // operand_segment_sizes = [ndeps=0, nidx=0, nmemref=1, noffsets=1, nsizes=1, nstrides=1]
      %tok0 = "air.channel.put"(%src, %c0, %c64, %c1)
          {chan_name = @mychan,
           operand_segment_sizes = array<i32: 0, 0, 1, 1, 1, 1>}
          : (memref<64xi32>, index, index, index) -> !air.async.token

      // air.channel.get async with same descriptor
      %tok1 = "air.channel.get"(%dst, %c0, %c64, %c1)
          {chan_name = @mychan,
           operand_segment_sizes = array<i32: 0, 0, 1, 1, 1, 1>}
          : (memref<64xi32>, index, index, index) -> !air.async.token

      return
    }
  }
}
