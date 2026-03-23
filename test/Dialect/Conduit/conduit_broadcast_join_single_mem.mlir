// RUN: aie-opt --objectfifo-to-conduit --conduit-to-dma %s | FileCheck %s
//
// Regression test for: Pass C emitting duplicate aie.mem blocks per compute
// tile when a tile is both a broadcast consumer AND a join source.
//
// Pattern: shim → broadcast → {tile_a, tile_b}
//                tile_a, tile_b → MemTile join → shim
//
// Each compute tile has two DMA channels:
//   S2MM ch0: receive broadcast data from shim
//   MM2S ch0: send output to MemTile
//
// Both channels must reside in ONE aie.mem block per tile.
// Bug: Pass C emitted two separate aie.mem blocks per tile (one per channel),
// which caused aiecc to silently discard the second block → hardware deadlock.
//
// CHECK-LABEL: aie.device(npu2)
// CHECK: %[[TILE_A:.*]] = aie.tile(0, 2)
// CHECK: %[[TILE_B:.*]] = aie.tile(0, 3)
//
// tile_a must have exactly ONE aie.mem block containing both S2MM and MM2S.
// CHECK:      = aie.mem(%[[TILE_A]])
// CHECK:        aie.dma_start(S2MM
// CHECK:        aie.dma_start(MM2S
//
// tile_b must have exactly ONE aie.mem block containing both S2MM and MM2S.
// CHECK:      = aie.mem(%[[TILE_B]])
// CHECK:        aie.dma_start(S2MM
// CHECK:        aie.dma_start(MM2S
//
// No further aie.mem for these tiles (would indicate the bug is present).
// CHECK-NOT: = aie.mem(%[[TILE_A]])
// CHECK-NOT: = aie.mem(%[[TILE_B]])

module @bcast_join_single_mem {
  aie.device(npu2) {
    %shim  = aie.tile(0, 0)
    %mem   = aie.tile(0, 1)
    %tile_a = aie.tile(0, 2)
    %tile_b = aie.tile(0, 3)

    // Broadcast: shim → {tile_a, tile_b}
    aie.objectfifo @bcast(%shim, {%tile_a, %tile_b}, [2, 2, 2])
        : !aie.objectfifo<memref<32xi32>>

    // Each tile produces output to MemTile
    aie.objectfifo @out_a(%tile_a, {%mem}, 2 : i32)
        : !aie.objectfifo<memref<32xi32>>
    aie.objectfifo @out_b(%tile_b, {%mem}, 2 : i32)
        : !aie.objectfifo<memref<32xi32>>

    // MemTile joins both outputs and sends to shim
    aie.objectfifo @out_mem(%mem, {%shim}, 2 : i32)
        : !aie.objectfifo<memref<64xi32>>
    aie.objectfifo.link [@out_a, @out_b] -> [@out_mem] ([0, 32] [])

    // Core for tile_a: passthrough from bcast → out_a (1 iteration)
    %core_a = aie.core(%tile_a) {
      %c0  = arith.constant 0  : index
      %c1  = arith.constant 1  : index
      %c32 = arith.constant 32 : index
      %in_sv  = aie.objectfifo.acquire @bcast(Consume, 1)
                    : !aie.objectfifosubview<memref<32xi32>>
      %in_buf  = aie.objectfifo.subview.access %in_sv[0]
                    : !aie.objectfifosubview<memref<32xi32>> -> memref<32xi32>
      %out_sv = aie.objectfifo.acquire @out_a(Produce, 1)
                    : !aie.objectfifosubview<memref<32xi32>>
      %out_buf = aie.objectfifo.subview.access %out_sv[0]
                    : !aie.objectfifosubview<memref<32xi32>> -> memref<32xi32>
      scf.for %i = %c0 to %c32 step %c1 {
        %val = memref.load %in_buf[%i]  : memref<32xi32>
        memref.store %val, %out_buf[%i] : memref<32xi32>
      }
      aie.objectfifo.release @out_a(Produce, 1)
      aie.objectfifo.release @bcast(Consume, 1)
      aie.end
    } {dynamic_objfifo_lowering = true}

    // Core for tile_b: passthrough from bcast → out_b (1 iteration)
    %core_b = aie.core(%tile_b) {
      %c0  = arith.constant 0  : index
      %c1  = arith.constant 1  : index
      %c32 = arith.constant 32 : index
      %in_sv  = aie.objectfifo.acquire @bcast(Consume, 1)
                    : !aie.objectfifosubview<memref<32xi32>>
      %in_buf  = aie.objectfifo.subview.access %in_sv[0]
                    : !aie.objectfifosubview<memref<32xi32>> -> memref<32xi32>
      %out_sv = aie.objectfifo.acquire @out_b(Produce, 1)
                    : !aie.objectfifosubview<memref<32xi32>>
      %out_buf = aie.objectfifo.subview.access %out_sv[0]
                    : !aie.objectfifosubview<memref<32xi32>> -> memref<32xi32>
      scf.for %i = %c0 to %c32 step %c1 {
        %val = memref.load %in_buf[%i]  : memref<32xi32>
        memref.store %val, %out_buf[%i] : memref<32xi32>
      }
      aie.objectfifo.release @out_b(Produce, 1)
      aie.objectfifo.release @bcast(Consume, 1)
      aie.end
    } {dynamic_objfifo_lowering = true}

    aie.runtime_sequence(%arg_in: memref<32xi32>, %arg_out: memref<64xi32>) {
      aiex.npu.dma_memcpy_nd (%arg_in[0, 0, 0, 0][1, 1, 1, 32][0, 0, 0, 1])
          {metadata = @bcast, id = 0 : i64} : memref<32xi32>
      aiex.npu.dma_memcpy_nd (%arg_out[0, 0, 0, 0][1, 1, 1, 64][0, 0, 0, 1])
          {metadata = @out_mem, id = 1 : i64} : memref<64xi32>
      aiex.npu.dma_wait {symbol = @out_mem}
    }
  }
}
