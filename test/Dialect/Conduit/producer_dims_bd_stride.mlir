// RUN: aie-opt --conduit-to-dma %s | FileCheck %s
//
// Regression test: producer_dimensions on conduit.create produces DMA BD
// ops with stride/size dimensions (not linear 1D).
//
// When conduit.create has producer_dimensions set, the --conduit-to-dma
// pipeline must emit aie.dma_bd with BDDimLayout dimensions on the
// producer MM2S side.  This ensures non-linear memory access patterns
// (e.g., tiled matrix layouts for flash attention) are correctly
// programmed into the DMA engine.
//
// Topology:
//   tile(0,2) = producer (compute tile)
//   tile(0,4) = consumer (compute tile, non-adjacent → requires DMA)
//
// The conduit has producer_dimensions = [<size=4, stride=8>, <size=8, stride=1>]
// which describes a 4x8 tile within a larger buffer with stride 8 on the
// outer dimension.
//
// After --conduit-to-dma, the producer tile's aie.mem should contain an
// aie.dma_bd with the dimensions attribute, while the consumer BD should
// have no dimensions (linear access).

// CHECK-LABEL: module @producer_dims_test

// Verify the producer tile DMA region contains an MM2S BD with dimensions:
// CHECK:       aie.mem(%{{.*}}tile_0_2)
// CHECK:         aie.dma_start(MM2S, 0,
// CHECK:         aie.dma_bd(%{{.*}} : memref<32xi32>, 0, 32, [<size = 4, stride = 8>, <size = 8, stride = 1>])

// Verify the consumer tile DMA region has a plain BD (no dimensions):
// CHECK:       aie.mem(%{{.*}}tile_0_4)
// CHECK:         aie.dma_start(S2MM, 0,
// CHECK:         aie.dma_bd(%{{.*}} : memref<32xi32>, 0, 32)
// CHECK-NOT:   <size =

module @producer_dims_test {
  aie.device(npu1_1col) {
    func.func @process(%buf: memref<32xi32>) -> () {
      return
    }

    %tile_0_2 = aie.tile(0, 2)
    %tile_0_4 = aie.tile(0, 4)

    // Conduit with producer_dimensions: 4x8 tiled access with stride 8.
    conduit.create @chan {
      element_type = memref<32xi32>,
      depth = 1 : i64,
      producer_dimensions = #aie<bd_dim_layout_array[<size = 4, stride = 8>, <size = 8, stride = 1>]>
    }

    // Producer core on tile(0,2).
    aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c4 = arith.constant 4 : index
      scf.for %i = %c0 to %c4 step %c1 {
        %w = conduit.acquire {name = @chan, count = 1 : i64,
                              port = #conduit.port<Produce>}
                : !conduit.window<memref<32xi32>>
        %buf = conduit.subview_access %w {index = 0 : i64}
                  : !conduit.window<memref<32xi32>> -> memref<32xi32>
        func.call @process(%buf) : (memref<32xi32>) -> ()
        conduit.release %w {count = 1 : i64, port = #conduit.port<Produce>}
            : !conduit.window<memref<32xi32>>
      }
      aie.end
    } {dynamic_objfifo_lowering = true}

    // Consumer core on tile(0,4) — non-adjacent to tile(0,2), requires DMA.
    aie.core(%tile_0_4) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c4 = arith.constant 4 : index
      scf.for %i = %c0 to %c4 step %c1 {
        %w = conduit.acquire {name = @chan, count = 1 : i64,
                              port = #conduit.port<Consume>}
                : !conduit.window<memref<32xi32>>
        %buf = conduit.subview_access %w {index = 0 : i64}
                  : !conduit.window<memref<32xi32>> -> memref<32xi32>
        func.call @process(%buf) : (memref<32xi32>) -> ()
        conduit.release %w {count = 1 : i64, port = #conduit.port<Consume>}
            : !conduit.window<memref<32xi32>>
      }
      aie.end
    } {dynamic_objfifo_lowering = true}
  }
}
