// RUN: aie-opt --objectfifo-to-conduit --conduit-fuse-relay %s | FileCheck %s
//
// Task #30 — Pass A iter_count inference followed by --conduit-fuse-relay.
//
// Closes the audit gap noted in the foundation Phase 1 lit review:
// existing tests pin Pass A × {fuse-channels, fuse-core-bodies, fuse-operators},
// but no test pinned Pass A × fuse-relay until now.
//
// What fuse-relay does (see ConduitFuseRelay.cpp): when a `conduit.gather`
// (N→1, from a join `aie.objectfifo.link`) feeds a `conduit.scatter`
// (1→M, from a distribute link) through the SAME memtile, and the
// intermediate channel has no acquire/release/put_memref/get_memref
// users (a pure relay), fuse-relay collapses the pair into one
// `conduit.transpose` and erases the intermediate `conduit.create`.
//
// Topology:
//   producer(0,2) ─┐
//                  ├─join─► @intermediate (memtile 0,1) ─dist─┬─► consumer(0,4)
//   producer(0,3) ─┘                                          └─► consumer(0,5)
//
// Pass A behavior on this shape (compute-to-compute fifos, emit.count == 0;
// the new emit.count == 1 skip in ObjectFifoToConduit.cpp does NOT apply):
//   * @s0, @s1, @d0, @d1 are all compute-↔-memtile fifos with finite
//     scf.for trip = 4 in their owning core → dma_repeat = 4 inferred.
//   * @intermediate has no compute-side acquire/release (it's a pure
//     relay between two link ops) → Pass A leaves dma_repeat unset.
//
// fuse-relay then erases @intermediate and replaces gather+scatter with
// one conduit.transpose; the dma_repeat attrs on the four endpoints are
// untouched because fuse-relay does not rewrite the source/dest createOps
// (it only reads gather/scatter offsets+memtile and erases the intermediate
// create). This test pins that preservation.

// CHECK-LABEL: module @infer_then_fuse_relay

// Endpoint createOps keep their inferred dma_repeat across fuse-relay.
// CHECK:       conduit.create @s0
// CHECK-SAME:  dma_repeat = 4
// CHECK:       conduit.create @s1
// CHECK-SAME:  dma_repeat = 4
// Intermediate relay createOp must be erased by fuse-relay.
// CHECK-NOT:   conduit.create @intermediate
// CHECK:       conduit.create @d0
// CHECK-SAME:  dma_repeat = 4
// CHECK:       conduit.create @d1
// CHECK-SAME:  dma_repeat = 4

// Gather + scatter must be collapsed into one conduit.transpose.
// CHECK-NOT:   conduit.gather
// CHECK-NOT:   conduit.scatter
// CHECK:       conduit.transpose
// CHECK-SAME:  srcs = {{[[]}}[@s0, @s1]{{[]]}}
// CHECK-SAME:  dsts = {{[[]}}[@d0, @d1]{{[]]}}
// CHECK-SAME:  memtile = "tile(0,1)"

module @infer_then_fuse_relay {
  aie.device(npu1_1col) {
    %tile_0_1 = aie.tile(0, 1)
    %tile_0_2 = aie.tile(0, 2)
    %tile_0_3 = aie.tile(0, 3)
    %tile_0_4 = aie.tile(0, 4)
    %tile_0_5 = aie.tile(0, 5)

    // Two compute→memtile inputs.
    aie.objectfifo @s0(%tile_0_2, {%tile_0_1}, 2 : i32)
        : !aie.objectfifo<memref<8xi32>>
    aie.objectfifo @s1(%tile_0_3, {%tile_0_1}, 2 : i32)
        : !aie.objectfifo<memref<8xi32>>

    // Memtile-internal relay fifo (pure relay, no compute users).
    aie.objectfifo @intermediate(%tile_0_1, {%tile_0_1}, 2 : i32)
        : !aie.objectfifo<memref<16xi32>>

    // Two memtile→compute outputs.
    aie.objectfifo @d0(%tile_0_1, {%tile_0_4}, 2 : i32)
        : !aie.objectfifo<memref<8xi32>>
    aie.objectfifo @d1(%tile_0_1, {%tile_0_5}, 2 : i32)
        : !aie.objectfifo<memref<8xi32>>

    // Join (N→1) lowers to conduit.gather; memtile = producer of @intermediate.
    aie.objectfifo.link [@s0, @s1] -> [@intermediate] ([0, 32] [])
    // Distribute (1→N) lowers to conduit.scatter; same memtile.
    aie.objectfifo.link [@intermediate] -> [@d0, @d1] ([] [0, 32])

    func.func private @produce(memref<8xi32>)
    func.func private @consume(memref<8xi32>)

    // Producer A.
    aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c4 = arith.constant 4 : index
      %c1 = arith.constant 1 : index
      scf.for %i = %c0 to %c4 step %c1 {
        %w = aie.objectfifo.acquire @s0(Produce, 1)
            : !aie.objectfifosubview<memref<8xi32>>
        %buf = aie.objectfifo.subview.access %w[0]
            : !aie.objectfifosubview<memref<8xi32>> -> memref<8xi32>
        func.call @produce(%buf) : (memref<8xi32>) -> ()
        aie.objectfifo.release @s0(Produce, 1)
      }
      aie.end
    }

    // Producer B.
    aie.core(%tile_0_3) {
      %c0 = arith.constant 0 : index
      %c4 = arith.constant 4 : index
      %c1 = arith.constant 1 : index
      scf.for %i = %c0 to %c4 step %c1 {
        %w = aie.objectfifo.acquire @s1(Produce, 1)
            : !aie.objectfifosubview<memref<8xi32>>
        %buf = aie.objectfifo.subview.access %w[0]
            : !aie.objectfifosubview<memref<8xi32>> -> memref<8xi32>
        func.call @produce(%buf) : (memref<8xi32>) -> ()
        aie.objectfifo.release @s1(Produce, 1)
      }
      aie.end
    }

    // Consumer A.
    aie.core(%tile_0_4) {
      %c0 = arith.constant 0 : index
      %c4 = arith.constant 4 : index
      %c1 = arith.constant 1 : index
      scf.for %i = %c0 to %c4 step %c1 {
        %w = aie.objectfifo.acquire @d0(Consume, 1)
            : !aie.objectfifosubview<memref<8xi32>>
        %buf = aie.objectfifo.subview.access %w[0]
            : !aie.objectfifosubview<memref<8xi32>> -> memref<8xi32>
        func.call @consume(%buf) : (memref<8xi32>) -> ()
        aie.objectfifo.release @d0(Consume, 1)
      }
      aie.end
    }

    // Consumer B.
    aie.core(%tile_0_5) {
      %c0 = arith.constant 0 : index
      %c4 = arith.constant 4 : index
      %c1 = arith.constant 1 : index
      scf.for %i = %c0 to %c4 step %c1 {
        %w = aie.objectfifo.acquire @d1(Consume, 1)
            : !aie.objectfifosubview<memref<8xi32>>
        %buf = aie.objectfifo.subview.access %w[0]
            : !aie.objectfifosubview<memref<8xi32>> -> memref<8xi32>
        func.call @consume(%buf) : (memref<8xi32>) -> ()
        aie.objectfifo.release @d1(Consume, 1)
      }
      aie.end
    }
  }
}
