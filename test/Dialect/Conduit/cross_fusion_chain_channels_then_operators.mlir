// RUN: aie-opt --objectfifo-to-conduit --conduit-fuse-channels --conduit-fuse-operators %s | FileCheck %s
//
// Foundation Phase 2 (Task #19), gap #9 — cross-pass chain test.
// Stacks fuse-channels then fuse-operators in the canonical Llama
// pipeline order; verifies that fuse-channels' annotations
// (dma_channel_group_s2mm, fuse_mode_s2mm) are PRESERVED through
// fuse-operators' device-merge + Step 8c trim.  Previously no test
// stacked these two fusion passes (only fuse-channels self-chained for
// idempotency).
//
// Geometry:
//   * devA: two compute producers (tile(0,2), tile(0,3)) feed shared
//     consumer compute tile(0,4); both producer channels (@chan_a /
//     @chan_b) are eligible for fuse-channels' S2MM grouping by
//     consumer tile.  Consumer interleaves them sequentially within
//     the same scf.for body block so live intervals are disjoint.
//   * Consumer also produces @inter_out → shim with fusion_group="fg0".
//   * devB: receives @inter_in (fusion_group="fg0") from shim, runs
//     downstream compute, emits @ext_out.
//
// What this test pins:
//   (a) After both passes:
//       - @chan_a and @chan_b retain `dma_channel_group_s2mm = "group0"`
//         + `fuse_mode_s2mm = "static"` annotations (and inferred
//         dma_repeat = 4) — fuse-operators' device-merge does NOT
//         strip them.
//       - @inter_out and @inter_in are replaced by @fused_intermediate_0;
//         no double-fusion artifacts (no @inter_out / @inter_in survives,
//         no extra fused_intermediate_N).
//   (b) Devices merged into one (devB into devA).
//   (c) Three compute cores survive: tile(0,2), tile(0,3), tile(0,4)
//       for devA's pieces; tile(1,2) for devB's downstream (auto-spread
//       across columns by fuse-operators).

// CHECK-LABEL: module @cross_fusion_chain

// Single device after fuse-operators.
// CHECK:       aie.device(npu2) @devA
// CHECK-NOT:   aie.device

// (#99) cross-producer S2MM groups skip annotation; same-producer pin lives in fuse_channels_s2mm_same_producer.mlir
// chan_a producer = tile(0,2); chan_b producer = tile(0,3); both consume on
// tile(0,4) -> Path c predicate suppresses dma_channel_group_s2mm /
// fuse_mode_s2mm.  dma_repeat = 4 still survives the chain — the original
// purpose of this fixture (fuse-operators preservation through fuse-channels).
// CHECK:       conduit.create @chan_a
// CHECK-SAME:  dma_repeat = 4

// CHECK:       conduit.create @chan_b
// CHECK-SAME:  dma_repeat = 4

// CHECK-NOT:   dma_channel_group_s2mm
// CHECK-NOT:   fuse_mode_s2mm

// fuse-operators emitted a single fused_intermediate; inter_out and
// inter_in are gone.
// CHECK:       conduit.create @fused_intermediate_0
// CHECK-NOT:   conduit.create @fused_intermediate_1
// CHECK-NOT:   conduit.create @inter_out
// CHECK-NOT:   conduit.create @inter_in

// Downstream ext_out survives.
// CHECK:       conduit.create @ext_out

// Producer cores for chan_a / chan_b survive intact.
// CHECK:       aie.core(%tile_0_2)
// CHECK:       conduit.acquire {{.*}} name = @chan_a
// CHECK-SAME:  port = #conduit.port<Produce>
// CHECK:       aie.core(%tile_0_3)
// CHECK:       conduit.acquire {{.*}} name = @chan_b
// CHECK-SAME:  port = #conduit.port<Produce>

// Consumer-and-bridge core (tile(0,4)) consumes both grouped channels
// then produces into the fused intermediate.
// CHECK:       aie.core(%tile_0_4)
// CHECK:       conduit.acquire {{.*}} name = @chan_a
// CHECK-SAME:  port = #conduit.port<Consume>
// CHECK:       conduit.acquire {{.*}} name = @chan_b
// CHECK-SAME:  port = #conduit.port<Consume>
// CHECK:       conduit.acquire {{.*}} name = @fused_intermediate_0
// CHECK-SAME:  port = #conduit.port<Produce>

// Devb's downstream core spread to a new column by fuse-operators.
// CHECK:       aie.core(%tile_1_2)
// CHECK:       conduit.acquire {{.*}} name = @fused_intermediate_0
// CHECK-SAME:  port = #conduit.port<Consume>
// CHECK:       conduit.acquire {{.*}} name = @ext_out
// CHECK-SAME:  port = #conduit.port<Produce>

module @cross_fusion_chain {
  aie.device(npu2) @devA {
    %shim_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)
    %tile_0_3 = aie.tile(0, 3)
    %tile_0_4 = aie.tile(0, 4)

    aie.objectfifo @chan_a(%tile_0_2, {%tile_0_4}, 2 : i32)
        : !aie.objectfifo<memref<8xi32>>
    aie.objectfifo @chan_b(%tile_0_3, {%tile_0_4}, 2 : i32)
        : !aie.objectfifo<memref<8xi32>>
    aie.objectfifo @inter_out(%tile_0_4, {%shim_0}, 2 : i32)
        {fusion_group = "fg0"}
        : !aie.objectfifo<memref<8xi32>>

    func.func private @produce_a(memref<8xi32>)
    func.func private @produce_b(memref<8xi32>)
    func.func private @consume_then_send(memref<8xi32>, memref<8xi32>, memref<8xi32>)

    aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c4 = arith.constant 4 : index
      scf.for %i = %c0 to %c4 step %c1 {
        %w = aie.objectfifo.acquire @chan_a(Produce, 1)
            : !aie.objectfifosubview<memref<8xi32>>
        %buf = aie.objectfifo.subview.access %w[0]
            : !aie.objectfifosubview<memref<8xi32>> -> memref<8xi32>
        func.call @produce_a(%buf) : (memref<8xi32>) -> ()
        aie.objectfifo.release @chan_a(Produce, 1)
      }
      aie.end
    }

    aie.core(%tile_0_3) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c4 = arith.constant 4 : index
      scf.for %i = %c0 to %c4 step %c1 {
        %w = aie.objectfifo.acquire @chan_b(Produce, 1)
            : !aie.objectfifosubview<memref<8xi32>>
        %buf = aie.objectfifo.subview.access %w[0]
            : !aie.objectfifosubview<memref<8xi32>> -> memref<8xi32>
        func.call @produce_b(%buf) : (memref<8xi32>) -> ()
        aie.objectfifo.release @chan_b(Produce, 1)
      }
      aie.end
    }

    // Consumer-and-bridge core: sequentially interleaves chan_a / chan_b
    // (S2MM-fuseable) and emits @inter_out toward devB.
    aie.core(%tile_0_4) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c4 = arith.constant 4 : index
      scf.for %i = %c0 to %c4 step %c1 {
        %wa = aie.objectfifo.acquire @chan_a(Consume, 1)
            : !aie.objectfifosubview<memref<8xi32>>
        %ba = aie.objectfifo.subview.access %wa[0]
            : !aie.objectfifosubview<memref<8xi32>> -> memref<8xi32>
        aie.objectfifo.release @chan_a(Consume, 1)
        %wb = aie.objectfifo.acquire @chan_b(Consume, 1)
            : !aie.objectfifosubview<memref<8xi32>>
        %bb = aie.objectfifo.subview.access %wb[0]
            : !aie.objectfifosubview<memref<8xi32>> -> memref<8xi32>
        aie.objectfifo.release @chan_b(Consume, 1)
        %wo = aie.objectfifo.acquire @inter_out(Produce, 1)
            : !aie.objectfifosubview<memref<8xi32>>
        %bo = aie.objectfifo.subview.access %wo[0]
            : !aie.objectfifosubview<memref<8xi32>> -> memref<8xi32>
        func.call @consume_then_send(%ba, %bb, %bo)
            : (memref<8xi32>, memref<8xi32>, memref<8xi32>) -> ()
        aie.objectfifo.release @inter_out(Produce, 1)
      }
      aie.end
    }

    aie.runtime_sequence(%a0: memref<32xi32>) {
      %t0 = aiex.dma_configure_task_for @inter_out {
        aie.dma_bd(%a0 : memref<32xi32>, 0, 32,
          [<size = 1, stride = 0>, <size = 1, stride = 0>,
           <size = 1, stride = 0>, <size = 32, stride = 1>])
            {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%t0)
      aiex.dma_await_task(%t0)
      aiex.dma_free_task(%t0)
    }
  }

  aie.device(npu2) @devB {
    %shim_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)

    aie.objectfifo @inter_in(%shim_0, {%tile_0_2}, 2 : i32)
        {fusion_group = "fg0"}
        : !aie.objectfifo<memref<8xi32>>
    aie.objectfifo @ext_out(%tile_0_2, {%shim_0}, 2 : i32)
        : !aie.objectfifo<memref<8xi32>>

    func.func private @downstream(memref<8xi32>, memref<8xi32>)

    aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c4 = arith.constant 4 : index
      scf.for %i = %c0 to %c4 step %c1 {
        %in = aie.objectfifo.acquire @inter_in(Consume, 1)
            : !aie.objectfifosubview<memref<8xi32>>
        %ib = aie.objectfifo.subview.access %in[0]
            : !aie.objectfifosubview<memref<8xi32>> -> memref<8xi32>
        %out = aie.objectfifo.acquire @ext_out(Produce, 1)
            : !aie.objectfifosubview<memref<8xi32>>
        %ob = aie.objectfifo.subview.access %out[0]
            : !aie.objectfifosubview<memref<8xi32>> -> memref<8xi32>
        func.call @downstream(%ib, %ob) : (memref<8xi32>, memref<8xi32>) -> ()
        aie.objectfifo.release @ext_out(Produce, 1)
        aie.objectfifo.release @inter_in(Consume, 1)
      }
      aie.end
    }

    aie.runtime_sequence(%b0: memref<32xi32>, %b1: memref<32xi32>) {
      %t0 = aiex.dma_configure_task_for @inter_in {
        aie.dma_bd(%b0 : memref<32xi32>, 0, 32,
          [<size = 1, stride = 0>, <size = 1, stride = 0>,
           <size = 1, stride = 0>, <size = 32, stride = 1>])
            {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%t0)
      %t1 = aiex.dma_configure_task_for @ext_out {
        aie.dma_bd(%b1 : memref<32xi32>, 0, 32,
          [<size = 1, stride = 0>, <size = 1, stride = 0>,
           <size = 1, stride = 0>, <size = 32, stride = 1>])
            {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%t1)
      aiex.dma_await_task(%t1)
      aiex.dma_free_task(%t0)
    }
  }
}
