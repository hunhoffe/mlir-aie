// RUN: aie-opt --conduit-prune-runtime-seq-args %s | FileCheck %s
//
// Test for --conduit-prune-runtime-seq-args pass.
//
// Models the post-fusion runtime_sequence shape that --aie-combine-device +
// --conduit-fuse-* produce when intermediate channels are erased: a merged
// aie.runtime_sequence with N declared block args but only M < N of them
// referenced by surviving conduit.put_memref_async / conduit.get_memref_async
// ops.  The pass must:
//   (1) drop the unreferenced args from the runtime_sequence signature,
//   (2) renumber surviving conduit ops' arg_index attrs to dense 0..M-1,
//   (3) leave alias-style cases (two ops sharing the same arg_index, e.g. a
//       K=2 convergent merge) intact — the slot stays live, both users keep
//       pointing at it (handling that aliasing is a separate fuse-operators
//       fix; this pass must not make it worse).
//
// Cases:
//   (a) Dead-args-after-fusion: 9 declared args, only {0, 7, 8} live (mirrors
//       the swiglu post-fusion shape).  Expect 3 surviving args; arg_index
//       remap 0→0, 7→1, 8→2.
//   (b) All-args-live: every declared arg is referenced.  Expect no-op
//       (signature and arg_index unchanged).
//   (c) Aliased-arg: two puts with arg_index = 0 (K=2 convergent merge);
//       middle args dead.  Expect single surviving slot; both puts still
//       point at arg_index = 0.

// CHECK-LABEL: aie.device(npu2_4col)
// Case (a) — dense 3-arg signature post-prune.
// CHECK: aie.runtime_sequence @case_a_dead_args(%arg0: memref<64xbf16>, %arg1: memref<64xbf16>, %arg2: memref<64xbf16>) {
// CHECK-NEXT: conduit.put_memref_async {arg_index = 0 : i64, name = @ext_in_a
// CHECK-NEXT: conduit.get_memref_async {arg_index = 1 : i64, name = @ext_out_a
// CHECK-NEXT: conduit.get_memref_async {arg_index = 2 : i64, name = @ext_out_b
module {
  aie.device(npu2_4col) {
    %shim_0_0 = aie.tile(0, 0)
    %shim_1_0 = aie.tile(1, 0)
    conduit.create @ext_in_a {consumer_tiles = array<i64: 0, 2>, depth = 2 : i64, element_type = memref<64xbf16>, producer_tile = array<i64: 0, 0>}
    aie.shim_dma_allocation @ext_in_a_shim_alloc(%shim_0_0, MM2S, 0) {conduit_channel = @ext_in_a}
    conduit.create @ext_out_a {depth = 1 : i64, element_type = memref<64xbf16>, producer_tile = array<i64: 0, 2>}
    aie.shim_dma_allocation @ext_out_a_shim_alloc(%shim_1_0, S2MM, 0) {conduit_channel = @ext_out_a}
    conduit.create @ext_out_b {depth = 1 : i64, element_type = memref<64xbf16>, producer_tile = array<i64: 0, 2>}
    aie.shim_dma_allocation @ext_out_b_shim_alloc(%shim_1_0, S2MM, 1) {conduit_channel = @ext_out_b}
    aie.runtime_sequence @case_a_dead_args(%arg0: memref<64xbf16>, %arg1: memref<64xbf16>, %arg2: memref<64xbf16>, %arg3: memref<64xbf16>, %arg4: memref<64xbf16>, %arg5: memref<64xbf16>, %arg6: memref<64xbf16>, %arg7: memref<64xbf16>, %arg8: memref<64xbf16>) {
      %0 = conduit.put_memref_async {arg_index = 0 : i64, name = @ext_in_a, num_elems = 64 : i64, offsets = array<i64: 0>, sizes = array<i64: 64>, strides = array<i64: 1>} : !conduit.dma.token
      %1 = conduit.get_memref_async {arg_index = 7 : i64, name = @ext_out_a, num_elems = 64 : i64, offsets = array<i64: 0>, sizes = array<i64: 64>, strides = array<i64: 1>} : !conduit.dma.token
      %2 = conduit.get_memref_async {arg_index = 8 : i64, name = @ext_out_b, num_elems = 64 : i64, offsets = array<i64: 0>, sizes = array<i64: 64>, strides = array<i64: 1>} : !conduit.dma.token
      conduit.wait_all %0 {token = false} : !conduit.dma.token
      conduit.wait_all %1 : !conduit.dma.token
      conduit.wait_all %2 : !conduit.dma.token
    }
  }
}

// -----

// Case (b) — all args live; pass must be a no-op (signature unchanged,
// arg_index unchanged).
// CHECK-LABEL: aie.device(npu2_4col)
// CHECK: aie.runtime_sequence @case_b_all_live(%arg0: memref<64xbf16>, %arg1: memref<64xbf16>) {
// CHECK-NEXT: conduit.put_memref_async {arg_index = 0 : i64, name = @ext_in_x
// CHECK-NEXT: conduit.get_memref_async {arg_index = 1 : i64, name = @ext_out_x
module {
  aie.device(npu2_4col) {
    %shim_0_0 = aie.tile(0, 0)
    conduit.create @ext_in_x {consumer_tiles = array<i64: 0, 2>, depth = 2 : i64, element_type = memref<64xbf16>, producer_tile = array<i64: 0, 0>}
    aie.shim_dma_allocation @ext_in_x_shim_alloc(%shim_0_0, MM2S, 0) {conduit_channel = @ext_in_x}
    conduit.create @ext_out_x {depth = 1 : i64, element_type = memref<64xbf16>, producer_tile = array<i64: 0, 2>}
    aie.shim_dma_allocation @ext_out_x_shim_alloc(%shim_0_0, S2MM, 0) {conduit_channel = @ext_out_x}
    aie.runtime_sequence @case_b_all_live(%arg0: memref<64xbf16>, %arg1: memref<64xbf16>) {
      %0 = conduit.put_memref_async {arg_index = 0 : i64, name = @ext_in_x, num_elems = 64 : i64, offsets = array<i64: 0>, sizes = array<i64: 64>, strides = array<i64: 1>} : !conduit.dma.token
      %1 = conduit.get_memref_async {arg_index = 1 : i64, name = @ext_out_x, num_elems = 64 : i64, offsets = array<i64: 0>, sizes = array<i64: 64>, strides = array<i64: 1>} : !conduit.dma.token
      conduit.wait_all %0 : !conduit.dma.token
      conduit.wait_all %1 : !conduit.dma.token
    }
  }
}

// -----

// Case (c) — convergent K=2 alias: two puts both at arg_index=0; middle args
// dead.  Single surviving slot; both puts retain arg_index=0; out gets
// renumbered to dense indices.
// CHECK-LABEL: aie.device(npu2_4col)
// CHECK: aie.runtime_sequence @case_c_alias(%arg0: memref<64xbf16>, %arg1: memref<64xbf16>) {
// CHECK-NEXT: conduit.put_memref_async {arg_index = 0 : i64, name = @ext_in_gate
// CHECK-NEXT: conduit.put_memref_async {arg_index = 0 : i64, name = @ext_in_up
// CHECK-NEXT: conduit.get_memref_async {arg_index = 1 : i64, name = @ext_out_a
module {
  aie.device(npu2_4col) {
    %shim_0_0 = aie.tile(0, 0)
    %shim_1_0 = aie.tile(1, 0)
    conduit.create @ext_in_gate {consumer_tiles = array<i64: 0, 2>, depth = 2 : i64, element_type = memref<64xbf16>, producer_tile = array<i64: 0, 0>}
    aie.shim_dma_allocation @ext_in_gate_shim_alloc(%shim_0_0, MM2S, 0) {conduit_channel = @ext_in_gate}
    conduit.create @ext_in_up {consumer_tiles = array<i64: 0, 2>, depth = 2 : i64, element_type = memref<64xbf16>, producer_tile = array<i64: 0, 0>}
    aie.shim_dma_allocation @ext_in_up_shim_alloc(%shim_0_0, MM2S, 1) {conduit_channel = @ext_in_up}
    conduit.create @ext_out_a {depth = 1 : i64, element_type = memref<64xbf16>, producer_tile = array<i64: 0, 2>}
    aie.shim_dma_allocation @ext_out_a_shim_alloc(%shim_1_0, S2MM, 0) {conduit_channel = @ext_out_a}
    aie.runtime_sequence @case_c_alias(%arg0: memref<64xbf16>, %arg1: memref<64xbf16>, %arg2: memref<64xbf16>, %arg3: memref<64xbf16>, %arg4: memref<64xbf16>, %arg5: memref<64xbf16>, %arg6: memref<64xbf16>, %arg7: memref<64xbf16>) {
      %0 = conduit.put_memref_async {arg_index = 0 : i64, name = @ext_in_gate, num_elems = 64 : i64, offsets = array<i64: 0>, sizes = array<i64: 64>, strides = array<i64: 1>} : !conduit.dma.token
      %1 = conduit.put_memref_async {arg_index = 0 : i64, name = @ext_in_up, num_elems = 64 : i64, offsets = array<i64: 0>, sizes = array<i64: 64>, strides = array<i64: 1>} : !conduit.dma.token
      %2 = conduit.get_memref_async {arg_index = 7 : i64, name = @ext_out_a, num_elems = 64 : i64, offsets = array<i64: 0>, sizes = array<i64: 64>, strides = array<i64: 1>} : !conduit.dma.token
      conduit.wait_all %0 {token = false} : !conduit.dma.token
      conduit.wait_all %1 {token = false} : !conduit.dma.token
      conduit.wait_all %2 : !conduit.dma.token
    }
  }
}
