// RUN: aie-opt --conduit-to-dma %s | FileCheck %s
// RUN: aie-opt --conduit-to-dma --aie-substitute-shim-dma-allocations --aie-assign-runtime-sequence-bd-ids %s | FileCheck %s
//
// Pass C Case A (shim MM2S → compute consumer S2MM) canonical pin —
// post-fix (Sprint N+2 Tier 0) for the nConsumerBuffers() putCount-override.
//
// Tier 0 of Sprint N+2 dropped the override branch in ConduitToDMACommon.h
// `nConsumerBuffers()` (the `if (putCount > 1 && dmaRepeat == 0) return
// putCount;` clause).  After the fix, all eight Pass C BD-emit call sites
// that read the helper size their consumer-side BD chains to depth (or
// max(depth, maxConsumerAcquire+1) for sliding-window patterns), independent
// of how many host-emit conduit.put_memref_async ops drive the channel.
//
// Companions in this set (each pinned at its own structural Pass C location):
//   * `..._case_b_..._canonical.mlir` (×2 homog/heterog) — compute MM2S → shim
//   * `..._join_..._canonical.mlir`   (×2 homog/heterog) — gather-source MM2S
//
// Fixture geometry (mirror of Case B but inverted producer/consumer):
//   * shim(0,0) producer (MM2S 0) → compute(0,2) consumer (S2MM)
//   * @chan: depth = 2, memref<8xi32>; conduit.create authored directly
//     (post-Pass-A IR; no objectfifo).
//   * aie.shim_dma_allocation @chan_shim_alloc(%shim, MM2S, 0) — shim is
//     producer (drives Case A: shim producer + compute consumer).
//   * Compute core: scf.for trip=∞ conduit.acquire/release(Consume, 1) —
//     consumes from @chan via S2MM into compute-tile buffers.
//   * 17 hand-authored conduit.put_memref_async ops on @chan in a regular
//     func.func (NOT aie.runtime_sequence — keeps Pass C's runtime-step
//     out of the path).  STRUCTURALLY IDENTICAL puts (offsets=0, sizes=8,
//     strides=1, num_elems=8) — IRON-loop-unroll match shape, paired with
//     wait_all{token=true} (await) + wait_all{token=false} (free).
//
// Why hand-authored puts here:
//   For shim→compute channels, IRON's runtime path naturally produces shim
//   MM2S configures via runtime_sequence + --dma-task-to-conduit, which
//   becomes one put on the canonical channel.  putCount > 1 on a Case A
//   shim-producer channel reaches Pass C via:
//     (a) `--conduit-fuse-channels` merging multiple distinct shim→compute
//         channels into one canonical name (per-channel offsets remain).
//     (b) Future canon symmetric expansion patterns.
//     (c) Hand-authored Conduit IR for non-IRON harnesses.
//   The hand-authored shape pins the structural latent bug at the LINK pass
//   independent of how the state arises upstream.
//
// Boundary choice: putCount = 17 = cap+1 for AIE2 compute (getNumBDs = 16).
// Pre-fix the override sized the chain at 17 → cap helper fired
// `BD chain length 17 on tile (0,2) exceeds cap 16`.  Post-fix the chain is
// depth-sized (2 BD blocks, circular), independent of the 17 host puts.

// Consumer compute tile gets exactly depth=2 buffers + a depth-many circular
// BD chain on its S2MM aie.mem.  The 17 host puts do NOT inflate either
// count; they live in the runtime-sequence-equivalent func.func and (post
// the second RUN line) become 17 shim DMA configures sharing the same
// consumer-side resources.
// CHECK-LABEL: aie.device(npu2)
// CHECK: aie.buffer({{.*}}) {{.*}}sym_name = "chan_cons_buff_0"{{.*}} memref<8xi32>
// CHECK: aie.buffer({{.*}}) {{.*}}sym_name = "chan_cons_buff_1"{{.*}} memref<8xi32>
// CHECK-NOT: sym_name = "chan_cons_buff_2"
// CHECK: aie.mem
// CHECK: aie.dma_start(S2MM
// CHECK-COUNT-2: aie.dma_bd
// CHECK-NOT: aie.dma_bd

module @path_c_case_a_consumer_s2mm_overlong_bd_chain {
  aie.device(npu2) {
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)

    conduit.create @chan {
      element_type = memref<8xi32>,
      depth = 2 : i64
    }

    // Shim is producer (MM2S) — makes compute(0,2) the consumer.
    aie.shim_dma_allocation @chan_shim_alloc(%tile_0_0, MM2S, 0) {conduit_channel = @chan}

    %core_0_2 = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %cmax = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %i = %c0 to %cmax step %c1 {
        %win = conduit.acquire {name = @chan, port = #conduit.port<Consume>, count = 1 : i64}
            : !conduit.window<memref<8xi32>>
        conduit.release %win {port = #conduit.port<Consume>, count = 1 : i64}
            : !conduit.window<memref<8xi32>>
      }
      aie.end
    }

    // 17 structurally-identical puts on @chan.  All offsets = 0; all sizes = 8;
    // all strides = 1; all num_elems = 8.  Each paired with wait_all{token=true}
    // (await) + wait_all{token=false} (free) — matches the canon collapse
    // predicate exactly (collapse not in this RUN line on purpose).
    func.func @sequence(%arg0: memref<8xi32>) {
      %t0 = conduit.put_memref_async {name = @chan, num_elems = 8 : i64,
            offsets = array<i64: 0>, sizes = array<i64: 8>,
            strides = array<i64: 1>} : !conduit.dma.token
      conduit.wait_all %t0 {token = true} : !conduit.dma.token
      conduit.wait_all %t0 {token = false} : !conduit.dma.token
      %t1 = conduit.put_memref_async {name = @chan, num_elems = 8 : i64,
            offsets = array<i64: 0>, sizes = array<i64: 8>,
            strides = array<i64: 1>} : !conduit.dma.token
      conduit.wait_all %t1 {token = true} : !conduit.dma.token
      conduit.wait_all %t1 {token = false} : !conduit.dma.token
      %t2 = conduit.put_memref_async {name = @chan, num_elems = 8 : i64,
            offsets = array<i64: 0>, sizes = array<i64: 8>,
            strides = array<i64: 1>} : !conduit.dma.token
      conduit.wait_all %t2 {token = true} : !conduit.dma.token
      conduit.wait_all %t2 {token = false} : !conduit.dma.token
      %t3 = conduit.put_memref_async {name = @chan, num_elems = 8 : i64,
            offsets = array<i64: 0>, sizes = array<i64: 8>,
            strides = array<i64: 1>} : !conduit.dma.token
      conduit.wait_all %t3 {token = true} : !conduit.dma.token
      conduit.wait_all %t3 {token = false} : !conduit.dma.token
      %t4 = conduit.put_memref_async {name = @chan, num_elems = 8 : i64,
            offsets = array<i64: 0>, sizes = array<i64: 8>,
            strides = array<i64: 1>} : !conduit.dma.token
      conduit.wait_all %t4 {token = true} : !conduit.dma.token
      conduit.wait_all %t4 {token = false} : !conduit.dma.token
      %t5 = conduit.put_memref_async {name = @chan, num_elems = 8 : i64,
            offsets = array<i64: 0>, sizes = array<i64: 8>,
            strides = array<i64: 1>} : !conduit.dma.token
      conduit.wait_all %t5 {token = true} : !conduit.dma.token
      conduit.wait_all %t5 {token = false} : !conduit.dma.token
      %t6 = conduit.put_memref_async {name = @chan, num_elems = 8 : i64,
            offsets = array<i64: 0>, sizes = array<i64: 8>,
            strides = array<i64: 1>} : !conduit.dma.token
      conduit.wait_all %t6 {token = true} : !conduit.dma.token
      conduit.wait_all %t6 {token = false} : !conduit.dma.token
      %t7 = conduit.put_memref_async {name = @chan, num_elems = 8 : i64,
            offsets = array<i64: 0>, sizes = array<i64: 8>,
            strides = array<i64: 1>} : !conduit.dma.token
      conduit.wait_all %t7 {token = true} : !conduit.dma.token
      conduit.wait_all %t7 {token = false} : !conduit.dma.token
      %t8 = conduit.put_memref_async {name = @chan, num_elems = 8 : i64,
            offsets = array<i64: 0>, sizes = array<i64: 8>,
            strides = array<i64: 1>} : !conduit.dma.token
      conduit.wait_all %t8 {token = true} : !conduit.dma.token
      conduit.wait_all %t8 {token = false} : !conduit.dma.token
      %t9 = conduit.put_memref_async {name = @chan, num_elems = 8 : i64,
            offsets = array<i64: 0>, sizes = array<i64: 8>,
            strides = array<i64: 1>} : !conduit.dma.token
      conduit.wait_all %t9 {token = true} : !conduit.dma.token
      conduit.wait_all %t9 {token = false} : !conduit.dma.token
      %t10 = conduit.put_memref_async {name = @chan, num_elems = 8 : i64,
            offsets = array<i64: 0>, sizes = array<i64: 8>,
            strides = array<i64: 1>} : !conduit.dma.token
      conduit.wait_all %t10 {token = true} : !conduit.dma.token
      conduit.wait_all %t10 {token = false} : !conduit.dma.token
      %t11 = conduit.put_memref_async {name = @chan, num_elems = 8 : i64,
            offsets = array<i64: 0>, sizes = array<i64: 8>,
            strides = array<i64: 1>} : !conduit.dma.token
      conduit.wait_all %t11 {token = true} : !conduit.dma.token
      conduit.wait_all %t11 {token = false} : !conduit.dma.token
      %t12 = conduit.put_memref_async {name = @chan, num_elems = 8 : i64,
            offsets = array<i64: 0>, sizes = array<i64: 8>,
            strides = array<i64: 1>} : !conduit.dma.token
      conduit.wait_all %t12 {token = true} : !conduit.dma.token
      conduit.wait_all %t12 {token = false} : !conduit.dma.token
      %t13 = conduit.put_memref_async {name = @chan, num_elems = 8 : i64,
            offsets = array<i64: 0>, sizes = array<i64: 8>,
            strides = array<i64: 1>} : !conduit.dma.token
      conduit.wait_all %t13 {token = true} : !conduit.dma.token
      conduit.wait_all %t13 {token = false} : !conduit.dma.token
      %t14 = conduit.put_memref_async {name = @chan, num_elems = 8 : i64,
            offsets = array<i64: 0>, sizes = array<i64: 8>,
            strides = array<i64: 1>} : !conduit.dma.token
      conduit.wait_all %t14 {token = true} : !conduit.dma.token
      conduit.wait_all %t14 {token = false} : !conduit.dma.token
      %t15 = conduit.put_memref_async {name = @chan, num_elems = 8 : i64,
            offsets = array<i64: 0>, sizes = array<i64: 8>,
            strides = array<i64: 1>} : !conduit.dma.token
      conduit.wait_all %t15 {token = true} : !conduit.dma.token
      conduit.wait_all %t15 {token = false} : !conduit.dma.token
      %t16 = conduit.put_memref_async {name = @chan, num_elems = 8 : i64,
            offsets = array<i64: 0>, sizes = array<i64: 8>,
            strides = array<i64: 1>} : !conduit.dma.token
      conduit.wait_all %t16 {token = true} : !conduit.dma.token
      conduit.wait_all %t16 {token = false} : !conduit.dma.token
      return
    }
  }
}
