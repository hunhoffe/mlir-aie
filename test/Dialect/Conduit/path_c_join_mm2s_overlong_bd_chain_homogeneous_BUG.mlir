// RUN: aie-opt --verify-diagnostics --conduit-to-dma %s
// RUN: aie-opt --verify-diagnostics --conduit-to-dma --aie-substitute-shim-dma-allocations --aie-assign-runtime-sequence-bd-ids %s
//
// Pass C join MM2S overlong-BD-chain BUG pin (homogeneous).
//
// Bug location: ConduitToDMALink.cpp join-source MM2S append/create paths,
// lines 1573-1717.
//   * L1637 (append into existing aie.mem): nBufs = info.nConsumerBuffers().
//   * L1683 (create new aie.mem): nBufs = info.nConsumerBuffers().
//   * Both paths emit `nBufs` sequential aie.dma_bd blocks chained via
//     aie.next_bd in a circular ring (`bdBlocks[(i+1) % nBufs]` at L1670/
//     L1713) on the producer compute tile (prodRow >= 2 guard at L1579).
//   * NO targetModel.getNumBDs(...) cap before block emission.
//
// Root cause: identical to Case A and Case B — `nConsumerBuffers()` returns
//   putCount when putCount > 1 && dmaRepeat == 0 (Common.h:315-326).
//   putCount is a host-emit accounting metric counted via module.walk over
//   conduit.put_memref_async ops (Collect.cpp:381-389) — it should not
//   influence hardware BD-chain length on a producer compute tile.
//
// On AIE2 compute tile, getNumBDs(...) = 16.  Any putCount > 16 on a join
// source produces > 16 aie.dma_bd blocks → AIEDialect.cpp:328-343
// HasValidBDs verifier rejects with `'aie.mem' op has more than 16 blocks`.
//
// Fixture geometry (gather topology):
//   * compute(0,2) → @joinA (17 puts) ┐
//   *                                  ├ memtile(0,1) → @joinDst → shim(0,0)
//   * compute(0,3) → @joinB (1 put)   ┘
//   * conduit.gather{srcs = [@joinA, @joinB], dst = @joinDst,
//                    memtile = "tile(0,1)"}
//   * @joinA / @joinB / @joinDst: depth = 2, memref<8xi32>; conduit.create
//     authored directly.
//   * aie.shim_dma_allocation @joinDst_shim_alloc(%shim, S2MM, 0)
//     {conduit_channel = @joinDst}
//   * Cores on (0,2) and (0,3): infinite scf.for with conduit.acquire/release
//     on Produce port of their respective channel.
//   * 17 hand-authored conduit.put_memref_async ops on @joinA.  All
//     STRUCTURALLY IDENTICAL (offsets=0, sizes=8, strides=1, num_elems=8,
//     no producer_dimensions) — IRON-loop-unroll match shape for canon.
//   * 1 put on @joinB just to anchor the gather's second src.
//
// Why hand-authored puts on a gather source:
//   conduit.gather is the structural form for N:1 fan-in via memtile relay.
//   IRON's natural compile path produces gather sources via per-tile
//   `aie.objectfifo` followed by an `aie.objectfifo.link [@srcA, @srcB] ->
//   [@dst]`, which becomes `conduit.gather` after Pass A.  putCount > 1 on
//   a gather source can arise via:
//     (a) `--conduit-fuse-channels` merging multiple distinct gather-source
//         channels into one canonical name (per-tile gather sources fold
//         after fusion).
//     (b) Future canon symmetric expansion patterns.
//     (c) Hand-authored Conduit IR for non-IRON harnesses that pre-fold
//         gather sources.
//   Each path is structural; this fixture pins the latent join MM2S bug
//   independent of the upstream reach.
//
// Today's behavior (this RUN line, with canon ABSENT from pipeline):
//   Pass C emits 17 sequential aie.dma_bd blocks in aie.mem on compute(0,2)
//   for @joinA's MM2S → memtile flow.  AIE verifier rejects with
//   `'aie.mem' op has more than 16 blocks`.
//
// Post-canon-stabilization flip plan (homogeneous variant):
//   --conduit-canonicalize-loop-unroll-puts (Task #11) collapses the 17
//   structurally-identical puts on @joinA → 1 put + dma_repeat = 17 on
//   conduit.create @joinA.  putCount drops to 1 and dmaRepeat = 17 →
//   nConsumerBuffers() = depth = 2; join MM2S emits 2 BDs (depth-many
//   circular chain).  Compute tile DMAs cycle infinitely (repeat_count = 0).
//   Flip this fixture (rename, drop _BUG suffix) to FileCheck pinning:
//     CHECK: conduit.create @joinA
//     CHECK-SAME: dma_repeat = 17
//     CHECK: aie.mem(%{{.*}}tile_0_2)
//     CHECK:   aie.dma_start(MM2S,
//     CHECK:   aie.dma_bd
//     CHECK:   aie.next_bd
//     CHECK:   aie.dma_bd
//     CHECK:   aie.next_bd
//     CHECK-NOT: aie.dma_bd
//     CHECK:   aie.end
//
// (See companion `..._heterogeneous_BUG.mlir` — canon refuses on
// non-identical puts, so that variant pins the cap-helper need; Task #15.)
//
// Sprint N+2 Tier 0 status (2026-04-30):
//   Tier 0 only adds a loop-context discriminator to nConsumerBuffers() in
//   ConduitToDMACommon.h — fixes Case A consumer S2MM (op7/op11 GEMV) by
//   suppressing the putCount override when consumer gets are inside a loop.
//   Join MM2S is the gather-source PRODUCER side; @joinA's downstream
//   consumer (memtile relay tile) has no consumer-side conduit ops at all
//   (consumerGetsInLoop = false), so the override still fires →
//   nConsumerBuffers() = 17 → BD chain length 17 > 16 cap.  Tier 0 does
//   NOT fix this BUG pin.  Flip → `_canonical.mlir` only after canon
//   collapse lands in the Pass C pre-pipeline.

module @path_c_join_mm2s_overlong_bd_chain_homogeneous {
  // expected-error@+1 {{conduit-to-dma: BD chain length}}
  aie.device(npu2) {
    %tile_0_0 = aie.tile(0, 0)
    %tile_0_1 = aie.tile(0, 1)
    %tile_0_2 = aie.tile(0, 2)
    %tile_0_3 = aie.tile(0, 3)

    conduit.create @joinA {
      element_type = memref<8xi32>,
      depth = 2 : i64
    }
    conduit.create @joinB {
      element_type = memref<8xi32>,
      depth = 2 : i64
    }
    conduit.create @joinDst {
      element_type = memref<8xi32>,
      depth = 2 : i64
    }

    // Gather: [@joinA, @joinB] → @joinDst via memtile(0,1).
    conduit.gather{srcs = [@joinA, @joinB], dst = @joinDst, memtile = %tile_0_1}

    // Shim consumes @joinDst (S2MM ch 0) — anchors the gather pipeline.
    aie.shim_dma_allocation @joinDst_shim_alloc(%tile_0_0, S2MM, 0) {conduit_channel = @joinDst}

    // Compute(0,2): produces on @joinA.
    %core_0_2 = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %cmax = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %i = %c0 to %cmax step %c1 {
        %win = conduit.acquire {name = @joinA, port = #conduit.port<Produce>, count = 1 : i64}
            : !conduit.window<memref<8xi32>>
        conduit.release %win {port = #conduit.port<Produce>, count = 1 : i64}
            : !conduit.window<memref<8xi32>>
      }
      aie.end
    }

    // Compute(0,3): produces on @joinB (single put, just anchors gather).
    %core_0_3 = aie.core(%tile_0_3) {
      %c0 = arith.constant 0 : index
      %cmax = arith.constant 9223372036854775807 : index
      %c1 = arith.constant 1 : index
      scf.for %i = %c0 to %cmax step %c1 {
        %win = conduit.acquire {name = @joinB, port = #conduit.port<Produce>, count = 1 : i64}
            : !conduit.window<memref<8xi32>>
        conduit.release %win {port = #conduit.port<Produce>, count = 1 : i64}
            : !conduit.window<memref<8xi32>>
      }
      aie.end
    }

    // 17 STRUCTURALLY-IDENTICAL puts on @joinA.  All offsets = 0; all
    // sizes = 8; all strides = 1; all num_elems = 8 — IRON-loop-unroll
    // match shape that --conduit-canonicalize-loop-unroll-puts WOULD
    // collapse if it were in the pipeline.  Plus 1 anchor put on @joinB.
    func.func @sequence(%arg0: memref<8xi32>) {
      // joinA: 17 identical puts.
      %t0 = conduit.put_memref_async {name = @joinA, num_elems = 8 : i64,
            offsets = array<i64: 0>, sizes = array<i64: 8>,
            strides = array<i64: 1>} : !conduit.dma.token
      conduit.wait_all %t0 {token = true} : !conduit.dma.token
      conduit.wait_all %t0 {token = false} : !conduit.dma.token
      %t1 = conduit.put_memref_async {name = @joinA, num_elems = 8 : i64,
            offsets = array<i64: 0>, sizes = array<i64: 8>,
            strides = array<i64: 1>} : !conduit.dma.token
      conduit.wait_all %t1 {token = true} : !conduit.dma.token
      conduit.wait_all %t1 {token = false} : !conduit.dma.token
      %t2 = conduit.put_memref_async {name = @joinA, num_elems = 8 : i64,
            offsets = array<i64: 0>, sizes = array<i64: 8>,
            strides = array<i64: 1>} : !conduit.dma.token
      conduit.wait_all %t2 {token = true} : !conduit.dma.token
      conduit.wait_all %t2 {token = false} : !conduit.dma.token
      %t3 = conduit.put_memref_async {name = @joinA, num_elems = 8 : i64,
            offsets = array<i64: 0>, sizes = array<i64: 8>,
            strides = array<i64: 1>} : !conduit.dma.token
      conduit.wait_all %t3 {token = true} : !conduit.dma.token
      conduit.wait_all %t3 {token = false} : !conduit.dma.token
      %t4 = conduit.put_memref_async {name = @joinA, num_elems = 8 : i64,
            offsets = array<i64: 0>, sizes = array<i64: 8>,
            strides = array<i64: 1>} : !conduit.dma.token
      conduit.wait_all %t4 {token = true} : !conduit.dma.token
      conduit.wait_all %t4 {token = false} : !conduit.dma.token
      %t5 = conduit.put_memref_async {name = @joinA, num_elems = 8 : i64,
            offsets = array<i64: 0>, sizes = array<i64: 8>,
            strides = array<i64: 1>} : !conduit.dma.token
      conduit.wait_all %t5 {token = true} : !conduit.dma.token
      conduit.wait_all %t5 {token = false} : !conduit.dma.token
      %t6 = conduit.put_memref_async {name = @joinA, num_elems = 8 : i64,
            offsets = array<i64: 0>, sizes = array<i64: 8>,
            strides = array<i64: 1>} : !conduit.dma.token
      conduit.wait_all %t6 {token = true} : !conduit.dma.token
      conduit.wait_all %t6 {token = false} : !conduit.dma.token
      %t7 = conduit.put_memref_async {name = @joinA, num_elems = 8 : i64,
            offsets = array<i64: 0>, sizes = array<i64: 8>,
            strides = array<i64: 1>} : !conduit.dma.token
      conduit.wait_all %t7 {token = true} : !conduit.dma.token
      conduit.wait_all %t7 {token = false} : !conduit.dma.token
      %t8 = conduit.put_memref_async {name = @joinA, num_elems = 8 : i64,
            offsets = array<i64: 0>, sizes = array<i64: 8>,
            strides = array<i64: 1>} : !conduit.dma.token
      conduit.wait_all %t8 {token = true} : !conduit.dma.token
      conduit.wait_all %t8 {token = false} : !conduit.dma.token
      %t9 = conduit.put_memref_async {name = @joinA, num_elems = 8 : i64,
            offsets = array<i64: 0>, sizes = array<i64: 8>,
            strides = array<i64: 1>} : !conduit.dma.token
      conduit.wait_all %t9 {token = true} : !conduit.dma.token
      conduit.wait_all %t9 {token = false} : !conduit.dma.token
      %t10 = conduit.put_memref_async {name = @joinA, num_elems = 8 : i64,
            offsets = array<i64: 0>, sizes = array<i64: 8>,
            strides = array<i64: 1>} : !conduit.dma.token
      conduit.wait_all %t10 {token = true} : !conduit.dma.token
      conduit.wait_all %t10 {token = false} : !conduit.dma.token
      %t11 = conduit.put_memref_async {name = @joinA, num_elems = 8 : i64,
            offsets = array<i64: 0>, sizes = array<i64: 8>,
            strides = array<i64: 1>} : !conduit.dma.token
      conduit.wait_all %t11 {token = true} : !conduit.dma.token
      conduit.wait_all %t11 {token = false} : !conduit.dma.token
      %t12 = conduit.put_memref_async {name = @joinA, num_elems = 8 : i64,
            offsets = array<i64: 0>, sizes = array<i64: 8>,
            strides = array<i64: 1>} : !conduit.dma.token
      conduit.wait_all %t12 {token = true} : !conduit.dma.token
      conduit.wait_all %t12 {token = false} : !conduit.dma.token
      %t13 = conduit.put_memref_async {name = @joinA, num_elems = 8 : i64,
            offsets = array<i64: 0>, sizes = array<i64: 8>,
            strides = array<i64: 1>} : !conduit.dma.token
      conduit.wait_all %t13 {token = true} : !conduit.dma.token
      conduit.wait_all %t13 {token = false} : !conduit.dma.token
      %t14 = conduit.put_memref_async {name = @joinA, num_elems = 8 : i64,
            offsets = array<i64: 0>, sizes = array<i64: 8>,
            strides = array<i64: 1>} : !conduit.dma.token
      conduit.wait_all %t14 {token = true} : !conduit.dma.token
      conduit.wait_all %t14 {token = false} : !conduit.dma.token
      %t15 = conduit.put_memref_async {name = @joinA, num_elems = 8 : i64,
            offsets = array<i64: 0>, sizes = array<i64: 8>,
            strides = array<i64: 1>} : !conduit.dma.token
      conduit.wait_all %t15 {token = true} : !conduit.dma.token
      conduit.wait_all %t15 {token = false} : !conduit.dma.token
      %t16 = conduit.put_memref_async {name = @joinA, num_elems = 8 : i64,
            offsets = array<i64: 0>, sizes = array<i64: 8>,
            strides = array<i64: 1>} : !conduit.dma.token
      conduit.wait_all %t16 {token = true} : !conduit.dma.token
      conduit.wait_all %t16 {token = false} : !conduit.dma.token

      // joinB anchor put.
      %tb0 = conduit.put_memref_async {name = @joinB, num_elems = 8 : i64,
            offsets = array<i64: 0>, sizes = array<i64: 8>,
            strides = array<i64: 1>} : !conduit.dma.token
      conduit.wait_all %tb0 {token = true} : !conduit.dma.token
      conduit.wait_all %tb0 {token = false} : !conduit.dma.token
      return
    }
  }
}
