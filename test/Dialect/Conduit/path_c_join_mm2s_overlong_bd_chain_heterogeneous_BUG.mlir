// RUN: aie-opt --verify-diagnostics --conduit-to-dma %s
// RUN: aie-opt --verify-diagnostics --conduit-to-dma --aie-substitute-shim-dma-allocations --aie-assign-runtime-sequence-bd-ids %s
//
// Pass C join MM2S overlong-BD-chain BUG pin (heterogeneous).
//
// Same Pass C bug location as `..._homogeneous_BUG.mlir`:
//   ConduitToDMALink.cpp join MM2S append/create paths (lines 1573-1717).
//   * L1637 / L1683: nBufs = info.nConsumerBuffers().
//   * L1670 / L1713: NextBD circular ring.
//   * NO targetModel.getNumBDs(...) cap — chain length is putCount-driven.
//
// Heterogeneous variant: 17 puts on @joinA have DIFFERENT per-put offsets
// (offset = i*8).  `--conduit-canonicalize-loop-unroll-puts`'s match
// predicate requires identical offsets across all candidates.  Different
// offsets per put → predicate FAILS, canon REFUSES to collapse, even
// after canon lands.
//
// Why this variant matters (Task #15 cap-helper need):
//   The IRON Python-unroll case is HOMOGENEOUS — canon collapses it.  But
//   gather-source channels can become heterogeneous via:
//     (a) `--conduit-fuse-channels` merging multiple distinct gather-source
//         channels into one canonical name (per-channel offsets remain
//         distinct).
//     (b) Future canon symmetric expansion patterns producing per-batch
//         offsets that don't match the canon predicate.
//     (c) Hand-authored Conduit IR for non-IRON harnesses pre-folding
//         heterogeneous gather-source dispatches.
//   For these, the only fix is a `targetModel.getNumBDs(tile)` cap in
//   Pass C itself — that's Task #15's `checkBDChainCap` helper.
//
// Fixture geometry (gather topology):
//   * compute(0,2) → @joinA (17 heterogeneous puts) ┐
//   *                                               ├ memtile(0,1) →
//   * compute(0,3) → @joinB (1 anchor put)          ┘   @joinDst → shim(0,0)
//   * 17 puts on @joinA with offsets = 0, 8, 16, ..., 128 (= i*8).  All
//     other attrs (sizes, strides, num_elems, no producer_dimensions) match.
//   * 1 anchor put on @joinB (depth=2 channel from compute(0,3)).
//
// Today's behavior (this RUN line):
//   Pass C join MM2S emit-loop runs over @joinA's putCount = 17 → 17
//   sequential aie.dma_bd blocks in aie.mem on compute(0,2).  AIE verifier
//   rejects with `'aie.mem' op has more than 16 blocks`.
//
// Post-canon flip plan:
//   Canon REFUSES (heterogeneous offsets fail predicate) — IR is unchanged,
//   Pass C still crashes.  This fixture STAYS as expected-error pin until
//   Task #15's cap helper lands at every Pass C BD-emit site.  Then the
//   cap helper either:
//     (a) emits a Pass C structured diagnostic naming the channel + cap
//         (preferred — clearer error than the verifier crash), or
//     (b) emits dma_repeat / temporal-mux equivalent that fits in the cap.
//   Whichever (a)/(b) lands, this fixture's CHECK lines flip then.

module @path_c_join_mm2s_overlong_bd_chain_heterogeneous {
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
    conduit.gather{srcs = [@joinA, @joinB], dst = @joinDst {memtile = "tile(0,1)"}}

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

    // 17 HETEROGENEOUS puts on @joinA.  offsets vary per put (i*8) — canon
    // refuses to collapse because its match predicate requires identical
    // offsets across all candidates.  Plus 1 anchor put on @joinB.
    func.func @sequence(%arg0: memref<136xi32>) {
      // joinA: 17 heterogeneous puts.
      %t0 = conduit.put_memref_async {name = @joinA, num_elems = 8 : i64,
            offsets = array<i64: 0>, sizes = array<i64: 8>,
            strides = array<i64: 1>} : !conduit.dma.token
      conduit.wait_all %t0 {token = true} : !conduit.dma.token
      conduit.wait_all %t0 {token = false} : !conduit.dma.token
      %t1 = conduit.put_memref_async {name = @joinA, num_elems = 8 : i64,
            offsets = array<i64: 8>, sizes = array<i64: 8>,
            strides = array<i64: 1>} : !conduit.dma.token
      conduit.wait_all %t1 {token = true} : !conduit.dma.token
      conduit.wait_all %t1 {token = false} : !conduit.dma.token
      %t2 = conduit.put_memref_async {name = @joinA, num_elems = 8 : i64,
            offsets = array<i64: 16>, sizes = array<i64: 8>,
            strides = array<i64: 1>} : !conduit.dma.token
      conduit.wait_all %t2 {token = true} : !conduit.dma.token
      conduit.wait_all %t2 {token = false} : !conduit.dma.token
      %t3 = conduit.put_memref_async {name = @joinA, num_elems = 8 : i64,
            offsets = array<i64: 24>, sizes = array<i64: 8>,
            strides = array<i64: 1>} : !conduit.dma.token
      conduit.wait_all %t3 {token = true} : !conduit.dma.token
      conduit.wait_all %t3 {token = false} : !conduit.dma.token
      %t4 = conduit.put_memref_async {name = @joinA, num_elems = 8 : i64,
            offsets = array<i64: 32>, sizes = array<i64: 8>,
            strides = array<i64: 1>} : !conduit.dma.token
      conduit.wait_all %t4 {token = true} : !conduit.dma.token
      conduit.wait_all %t4 {token = false} : !conduit.dma.token
      %t5 = conduit.put_memref_async {name = @joinA, num_elems = 8 : i64,
            offsets = array<i64: 40>, sizes = array<i64: 8>,
            strides = array<i64: 1>} : !conduit.dma.token
      conduit.wait_all %t5 {token = true} : !conduit.dma.token
      conduit.wait_all %t5 {token = false} : !conduit.dma.token
      %t6 = conduit.put_memref_async {name = @joinA, num_elems = 8 : i64,
            offsets = array<i64: 48>, sizes = array<i64: 8>,
            strides = array<i64: 1>} : !conduit.dma.token
      conduit.wait_all %t6 {token = true} : !conduit.dma.token
      conduit.wait_all %t6 {token = false} : !conduit.dma.token
      %t7 = conduit.put_memref_async {name = @joinA, num_elems = 8 : i64,
            offsets = array<i64: 56>, sizes = array<i64: 8>,
            strides = array<i64: 1>} : !conduit.dma.token
      conduit.wait_all %t7 {token = true} : !conduit.dma.token
      conduit.wait_all %t7 {token = false} : !conduit.dma.token
      %t8 = conduit.put_memref_async {name = @joinA, num_elems = 8 : i64,
            offsets = array<i64: 64>, sizes = array<i64: 8>,
            strides = array<i64: 1>} : !conduit.dma.token
      conduit.wait_all %t8 {token = true} : !conduit.dma.token
      conduit.wait_all %t8 {token = false} : !conduit.dma.token
      %t9 = conduit.put_memref_async {name = @joinA, num_elems = 8 : i64,
            offsets = array<i64: 72>, sizes = array<i64: 8>,
            strides = array<i64: 1>} : !conduit.dma.token
      conduit.wait_all %t9 {token = true} : !conduit.dma.token
      conduit.wait_all %t9 {token = false} : !conduit.dma.token
      %t10 = conduit.put_memref_async {name = @joinA, num_elems = 8 : i64,
            offsets = array<i64: 80>, sizes = array<i64: 8>,
            strides = array<i64: 1>} : !conduit.dma.token
      conduit.wait_all %t10 {token = true} : !conduit.dma.token
      conduit.wait_all %t10 {token = false} : !conduit.dma.token
      %t11 = conduit.put_memref_async {name = @joinA, num_elems = 8 : i64,
            offsets = array<i64: 88>, sizes = array<i64: 8>,
            strides = array<i64: 1>} : !conduit.dma.token
      conduit.wait_all %t11 {token = true} : !conduit.dma.token
      conduit.wait_all %t11 {token = false} : !conduit.dma.token
      %t12 = conduit.put_memref_async {name = @joinA, num_elems = 8 : i64,
            offsets = array<i64: 96>, sizes = array<i64: 8>,
            strides = array<i64: 1>} : !conduit.dma.token
      conduit.wait_all %t12 {token = true} : !conduit.dma.token
      conduit.wait_all %t12 {token = false} : !conduit.dma.token
      %t13 = conduit.put_memref_async {name = @joinA, num_elems = 8 : i64,
            offsets = array<i64: 104>, sizes = array<i64: 8>,
            strides = array<i64: 1>} : !conduit.dma.token
      conduit.wait_all %t13 {token = true} : !conduit.dma.token
      conduit.wait_all %t13 {token = false} : !conduit.dma.token
      %t14 = conduit.put_memref_async {name = @joinA, num_elems = 8 : i64,
            offsets = array<i64: 112>, sizes = array<i64: 8>,
            strides = array<i64: 1>} : !conduit.dma.token
      conduit.wait_all %t14 {token = true} : !conduit.dma.token
      conduit.wait_all %t14 {token = false} : !conduit.dma.token
      %t15 = conduit.put_memref_async {name = @joinA, num_elems = 8 : i64,
            offsets = array<i64: 120>, sizes = array<i64: 8>,
            strides = array<i64: 1>} : !conduit.dma.token
      conduit.wait_all %t15 {token = true} : !conduit.dma.token
      conduit.wait_all %t15 {token = false} : !conduit.dma.token
      %t16 = conduit.put_memref_async {name = @joinA, num_elems = 8 : i64,
            offsets = array<i64: 128>, sizes = array<i64: 8>,
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
