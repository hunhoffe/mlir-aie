// RUN: aie-opt --objectfifo-to-conduit --dma-task-to-conduit --conduit-fuse-core-bodies --split-input-file --verify-diagnostics %s
//
// Task #55 — Pattern E (forward-chain endpoint) fusion-skip remark for
// `--conduit-fuse-core-bodies`.  Companion to
// `pattern_e_no_fuse_core_bodies_with_neighbor.mlir`: pins the explicit
// `expected-remark` so the diagnostic text is part of the regression net
// and a future refactor cannot silently demote it.
//
// `--conduit-fuse-core-bodies` MUST emit a remark in its Step 0 cross-
// device matcher when it rejects a fusion_group match because one side is
// a forward-chain endpoint (referenced by `conduit.scatter` /
// `conduit.gather`).  The remark is anchored on the `conduit.create` (Pass
// A lowers `aie.objectfifo` into `conduit.create`, so `expected-remark@+1`
// targets the source `aie.objectfifo` line that lowered to it).
//
// Two split-input-file modules:
//   1. Pattern E producer in devA matched by fusion_group → producer-side
//      remark fires on the output channel (@fwd_out).
//   2. Pattern E consumer in devB matched by fusion_group → consumer-side
//      remark fires on the input channel (@fwd_in_b).

// -----

module @remark_producer_side_core_bodies {
  aie.device(npu2) @devA {
    %shim_0 = aie.tile(0, 0)
    %memtile_0 = aie.tile(0, 1)
    %shim_1 = aie.tile(1, 0)

    aie.objectfifo @fwd_in(%shim_0, {%memtile_0}, 2 : i32)
        {fusion_group = "fg0"}
        : !aie.objectfifo<memref<8xbf16>>
    // expected-remark@+1 {{conduit-fuse-core-bodies: skipping fusion_group match for output channel @fwd_out — forward-chain / link-only endpoint (Pattern E); scatter/gather references cannot be safely renamed}}
    aie.objectfifo @fwd_out(%memtile_0, {%shim_1}, 2 : i32)
        {fusion_group = "fg0"}
        : !aie.objectfifo<memref<8xbf16>>
    aie.objectfifo.link [@fwd_in] -> [@fwd_out]([] [])

    aie.runtime_sequence(%a0: memref<64xbf16>, %a1: memref<64xbf16>) {
      %t0 = aiex.dma_configure_task_for @fwd_in {
        aie.dma_bd(%a0 : memref<64xbf16>, 0, 64,
          [<size = 1, stride = 0>, <size = 1, stride = 0>,
           <size = 1, stride = 0>, <size = 64, stride = 1>])
            {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%t0)
      %t1 = aiex.dma_configure_task_for @fwd_out {
        aie.dma_bd(%a1 : memref<64xbf16>, 0, 64,
          [<size = 1, stride = 0>, <size = 1, stride = 0>,
           <size = 1, stride = 0>, <size = 64, stride = 1>])
            {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%t1)
      aiex.dma_await_task(%t1)
      aiex.dma_free_task(%t0)
    }
  }

  aie.device(npu2) @devB {
    %shim_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)

    // expected-remark@+1 {{conduit-objectfifo: dma_repeat inference skipped: host-side num_invocations not observable in IR (single shim BD def); deferring dma_repeat to runtime}}
    aie.objectfifo @comp_in(%shim_0, {%tile_0_2}, 2 : i32)
        {fusion_group = "fg0"}
        : !aie.objectfifo<memref<8xbf16>>
    // expected-remark@+1 {{conduit-objectfifo: dma_repeat inference skipped: host-side num_invocations not observable in IR (single shim BD def); deferring dma_repeat to runtime}}
    aie.objectfifo @comp_out(%tile_0_2, {%shim_0}, 2 : i32)
        : !aie.objectfifo<memref<8xbf16>>

    func.func private @kernel(memref<8xbf16>, memref<8xbf16>)

    %core = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c8 = arith.constant 8 : index
      scf.for %i = %c0 to %c8 step %c1 {
        %in = aie.objectfifo.acquire @comp_in(Consume, 1)
            : !aie.objectfifosubview<memref<8xbf16>>
        %in_buf = aie.objectfifo.subview.access %in[0]
            : !aie.objectfifosubview<memref<8xbf16>> -> memref<8xbf16>
        %out = aie.objectfifo.acquire @comp_out(Produce, 1)
            : !aie.objectfifosubview<memref<8xbf16>>
        %out_buf = aie.objectfifo.subview.access %out[0]
            : !aie.objectfifosubview<memref<8xbf16>> -> memref<8xbf16>
        func.call @kernel(%in_buf, %out_buf)
            : (memref<8xbf16>, memref<8xbf16>) -> ()
        aie.objectfifo.release @comp_out(Produce, 1)
        aie.objectfifo.release @comp_in(Consume, 1)
      }
      aie.end
    } {link_with = "kernel.a"}

    aie.runtime_sequence(%b0: memref<64xbf16>, %b1: memref<64xbf16>) {
      %t0 = aiex.dma_configure_task_for @comp_in {
        aie.dma_bd(%b0 : memref<64xbf16>, 0, 64,
          [<size = 1, stride = 0>, <size = 1, stride = 0>,
           <size = 1, stride = 0>, <size = 64, stride = 1>])
            {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%t0)
      %t1 = aiex.dma_configure_task_for @comp_out {
        aie.dma_bd(%b1 : memref<64xbf16>, 0, 64,
          [<size = 1, stride = 0>, <size = 1, stride = 0>,
           <size = 1, stride = 0>, <size = 64, stride = 1>])
            {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%t1)
      aiex.dma_await_task(%t1)
      aiex.dma_free_task(%t0)
    }
  }
}

// -----

// Symmetric case: devA is Pattern A (compute core), devB is Pattern E
// (forward chain).  The consumer-side remark fires on devB's input
// channel (@fwd_in_b), which receives the forwarded data via scatter.

module @remark_consumer_side_core_bodies {
  aie.device(npu2) @devA {
    %shim_0 = aie.tile(0, 0)
    %tile_0_2 = aie.tile(0, 2)

    // expected-remark@+1 {{conduit-objectfifo: dma_repeat inference skipped: host-side num_invocations not observable in IR (single shim BD def); deferring dma_repeat to runtime}}
    aie.objectfifo @comp_in_a(%shim_0, {%tile_0_2}, 2 : i32)
        : !aie.objectfifo<memref<8xbf16>>
    // expected-remark@+1 {{conduit-objectfifo: dma_repeat inference skipped: host-side num_invocations not observable in IR (single shim BD def); deferring dma_repeat to runtime}}
    aie.objectfifo @comp_out_a(%tile_0_2, {%shim_0}, 2 : i32)
        {fusion_group = "fg1"}
        : !aie.objectfifo<memref<8xbf16>>

    func.func private @kernel_a(memref<8xbf16>, memref<8xbf16>)

    %core_a = aie.core(%tile_0_2) {
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %c8 = arith.constant 8 : index
      scf.for %i = %c0 to %c8 step %c1 {
        %in = aie.objectfifo.acquire @comp_in_a(Consume, 1)
            : !aie.objectfifosubview<memref<8xbf16>>
        %in_buf = aie.objectfifo.subview.access %in[0]
            : !aie.objectfifosubview<memref<8xbf16>> -> memref<8xbf16>
        %out = aie.objectfifo.acquire @comp_out_a(Produce, 1)
            : !aie.objectfifosubview<memref<8xbf16>>
        %out_buf = aie.objectfifo.subview.access %out[0]
            : !aie.objectfifosubview<memref<8xbf16>> -> memref<8xbf16>
        func.call @kernel_a(%in_buf, %out_buf)
            : (memref<8xbf16>, memref<8xbf16>) -> ()
        aie.objectfifo.release @comp_out_a(Produce, 1)
        aie.objectfifo.release @comp_in_a(Consume, 1)
      }
      aie.end
    } {link_with = "kernel_a.a"}

    aie.runtime_sequence(%a0: memref<64xbf16>, %a1: memref<64xbf16>) {
      %t0 = aiex.dma_configure_task_for @comp_in_a {
        aie.dma_bd(%a0 : memref<64xbf16>, 0, 64,
          [<size = 1, stride = 0>, <size = 1, stride = 0>,
           <size = 1, stride = 0>, <size = 64, stride = 1>])
            {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%t0)
      %t1 = aiex.dma_configure_task_for @comp_out_a {
        aie.dma_bd(%a1 : memref<64xbf16>, 0, 64,
          [<size = 1, stride = 0>, <size = 1, stride = 0>,
           <size = 1, stride = 0>, <size = 64, stride = 1>])
            {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%t1)
      aiex.dma_await_task(%t1)
      aiex.dma_free_task(%t0)
    }
  }

  aie.device(npu2) @devB {
    %shim_0 = aie.tile(0, 0)
    %memtile_0 = aie.tile(0, 1)
    %shim_1 = aie.tile(1, 0)

    // expected-remark@+1 {{conduit-fuse-core-bodies: skipping fusion_group match for input channel @fwd_in_b — forward-chain / link-only endpoint (Pattern E); scatter/gather references cannot be safely renamed}}
    aie.objectfifo @fwd_in_b(%shim_0, {%memtile_0}, 2 : i32)
        {fusion_group = "fg1"}
        : !aie.objectfifo<memref<8xbf16>>
    aie.objectfifo @fwd_out_b(%memtile_0, {%shim_1}, 2 : i32)
        : !aie.objectfifo<memref<8xbf16>>
    aie.objectfifo.link [@fwd_in_b] -> [@fwd_out_b]([] [])

    aie.runtime_sequence(%b0: memref<64xbf16>, %b1: memref<64xbf16>) {
      %t0 = aiex.dma_configure_task_for @fwd_in_b {
        aie.dma_bd(%b0 : memref<64xbf16>, 0, 64,
          [<size = 1, stride = 0>, <size = 1, stride = 0>,
           <size = 1, stride = 0>, <size = 64, stride = 1>])
            {burst_length = 0 : i32}
        aie.end
      }
      aiex.dma_start_task(%t0)
      %t1 = aiex.dma_configure_task_for @fwd_out_b {
        aie.dma_bd(%b1 : memref<64xbf16>, 0, 64,
          [<size = 1, stride = 0>, <size = 1, stride = 0>,
           <size = 1, stride = 0>, <size = 64, stride = 1>])
            {burst_length = 0 : i32}
        aie.end
      } {issue_token = true}
      aiex.dma_start_task(%t1)
      aiex.dma_await_task(%t1)
      aiex.dma_free_task(%t0)
    }
  }
}
