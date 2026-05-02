// RUN: aie-opt --conduit-depth-promote --verify-diagnostics %s | FileCheck %s
//
// Task #102 regression pin.
//
// `--conduit-fuse-channels` may stamp `dma_channel_group` + `fuse_mode`
// on depth=1 conduits as part of its Tier-3 grouping (the grouping is
// only valid for depth=1). When `--conduit-depth-promote` then promotes
// such a conduit from depth=1 → depth>1, those grouping annotations
// become stale: Pass C's grouped code path (ConduitToDMACollect /
// ConduitToDMARoute / Phase 4.5a lock sharing) silently corrupts
// lock-init, flow-key dedup, and packet-ID bookkeeping for the now-
// promoted (formerly grouped) channels. Symptom on swiglu HW: out_a +
// out_b sinks dead (all-zero output).
//
// Fix: depth-promote strips both `dma_channel_group` and `fuse_mode`
// from the conduit op when it bumps depth from 1 → depth>1, forcing
// Pass C onto its simple ungrouped path.

// CHECK: conduit.create @grouped_out
// CHECK-SAME: depth = 2 : i64
// CHECK-NOT: dma_channel_group
// CHECK-NOT: fuse_mode

// expected-remark @+1 {{conduit-depth-promote: promoted 1 conduit(s)}}
module {
aie.device(npu1) {

// expected-remark @+1 {{conduit-depth-promote: promoted 'grouped_out' from depth-1 to depth-2}}
conduit.create @grouped_out {
  element_type = memref<8xi32>,
  depth = 1 : i64,
  dma_channel_group = "group0",
  fuse_mode = "static"
}

func.func @consumer(%result: memref<8xi32>) {
  %c0 = arith.constant 0 : index
  %c1 = arith.constant 1 : index
  %c8 = arith.constant 8 : index
  scf.for %i = %c0 to %c8 step %c1 {
    %win = conduit.acquire {name = @grouped_out, count = 1 : i64, port = #conduit.port<Consume>}
               : !conduit.window<memref<8xi32>>
    %elem = conduit.subview_access %win {index = 0 : i64}
               : !conduit.window<memref<8xi32>> -> memref<8xi32>
    memref.copy %elem, %result : memref<8xi32> to memref<8xi32>
    conduit.release %win {count = 1 : i64, port = #conduit.port<Consume>}
        : !conduit.window<memref<8xi32>>
  }
  return
}

} // aie.device
} // module
