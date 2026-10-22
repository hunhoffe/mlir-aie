//===- passes.mlir ---------------------------------------------*- MLIR -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// AIE_COMPILE_TRACE makes each traced pass write its decisions as JSON lines
// (AIECompileTrace.h). The inputs are other passes' tests; this one checks
// only the events, and that the passes' output is the same with the trace on.

// Sequential placer: the model, its order, each tile it placed, the result.
// RUN: env AIE_COMPILE_TRACE=%t.seq.jsonl aie-opt --aie-place-tiles %S/../place-tiles/test_shim_follows_pinned_memtile.mlir -o %t.seq.mlir
// RUN: FileCheck %s --check-prefix=SEQ < %t.seq.jsonl
// SEQ: {"t":{{[0-9.]+}},"ev":"place.model",{{.*}}"placer":"sequential"
// SEQ-NEXT: "ev":"seq.order"
// SEQ-NEXT: "ev":"seq.place",{{.*}}"at":[
// SEQ: "ev":"place.final",{{.*}}"legal":true

// Simulated annealing: the starting placement and the one it kept.
// RUN: env AIE_COMPILE_TRACE=%t.sa.jsonl aie-opt --split-input-file --aie-place-tiles='placer=sa_placer sa-seed=42' %S/../place-tiles/sa_placer/test_sa_basic.mlir -o /dev/null
// RUN: FileCheck %s --check-prefix=SA < %t.sa.jsonl
// SA: "ev":"place.model",{{.*}}"placer":"sa"
// SA-NEXT: "ev":"place.init",{{.*}}"seed":42
// SA: "ev":"place.final",{{.*}}"legal":true

// Router: one run, its flows, each iteration, and how it ended.
// RUN: env AIE_COMPILE_TRACE=%t.route.jsonl aie-opt --aie-create-pathfinder-flows %S/../create-flows/broadcast.mlir -o /dev/null
// RUN: FileCheck %s --check-prefix=ROUTE < %t.route.jsonl
// ROUTE: "ev":"route.begin",{{.*}}"run":0,{{.*}}"flows":[{"id":0,"src":[2,0,"DMA0"]
// ROUTE-NEXT: "ev":"route.iter",{{.*}}"i":0,
// ROUTE-NEXT: "ev":"route.end",{{.*}}"ok":true,"iters":1}

// Buffer allocator: each tile's layout and the search that found it.
// RUN: env AIE_COMPILE_TRACE=%t.alloc.jsonl aie-opt --aie-assign-buffer-addresses %S/../assign-buffer-addresses/backtracking_dense_pack.mlir -o /dev/null
// RUN: FileCheck %s --check-prefix=ALLOC < %t.alloc.jsonl
// ALLOC: "ev":"alloc.tile",{{.*}}"tile":[0,2],"ok":true,{{.*}}"buffers":[{{.*}}"search":{{\[\[}}0,

// Without it, nothing is written and the result is the same.
// RUN: aie-opt --aie-place-tiles %S/../place-tiles/test_shim_follows_pinned_memtile.mlir -o %t.off.mlir
// RUN: diff %t.seq.mlir %t.off.mlir
