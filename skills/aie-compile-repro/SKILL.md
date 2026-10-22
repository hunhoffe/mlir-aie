---
name: aie-compile-repro
description: Guide to turning an aiecc compile failure (placement, routing or buffer allocation) or a slow compile into a small, traced reproducer — recording a compile trace, finding which phase failed and why, saving the file that phase starts from, shrinking a routing failure to a minimal design with router_shrink.py while tracing each step, and replaying it all in a compile replay page. Use this whenever aiecc fails with "Unable to find a legal routing", a placer or allocator error, or routing that takes many iterations; when someone wants a minimal reproducer for a router or allocator bug; or when comparing how a compiler change affects a design's placement, routing or allocation.
license: Apache-2.0 WITH LLVM-exception
---

<!--
Copyright (C) 2026 Advanced Micro Devices, Inc.
SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
-->

# From a compile failure to a traced, minimal reproducer

A big design that fails to route or allocate is slow to debug: each try
takes seconds to minutes, and the failure is buried among hundreds of
flows. This guide works the failure down to a small design that still
fails, recording what the compiler did at each step so you can see it.

## 1. Record a compile trace

```shell
aiecc design.mlir <your usual flags> --compile-trace=trace.jsonl
# or, through IRON's Python flow:
AIE_COMPILE_TRACE=trace.jsonl python3 my_design.py ...
```

The trace is JSON lines (one event per line) and does not change what
aiecc produces. It records, per device: the placer's model and each of its
decisions (`place.*`, `seq.*`), each router run and iteration (`route.*`),
each tile's buffer allocation search, lock ids and BD chains (`alloc.*`),
each per-core backend stage's time (`stage.end` with a `key`), and each
core ELF's program memory, stack, sections and symbols (`core.elf`).
`aie-opt` honours `AIE_COMPILE_TRACE` too.

## 2. Find what failed

Read the trace's last events, or fold it in Python:

```python
from iron_explorer import compile_trace as ct   # iron-design-explorer
r = ct.replay(ct.read(Path("trace.jsonl")))
for f in ct.findings(r):
    print(f["severity"], f["message"], f.get("repro"))
```

`findings` lists failures (placement, routing, allocation), nearly full
memory, locks or program memory, a stack too small for what the probe link
measured, and routing that took many iterations. Each failure carries the
commands below, filled in from the trace's `argv`.

## 3. Save the file the failing phase starts from

Rerun aiecc with the same flags, asking for that phase's input:

| Phase | `--get=` |
| --- | --- |
| routing | `input_with_symbols.mlir` |
| buffer allocation | `input_physical.mlir` |

```shell
aiecc design.mlir <same flags> --get=input_with_symbols.mlir --tmpdir=repro --output-dir=repro
```

Check that the pass alone still fails on it before shrinking:
`aie-opt --aie-create-pathfinder-flows repro/input_with_symbols.mlir`.

## 4. Shrink a routing failure, tracing each step

```shell
python utils/router_testing/router_shrink.py repro/input_with_symbols.mlir \
    --match "Unable to find a legal routing" \
    --out repro/min.mlir --trace-dir repro/shrink --budget-s 1800 -v
```

- `--match` is a regex on the failure you keep: aie-opt's stderr, or a
  verifier problem when routing succeeds but is wrong. Make it specific
  enough that the shrink cannot wander to a different failure.
- Each try runs aie-opt once (seconds on a large design), so set
  `--budget-s`; the smallest design so far is kept when time runs out.
- `--trace-dir` saves every design the shrink keeps as `step_NNN.mlir`
  with a compile trace of routing it (`step_NNN.jsonl`) and `steps.json`.
  The last step is the reproducer.
- When `aie-opt` was built with assertions, `-debug-only=aie-pathfinder`
  on the reproducer shows the router's own reasoning.

There is no shrinker for allocation yet: cut buffers from
`input_physical.mlir` by hand, rerunning
`aie-opt --aie-assign-buffer-addresses` with `AIE_COMPILE_TRACE` set to
watch its search.

## 5. Replay it

iron-design-explorer's Compile page (`iron-explorer compile`) turns traces
into one static page: placement, routing switchbox by switchbox, each
tile's allocation search, and each core's Peano backend, with the findings
at the top.

```shell
iron-explorer compile \
    "MyDesign/before=trace.jsonl" \
    --shrink "MyDesign/before=repro/shrink" -o compile.html
```

With `--shrink`, the routing phase shows each step of the shrink, and the
reproducer's routing replays iteration by iteration. Add a second trace
(`"MyDesign/after=trace2.jsonl"`) from a changed compiler to compare
placement quality, router iterations and time side by side.

## When the reproducer is small

- Keep `repro/min.mlir` as a lit test next to the router's tests
  (`test/create-packet-flows/`), with the `--match` text as the expected
  error, so the fix is checked.
- Say in the issue or PR which design it came from and the aiecc flags,
  and attach the replay page when the failure is easier seen than read.
