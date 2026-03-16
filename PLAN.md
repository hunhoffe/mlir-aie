# Conduit — Research Plan

**Last updated:** 2026-03-16
**Lit suite:** 65 PASS | **Corpus:** 96/118 = 81% | **Paper:** workshop-submittable

---

## Immediate Priorities

### 1. Hardware Validation (Phoenix/Strix when available)

| Experiment | Input | Gate |
|---|---|---|
| **Exp 1** — Depth-1 SPSC | `depth_one_objectfifo_test.mlir` | **Unblocked — run now** |
| **Exp 2** — Depth-2 double-buffer | `same_depth_objectfifos_test.mlir` | **Unblocked** (rotation counter, distribute/join all fixed) |
| **Exp 3** — Distribute/broadcast | `conduit_to_dma_distribute.mlir` | After Exp 1 passes |
| **Exp 4** — acquire_async stall reduction | New micro-benchmark | After Exp 2 passes |

```bash
# Experiment 1
build/bin/aie-opt --objectfifo-to-conduit --conduit-to-dma \
  test/objectFifo-stateful-transform/dynamic_lowering/depth_one_objectfifo_test.mlir
# vs oracle:
build/bin/aie-opt --aie-objectFifo-stateful-transform <same file>
```

Success criterion: byte-identical output tensors; no surviving `conduit.*` ops.

### 2. Paper (workshop submission)

**State:** Fully updated to reflect all implemented features. Honest about no hardware results.

**Before submission:**
- Experiment 1 passes on hardware
- Render in ACM 2-column to confirm 6-page fit
- Add one sentence to §4.3 (Pass B validation timeline)

**Target venues:** MLIR Workshop (LLVM DevMtg), LCTES, or CGO Workshop.
**NOT ready for CGO/PLDI main track** — needs hardware results on ≥2 experiments.

---

## Open Bugs

| ID | Severity | Description |
|---|---|---|
| depth-2 resource parity | HIGH | buffer +1, use_lock -8 vs stateful transform for same_depth_objectfifos_test. |
| aie_stream | HIGH | ~6 corpus files: stream-mode objectfifos (WireBundle::Core) get DMA resources allocated; oracle emits none. Pass A needs stream-routing detection. |
| AIE2_delayed_release | CORPUS | Correctly rejected by M8a. 104/104 needs Pass A sequential acquire fix. |
| plio_test | CORPUS | Symbol collision when multiple objectfifos share a PLIO tile. |
| broadcast-enforced-depths | CORPUS | subview_access index OOB for depth-2 broadcast. |

**Recently resolved (Round 2 — 2026-03-15):**
- `depth-2-lock-deficit` — missing shim consumer lock in sub-case 4b. Experiment 2 now unblocked.
- `nd_dma_distribute` — duplicate BD emission in Phase 5.5 Case C. Fixed by `linkDstNames` skip.
- `link-distribute + repeat_count` — Phase 5 MemTile MM2S not scaling BD count or lock inits. Fixed.

**Recently resolved (Round 2b + 3 — 2026-03-16):**
- `broadcast-subview-crash` — crash on broadcast objectFIFO with SubviewAccess. Fixed in allocPhase.
- `produce-rotation-missing` — producer cores with depth>1 always selected buff_0. Full rotation counter for Port::Produce now implemented.
- `distribute-over-alloc` / `join-over-alloc` — duplicate buffer/lock/flow allocation for link-dst conduits. Fixed; distribute and join now resource-equivalent to oracle (delta=0).
- `rotation-counter-over-alloc` — two conduits on same tile each allocated separate `memref<1xi32>`; now share one `memref<N xi32>` per tile via prescan.
- `shim-prod-lock-init` — shim producer lock init corrected to match oracle.
- `produce-rotation-modulus` — rotation counter wrapped at wrong modulus; silent wrong-buffer-access. Fixed.
- `release-async-produce-counter` — `conduit.release_async` on Port::Produce did not increment counter. Fixed.

---

## Feature Backlog

| Feature | Files affected | Status |
|---|---|---|
| N-D DMA transforms (`dimensionsToStream`/`dimensionsFromStream`) | ~7 | **DONE** — Pass A propagates; Pass C applies to MM2S/S2MM BDs; auto-viaDMA when dims non-empty |
| `repeat_count` | ~14 | **DONE** — BD chain unrolled; lock inits scaled; core ops CSDF-correct; link-distribute fixed |
| `iter_count` | ~10 | **DONE** — DMAStartOp.repeat_count = K-1; non-circular BD chain |
| `disable_synchronization` | 5 | **DONE** — lock suppressed; BD chains intact |
| `viaDMA` force | 1 | **DONE** — renamed from `via_DMA`; auto-set when dims non-empty |
| External buffers | excluded | Pass C integration pending (Conduit.td + Pass A done) |
| `register_process` | unknown | Hard error; design full lowering |
| Cascade mode | N/A | Requires `aie.cascade_flow`; fundamentally different path |

---

## Research Plan

**Team:** 3 students, 2-year timeline
**Target hardware:** Strix (npu2, AIE2p) primary; Phoenix (npu1, AIE2) secondary; xcvc1902 (AIE1) tertiary

**Research statement:** Conduit IR (1) enables programs combining ObjectFIFO's zero-copy window semantics with AIR Channel's N-D DMA efficiency — inexpressible in either alone — and (2) enables optimizations from one framework to apply to programs from both via a shared lowering path.

**Two claims to validate:**
- **Expressiveness:** cross-tier `put_memref_async` + `acquire_async` + `wait_all` expresses programs neither source dialect can
- **Optimization:** `--conduit-depth-promote` applies unchanged to ObjectFIFO-origin and AIR-origin programs

### Track A — Gap Verification + Benchmarking (Student 1)

1. Create `test/gap_verification/` — one `.mlir` per gap row in the comparison table
2. Extend `tools/conduit/compare_lowering_resources.py` to measure hardware performance
3. Run 3 benchmarks: `passthrough_dmas`, `vector_vector_add`, `matrix_scalar_add`
4. Write the paper's evaluation comparison table from hardware measurements

### Track B — Pass C Correctness + Optimization (Student 2)

**Current state:** 65/65 lit tests. Depth-1/2/N correct. Distribute/join resource-equivalent to oracle. All feature gaps closed except external buffers and register_process. depth-2 resource parity (buffer +1, use_lock -8) open.

**Remaining:**
1. Resolve depth-2 resource parity (buffer +1, use_lock -8 vs oracle)
2. External buffers Pass C integration
3. aie_stream corpus gap (stream-routing detection in Pass A)
4. Validate `--conduit-depth-promote` on hardware (Experiment 4 prerequisite)

### Track C — Pass B + Cross-Dialectical Validation (Student 3)

**Current state:** Static shapes handled. Dynamic strides hard-error. Indexed channels not yet supported.

**Dependency:** Track B must reach Experiment 2 pass before cross-dialectical validation is meaningful.

**Remaining:**
1. Pass B indexed channel support (`air.channel @foo[%i][%j]` → flattened named conduits)
2. Pass B async token propagation
3. Cross-tier expressiveness demo on Strix (Empirical Result #1)
4. Cross-dialectical optimization demo: AIR program through `--conduit-depth-promote`

### The Four Hardware Experiments

**Exp 1 — Depth-1 SPSC:** byte-identical output, no deadlock. Validates base correctness claim.
**Exp 2 — Depth-2 double-buffer:** output varies across iterations correctly. Validates rotation counter.
**Exp 3 — Distribute/broadcast:** 3 destinations each receive correct data slice. Validates link lowering.
**Exp 4 — acquire_async stall reduction:** statistically significant stall-cycle reduction. Validates expressiveness claim. Measurement: `AIE_CORE_STALL_CYCLES` event `0x0044` or wall-clock (10 runs, drop 2 warmup, average 8).

**Minimum for workshop paper:** Experiment 1 passes + any statistically significant Experiment 4 result.

---

## Convergence: The Paper

| Section | Track |
|---|---|
| Motivation + gap table | Track A |
| Conduit design + type system | All |
| Expressiveness claim + cross-tier demo | Track B + C |
| Correctness evaluation | Track A + B |
| Optimization claim (depth-promote) | Track A + B |
| Cross-dialectical demo | Track C |

**Year 1 target:** Workshop at LLVM DevMtg, CGO Workshop, or LCTES
**Year 2 target:** Full paper at CGO, PLDI, or MLSys

---

## Reference

| What | Path |
|---|---|
| Dialect spec | `include/aie/Dialect/Conduit/IR/Conduit.td` |
| Pass C (split) | `lib/Dialect/Conduit/Transforms/ConduitToDMA*.cpp` |
| Pass A | `lib/Dialect/Conduit/Transforms/ObjectFifoToConduit.cpp` |
| Pass B | `lib/Dialect/Conduit/Transforms/AirChannelToConduit.cpp` |
| Stateful transform oracle | `lib/Dialect/AIE/Transforms/AIEObjectFifoStatefulTransform.cpp` |
| Hardware benchmarks | paper §6.6 |
| Bug list | CLAUDE.md |
