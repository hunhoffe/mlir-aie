# RFC 000 — Promote `conduit.objectfifo_link` to First-Class Op

**Status:** Draft
**Feature:** `conduit.objectfifo_link`
**Target backend:** mlir-aie (`aie.objectfifo.link`)
**Evidence:** `artifacts/roundtrip_expanded_results.json`

---

## Summary

Promote `conduit.objectfifo_link` from an annotation-based lowering hint to a
first-class Conduit IR operation with a well-defined signature, verifier, and
round-trippable lowering to AIE objectfifo.link.

---

## Motivation

`aie.objectfifo.link` appears in **53 upstream mlir-aie files** (98 total line
occurrences across those files) and enables:

- **Distribute** (1 source → N destinations): split a wide objectfifo across
  multiple consumer tiles at byte offsets.
- **Join** (N sources → 1 destination): concatenate multiple producer FIFOs
  into a single consumer FIFO via MemTile relay.

Without a first-class Conduit representation, any program using objectfifo.link
cannot be round-tripped through the Conduit IR layer without losing the MemTile
relay topology, offset splits, and lock-chain semantics.

The expanded roundtrip validation (100 files, 99% pass rate) confirmed that the
`objectfifo_link_to_conduit.py` script correctly translates all link forms
including `repeat_count`, `via_DMA`, and per-consumer depth lists, emitting
unknown attributes as `conduit.annotate` entries for lossless preservation.

---

## Proposed Conduit Syntax

```
// Distribute: one source, N destinations
conduit.objectfifo_link
    src=[%of_in]
    dsts=[%of_out0, %of_out1]
    mode="distribute"
    memtile="tile(0,1)"
    offsets=[0, 1024]

// Join: N sources, one destination
conduit.objectfifo_link
    src=[%of_src0, %of_src1]
    dsts=[%of_dst]
    mode="join"
    memtile="tile(0,1)"
    offsets=[0, 16]

// Optional preserved attributes
conduit.annotate unknown_attr="repeat_count = 3 : i32"
```

**Attribute semantics:**

| Attribute | Type | Description |
|-----------|------|-------------|
| `src` | list of conduit refs | Source objectfifo(s) |
| `dsts` | list of conduit refs | Destination objectfifo(s) |
| `mode` | `"distribute"` \| `"join"` | Routing direction |
| `memtile` | string | MemTile tile coordinate hint |
| `offsets` | list of ints | Byte offsets for splits (distribute) or insertions (join) |
| `lock_id` | optional int | Preserved synchronisation lock index |

---

## Lowering Notes

### conduit.objectfifo_link → aie.objectfifo.link

```
conduit.objectfifo_link src=[%of_in] dsts=[%of_out0, %of_out1]
    mode="distribute" memtile="tile(0,1)" offsets=[0, 1024]
```
→
```mlir
aie.objectfifo.link [@of_in] -> [@of_out0, @of_out1] ([] [0, 1024])
```

The MemTile coordinate is used to assign the producer tile of the destination
objectfifos. BD chains are generated one per destination.

### conduit.objectfifo_link → objectfifo.create + acquire/release (flattened)

For backends without native link support, the link is expanded into explicit
objectfifo.create declarations plus a software relay kernel using
objectfifo.acquire / subview.access / objectfifo.release at the MemTile core.

### AIR fallback

No direct AIR equivalent. The lowering emits one `air.channel` declaration per
destination and inserts a comment flagging the missing MemTile relay. This is
intentionally lossy and triggers a warning.

---

## Representative Passing Roundtrip Files

The following files from the expanded validation set (100 files, 99% pass)
demonstrate correct roundtrip of objectfifo.link:

1. `mlir-aie/test/dialect/AIE/roundtrip.mlir`
   Category: `objectfifo.link` (distribute, 2 destinations, dimensionsToStream)

2. `mlir-aie/test/objectFifo-stateful-transform/repeat_count/link_repeat_count_test.mlir`
   Category: `objectfifo.link` (distribute with `repeat_count` preserved via `conduit.annotate`)

3. `mlir-aie/test/objectFifo-stateful-transform/data_movement_patterns/link/link_test_broadcast.mlir`
   Category: `objectfifo.link` (broadcast pattern, multiple consumers)

All three files verified in `artifacts/roundtrip_expanded_results.json` with
`pass_metadata_integrity: true`.

---

## Evidence

| Artifact | Path |
|----------|------|
| Roundtrip results (100 files) | `artifacts/roundtrip_expanded_results.json` |
| Parser validation (20-file canonical) | `artifacts/parser_validation.json` |
| Warning triage | `artifacts/triage_warnings.json` |
| Evidence bundle | `artifacts/evidence_bundle.tar` |
| CI smoke log | `artifacts/ci_run.log` |

---

## Acceptance Gates

| Gate | Requirement | Current |
|------|-------------|---------|
| Roundtrip pass rate | ≥ 90% | 99% (99/100) |
| Canonical set (20 files) | 100% | 100% (20/20) |
| CI smoke test | Pass | Pass (20/20) |
| Metadata integrity | src, dst, offsets preserved | Verified |
| Unknown attr preservation | conduit.annotate emitted | Verified |

> **Evidence caveat:** All "pass" results above reflect structural text-transform
> roundtrip correctness (the Python script correctly identifies and reformats
> fields).  They do **not** mean the generated Conduit IR is valid MLIR or that
> any lowering is semantically correct.  No code has been compiled or run on
> hardware.  These gates will need to be re-evaluated once the Conduit dialect
> is registered and `aie-opt` can parse the output files.
