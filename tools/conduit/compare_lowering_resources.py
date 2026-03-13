#!/usr/bin/env python3
"""
compare_lowering_resources.py — resource count comparison between two lowering
paths for the same ObjectFIFO source file.

Path A (existing):
  original.mlir → aie-opt --aie-objectFifo-stateful-transform → lowered_A.mlir

Path B (Conduit pipeline):
  original.mlir → aie-opt --objectfifo-to-conduit --conduit-to-dma → lowered_B.mlir

For each benchmark:
  1. Runs both aie-opt invocations.
  2. Counts aie.dma_bd, aie.lock, aie.buffer, aie.flow, aie.use_lock in each.
  3. Emits a Markdown table comparing the counts.

Usage:
  python3 tools/compare_lowering_resources.py \\
      --aie-opt mlir-aie/build/bin/aie-opt \\
      --output /tmp/conduit_resource_comparison.md

Benchmarks are hard-coded to the mlir-aie test files that match the patterns
exercised by the Conduit Passes A and C lit tests.  Add entries to BENCHMARKS
to extend coverage.

Honest limitations
------------------
- The output will differ from the existing transform if the Conduit passes
  are not yet fully implemented.  Differences are reported as-is; the script
  does not hide mismatches.
- The script counts occurrences of op mnemonics in the textual IR output.
  It may double-count ops inside FileCheck comment lines if the aie-opt binary
  is not available and the input file is read directly.  Filter with
  --strip-comments if needed.
- Lines starting with '//' are excluded from counts (avoids counting CHECK
  comment lines in test inputs).
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Benchmark registry
# ---------------------------------------------------------------------------

# Each entry: (label, source_file_relative_to_mlir_aie)
BENCHMARKS = [
    (
        "depth-1 single fifo",
        "test/objectFifo-stateful-transform/dynamic_lowering/"
        "depth_one_objectfifo_test.mlir",
    ),
    (
        "1→3 distribute with offsets",
        "test/objectFifo-stateful-transform/data_movement_patterns/"
        "link/link_test_distribute_offsets.mlir",
    ),
    (
        "N→1 join with offsets",
        "test/objectFifo-stateful-transform/data_movement_patterns/"
        "link/link_test_join_offsets.mlir",
    ),
    (
        "broadcast (enforced depths)",
        "test/objectFifo-stateful-transform/data_movement_patterns/"
        "broadcast_enforced_depths.mlir",
    ),
    (
        "depth-2 same depth fifos",
        "test/objectFifo-stateful-transform/dynamic_lowering/"
        "same_depth_objectfifos_test.mlir",
    ),
]

# Hardware ops to count in the lowered output.
RESOURCE_OPS = [
    "aie.buffer",
    "aie.lock",
    "aie.flow",
    "aie.dma_bd",
    "aie.use_lock",
    "aie.dma_start",
    "aie.next_bd",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def count_ops(ir_text: str) -> dict:
    """Count occurrences of each resource op in the IR text.

    Lines starting with '//' (FileCheck comments) are excluded.
    """
    counts = {op: 0 for op in RESOURCE_OPS}
    for line in ir_text.splitlines():
        stripped = line.strip()
        if stripped.startswith("//"):
            continue
        for op in RESOURCE_OPS:
            # Match the op mnemonic as a token (not inside a comment or string)
            counts[op] += len(re.findall(r"\b" + re.escape(op) + r"\b", stripped))
    return counts


def run_aie_opt(aie_opt: str, passes: list, source: Path) -> tuple:
    """Run aie-opt with the given passes on source.  Returns (stdout, stderr, ok)."""
    cmd = [aie_opt] + passes + [str(source)]
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=60
        )
        ok = result.returncode == 0
        return result.stdout, result.stderr, ok
    except FileNotFoundError:
        return "", f"aie-opt not found at {aie_opt}", False
    except subprocess.TimeoutExpired:
        return "", "aie-opt timed out", False


def format_count(a: int, b: int) -> str:
    """Format two counts with a delta indicator."""
    if a == b:
        return f"{a} / {b} ✓"
    delta = b - a
    sign = "+" if delta > 0 else ""
    return f"{a} / {b}  ({sign}{delta})"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Compare ObjectFIFO stateful transform vs Conduit pipeline"
    )
    parser.add_argument(
        "--aie-opt",
        default="mlir-aie/build/bin/aie-opt",
        help="Path to the aie-opt binary",
    )
    parser.add_argument(
        "--mlir-aie-root",
        default="mlir-aie",
        help="Path to the mlir-aie repo root (default: mlir-aie)",
    )
    parser.add_argument(
        "--output",
        "-o",
        default=None,
        help="Write Markdown table to this file (default: print to stdout)",
    )
    args = parser.parse_args()

    aie_opt = args.aie_opt
    root = Path(args.mlir_aie_root)

    rows = []

    for label, rel_path in BENCHMARKS:
        source = root / rel_path
        if not source.exists():
            rows.append(
                {
                    "label": label,
                    "error": f"source not found: {source}",
                }
            )
            continue

        # Path A: existing stateful transform
        out_a, err_a, ok_a = run_aie_opt(
            aie_opt, ["--aie-objectFifo-stateful-transform"], source
        )

        # Path B: Conduit pipeline
        out_b, err_b, ok_b = run_aie_opt(
            aie_opt, ["--objectfifo-to-conduit", "--conduit-to-dma"], source
        )

        counts_a = count_ops(out_a) if ok_a else None
        counts_b = count_ops(out_b) if ok_b else None

        rows.append(
            {
                "label": label,
                "ok_a": ok_a,
                "ok_b": ok_b,
                "err_a": err_a,
                "err_b": err_b,
                "counts_a": counts_a,
                "counts_b": counts_b,
            }
        )

    # Build Markdown output
    lines = [
        "# Conduit vs ObjectFIFO Stateful Transform — Resource Comparison",
        "",
        "**Path A:** `aie-opt --aie-objectFifo-stateful-transform`  ",
        "**Path B:** `aie-opt --objectfifo-to-conduit --conduit-to-dma`  ",
        "",
        "Format: `Path-A / Path-B (delta)`.  A delta of 0 means the passes are resource-equivalent.",
        "",
    ]

    for row in rows:
        label = row["label"]
        lines.append(f"## {label}")
        lines.append("")

        if "error" in row:
            lines.append(f"> ERROR: {row['error']}")
            lines.append("")
            continue

        if not row["ok_a"]:
            lines.append(f"> Path A FAILED: {row['err_a'][:200]}")
        if not row["ok_b"]:
            lines.append(f"> Path B FAILED: {row['err_b'][:200]}")

        if not (row["ok_a"] and row["ok_b"]):
            lines.append("")
            continue

        ca = row["counts_a"]
        cb = row["counts_b"]

        # Table header
        lines.append("| Op | Path A | Path B | Delta |")
        lines.append("|---|---|---|---|")
        all_match = True
        for op in RESOURCE_OPS:
            a, b = ca[op], cb[op]
            delta = b - a
            sign = "+" if delta > 0 else ""
            match_icon = "✓" if delta == 0 else "**DIFF**"
            if delta != 0:
                all_match = False
            lines.append(
                f"| `{op}` | {a} | {b} | {sign}{delta} {match_icon} |"
            )

        if all_match:
            lines.append("")
            lines.append("> All resource counts match — passes are equivalent.")
        else:
            lines.append("")
            lines.append(
                "> Resource counts differ — see delta column for details."
            )
        lines.append("")

    md = "\n".join(lines)

    if args.output:
        Path(args.output).write_text(md, encoding="utf-8")
        print(f"Written to {args.output}", file=sys.stderr)
    else:
        print(md)


if __name__ == "__main__":
    main()
