#!/usr/bin/env python3
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Turn an lcov export into the maintainer dashboard's coverage series.

    coverage_summary.py build/report/coverage.lcov --out coverage-rows.json \
        [--root /path/to/mlir-aie] [--depth 2] [--summary coverage-summary.json]

The rows (``--out``) are what ``utils/dashboard/publish.py record --target
coverage`` takes: one series ``total`` for the whole export, plus one per
directory prefix of up to ``--depth`` components (``lib``, ``lib/Dialect``,
``include/aie``, ...). Every series has

    <series>/line_pct         %       lines hit / lines instrumented
    <series>/lines            lines   instrumented lines
    <series>/lines_covered    lines   lines hit
    <series>/function_pct     %
    <series>/functions        functions
    <series>/functions_covered functions
    <series>/branch_pct       %       only when the export has BRDA records
    <series>/branches         branches
    <series>/branches_covered branches

and a percentage is left out when its denominator is zero. ``--summary``
writes the same counts nested by series, for reading by hand. Standard
library only: the workflow runs it from a bare checkout.
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Sources nobody's tests are expected to cover, and out-of-tree build products.
SKIPPED_COMPONENTS = ("third_party",)
SKIPPED_PREFIXES = ("build",)
KINDS = ("lines", "functions", "branches")


def parse_lcov(text: str) -> tuple[dict[str, dict], bool]:
    """Per source file, the hit count of each line, function and branch.

    Records for the same file (one per object llvm-cov was given) are merged
    by keeping the higher count, so a repeated file is not counted twice.
    Returns the files and whether any BRDA record was seen at all.
    """
    files: dict[str, dict] = {}
    current = None
    has_branches = False
    for raw in text.splitlines():
        line = raw.strip()
        if line.startswith("SF:"):
            current = files.setdefault(
                line[3:].strip(), {"lines": {}, "functions": {}, "branches": {}}
            )
        elif line == "end_of_record":
            current = None
        elif current is None:
            continue
        elif line.startswith("DA:"):
            fields = line[3:].split(",")
            if len(fields) >= 2:
                _bump(current["lines"], fields[0], _count(fields[1]))
        elif line.startswith("FNDA:"):
            hits, _, name = line[5:].partition(",")
            _bump(current["functions"], name, _count(hits))
        elif line.startswith("BRDA:"):
            has_branches = True
            fields = line[5:].split(",")
            if len(fields) >= 4:
                # '-' means the block was never entered: a branch not taken.
                taken = fields[3].strip()
                hits = 0 if taken == "-" else _count(taken)
                _bump(current["branches"], tuple(fields[:3]), hits)
    return files, has_branches


def _count(text: str) -> int:
    try:
        return int(text.strip())
    except ValueError:
        return 0


def _bump(counts: dict, key, hits: int) -> None:
    counts[key] = max(counts.get(key, 0), hits)


def relative_path(path: str, root: str | None) -> str:
    """The path as the dashboard names it: relative to the repository root.

    With no ``--root``, everything through the last ``/mlir-aie/`` component
    is dropped (the Actions workspace is ``.../work/mlir-aie/mlir-aie/``), so
    a locally produced lcov and the nightly's name their files the same.
    """
    path = path.strip().replace("\\", "/")
    if root:
        root = root.rstrip("/") + "/"
        if path.startswith(root):
            path = path[len(root) :]
    elif "/mlir-aie/" in path:
        path = path.rsplit("/mlir-aie/", 1)[1]
    while path.startswith("./"):
        path = path[2:]
    return os.path.normpath(path) if path else path


def is_skipped(path: str) -> bool:
    parts = path.split("/")[:-1]
    return any(p in SKIPPED_COMPONENTS or p.startswith(SKIPPED_PREFIXES) for p in parts)


def series_of(path: str, depth: int) -> list[str]:
    """``total`` and each directory prefix of ``path`` up to ``depth`` deep."""
    dirs = path.split("/")[:-1]
    if path.startswith("/"):
        # An absolute path outside the root belongs to no directory series.
        return ["total"]
    return ["total"] + ["/".join(dirs[:n]) for n in range(1, min(depth, len(dirs)) + 1)]


def aggregate(
    files: dict[str, dict], *, root: str | None, depth: int, has_branches: bool
) -> tuple[dict[str, dict], int]:
    """Totals and hits per series, and the number of files counted."""
    series: dict[str, dict] = {}
    counted = 0
    kinds = KINDS if has_branches else KINDS[:2]
    for path, counts in files.items():
        rel = relative_path(path, root)
        if not rel or is_skipped(rel):
            continue
        counted += 1
        for name in series_of(rel, depth):
            acc = series.setdefault(
                name, {k: {"total": 0, "covered": 0} for k in kinds}
            )
            for kind in kinds:
                acc[kind]["total"] += len(counts[kind])
                acc[kind]["covered"] += sum(1 for h in counts[kind].values() if h > 0)
    ordered = {
        "total": series.pop("total", {k: {"total": 0, "covered": 0} for k in kinds})
    }
    ordered.update(sorted(series.items()))
    return ordered, counted


def rows_of(series: dict[str, dict]) -> list[dict]:
    """The publish.py rows: ``<series>/<metric>`` with a unit and a value."""
    rows = []
    for name, acc in series.items():
        for kind, singular in (
            ("lines", "line"),
            ("functions", "function"),
            ("branches", "branch"),
        ):
            if kind not in acc:
                continue
            total, covered = acc[kind]["total"], acc[kind]["covered"]
            if total:
                rows.append(
                    {
                        "name": f"{name}/{singular}_pct",
                        "unit": "%",
                        "value": round(100 * covered / total, 2),
                    }
                )
            rows.append({"name": f"{name}/{kind}", "unit": kind, "value": total})
            rows.append(
                {"name": f"{name}/{kind}_covered", "unit": kind, "value": covered}
            )
    return rows


def one_line(total: dict) -> str:
    parts = []
    for kind in KINDS:
        if kind not in total:
            continue
        n, covered = total[kind]["total"], total[kind]["covered"]
        pct = f"{100 * covered / n:.1f}%" if n else "n/a"
        parts.append(f"{pct} of {n} {kind}")
    return "total: " + ", ".join(parts)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n", 1)[0])
    parser.add_argument("lcov", type=Path, help="the lcov export to summarize")
    parser.add_argument("--out", required=True, type=Path, help="rows JSON to write")
    parser.add_argument(
        "--root",
        default=None,
        help="prefix to strip from SF paths (default: through the last /mlir-aie/)",
    )
    parser.add_argument(
        "--depth", type=int, default=2, help="directory levels to report (default 2)"
    )
    parser.add_argument(
        "--summary", type=Path, default=None, help="nested counts JSON to write"
    )
    args = parser.parse_args(argv)

    files, has_branches = parse_lcov(args.lcov.read_text(errors="replace"))
    series, counted = aggregate(
        files, root=args.root, depth=args.depth, has_branches=has_branches
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rows_of(series), indent=1) + "\n")
    if args.summary:
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(
            json.dumps({"series": series, "files": counted}, indent=1) + "\n"
        )
    print(one_line(series["total"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
