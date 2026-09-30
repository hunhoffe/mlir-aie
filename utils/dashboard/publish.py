#!/usr/bin/env python3
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Record one run of a dashboard metric target on the publication branch.

The maintainer dashboard keeps every non-kernel series (test coverage, wheel
sizes, ...) under ``dashboard/metrics/<target>/`` on the publication branch,
in the files ``kernel-checks/<npu>/`` already uses, written by the same code:
``utils/kernel_checks/publish.py`` is loaded by path and its ``rebuild`` does
the pruning (every run of the last 90 days, then one per ISO week, tagged
releases forever) and writes ``runs.json``, ``latest.json`` and one
``history/<metric>.json`` per metric. The dashboard page reads those.

    publish.py record --target coverage --rows coverage-rows.json \
        --run-id 42 --run-url https://github.com/.../actions/runs/42 \
        [--provenance "clang 20.1 | os ubuntu-24.04"] [--tag v1.2.0] \
        --out gh-pages/dashboard/metrics/coverage

``--rows`` is a JSON list in the perf.json shape the kernel checks write,
``[{"name": "<series>/<metric>", "unit": ..., "value": ..., "range"?: ...}]``;
the part after the last slash is the metric, the rest the series (a file,
a wheel, a directory). Standard library only.
"""

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

KERNEL_CHECKS = Path(__file__).resolve().parents[1] / "kernel_checks" / "publish.py"
# The targets a workflow may record, and what a larger value of their series means.
TARGETS = {
    "coverage": {"better": "higher", "about": "test coverage of the C++ sources"},
    "wheels": {"better": "lower", "about": "size of the mlir_aie wheels"},
}


def kernel_checks():
    """The kernel checks publisher, whose record format and rebuild this shares."""
    spec = importlib.util.spec_from_file_location(
        "kernel_checks_publish", KERNEL_CHECKS
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def record_rows(
    rows: list[dict], *, target: str, run: dict, provenance: str = ""
) -> dict:
    """The run record of ``rows``, in the kernel checks' record shape."""
    kc = kernel_checks()
    return {
        "schema": kc.SCHEMA,
        "target": target,
        **run,
        "pmode": None,
        "provenance": kc.provenance_fields(provenance),
        "better": TARGETS.get(target, {}).get("better"),
        "sane": True,
        "published": bool(rows),
        "n_rows": len(rows),
        "failed": [],
        "rows": kc.rows_by_case(rows),
    }


def publish(
    out: Path, rows: list[dict], *, target: str, run: dict, provenance="", now=None
) -> dict:
    """Write the run's record under ``out`` and rebuild the derived files."""
    kc = kernel_checks()
    index = out / "runs.json"
    if index.exists():
        kc._check_schema(json.loads(index.read_text()), index)
    record = record_rows(rows, target=target, run=run, provenance=provenance)
    (out / "runs").mkdir(parents=True, exist_ok=True)
    (out / "runs" / f"{run['id']}.json").write_text(json.dumps(record, indent=1))
    kc.rebuild(out, now)
    return record


def main(argv=None) -> int:
    kc = kernel_checks()
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n", 1)[0])
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("record", help="record one run of a target and rebuild it")
    p.add_argument("--target", required=True, choices=sorted(TARGETS))
    p.add_argument("--rows", required=True, type=Path, help="JSON list of rows")
    p.add_argument("--run-id", required=True)
    p.add_argument("--run-url", default="")
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--provenance", default="", help='"key value | key value"')
    p.add_argument("--tag", default="", help="the release tag this run measured")
    p.add_argument("--commit", default=os.environ.get("GITHUB_SHA", ""))
    p.add_argument("--commit-message", default="")
    p.add_argument("--commit-date", default="")
    p.add_argument("--date", default="", help="ISO date of the run (default: now)")
    p = sub.add_parser("rebuild", help="rewrite the derived files from the records")
    p.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)

    if args.command == "rebuild":
        print(json.dumps(kc.rebuild(args.out)))
        return 0
    run = {
        "id": args.run_id,
        "url": args.run_url,
        "date": args.date or kc.iso(kc.now_utc()),
        "commit": kc.commit_info(args.commit, args.commit_message, args.commit_date),
    }
    if args.tag:
        run["tag"] = args.tag
    rows = json.loads(args.rows.read_text())
    record = publish(
        args.out, rows, target=args.target, run=run, provenance=args.provenance
    )
    print(
        f"record {args.target}: run {record['id']}, {record['n_rows']} rows, published={record['published']}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
