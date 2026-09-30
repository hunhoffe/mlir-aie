#!/usr/bin/env python3
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Measure the mlir_aie wheels of one build as dashboard rows.

    wheel_sizes.py <dir> [<dir>...] --out rows.json

Every ``*.whl`` under the directories is one series,
``<platform>/<rtti>/<python>`` (``manylinux_2_28_x86_64/rtti_ON/cp312``):
platform and python tag come from the PEP 427 filename
(``mlir_aie-<version>-<pytag>-<abitag>-<platform>.whl``), RTTI from the
artifact directory the wheel was downloaded into (``mlir_aie_rtti_ON-3.12``),
since the filename does not carry it. Each series has ``bytes`` (the file),
``uncompressed_bytes`` (its members) and ``files`` (member count); ``all``
sums them (``bytes``, ``wheels``). ``--out`` gets the rows in the shape
``publish.py record --target wheels --rows`` reads. Standard library only.
"""

import argparse
import json
import re
import sys
import zipfile
from pathlib import Path

RTTI = re.compile(r"rtti_(ON|OFF)(?![A-Za-z0-9])")


def wheel_tags(filename: str):
    """``(python, platform)`` of a PEP 427 wheel filename, or None."""
    parts = Path(filename).stem.split("-")
    # distribution-version[-build]-python-abi-platform
    if len(parts) < 5 or not filename.endswith(".whl"):
        return None
    return parts[-3], parts[-1]


def rtti_of(path: Path, root: Path) -> str:
    """The RTTI setting named by the nearest enclosing directory, else unknown."""
    names = [p.name for p in path.relative_to(root).parents] + [root.resolve().name]
    for name in names:
        m = RTTI.search(name)
        if m:
            return f"rtti_{m.group(1)}"
    return "rtti_unknown"


def measure(path: Path) -> dict:
    with zipfile.ZipFile(path) as zf:
        infos = zf.infolist()
    return {
        "bytes": path.stat().st_size,
        "uncompressed_bytes": sum(i.file_size for i in infos),
        "files": len(infos),
    }


def collect(dirs: list[Path], log=print) -> dict[str, dict]:
    """Series -> measurement of the wheels under ``dirs``; larger wins a clash."""
    wheels: dict[str, dict] = {}
    for root in dirs:
        for path in sorted(root.rglob("*.whl")):
            tags = wheel_tags(path.name)
            if tags is None:
                log(f"::warning::not a wheel filename, skipped: {path}")
                continue
            python, platform = tags
            series = f"{platform}/{rtti_of(path, root)}/{python}"
            found = {"path": str(path), **measure(path)}
            other = wheels.get(series)
            if other is not None:
                # Two builds of one series: chart the worse case.
                log(
                    f"::warning::two wheels for {series}: {other['path']} and "
                    f"{path}; keeping the larger"
                )
                if other["bytes"] >= found["bytes"]:
                    continue
            wheels[series] = found
    return wheels


UNITS = {"bytes": "bytes", "uncompressed_bytes": "bytes", "files": "files"}


def rows_of(wheels: dict[str, dict]) -> list[dict]:
    rows = [
        {"name": f"{series}/{metric}", "unit": unit, "value": wheel[metric]}
        for series, wheel in sorted(wheels.items())
        for metric, unit in UNITS.items()
    ]
    rows.append(
        {
            "name": "all/bytes",
            "unit": "bytes",
            "value": sum(w["bytes"] for w in wheels.values()),
        }
    )
    rows.append({"name": "all/wheels", "unit": "wheels", "value": len(wheels)})
    return rows


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n", 1)[0])
    parser.add_argument("dirs", nargs="+", type=Path, help="directories of wheels")
    parser.add_argument("--out", required=True, type=Path, help="rows JSON to write")
    args = parser.parse_args(argv)

    wheels = collect(args.dirs)
    for series, wheel in sorted(wheels.items()):
        print(f"{series}: {wheel['bytes']} bytes")
    total = sum(w["bytes"] for w in wheels.values())
    print(f"total: {len(wheels)} wheels, {total} bytes")
    if not wheels:
        print("::warning::no wheels found")
    args.out.write_text(json.dumps(rows_of(wheels), indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
