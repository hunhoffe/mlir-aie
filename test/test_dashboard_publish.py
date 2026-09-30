# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# RUN: %pytest %s

"""Record dashboard metric runs in the kernel checks' files; standard library only."""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "utils/dashboard/publish.py"


@pytest.fixture(scope="module")
def publish():
    spec = importlib.util.spec_from_file_location("dashboard_publish", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ROWS = [
    {"name": "total/line_pct", "unit": "%", "value": 61.2},
    {"name": "total/lines", "unit": "lines", "value": 40210},
    {"name": "lib/Dialect/line_pct", "unit": "%", "value": 70.5},
]


def run(id, date, tag=None):
    out = {
        "id": id,
        "url": f"https://example.com/runs/{id}",
        "date": date,
        "commit": {"id": "abc", "url": "", "message": "m", "timestamp": date},
    }
    if tag:
        out["tag"] = tag
    return out


def test_records_share_the_kernel_checks_format(publish, tmp_path):
    out = tmp_path / "coverage"
    rec = publish.publish(
        out,
        ROWS,
        target="coverage",
        run=run("1", "2026-09-01T00:00:00+00:00"),
        provenance="clang 20 | os ubuntu",
    )
    assert rec["target"] == "coverage" and rec["published"] and rec["n_rows"] == 3
    assert rec["provenance"] == {"clang": "20", "os": "ubuntu"}
    assert rec["better"] == "higher"
    assert rec["rows"]["total"]["line_pct"] == {"value": 61.2, "unit": "%"}
    publish.publish(
        out,
        [r | {"value": r["value"] + 1} for r in ROWS],
        target="coverage",
        run=run("2", "2026-09-02T00:00:00+00:00"),
    )
    index = json.loads((out / "runs.json").read_text())
    assert [r["id"] for r in index["runs"]] == ["1", "2"]
    assert "rows" not in index["runs"][0]
    latest = json.loads((out / "latest.json").read_text())
    assert latest["id"] == "2"
    history = json.loads((out / "history/line_pct.json").read_text())
    assert history["unit"] == "%"
    assert history["series"]["total"]["values"] == [61.2, 62.2]
    assert history["series"]["lib/Dialect"]["values"] == [70.5, 71.5]
    assert [r["id"] for r in history["runs"]] == ["1", "2"]
    assert sorted(p.name for p in (out / "history").glob("*.json")) == [
        "line_pct.json",
        "lines.json",
    ]


def test_a_run_without_rows_is_recorded_but_not_charted(publish, tmp_path):
    out = tmp_path / "wheels"
    rec = publish.publish(
        out, [], target="wheels", run=run("1", "2026-09-01T00:00:00+00:00")
    )
    assert rec["published"] is False and rec["better"] == "lower"
    assert not (out / "latest.json").exists()
    assert json.loads((out / "runs.json").read_text())["runs"][0]["published"] is False


def test_command_line_records_a_rows_file(tmp_path):
    rows = tmp_path / "rows.json"
    rows.write_text(json.dumps(ROWS))
    out = tmp_path / "coverage"
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "record",
            "--target",
            "coverage",
            "--rows",
            str(rows),
            "--run-id",
            "7",
            "--run-url",
            "https://example.com/7",
            "--out",
            str(out),
            "--commit",
            "",
            "--date",
            "2026-09-03T01:02:03+00:00",
            "--tag",
            "v9.9.9",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "record coverage: run 7, 3 rows, published=True" in result.stdout
    latest = json.loads((out / "latest.json").read_text())
    assert latest["tag"] == "v9.9.9" and latest["date"] == "2026-09-03T01:02:03+00:00"
    rebuilt = subprocess.run(
        [sys.executable, str(SCRIPT), "rebuild", "--out", str(out)],
        capture_output=True,
        text=True,
        check=True,
    )
    assert json.loads(rebuilt.stdout)["metrics"] == ["line_pct", "lines"]


def test_unknown_target_is_refused(tmp_path):
    rows = tmp_path / "rows.json"
    rows.write_text("[]")
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "record",
            "--target",
            "mystery",
            "--rows",
            str(rows),
            "--run-id",
            "1",
            "--out",
            str(tmp_path / "x"),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "invalid choice" in result.stderr
