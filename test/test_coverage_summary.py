# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# RUN: %pytest %s

"""Summarize an lcov export into dashboard series; standard library only."""

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "utils/dashboard/coverage_summary.py"
PUBLISH = ROOT / "utils/dashboard/publish.py"
WORKFLOW = ROOT / ".github/workflows/codeCoverage.yml"

WORKSPACE = "/home/runner/work/mlir-aie/mlir-aie"
# Two directories deep enough to test the depth rule, and a third_party file
# that must not count. A.cpp: 3 of 4 lines, 1 of 2 functions, 1 of 2
# branches; main.cpp: 2 of 2 lines, 1 function, 0 of 1 branches.
LCOV = f"""SF:{WORKSPACE}/lib/Dialect/AIE/IR/A.cpp
FN:1,_Zf
FN:5,_Zg
FNDA:3,_Zf
FNDA:0,_Zg
DA:1,3
DA:2,0
DA:3,1
DA:4,7
BRDA:1,0,0,2
BRDA:1,0,1,-
LF:4
LH:3
end_of_record
SF:./tools/aie-opt/main.cpp
FNDA:1,main
DA:1,1
DA:2,1
BRDA:2,0,0,0
end_of_record
SF:{WORKSPACE}/third_party/x/y.cpp
FNDA:0,_Zy
DA:1,0
DA:2,0
end_of_record
"""


@pytest.fixture(scope="module")
def summary():
    spec = importlib.util.spec_from_file_location("coverage_summary", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run_cli(tmp_path, text=LCOV, *args):
    lcov = tmp_path / "coverage.lcov"
    lcov.write_text(text)
    rows = tmp_path / "rows.json"
    nested = tmp_path / "summary.json"
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            str(lcov),
            "--out",
            str(rows),
            "--summary",
            str(nested),
            *args,
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return (
        result.stdout,
        json.loads(rows.read_text()),
        json.loads(nested.read_text()),
    )


def by_name(rows):
    return {r["name"]: r for r in rows}


def test_series_count_lines_functions_and_branches_per_directory(tmp_path):
    stdout, rows, nested = run_cli(tmp_path)
    assert (
        stdout.strip()
        == "total: 83.3% of 6 lines, 66.7% of 3 functions, 33.3% of 3 branches"
    )
    assert nested["files"] == 2
    assert list(nested["series"]) == [
        "total",
        "lib",
        "lib/Dialect",
        "tools",
        "tools/aie-opt",
    ]
    assert nested["series"]["total"] == {
        "lines": {"total": 6, "covered": 5},
        "functions": {"total": 3, "covered": 2},
        "branches": {"total": 3, "covered": 1},
    }
    assert nested["series"]["lib/Dialect"] == nested["series"]["lib"]
    assert nested["series"]["lib"]["lines"] == {"total": 4, "covered": 3}
    assert nested["series"]["tools"]["branches"] == {"total": 1, "covered": 0}

    named = by_name(rows)
    assert len(named) == len(rows) == 5 * 9
    assert named["total/line_pct"] == {
        "name": "total/line_pct",
        "unit": "%",
        "value": 83.33,
    }
    assert named["total/lines"] == {"name": "total/lines", "unit": "lines", "value": 6}
    assert named["total/lines_covered"]["value"] == 5
    assert named["lib/function_pct"]["value"] == 50.0
    assert named["lib/functions"] == {
        "name": "lib/functions",
        "unit": "functions",
        "value": 2,
    }
    assert named["lib/functions_covered"]["value"] == 1
    assert named["lib/Dialect/branch_pct"]["value"] == 50.0
    assert named["tools/aie-opt/branch_pct"]["value"] == 0.0
    assert named["tools/aie-opt/branches"] == {
        "name": "tools/aie-opt/branches",
        "unit": "branches",
        "value": 1,
    }
    assert named["tools/branches_covered"]["value"] == 0
    assert not any(name.startswith("third_party") for name in named)
    assert not any("/AIE/" in name or "/IR/" in name for name in named)
    # Every row is what publish.py takes: series/metric, a unit, a number.
    for row in rows:
        assert set(row) == {"name", "unit", "value"}
        assert row["name"].count("/") >= 1
        assert row["unit"] in {"%", "lines", "functions", "branches"}
        assert isinstance(row["value"], (int, float))
        if row["unit"] == "%":
            assert 0 <= row["value"] <= 100


def test_depth_root_and_skipped_directories(tmp_path):
    _, rows, nested = run_cli(tmp_path, LCOV, "--depth", "3")
    assert "lib/Dialect/AIE" in nested["series"]
    assert "lib/Dialect/AIE/IR" not in nested["series"]
    assert by_name(rows)["lib/Dialect/AIE/lines"]["value"] == 4

    # With an explicit root the same files are named the same...
    _, rows_root, nested_root = run_cli(tmp_path, LCOV, "--root", WORKSPACE + "/")
    assert nested_root == run_cli(tmp_path)[2]
    assert rows_root == run_cli(tmp_path)[1]
    # ...and a root that matches nothing leaves absolute paths in no series
    # but total, while a build directory is skipped like third_party.
    text = LCOV.replace(f"{WORKSPACE}/lib/", "/elsewhere/lib/").replace(
        "./tools/", "./build_release/tools/"
    )
    _, rows_other, nested_other = run_cli(tmp_path, text, "--root", "/nowhere")
    assert list(nested_other["series"]) == ["total"]
    assert nested_other["files"] == 1
    assert nested_other["series"]["total"]["lines"] == {"total": 4, "covered": 3}
    assert by_name(rows_other)["total/functions"]["value"] == 2


def test_no_branch_rows_without_brda_and_no_percentage_of_nothing(tmp_path):
    text = "".join(l for l in LCOV.splitlines(True) if not l.startswith("BRDA:"))
    empty = "SF:lib/Empty.cpp\nend_of_record\n"
    stdout, rows, nested = run_cli(tmp_path, text + empty)
    assert stdout.strip() == "total: 83.3% of 6 lines, 66.7% of 3 functions"
    assert set(nested["series"]["total"]) == {"lines", "functions"}
    assert not any("branch" in r["name"] for r in rows)
    # Empty.cpp counts as a file with nothing in it, so no percentages.
    assert nested["files"] == 3
    assert nested["series"]["lib"]["lines"] == {"total": 4, "covered": 3}
    _, rows_empty, nested_empty = run_cli(tmp_path, empty)
    assert nested_empty["series"]["lib"] == {
        "lines": {"total": 0, "covered": 0},
        "functions": {"total": 0, "covered": 0},
    }
    assert [r["name"] for r in rows_empty] == [
        "total/lines",
        "total/lines_covered",
        "total/functions",
        "total/functions_covered",
        "lib/lines",
        "lib/lines_covered",
        "lib/functions",
        "lib/functions_covered",
    ]


def test_repeated_records_of_a_file_are_merged_not_summed(summary):
    text = (
        "SF:lib/A.cpp\nDA:1,0\nDA:2,1\nFNDA:0,f\nend_of_record\n"
        "SF:lib/A.cpp\nDA:1,4\nDA:2,0\nFNDA:2,f\nend_of_record\n"
    )
    files, has_branches = summary.parse_lcov(text)
    assert not has_branches
    assert files == {
        "lib/A.cpp": {"lines": {"1": 4, "2": 1}, "functions": {"f": 2}, "branches": {}}
    }
    series, counted = summary.aggregate(files, root=None, depth=2, has_branches=False)
    assert counted == 1
    assert series["lib"]["lines"] == {"total": 2, "covered": 2}
    assert series["lib"]["functions"] == {"total": 1, "covered": 1}


@pytest.mark.parametrize(
    "path,root,expected",
    [
        (f"{WORKSPACE}/lib/A.cpp", None, "lib/A.cpp"),
        ("/tmp/mlir-aie/include/aie/B.h", None, "include/aie/B.h"),
        ("./python/c.cpp", None, "python/c.cpp"),
        ("lib/x/../A.cpp", None, "lib/A.cpp"),
        ("/opt/src/lib/A.cpp", "/opt/src", "lib/A.cpp"),
        ("/opt/src/lib/A.cpp", "/opt/src/", "lib/A.cpp"),
        ("/opt/other/lib/A.cpp", "/opt/src", "/opt/other/lib/A.cpp"),
    ],
)
def test_paths_become_repository_relative(summary, path, root, expected):
    assert summary.relative_path(path, root) == expected


def test_rows_round_trip_through_the_dashboard_publisher(tmp_path):
    _, rows, _ = run_cli(tmp_path)
    rows_file = tmp_path / "coverage-rows.json"
    rows_file.write_text(json.dumps(rows))
    out = tmp_path / "metrics" / "coverage"
    result = subprocess.run(
        [
            sys.executable,
            str(PUBLISH),
            "record",
            "--target",
            "coverage",
            "--rows",
            str(rows_file),
            "--run-id",
            "11",
            "--run-url",
            "https://example.com/runs/11",
            "--provenance",
            "compiler clang-20 | os ubuntu24",
            "--out",
            str(out),
            "--commit",
            "",
            "--date",
            "2026-09-30T00:10:00+00:00",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    assert "record coverage: run 11, 45 rows, published=True" in result.stdout
    history = json.loads((out / "history/line_pct.json").read_text())
    assert history["unit"] == "%"
    assert history["series"]["total"]["values"] == [83.33]
    assert history["series"]["lib/Dialect"]["values"] == [75.0]
    assert [r["id"] for r in history["runs"]] == ["11"]
    assert history["runs"][0]["date"] == "2026-09-30T00:10:00+00:00"
    assert sorted(p.stem for p in (out / "history").glob("*.json")) == [
        "branch_pct",
        "branches",
        "branches_covered",
        "function_pct",
        "functions",
        "functions_covered",
        "line_pct",
        "lines",
        "lines_covered",
    ]
    latest = json.loads((out / "latest.json").read_text())
    assert latest["provenance"] == {"compiler": "clang-20", "os": "ubuntu24"}
    assert latest["rows"]["total"]["lines"] == {"value": 6, "unit": "lines"}


def workflow():
    # Avoid YAML 1.1 interpreting GitHub's "on" key as a boolean.
    return yaml.load(WORKFLOW.read_text(), Loader=yaml.BaseLoader)


def test_workflow_summarizes_then_publishes_under_the_gh_pages_lock():
    jobs = workflow()["jobs"]
    coverage = jobs["code-coverage"]
    summarize = next(
        s["run"] for s in coverage["steps"] if "coverage_summary.py" in s.get("run", "")
    )
    assert '--root "$GITHUB_WORKSPACE"' in summarize
    assert "GITHUB_STEP_SUMMARY" in summarize
    upload = next(
        s
        for s in coverage["steps"]
        if s.get("uses", "").startswith("actions/upload-artifact@")
    )
    assert upload["with"]["name"] == "coverage-report"
    assert "--out " + upload["with"]["path"] + "/coverage-rows.json" in summarize
    assert "--summary " + upload["with"]["path"] + "/coverage-summary.json" in summarize

    publish = jobs["publish"]
    assert publish["needs"] == "code-coverage"
    assert "refs/heads/main" in publish["if"]
    assert publish["permissions"] == {"contents": "write"}
    assert publish["concurrency"] == {
        "group": "gh-pages-publish",
        "cancel-in-progress": "false",
        "queue": "max",
    }
    steps = publish["steps"]
    download = next(
        s for s in steps if s.get("uses", "").startswith("actions/download-artifact@")
    )
    assert download["with"] == {"name": "coverage-report", "path": "report"}
    scripts = "\n".join(s.get("run", "") for s in steps)
    assert "--target coverage" in scripts
    assert "--rows report/coverage-rows.json" in scripts
    assert "--out gh-pages-wt/dashboard/metrics/coverage" in scripts
    assert "push origin HEAD:gh-pages" in scripts
    drilldown = next(s for s in steps if s.get("id") == "drilldown")
    assert drilldown["env"]["MAX_MB"] == "60"
    assert "::warning::" in drilldown["run"]
    assert steps[-1]["uses"] == "./.github/actions/squash-gh-pages"
    assert "squash-gh-pages" in steps[-1]["uses"]
    # The pinned SHAs match the rest of the file.
    pins = {}
    for job in jobs.values():
        for step in job["steps"]:
            action, _, sha = step.get("uses", "").partition("@")
            if action.startswith("actions/"):
                assert pins.setdefault(action, sha) == sha


def run_drilldown(tmp_path, max_mb, report_bytes):
    step = next(
        s for s in workflow()["jobs"]["publish"]["steps"] if s.get("id") == "drilldown"
    )
    src = tmp_path / "report"
    src.mkdir(exist_ok=True)
    (src / "index.html").write_bytes(b"<html>" + b"x" * report_bytes)
    (src / "coverage-summary.json").write_text('{"files": 2}')
    (src / "coverage-rows.json").write_text("[]")
    (src / "provenance.txt").write_text("compiler clang")
    # The lcov weighs more than the cap, and must not count.
    (src / "coverage.lcov").write_bytes(b"D" * (2 * 1024 * 1024))
    dest = tmp_path / "gh-pages-wt/dashboard/coverage"
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "report").mkdir(exist_ok=True)
    (dest / "report/stale.html").write_text("old")
    output = tmp_path / "github_output"
    output.write_text("")
    result = subprocess.run(
        ["bash", "-eo", "pipefail", "-c", step["run"]],
        cwd=tmp_path,
        env={
            **os.environ,
            **step["env"],
            "MAX_MB": str(max_mb),
            "GITHUB_OUTPUT": str(output),
        },
        capture_output=True,
        text=True,
        check=True,
    )
    outputs = dict(l.split("=", 1) for l in output.read_text().splitlines())
    return result, outputs, dest


def test_drilldown_is_published_only_under_the_size_cap(tmp_path):
    result, outputs, dest = run_drilldown(tmp_path, max_mb=1, report_bytes=100)
    assert outputs == {"published": "true"}
    assert "::warning::" not in result.stdout
    assert json.loads((dest / "summary.json").read_text()) == {"files": 2}
    assert (dest / "report/index.html").exists()
    assert not (dest / "report/stale.html").exists()
    assert sorted(p.name for p in (dest / "report").iterdir()) == ["index.html"]

    result, outputs, dest = run_drilldown(
        tmp_path, max_mb=1, report_bytes=2 * 1024 * 1024
    )
    assert outputs == {"published": "false"}
    assert "::warning::" in result.stdout and "1 MB" in result.stdout
    # The series' summary still lands; the old drill-down is left in place.
    assert json.loads((dest / "summary.json").read_text()) == {"files": 2}
    assert (dest / "report/stale.html").exists()
