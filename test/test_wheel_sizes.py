# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# RUN: %pytest %s

"""Measure synthetic wheels as dashboard rows, and check the nightly job that records them."""

import importlib.util
import json
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "utils/dashboard/wheel_sizes.py"
WORKFLOW = ROOT / ".github/workflows/buildRyzenWheels.yml"


@pytest.fixture(scope="module")
def sizes():
    spec = importlib.util.spec_from_file_location("wheel_sizes", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_wheel(directory: Path, name: str, members: dict[str, bytes]) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        for member, data in members.items():
            zf.writestr(member, data)
    return path


LINUX = "mlir_aie-1.0.0.dev0-cp312-cp312-manylinux_2_28_x86_64.whl"
WINDOWS = "mlir_aie-1.0.0.dev0-cp312-cp312-win_amd64.whl"
MEMBERS = {"aie/__init__.py": b"x" * 1000, "aie/_mlir.so": b"\0" * 50000}


@pytest.fixture
def artifacts(tmp_path):
    """Three downloaded artifacts, as the workflow lays them out under wheels/."""
    wheels = tmp_path / "wheels"
    on = make_wheel(wheels / "mlir_aie_rtti_ON-3.12", LINUX, MEMBERS)
    off = make_wheel(wheels / "mlir_aie_rtti_OFF-3.12", LINUX, {"a.py": b"y" * 10})
    win = make_wheel(wheels / "mlir_aie_windows_rtti_ON-3.12", WINDOWS, MEMBERS)
    return wheels, {"on": on, "off": off, "win": win}


def test_series_come_from_filename_and_artifact_directory(sizes, artifacts):
    wheels, paths = artifacts
    found = sizes.collect([wheels])
    assert sorted(found) == [
        "manylinux_2_28_x86_64/rtti_OFF/cp312",
        "manylinux_2_28_x86_64/rtti_ON/cp312",
        "win_amd64/rtti_ON/cp312",
    ]
    on = found["manylinux_2_28_x86_64/rtti_ON/cp312"]
    assert on["bytes"] == paths["on"].stat().st_size
    assert on["files"] == 2
    assert on["uncompressed_bytes"] == 51000
    # The zip compresses the members, so the wheel is smaller than its content.
    assert on["bytes"] < on["uncompressed_bytes"]
    off = found["manylinux_2_28_x86_64/rtti_OFF/cp312"]
    assert (off["files"], off["uncompressed_bytes"]) == (1, 10)


def test_rows_carry_units_and_aggregates(sizes, artifacts):
    wheels, paths = artifacts
    rows = sizes.rows_of(sizes.collect([wheels]))
    by_name = {row["name"]: row for row in rows}
    assert len(by_name) == len(rows) == 3 * 3 + 2
    assert by_name["win_amd64/rtti_ON/cp312/bytes"] == {
        "name": "win_amd64/rtti_ON/cp312/bytes",
        "unit": "bytes",
        "value": paths["win"].stat().st_size,
    }
    assert by_name["win_amd64/rtti_ON/cp312/files"]["unit"] == "files"
    assert by_name["win_amd64/rtti_ON/cp312/uncompressed_bytes"]["unit"] == "bytes"
    assert by_name["all/wheels"] == {"name": "all/wheels", "unit": "wheels", "value": 3}
    assert by_name["all/bytes"]["value"] == sum(
        p.stat().st_size for p in paths.values()
    )
    # Every name splits into a series and a metric, as publish.py groups them.
    assert all(row["name"].count("/") >= 1 for row in rows)


def test_rtti_falls_back_to_unknown_and_nearest_directory_wins(sizes, tmp_path):
    make_wheel(tmp_path / "plain", LINUX, MEMBERS)
    make_wheel(tmp_path / "rtti_OFF-outer" / "mlir_aie_rtti_ON-3.12", WINDOWS, MEMBERS)
    found = sizes.collect([tmp_path])
    assert sorted(found) == [
        "manylinux_2_28_x86_64/rtti_unknown/cp312",
        "win_amd64/rtti_ON/cp312",
    ]
    # An artifact directory given directly still names its RTTI setting.
    artifact = tmp_path / "mlir_aie_rtti_OFF-3.13"
    make_wheel(artifact, LINUX.replace("cp312", "cp313"), MEMBERS)
    assert list(sizes.collect([artifact])) == ["manylinux_2_28_x86_64/rtti_OFF/cp313"]


def test_duplicate_series_keeps_the_larger_and_warns(sizes, tmp_path, capsys):
    directory = tmp_path / "mlir_aie_rtti_ON-3.12"
    small = make_wheel(directory, LINUX, {"a.py": b"z" * 10})
    big = make_wheel(directory, LINUX.replace("1.0.0", "1.0.1"), MEMBERS)
    found = sizes.collect([tmp_path])
    out = capsys.readouterr().out
    assert list(found) == ["manylinux_2_28_x86_64/rtti_ON/cp312"]
    assert found["manylinux_2_28_x86_64/rtti_ON/cp312"]["path"] == str(big)
    assert "::warning::" in out and "manylinux_2_28_x86_64/rtti_ON/cp312" in out
    assert str(small) in out and str(big) in out
    # Order does not matter: the same wheel wins when found second.
    (tmp_path / "other").mkdir()
    keep = make_wheel(tmp_path / "other/mlir_aie_rtti_ON-3.12", LINUX, MEMBERS)
    small.rename(directory / LINUX.replace("1.0.0", "1.0.2"))
    big.unlink()
    found = sizes.collect([directory.parent / "other", directory])
    assert found["manylinux_2_28_x86_64/rtti_ON/cp312"]["path"] == str(keep)


def test_non_wheel_filenames_are_skipped_with_a_warning(sizes, tmp_path, capsys):
    make_wheel(tmp_path / "mlir_aie_rtti_ON-3.12", "odd-name.whl", MEMBERS)
    make_wheel(tmp_path / "mlir_aie_rtti_ON-3.12", LINUX, MEMBERS)
    assert list(sizes.collect([tmp_path])) == ["manylinux_2_28_x86_64/rtti_ON/cp312"]
    assert "::warning::" in capsys.readouterr().out
    assert sizes.wheel_tags("mlir_aie-1.0-1-cp311-cp311-win_amd64.whl") == (
        "cp311",
        "win_amd64",
    )
    assert sizes.wheel_tags("mlir_aie-1.0-cp311-cp311-win_amd64.zip") is None


def test_command_line_writes_rows_and_prints_one_line_per_wheel(artifacts, tmp_path):
    wheels, paths = artifacts
    out = tmp_path / "rows.json"
    result = subprocess.run(
        [sys.executable, SCRIPT, wheels, "--out", out],
        capture_output=True,
        text=True,
        check=True,
    )
    lines = result.stdout.splitlines()
    assert f"win_amd64/rtti_ON/cp312: {paths['win'].stat().st_size} bytes" in lines
    assert lines[-1].startswith("total: 3 wheels, ")
    rows = json.loads(out.read_text())
    assert {row["name"] for row in rows} >= {"all/bytes", "all/wheels"}
    assert all(set(row) == {"name", "unit", "value"} for row in rows)


def test_record_sizes_job_writes_gh_pages_only_from_the_nightly():
    # Avoid YAML 1.1 interpreting GitHub's "on" key as a boolean.
    jobs = yaml.load(WORKFLOW.read_text(), Loader=yaml.BaseLoader)["jobs"]
    job = jobs["record-sizes"]
    assert job["if"].strip() == "github.event_name == 'schedule'"
    assert set(job["needs"]) == {"build-repo", "build-windows"}
    assert job["permissions"] == {"contents": "write"}
    assert job["concurrency"]["group"] == "gh-pages-publish"
    assert job["concurrency"]["cancel-in-progress"] == "false"
    uses = [step.get("uses", "") for step in job["steps"]]
    assert "./.github/actions/squash-gh-pages" in uses
    download = next(
        step
        for step in job["steps"]
        if step.get("uses", "").startswith("actions/download-artifact@")
    )
    assert download["with"]["pattern"].startswith("mlir_aie")
    assert "merge-multiple" not in download["with"]
    runs = "\n".join(step.get("run", "") for step in job["steps"])
    assert "utils/dashboard/wheel_sizes.py" in runs
    assert "publish.py record --target wheels" in runs
    assert "HEAD:gh-pages" in runs
    # The other gh-pages writer names the same lock, so the two queue.
    other = yaml.load(
        (ROOT / ".github/workflows/publishKernelResults.yml").read_text(),
        Loader=yaml.BaseLoader,
    )
    assert other["jobs"]["publish"]["concurrency"]["group"] == "gh-pages-publish"
