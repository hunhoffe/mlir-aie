# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# RUN: %pytest %s

"""Check the sanitizer nightly's shape and run its shell steps on synthetic inputs."""

import os
import re
import subprocess
from pathlib import Path

import pytest
import yaml

WORKFLOW = (
    Path(__file__).resolve().parents[1] / ".github/workflows/nightlySanitizers.yml"
)


@pytest.fixture(scope="module")
def workflow():
    # Avoid YAML 1.1 interpreting GitHub's "on" key as a boolean.
    return yaml.load(WORKFLOW.read_text(), Loader=yaml.BaseLoader)


@pytest.fixture(scope="module")
def steps(workflow):
    (job,) = workflow["jobs"].values()
    return job["steps"]


def step(steps, id):
    return next(step for step in steps if step.get("id") == id)


def run_step(run, cwd, env=None, stubs=""):
    # GitHub runs `bash --noprofile --norc -eo pipefail` for `shell: bash`.
    return subprocess.run(
        ["bash", "-eo", "pipefail", "-c", stubs + run],
        cwd=cwd,
        env={**os.environ, **(env or {})},
        capture_output=True,
        text=True,
    )


def test_is_a_scheduled_single_job_nightly(workflow):
    assert workflow["permissions"] == {"contents": "read"}
    assert "workflow_dispatch" in workflow["on"]
    assert workflow["on"]["schedule"][0]["cron"]
    assert len(workflow["jobs"]) == 1
    (job,) = workflow["jobs"].values()
    assert job["runs-on"] == "ubuntu-latest"
    assert int(job["timeout-minutes"]) >= 240


def test_every_action_is_pinned_to_a_commit(steps):
    uses = [step["uses"] for step in steps if "uses" in step]
    assert uses
    for ref in uses:
        assert re.fullmatch(r"[\w.-]+/[\w.-]+(/[\w./-]+)?@[0-9a-f]{40}", ref), ref


def test_configure_instruments_compile_and_link_without_coverage(steps):
    run = step(steps, "configure")["run"]
    for var in ["CMAKE_C_FLAGS", "CMAKE_CXX_FLAGS"]:
        assert re.search(rf'-D{var}="[^"]*-fsanitize=address,undefined', run), var
    for var in ["EXE", "SHARED", "MODULE"]:
        flags = re.search(rf'-DCMAKE_{var}_LINKER_FLAGS_INIT="([^"]*)"', run)
        assert flags, var
        assert "-fsanitize=address,undefined" in flags.group(1)
        assert "-fuse-ld=lld" in flags.group(1)
        # Executables and the preloaded runtime must be the same shared one,
        # and the tools must start without the preload.
        assert "-shared-libsan" in flags.group(1)
        assert "-Wl,-rpath,$rt_dir" in flags.group(1)
    assert "BUILD_INSTRUMENTED_COVERAGE" not in run
    assert "-DCMAKE_BUILD_TYPE=RelWithDebInfo" in run
    assert "-DLLVM_ENABLE_ASSERTIONS=ON" in run
    assert "-DMLIR_DIR=$PWD/../mlir/lib/cmake/mlir" in run


def test_check_step_runs_under_the_sanitizer_runtime(steps):
    check = step(steps, "check")
    env = check["env"]
    assert env["LD_PRELOAD"] == "${{ env.ASAN_RUNTIME }}"
    asan = env["ASAN_OPTIONS"].split(":")
    assert "detect_leaks=0" in asan and "halt_on_error=1" in asan
    assert "halt_on_error=1" in env["UBSAN_OPTIONS"].split(":")
    assert "--timeout 900" in env["LIT_OPTS"]
    # check-aie's own filter is replaced, not extended, by LIT_OPTS.
    assert re.search(r"--filter-out \S*python-concurrency", env["LIT_OPTS"])
    assert "|| status=$?" in check["run"]
    assert check["run"].rstrip().endswith("exit $status")
    # The locate step feeds both the configure rpath and the check preload.
    ids = [step.get("id") for step in steps]
    assert ids.index("runtime") < ids.index("configure") < ids.index("check")
    assert "ASAN_RUNTIME=" in step(steps, "runtime")["run"]
    assert "ASAN_SYMBOLIZER_PATH=" in step(steps, "runtime")["run"]


def test_log_is_summarised_and_uploaded_even_on_failure(steps):
    assert step(steps, "summary")["if"] == "always()"
    upload = next(
        step
        for step in steps
        if step.get("uses", "").startswith("actions/upload-artifact@")
    )
    assert upload["if"] == "always()"
    assert upload["with"]["path"] == "sanitizers.log"
    assert "${{ github.run_id }}" in upload["with"]["name"]


def run_configure(steps, tmp_path, env):
    return run_step(
        step(steps, "configure")["run"],
        tmp_path,
        env=env,
        stubs='cmake() { printf "%s\\n" "$@" > cmake-args; }\n',
    )


def test_configure_expands_the_runtime_dir_into_every_link_line(steps, tmp_path):
    result = run_configure(
        steps, tmp_path, {"ASAN_RUNTIME": "/opt/rt/lib/libclang_rt.asan-x86_64.so"}
    )
    assert result.returncode == 0, result.stdout + result.stderr
    args = (tmp_path / "build_release/cmake-args").read_text().splitlines()
    for var in ["EXE", "SHARED", "MODULE"]:
        assert (
            f"-DCMAKE_{var}_LINKER_FLAGS_INIT="
            + (
                "-fuse-ld=lld -fsanitize=address,undefined -shared-libsan "
                "-Wl,-rpath,/opt/rt/lib"
            )
            in args
        ), var
    assert not any("BUILD_INSTRUMENTED_COVERAGE" in arg for arg in args)
    assert "-DCMAKE_BUILD_TYPE=RelWithDebInfo" in args


def test_configure_refuses_to_run_without_a_located_runtime(steps, tmp_path):
    result = run_configure(steps, tmp_path, {})
    assert result.returncode != 0
    assert not (tmp_path / "build_release/cmake-args").exists()


def test_check_step_keeps_the_log_and_ninja_exit_status(steps, tmp_path):
    result = run_step(
        step(steps, "check")["run"],
        tmp_path,
        stubs='ninja() { echo "lit output for $*"; return 3; }\n',
    )
    assert result.returncode == 3
    assert (tmp_path / "sanitizers.log").read_text() == (
        "lit output for -C build_release check-aie\n"
    )


def run_runtime_step(steps, tmp_path, present):
    # clang -print-file-name echoes the bare name when the file is missing.
    if present:
        (tmp_path / present).write_text("")
    (tmp_path / "bin").mkdir()
    symbolizer = tmp_path / "bin/llvm-symbolizer"
    symbolizer.write_text("#!/bin/sh\n")
    symbolizer.chmod(0o755)
    github_env = tmp_path / "github_env"
    github_env.write_text("")
    stubs = (
        'clang() { f="$PWD/${1#-print-file-name=}"; '
        '[ -f "$f" ] && echo "$f" || echo "${1#-print-file-name=}"; }\n'
    )
    result = run_step(
        step(steps, "runtime")["run"],
        tmp_path,
        env={
            "GITHUB_ENV": str(github_env),
            "PATH": f"{tmp_path / 'bin'}{os.pathsep}{os.environ['PATH']}",
        },
        stubs=stubs,
    )
    return result, dict(
        line.split("=", 1) for line in github_env.read_text().splitlines()
    )


@pytest.mark.parametrize("name", ["libclang_rt.asan-x86_64.so", "libclang_rt.asan.so"])
def test_runtime_step_exports_whichever_layout_clang_has(steps, tmp_path, name):
    result, env = run_runtime_step(steps, tmp_path, name)
    assert result.returncode == 0, result.stdout + result.stderr
    assert env["ASAN_RUNTIME"] == str(tmp_path / name)
    assert env["ASAN_SYMBOLIZER_PATH"] == str(tmp_path / "bin/llvm-symbolizer")


def test_runtime_step_fails_loudly_without_a_runtime(steps, tmp_path):
    result, env = run_runtime_step(steps, tmp_path, None)
    assert result.returncode != 0
    assert "::error::" in result.stdout
    assert env == {}


SYNTHETIC_LOG = """\
-- Testing: 3 tests, 3 workers --
PASS: AIE_TEST :: dialect/ok.mlir (1 of 3)
FAIL: AIE_TEST :: dialect/bad.mlir (2 of 3)
==123==ERROR: AddressSanitizer: heap-use-after-free on address 0x1
==124==ERROR: AddressSanitizer: heap-use-after-free on address 0x1
lib/Foo.cpp:10:5: runtime error: signed integer overflow
TIMEOUT: AIE_TEST :: python/slow.py (3 of 3)
********************
Failed Tests (1):
  AIE_TEST :: dialect/bad.mlir
"""


def run_summary(steps, tmp_path, log):
    if log is not None:
        (tmp_path / "sanitizers.log").write_text(log)
    summary = tmp_path / "summary.md"
    result = run_step(
        step(steps, "summary")["run"],
        tmp_path,
        env={"GITHUB_STEP_SUMMARY": str(summary)},
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return summary.read_text()


def test_summary_counts_reports_and_lists_failing_tests(steps, tmp_path):
    text = run_summary(steps, tmp_path, SYNTHETIC_LOG)
    assert "| ASan reports (`ERROR: AddressSanitizer`) | 2 |" in text
    assert "| UBSan reports (`runtime error:`) | 1 |" in text
    assert "| Failed or timed-out lit tests | 2 |" in text
    failing = text[text.index("### Failing tests") : text.index("### Log tail")]
    assert "FAIL: AIE_TEST :: dialect/bad.mlir" in failing
    assert "TIMEOUT: AIE_TEST :: python/slow.py" in failing
    assert "PASS:" not in failing
    assert text.rstrip().endswith("AIE_TEST :: dialect/bad.mlir\n```")


def test_summary_of_a_clean_run_has_zero_counts_and_no_failure_list(steps, tmp_path):
    text = run_summary(steps, tmp_path, "PASS: AIE_TEST :: ok.mlir (1 of 1)\n")
    assert "| ASan reports (`ERROR: AddressSanitizer`) | 0 |" in text
    assert "| Failed or timed-out lit tests | 0 |" in text
    assert "### Failing tests" not in text


def test_summary_explains_a_missing_log(steps, tmp_path):
    text = run_summary(steps, tmp_path, None)
    assert "No lit log" in text
