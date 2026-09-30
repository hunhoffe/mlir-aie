<!---//===- README.md --------------------------*- Markdown -*-===//
//
// Copyright (C) 2026 Advanced Micro Devices, Inc.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//-->

# Maintainer dashboard

A page for the people who keep CI green, published beside the docs at
`https://xilinx.github.io/mlir-aie/dashboard/`. It shows, on one page, what
every nightly did last night and how the numbers that creep are trending:
kernel timings and sizes (from the kernel checks), test coverage, wheel
sizes, and the age of the toolchain pins. Everything is static files on the
`gh-pages` branch, written by the workflows below and read by
`index.html`; there is no server and no external service.

## What writes what

| Workflow | Writes on `gh-pages` | From |
| --- | --- | --- |
| `nightlyDashboard.yml` (daily, 09:23 UTC) | `dashboard/status/latest.json`, `dashboard/status/history.json`, `dashboard/index.html` | `collect.py`: the Actions API, for every workflow with a `schedule` trigger, and the pin files |
| `codeCoverage.yml` (nightly, main) | `dashboard/metrics/coverage/`, `dashboard/coverage/report/`, `dashboard/coverage/summary.json` | `coverage_summary.py` over the lcov the coverage build already writes |
| `buildRyzenWheels.yml` (nightly) | `dashboard/metrics/wheels/` | `wheel_sizes.py` over the wheel artifacts |
| `nightlyKernelChecks.yml` via `publishKernelResults.yml` | `kernel-checks/<npu>/` | `../kernel_checks/publish.py` (unchanged format; the dashboard summarizes and links it) |

`dashboard/metrics/<target>/` uses the kernel checks' record format, written
by `publish.py record`, which loads `../kernel_checks/publish.py` for its
`rebuild`: `runs/<id>.json`, `runs.json`, `latest.json`, and one
`history/<metric>.json` per metric. Retention is the same everywhere: every
run of the last 90 days, one per ISO week before that, at most 400 runs,
and runs of a release tag (`--tag`) for good.

New scheduled workflows appear on the dashboard by themselves;
`nightlySanitizers.yml` was added with this dashboard and is one of them.

## Keeping `gh-pages` small

Nothing on the branch needs history, and the docs deploy regenerates tens of
MB of Doxygen HTML per push, which every clone of the repository then
downloads. Every writer therefore ends with `.github/actions/squash-gh-pages`
(`squash_gh_pages.sh`): the branch tip's tree is committed again with no
parent and force-pushed under a lease on the tip that was fetched, so the
content cannot change and a concurrent push fails rather than being
overwritten. `squashGhPages.yml` runs the same step on demand and weekly.
All writers share the `gh-pages-publish` concurrency group.

## Adding a series

Emit rows `[{"name": "<series>/<metric>", "unit": ..., "value": ...}]`,
add the target to `TARGETS` in `publish.py`, record it from a workflow job
that holds the `gh-pages-publish` lock and squashes afterwards (copy the
`record-sizes` job of `buildRyzenWheels.yml`), and read
`metrics/<target>/history/<metric>.json` in `index.html`.

## Tests

`test/test_dashboard_*.py`, `test/test_coverage_summary.py`,
`test/test_wheel_sizes.py`, `test/test_gh_pages_squash.py` and
`test/test_sanitizer_workflow.py` run under `pytest` with no hardware and
no network: the page under node with a minimal DOM, the collector against a
fake API, the squash against a local bare repository, and the workflow
YAML for the invariants above.
