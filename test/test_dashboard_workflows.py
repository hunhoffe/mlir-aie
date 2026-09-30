# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# RUN: %pytest %s

"""The invariants every gh-pages writer keeps: one lock, a squash after the push, pinned actions."""

import re
from pathlib import Path

import yaml

WORKFLOWS = Path(__file__).resolve().parents[1] / ".github" / "workflows"
LOCK = "gh-pages-publish"
SQUASH = "./.github/actions/squash-gh-pages"
NEW = [
    "nightlyDashboard.yml",
    "nightlySanitizers.yml",
    "squashGhPages.yml",
]


def workflow(name):
    # Avoid YAML 1.1 interpreting GitHub's "on" key as a boolean.
    return yaml.load((WORKFLOWS / name).read_text(), Loader=yaml.BaseLoader)


def steps_of(job):
    return job.get("steps", [])


def pushes_gh_pages(job):
    return any(
        "gh-pages" in step.get("run", "") and "push" in step.get("run", "")
        for step in steps_of(job)
    )


def writers():
    """Every (workflow, job name, job) whose steps push gh-pages, or that mike deploys from."""
    out = []
    for path in sorted(WORKFLOWS.glob("*.yml")):
        data = workflow(path.name)
        for name, job in (data.get("jobs") or {}).items():
            if pushes_gh_pages(job) or any(
                "mike deploy" in step.get("run", "") for step in steps_of(job)
            ):
                out.append((path.name, name, job, data))
    return out


def test_every_gh_pages_writer_holds_the_lock_and_squashes_after_its_push():
    found = writers()
    assert {w[0] for w in found} >= {
        "generateDocs.yml",
        "publishKernelResults.yml",
        "nightlyDashboard.yml",
        "codeCoverage.yml",
        "buildRyzenWheels.yml",
    }
    for file, name, job, data in found:
        group = (job.get("concurrency") or data.get("concurrency") or {}).get("group")
        assert group == LOCK, f"{file}:{name} does not take the {LOCK} lock"
        steps = steps_of(job)
        squash = [i for i, s in enumerate(steps) if s.get("uses") == SQUASH]
        assert squash, f"{file}:{name} pushes gh-pages without squashing it"
        pushes = [
            i
            for i, s in enumerate(steps)
            if "gh-pages" in s.get("run", "") and "push" in s.get("run", "")
        ]
        assert (
            not pushes or max(pushes) < squash[-1]
        ), f"{file}:{name} squashes before its last push"
        assert (job.get("permissions") or data.get("permissions") or {}).get(
            "contents"
        ) == "write"


def test_docs_deploy_keeps_the_dashboard_and_kernel_checks():
    job = workflow("generateDocs.yml")["jobs"]["build-docs"]
    prune = next(s["run"] for s in steps_of(job) if "git rm -rf" in s.get("run", ""))
    for kept in ("kernel-checks/", "dashboard/"):
        assert f"grep -zv '^{kept}'" in prune, f"the docs deploy would delete {kept}"


def test_the_standalone_squash_runs_weekly_and_on_demand():
    data = workflow("squashGhPages.yml")
    assert "workflow_dispatch" in data["on"]
    assert data["on"]["schedule"]
    assert data["concurrency"]["group"] == LOCK
    assert data["jobs"]["squash"]["steps"][-1]["uses"] == SQUASH


def test_new_workflows_pin_every_action_to_a_commit():
    pinned = re.compile(r"^(\./|[\w.-]+/[\w.-]+@[0-9a-f]{40}$)")
    for name in NEW:
        for job in workflow(name)["jobs"].values():
            for step in steps_of(job):
                if "uses" in step:
                    assert pinned.match(
                        step["uses"]
                    ), f"{name}: {step['uses']} is not pinned"


def test_the_dashboard_lists_only_scheduled_workflows_it_can_find():
    """The collector discovers schedules from the tree; the ones the README names must be there."""
    for name in ("nightlyDashboard.yml", "nightlySanitizers.yml", "codeCoverage.yml"):
        assert "schedule" in workflow(name)["on"], f"{name} is not scheduled"
