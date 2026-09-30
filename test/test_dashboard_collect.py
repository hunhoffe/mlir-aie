# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# RUN: %pytest %s

"""Collect the nightlies' status for the dashboard against a fake GitHub API."""

import datetime
import importlib.util
import json
import urllib.error
from pathlib import Path

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "utils/dashboard/collect.py"
WORKFLOWS = ROOT / ".github/workflows"
REPO = "example/synthetic"
NOW = datetime.datetime(2026, 10, 1, 9, 23, tzinfo=datetime.timezone.utc)


@pytest.fixture(scope="module")
def collect():
    spec = importlib.util.spec_from_file_location("dashboard_collect", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# ------------------------------------------------------------ fake GitHub


def run(id, conclusion, created, started, updated, sha, status="completed", **more):
    return {
        "id": id,
        "html_url": f"https://github.com/{REPO}/actions/runs/{id}",
        "event": "schedule",
        "status": status,
        "conclusion": conclusion,
        "head_sha": sha,
        "created_at": created,
        "run_started_at": started,
        "updated_at": updated,
        "run_attempt": 1,
        **more,
    }


ALPHA_RUNS = [
    # Newest first as the API lists them, one still running.
    run(
        105,
        None,
        "2026-10-01T00:00:10Z",
        "2026-10-01T00:00:12Z",
        "2026-10-01T00:00:12Z",
        "e" * 40,
        status="in_progress",
    ),
    run(
        104,
        "failure",
        "2026-09-30T00:00:10Z",
        "2026-09-30T00:05:10Z",
        "2026-09-30T01:05:10Z",
        "d" * 40,
        run_attempt=2,
    ),
    run(
        103,
        "success",
        "2026-09-29T00:00:10Z",
        "2026-09-29T00:00:20Z",
        "2026-09-29T00:50:20Z",
        "c" * 40,
    ),
] + [
    run(
        100 - i,
        "success",
        f"2026-09-{28 - i:02d}T00:00:10Z",
        f"2026-09-{28 - i:02d}T00:00:20Z",
        f"2026-09-{28 - i:02d}T00:40:20Z",
        "b" * 40,
    )
    for i in range(16)
]
ALPHA_JOBS = {
    "jobs": [
        {
            "name": "build (ubuntu)",
            "conclusion": "success",
            "runner_name": "GitHub Actions 12",
            "labels": ["ubuntu-latest"],
            "created_at": "2026-09-30T00:00:11Z",
            "started_at": "2026-09-30T00:05:10Z",
            "completed_at": "2026-09-30T00:35:10Z",
            "html_url": f"https://github.com/{REPO}/actions/runs/104/job/1",
        },
        {
            "name": "test (npu1)",
            "conclusion": "failure",
            "runner_name": "bench-3",
            "labels": ["self-hosted", "bench"],
            # Clock skew: started before created, so the wait reads 0.
            "created_at": "2026-09-30T00:35:12Z",
            "started_at": "2026-09-30T00:35:11Z",
            "completed_at": "2026-09-30T01:05:10Z",
            "html_url": f"https://github.com/{REPO}/actions/runs/104/job/2",
        },
    ]
}
GAMMA_MAIN_RUNS = [
    {
        **run(
            7,
            "success",
            "2026-09-30T12:00:00Z",
            "2026-09-30T12:00:30Z",
            "2026-09-30T12:10:30Z",
            "f" * 40,
        ),
        "event": "workflow_dispatch",
    }
]
BUMP_PR = {
    "number": 4242,
    "html_url": f"https://github.com/{REPO}/pull/4242",
    "title": "Update Peano version to 22.0.0.2026093001",
    "created_at": "2026-09-28T05:53:00Z",
    "head": {"sha": "1" * 40},
}
CHECK_RUNS = {
    "check_runs": [
        {"status": "completed", "conclusion": "success"},
        {"status": "completed", "conclusion": "success"},
        {"status": "completed", "conclusion": "failure"},
        {"status": "completed", "conclusion": "timed_out"},
        {"status": "completed", "conclusion": "skipped"},
        {"status": "in_progress", "conclusion": None},
        {"status": "queued", "conclusion": None},
    ]
}
BUMP_RUNS = [
    run(
        9001,
        None,
        "2026-10-01T05:53:00Z",
        "2026-10-01T05:53:10Z",
        "2026-10-01T05:53:10Z",
        "0" * 40,
        status="queued",
    ),
    run(
        9000,
        "success",
        "2026-09-28T05:53:00Z",
        "2026-09-28T05:53:10Z",
        "2026-09-28T06:03:10Z",
        "0" * 40,
    ),
]

# The API, keyed by a substring of the URL; first match wins, so the more
# specific keys come first. A value that is an exception is raised.
API = {
    "workflows/alpha.yml/runs?event=schedule": {"workflow_runs": ALPHA_RUNS},
    "actions/runs/104/jobs": ALPHA_JOBS,
    f"compare/{'c' * 40}...{'d' * 40}": {
        "html_url": f"https://github.com/{REPO}/compare/ccc...ddd",
        "ahead_by": 3,
    },
    "workflows/beta.yml/runs?event=schedule": urllib.error.HTTPError(
        "https://api.github.com/x", 404, "Not Found", {}, None
    ),
    "workflows/gamma.yml/runs?event=schedule": {"workflow_runs": []},
    "workflows/gamma.yml/runs?branch=main": {"workflow_runs": GAMMA_MAIN_RUNS},
    "actions/runs/7/jobs": {"jobs": []},
    "pulls?state=open&head=example%3Aupdate-peano-version": [BUMP_PR],
    "pulls?state=open&head=example%3Aupdate-llvm-version": [],
    f"commits/{'1' * 40}/check-runs": CHECK_RUNS,
    "workflows/update-peano.yml/runs?per_page": {"workflow_runs": BUMP_RUNS},
    "workflows/update-llvm.yml/runs?per_page": {"workflow_runs": []},
    # Any other run's jobs: none.
    "/jobs?per_page=100": {"jobs": []},
}


class FakeFetch:
    def __init__(self, api=API):
        self.api = api
        self.urls = []

    def __call__(self, url):
        self.urls.append(url)
        for needle, payload in self.api.items():
            if needle in url:
                if isinstance(payload, Exception):
                    raise payload
                return json.loads(json.dumps(payload))
        raise urllib.error.HTTPError(url, 404, "no fake for this URL", {}, None)


WORKFLOW_FILES = {
    "alpha.yml": "name: Alpha\non:\n  schedule:\n    # nightly\n    - cron: '0 0 * * *'\n  push:\n",
    "beta.yml": 'name: "Beta"\npermissions:\n  contents: read\non:\n  workflow_dispatch:\n  schedule:\n    - cron: "0 4 * * *"\n    - cron: 0 16 * * *\njobs:\n  x:\n    runs-on: ubuntu-latest\n',
    "gamma.yml": "name: Gamma\non: [push, schedule]\n",
    "delta.yml": "name: Delta\non:\n  push:\n  workflow_dispatch:\n    inputs:\n      schedule:\n        description: not a trigger\n",
}


@pytest.fixture
def workflows(tmp_path):
    d = tmp_path / "workflows"
    d.mkdir()
    for name, text in WORKFLOW_FILES.items():
        (d / name).write_text(text)
    return d


@pytest.fixture
def collected(collect, workflows):
    fetch = FakeFetch()
    collector = collect.Collector(REPO, fetch=fetch, api="https://api.github.com")
    found = collect.discover_workflows(workflows, REPO, "https://github.com")
    latest = collect.collect(collector, found, root=ROOT, now=NOW, commit="abc123")
    return latest, fetch


# ---------------------------------------------------------------- discovery


def test_discovers_the_scheduled_workflows_of_the_repository(collect):
    found = {
        w["file"]: w
        for w in collect.discover_workflows(
            WORKFLOWS, "Xilinx/mlir-aie", "https://github.com"
        )
    }
    assert {
        "codeCoverage.yml",
        "nightlyKernelChecks.yml",
        "nightlyDashboard.yml",
    } <= set(found)
    assert found["nightlyKernelChecks.yml"]["schedule"] == ["0 6 * * *"]
    assert (
        found["nightlyKernelChecks.yml"]["name"] == "Nightly Kernel Checks on Ryzen AI"
    )
    assert found["nightlyDashboard.yml"]["schedule"] == ["23 9 * * *"]
    assert (
        found["codeCoverage.yml"]["url"]
        == "https://github.com/Xilinx/mlir-aie/actions/workflows/codeCoverage.yml"
    )
    assert "generateDocs.yml" not in found
    assert "publishKernelResults.yml" not in found
    names = [w["name"] for w in found.values()]
    assert names == sorted(names)
    # Every file the grep-level check calls scheduled, and only those.
    expected = {
        p.name
        for p in WORKFLOWS.glob("*.yml")
        if any(line.strip() == "schedule:" for line in p.read_text().splitlines())
    }
    assert set(found) == expected


def test_discovery_reads_names_crons_and_flow_style_triggers(collect, workflows):
    found = {
        w["file"]: w for w in collect.discover_workflows(workflows, REPO, "https://x")
    }
    assert set(found) == {"alpha.yml", "beta.yml", "gamma.yml"}
    assert found["alpha.yml"]["schedule"] == ["0 0 * * *"]
    assert found["beta.yml"] == {
        "file": "beta.yml",
        "name": "Beta",
        "schedule": ["0 4 * * *", "0 16 * * *"],
        "url": "https://x/example/synthetic/actions/workflows/beta.yml",
    }
    assert found["gamma.yml"]["schedule"] == []


# --------------------------------------------------------------- collection


def by_file(latest):
    return {w["file"]: w for w in latest["workflows"]}


def test_latest_run_and_its_jobs(collected):
    latest, fetch = collected
    alpha = by_file(latest)["alpha.yml"]
    assert "error" not in alpha and "event_fallback" not in alpha
    run = alpha["latest"]
    assert run["id"] == 104 and run["conclusion"] == "failure" and run["attempt"] == 2
    assert run["head_sha"] == "d" * 40
    assert run["url"] == f"https://github.com/{REPO}/actions/runs/104"
    assert run["started_at"] == "2026-09-30T00:05:10Z"
    assert run["completed_at"] == "2026-09-30T01:05:10Z"
    assert run["queue_s"] == 300 and run["duration_s"] == 3600
    build, test = run["jobs"]
    assert build["runner"] == "GitHub Actions 12" and build["labels"] == [
        "ubuntu-latest"
    ]
    assert build["queue_s"] == 299 and build["duration_s"] == 1800
    assert test == {
        "name": "test (npu1)",
        "conclusion": "failure",
        "runner": "bench-3",
        "labels": ["self-hosted", "bench"],
        "queue_s": 0,
        "duration_s": 1799,
        "url": f"https://github.com/{REPO}/actions/runs/104/job/2",
    }
    # Only the latest run's jobs are fetched.
    assert [u for u in fetch.urls if "/jobs" in u and "alpha" not in u].count(
        f"https://api.github.com/repos/{REPO}/actions/runs/104/jobs?per_page=100"
    ) == 1


def test_in_progress_last_success_since_green_and_recent(collected):
    latest, _ = collected
    alpha = by_file(latest)["alpha.yml"]
    assert alpha["in_progress"] == {
        "id": 105,
        "url": f"https://github.com/{REPO}/actions/runs/105",
        "created_at": "2026-10-01T00:00:10Z",
    }
    assert alpha["last_success"] == {
        "id": 103,
        "url": f"https://github.com/{REPO}/actions/runs/103",
        "head_sha": "c" * 40,
        "completed_at": "2026-09-29T00:50:20Z",
    }
    assert alpha["since_green"] == {
        "compare_url": f"https://github.com/{REPO}/compare/ccc...ddd",
        "commits": 3,
    }
    recent = alpha["recent"]
    assert len(recent) == 14
    assert [r["id"] for r in recent[:3]] == [104, 103, 100]
    assert recent[0] == {
        "id": 104,
        "url": f"https://github.com/{REPO}/actions/runs/104",
        "conclusion": "failure",
        "completed_at": "2026-09-30T01:05:10Z",
        "duration_s": 3600,
    }


def test_since_green_falls_back_to_a_compare_link(collect):
    api = {k: v for k, v in API.items() if not k.startswith("compare/")}
    collector = collect.Collector(
        REPO, fetch=FakeFetch(api), api="https://api.github.com"
    )
    status = collector.workflow_status("alpha.yml")
    assert status["since_green"] == {
        "compare_url": f"https://github.com/{REPO}/compare/{'c' * 40}...{'d' * 40}",
        "commits": None,
    }


def test_since_green_is_null_when_green_or_same_commit(collect):
    green = {
        **API,
        "workflows/alpha.yml/runs?event=schedule": {"workflow_runs": ALPHA_RUNS[2:]},
    }
    collector = collect.Collector(
        REPO, fetch=FakeFetch(green), api="https://api.github.com"
    )
    status = collector.workflow_status("alpha.yml")
    assert status["latest"]["conclusion"] == "success" and status["since_green"] is None
    assert status["in_progress"] is None
    same = {
        **API,
        "workflows/alpha.yml/runs?event=schedule": {
            "workflow_runs": [ALPHA_RUNS[1], {**ALPHA_RUNS[2], "head_sha": "d" * 40}]
        },
    }
    collector = collect.Collector(
        REPO, fetch=FakeFetch(same), api="https://api.github.com"
    )
    assert collector.workflow_status("alpha.yml")["since_green"] is None


def test_a_workflow_without_scheduled_runs_falls_back_to_main(collected):
    latest, _ = collected
    gamma = by_file(latest)["gamma.yml"]
    assert gamma["event_fallback"] is True
    assert (
        gamma["latest"]["id"] == 7 and gamma["latest"]["event"] == "workflow_dispatch"
    )
    assert gamma["latest"]["jobs"] == []
    assert gamma["last_success"]["id"] == 7 and gamma["since_green"] is None


def test_one_failing_workflow_is_recorded_and_the_rest_publish(capsys, collected):
    latest, _ = collected
    beta = by_file(latest)["beta.yml"]
    assert beta["error"].startswith("HTTP 404 Not Found")
    assert "latest" not in beta
    assert beta["schedule"] == ["0 4 * * *", "0 16 * * *"]
    assert by_file(latest)["alpha.yml"]["latest"]["id"] == 104
    assert "::warning::beta.yml: HTTP 404" in capsys.readouterr().out
    assert [w["name"] for w in latest["workflows"]] == ["Alpha", "Beta", "Gamma"]
    assert latest["schema"] == 1 and latest["repo"] == REPO
    assert (
        latest["date"] == "2026-10-01T09:23:00+00:00" and latest["commit"] == "abc123"
    )


def test_every_workflow_failing_exits_non_zero_without_writing(
    collect, workflows, tmp_path, monkeypatch, capsys
):
    def unauthorized(url):
        raise urllib.error.HTTPError(url, 401, "Bad credentials", {}, None)

    monkeypatch.setattr(collect, "github_fetch", unauthorized)
    out = tmp_path / "status"
    argv = ["--repo", REPO, "--workflows", str(workflows), "--out", str(out)]
    assert collect.main(argv) == 1
    assert not out.exists()
    assert "::error::all 3 workflows failed" in capsys.readouterr().out


# ------------------------------------------------------------- dependencies


def test_dependencies_parse_the_real_pins(collect, collected):
    latest, _ = collected
    peano, llvm = latest["dependencies"]["peano"], latest["dependencies"]["llvm"]
    pin = (ROOT / "utils/peano-requirements.txt").read_text()
    assert f"llvm-aie=={peano['pin']}" in pin
    assert peano["file"] == "utils/peano-requirements.txt"
    stamp = peano["pin"].split(".")[3]
    assert peano["date"] == f"{stamp[:4]}-{stamp[4:6]}-{stamp[6:8]}"
    assert peano["commit"] == peano["pin"].split("+")[1]
    assert (
        peano["age_days"]
        == (NOW.date() - datetime.date.fromisoformat(peano["date"])).days
    )
    clone = (ROOT / "utils/clone-llvm.sh").read_text()
    assert f"LLVM_PROJECT_COMMIT={llvm['commit']}\n" in clone
    assert llvm["pin"] == llvm["commit"][:8]
    assert f"DATETIME={llvm['date'].replace('-', '')}" in clone
    assert (
        llvm["age_days"]
        == (NOW.date() - datetime.date.fromisoformat(llvm["date"])).days
    )


def test_pin_parsers(collect):
    peano = collect.parse_peano(
        "-f https://x\nllvm-aie==22.0.0.2026092801+b0d37423\n", NOW
    )
    assert peano == {
        "pin": "22.0.0.2026092801+b0d37423",
        "commit": "b0d37423",
        "date": "2026-09-28",
        "age_days": 3,
    }
    llvm = collect.parse_llvm(
        "LLVM_PROJECT_COMMIT=e4fcd12811396e394eab2570f7f9bd25c6369811\nDATETIME=2026091506\n",
        NOW,
    )
    assert llvm == {
        "pin": "e4fcd128",
        "commit": "e4fcd12811396e394eab2570f7f9bd25c6369811",
        "date": "2026-09-15",
        "age_days": 16,
    }
    with pytest.raises(ValueError):
        collect.parse_peano("llvm-aie==1.0\n", NOW)
    with pytest.raises(ValueError):
        collect.parse_llvm("DATETIME=2026091506\n", NOW)


def test_bump_pr_with_check_counts_and_last_run(collected):
    latest, _ = collected
    bump = latest["dependencies"]["peano"]["bump"]
    assert bump["workflow"] == "update-peano.yml"
    assert (
        bump["url"] == f"https://github.com/{REPO}/actions/workflows/update-peano.yml"
    )
    assert bump["pr"] == {
        "number": 4242,
        "url": f"https://github.com/{REPO}/pull/4242",
        "title": "Update Peano version to 22.0.0.2026093001",
        "created_at": "2026-09-28T05:53:00Z",
        "age_days": 3,
        "checks": {"success": 2, "failure": 2, "pending": 2},
    }
    assert bump["last_run"] == {
        "id": 9000,
        "url": f"https://github.com/{REPO}/actions/runs/9000",
        "conclusion": "success",
        "completed_at": "2026-09-28T06:03:10Z",
    }
    llvm = latest["dependencies"]["llvm"]["bump"]
    assert llvm["pr"] is None and llvm["last_run"] is None and "error" not in llvm


def test_bump_api_failure_is_recorded(collect):
    api = {k: v for k, v in API.items() if "update-peano" not in k}
    collector = collect.Collector(
        REPO, fetch=FakeFetch(api), api="https://api.github.com"
    )
    bump = collector.bump("update-peano.yml", "update-peano-version", NOW)
    assert bump["pr"] is None and bump["last_run"] is None
    assert bump["error"].startswith("HTTP 404")


# ------------------------------------------------------------------ history


def latest_on(collect, day, conclusion="success"):
    return {
        "date": day,
        "workflows": [
            {
                "file": "alpha.yml",
                "latest": {
                    "id": 1,
                    "conclusion": conclusion,
                    "duration_s": 10,
                    "queue_s": 1,
                    "completed_at": day,
                    "jobs": [],
                    "head_sha": "x",
                },
            },
            {"file": "beta.yml", "error": "HTTP 404"},
        ],
    }


def test_history_appends_replaces_the_same_day_and_caps(collect):
    h = collect.update_history(None, latest_on(collect, "2026-09-30T09:23:00+00:00"))
    assert h["schema"] == 1
    assert h["days"] == [
        {
            "date": "2026-09-30T09:23:00+00:00",
            "workflows": {
                "alpha.yml": {
                    "conclusion": "success",
                    "id": 1,
                    "duration_s": 10,
                    "queue_s": 1,
                    "completed_at": "2026-09-30T09:23:00+00:00",
                }
            },
        }
    ]
    h = collect.update_history(h, latest_on(collect, "2026-10-01T09:23:00+00:00"))
    # A second collection the same day (a dispatch) replaces the day's entry.
    h = collect.update_history(
        h, latest_on(collect, "2026-10-01T15:00:00+00:00", "failure")
    )
    assert [d["date"] for d in h["days"]] == [
        "2026-09-30T09:23:00+00:00",
        "2026-10-01T15:00:00+00:00",
    ]
    assert h["days"][-1]["workflows"]["alpha.yml"]["conclusion"] == "failure"
    start = datetime.datetime(2025, 1, 1, tzinfo=datetime.timezone.utc)
    for i in range(450):
        h = collect.update_history(
            h, latest_on(collect, collect.iso(start + datetime.timedelta(days=i)))
        )
    assert len(h["days"]) == 400
    assert h["days"][-1]["date"] == "2026-10-01T15:00:00+00:00"
    assert h["days"][0]["date"] < h["days"][1]["date"]


def test_a_newer_history_schema_is_refused(collect, tmp_path):
    with pytest.raises(collect.NewerSchema):
        collect.update_history(
            {"schema": 2, "days": []}, latest_on(collect, "2026-10-01T09:23:00+00:00")
        )
    out = tmp_path / "status"
    out.mkdir()
    (out / "history.json").write_text('{"schema": 2, "days": []}')
    (out / "latest.json").write_text("old")
    with pytest.raises(collect.NewerSchema):
        collect.write(out, latest_on(collect, "2026-10-01T09:23:00+00:00"))
    assert (out / "latest.json").read_text() == "old"


def test_main_writes_both_files_and_a_summary(
    collect, workflows, tmp_path, monkeypatch, capsys
):
    monkeypatch.setattr(collect, "github_fetch", FakeFetch())
    out = tmp_path / "status"
    (out).mkdir()
    (out / "history.json").write_text(
        json.dumps(
            {
                "schema": 1,
                "days": [{"date": "2026-09-30T09:23:00+00:00", "workflows": {}}],
            }
        )
    )
    argv = [
        "--repo",
        REPO,
        "--workflows",
        str(workflows),
        "--out",
        str(out),
        "--now",
        "2026-10-01T09:23:00Z",
        "--commit",
        "abc",
        "--server",
        "https://github.com",
    ]
    assert collect.main(argv) == 0
    latest = json.loads((out / "latest.json").read_text())
    history = json.loads((out / "history.json").read_text())
    assert latest["commit"] == "abc" and len(latest["workflows"]) == 3
    assert [d["date"][:10] for d in history["days"]] == ["2026-09-30", "2026-10-01"]
    assert set(history["days"][1]["workflows"]) == {"alpha.yml", "gamma.yml"}
    out_text = capsys.readouterr().out
    assert "Alpha: failure (2026-09-30T01:05:10Z)" in out_text
    assert "Beta: error (HTTP 404" in out_text
    assert "::warning::beta.yml" in out_text


# ----------------------------------------------------------------- workflow


def test_workflow_publishes_under_the_gh_pages_lock():
    wf = yaml.load(
        (WORKFLOWS / "nightlyDashboard.yml").read_text(), Loader=yaml.BaseLoader
    )
    assert wf["name"] == "Nightly dashboard status"
    assert wf["on"]["schedule"] == [{"cron": "23 9 * * *"}]
    assert "workflow_dispatch" in wf["on"]
    assert wf["permissions"] == {"contents": "read"}
    (job,) = wf["jobs"].values()
    assert job["concurrency"] == {
        "group": "gh-pages-publish",
        "cancel-in-progress": "false",
        "queue": "max",
    }
    assert (
        job["permissions"]["contents"] == "write"
        and job["permissions"]["actions"] == "read"
    )
    steps = job["steps"]
    assert steps[-1]["uses"] == "./.github/actions/squash-gh-pages"
    runs = "\n".join(s.get("run", "") for s in steps)
    assert (
        "utils/dashboard/collect.py" in runs
        and "--out gh-pages-wt/dashboard/status" in runs
    )
    collect_step = next(s for s in steps if "collect.py" in s.get("run", ""))
    assert collect_step["env"]["GITHUB_TOKEN"] == "${{ github.token }}"
    assert (
        runs.index("git worktree add")
        < runs.index("collect.py")
        < runs.index("push origin HEAD:gh-pages")
    )
    upload = next(
        s for s in steps if s.get("uses", "").startswith("actions/upload-artifact@")
    )
    assert upload["with"]["name"] == "dashboard-status-${{ github.run_id }}"
