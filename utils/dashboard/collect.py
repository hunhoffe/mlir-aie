#!/usr/bin/env python3
# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Record what every scheduled workflow did last night, for the dashboard.

The maintainer dashboard shows a status matrix of the nightlies: freshness,
conclusion, duration, queue wait, the runner each job landed on, and how
many commits went in since the last green run, beside the age of the Peano
and LLVM pins and the state of their bump PRs. This collects all of that
once a day from the GitHub REST API into two files under
``dashboard/status/`` on the publication branch:

    latest.json    every tracked workflow's newest runs and jobs, and the
                   dependency pins, as of the collection
    history.json   one entry per day with each workflow's conclusion,
                   duration and queue wait, for the matrix's past columns

Tracked workflows are found in the checkout, not listed here: every file in
``.github/workflows/`` whose ``on:`` has a ``schedule:`` is one, so a new
nightly appears on the dashboard by itself. Standard library only: the
workflow runs it on a bare runner.

    collect.py --repo Xilinx/mlir-aie --workflows .github/workflows --out status
        [--server https://github.com] [--now ISO] [--commit SHA]

``GITHUB_TOKEN`` authenticates the API calls (``actions: read`` for the runs,
``pull-requests: read`` and ``checks: read`` for the bump PRs). One
workflow's API failure is recorded on its entry and the rest still publish;
only when every workflow failed (a bad token, an outage) does this exit
non-zero, leaving the previous files alone.
"""

import argparse
import datetime
import json
import os
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

# The format of both files. A reader that knows an older one refuses the
# file rather than misreading it; bump it with any change a reader of the
# old format would get wrong.
SCHEMA = 1
ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BRANCH = "main"
# Runs listed per workflow; `recent` keeps the newest RECENT completed ones.
PER_PAGE = 20
RECENT = 14
# Days of history kept in history.json, newest last.
HISTORY_DAYS = 400
# Check-run conclusions the bump PR's counters call a failure.
FAILED_CHECKS = (
    "failure",
    "timed_out",
    "cancelled",
    "action_required",
    "startup_failure",
)
# The pins the dashboard ages, and the workflow and PR branch that bump each.
DEPENDENCIES = {
    "peano": {
        "file": "utils/peano-requirements.txt",
        "workflow": "update-peano.yml",
        "branch": "update-peano-version",
    },
    "llvm": {
        "file": "utils/clone-llvm.sh",
        "workflow": "update-llvm.yml",
        "branch": "update-llvm-version",
    },
}


class NewerSchema(Exception):
    """A file on the branch was written by a newer collect.py."""


class CollectionFailed(Exception):
    """No workflow could be collected; nothing was written."""


def now_utc() -> datetime.datetime:
    return datetime.datetime.now(datetime.timezone.utc)


def iso(dt: datetime.datetime) -> str:
    return dt.astimezone(datetime.timezone.utc).isoformat(timespec="seconds")


def parse_date(text: str) -> datetime.datetime:
    dt = datetime.datetime.fromisoformat(text.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=datetime.timezone.utc)
    return dt


def seconds(start, end, floor: bool = True):
    """Whole seconds from ``start`` to ``end`` (ISO strings), None when either is missing.

    GitHub's ``run_started_at`` can precede ``created_at`` by a moment; the
    clamp keeps a queue wait from reading negative.
    """
    if not start or not end:
        return None
    delta = int((parse_date(end) - parse_date(start)).total_seconds())
    return max(delta, 0) if floor else delta


# ---------------------------------------------------------------- workflows

_TOP_KEY = re.compile(r"^(\"?on\"?|name)\s*:\s*(.*?)\s*$")


def _unquote(value: str) -> str:
    value = value.split(" #", 1)[0].strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
        return value[1:-1]
    return value


def scheduled_triggers(text: str) -> dict:
    """Return ``{"name": ..., "schedule": [cron, ...]}`` of a workflow, or None.

    A minimal read of the two top-level keys the dashboard needs, so the
    runner needs no YAML library: ``name:`` and, under ``on:``, the ``cron``
    entries of ``schedule:``. Comments and blank lines inside the block are
    skipped; ``on: [push, schedule]`` counts as scheduled without a cron.
    """
    name = None
    crons = None
    in_on = False
    # The indent of on:'s direct children (the triggers), fixed by the first
    # one: a `schedule:` deeper down is an input or a step, not a trigger.
    trigger_indent = None
    schedule_indent = None
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        indent = len(line) - len(line.lstrip())
        if indent == 0:
            in_on = False
            trigger_indent = schedule_indent = None
            top = _TOP_KEY.match(line)
            if not top:
                continue
            key, value = top.group(1).strip('"'), top.group(2)
            if key == "name":
                name = _unquote(value)
            elif value.startswith("["):
                if "schedule" in re.split(r"[\[\],\s]+", value):
                    crons = crons or []
            else:
                in_on = True
            continue
        if not in_on:
            continue
        if schedule_indent is not None and indent > schedule_indent:
            cron = re.match(r"^-\s*cron\s*:\s*(.+)$", stripped)
            if cron:
                crons.append(_unquote(cron.group(1)))
            continue
        schedule_indent = None
        trigger_indent = indent if trigger_indent is None else trigger_indent
        if indent == trigger_indent and re.match(r"^schedule\s*:\s*(#.*)?$", stripped):
            schedule_indent = indent
            crons = crons or []
    if crons is None:
        return None
    return {"name": name, "schedule": crons}


def discover_workflows(workflows: Path, repo: str, server: str) -> list[dict]:
    """Every workflow under ``workflows`` with a schedule, sorted by name."""
    found = []
    for path in sorted(workflows.glob("*.yml")) + sorted(workflows.glob("*.yaml")):
        triggers = scheduled_triggers(path.read_text())
        if triggers is None:
            continue
        found.append(
            {
                "file": path.name,
                "name": triggers["name"] or path.stem,
                "schedule": triggers["schedule"],
                "url": f"{server}/{repo}/actions/workflows/{path.name}",
            }
        )
    return sorted(found, key=lambda w: (w["name"], w["file"]))


# ------------------------------------------------------------------- GitHub


def github_fetch(url: str) -> dict | list:
    """GET one API URL as JSON, authenticated with ``GITHUB_TOKEN`` when set."""
    headers = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
        "User-Agent": "mlir-aie-dashboard",
    }
    token = os.environ.get("GITHUB_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"
    with urllib.request.urlopen(
        urllib.request.Request(url, headers=headers), timeout=60
    ) as r:
        return json.loads(r.read().decode())


def _describe(exc: Exception) -> str:
    if isinstance(exc, urllib.error.HTTPError):
        return f"HTTP {exc.code} {exc.reason} for {exc.url}"
    return f"{type(exc).__name__}: {exc}"


# Everything one API call or one malformed payload can raise; nothing else
# is caught, so a bug in this file still fails the run loudly.
API_ERRORS = (OSError, ValueError, KeyError, TypeError, IndexError)


class Collector:
    """The API queries, over an injectable ``fetch(url) -> dict | list``."""

    def __init__(self, repo: str, *, fetch=None, api=None, server=None):
        self.repo = repo
        self.fetch = fetch or github_fetch
        self.api = (
            api or os.environ.get("GITHUB_API_URL") or "https://api.github.com"
        ).rstrip("/")
        self.server = (server or "https://github.com").rstrip("/")

    def get(self, path: str, **params):
        url = f"{self.api}/repos/{self.repo}/{path}"
        if params:
            url += "?" + urllib.parse.urlencode(params)
        return self.fetch(url)

    def runs(self, file: str, **params) -> list[dict]:
        return self.get(f"actions/workflows/{file}/runs", **params, per_page=PER_PAGE)[
            "workflow_runs"
        ]

    def jobs(self, run_id) -> list[dict]:
        jobs = self.get(f"actions/runs/{run_id}/jobs", per_page=100)["jobs"]
        return [
            {
                "name": j.get("name"),
                "conclusion": j.get("conclusion"),
                "runner": j.get("runner_name"),
                "labels": list(j.get("labels") or []),
                "queue_s": seconds(j.get("created_at"), j.get("started_at")),
                "duration_s": seconds(j.get("started_at"), j.get("completed_at")),
                "url": j.get("html_url"),
            }
            for j in jobs
        ]

    def run_record(self, run: dict) -> dict:
        return {
            "id": run["id"],
            "url": run.get("html_url"),
            "event": run.get("event"),
            "conclusion": run.get("conclusion"),
            "status": run.get("status"),
            "head_sha": run.get("head_sha"),
            "created_at": run.get("created_at"),
            "started_at": run.get("run_started_at"),
            "completed_at": run.get("updated_at"),
            "duration_s": seconds(run.get("run_started_at"), run.get("updated_at")),
            "queue_s": seconds(run.get("created_at"), run.get("run_started_at")),
            "attempt": run.get("run_attempt"),
        }

    def since_green(self, good: str, bad: str) -> dict:
        """Commits between the last green run's commit and the failing one."""
        html = f"{self.server}/{self.repo}/compare/{good}...{bad}"
        try:
            compare = self.get(f"compare/{good}...{bad}")
            return {
                "compare_url": compare.get("html_url") or html,
                "commits": compare["ahead_by"],
            }
        except API_ERRORS:
            return {"compare_url": html, "commits": None}

    def workflow_status(self, file: str) -> dict:
        """Last night's runs of one workflow; raises on an API failure."""
        runs = self.runs(file, event="schedule")
        out = {}
        if not runs:
            # A new nightly that has not fired yet, or a dispatched-only one:
            # show what main did rather than nothing.
            runs = self.runs(file, branch=DEFAULT_BRANCH)
            out["event_fallback"] = True
        runs = sorted(runs, key=lambda r: r.get("created_at") or "", reverse=True)
        completed = [r for r in runs if r.get("status") == "completed"]
        pending = [r for r in runs if r.get("status") != "completed"]
        green = [r for r in completed if r.get("conclusion") == "success"]

        latest = None
        if completed:
            latest = self.run_record(completed[0])
            latest["jobs"] = self.jobs(latest["id"])
        out["latest"] = latest
        out["in_progress"] = (
            {
                "id": pending[0]["id"],
                "url": pending[0].get("html_url"),
                "created_at": pending[0].get("created_at"),
            }
            if pending
            else None
        )
        out["last_success"] = (
            {
                "id": green[0]["id"],
                "url": green[0].get("html_url"),
                "head_sha": green[0].get("head_sha"),
                "completed_at": green[0].get("updated_at"),
            }
            if green
            else None
        )
        out["since_green"] = None
        if (
            latest
            and latest["conclusion"] != "success"
            and out["last_success"]
            and out["last_success"]["head_sha"] != latest["head_sha"]
        ):
            out["since_green"] = self.since_green(
                out["last_success"]["head_sha"], latest["head_sha"]
            )
        out["recent"] = [
            {
                "id": r["id"],
                "url": r.get("html_url"),
                "conclusion": r.get("conclusion"),
                "completed_at": r.get("updated_at"),
                "duration_s": seconds(r.get("run_started_at"), r.get("updated_at")),
            }
            for r in completed[:RECENT]
        ]
        return out

    def bump(self, file: str, branch: str, now: datetime.datetime) -> dict:
        """The bump workflow's open PR (with its check counts) and last run."""
        owner = self.repo.split("/", 1)[0]
        out = {
            "workflow": file,
            "url": f"{self.server}/{self.repo}/actions/workflows/{file}",
        }
        try:
            pulls = self.get(
                "pulls", state="open", head=f"{owner}:{branch}", per_page=1
            )
            pr = None
            if pulls:
                p = pulls[0]
                checks = self.get(
                    f"commits/{p['head']['sha']}/check-runs", per_page=100
                )["check_runs"]
                pr = {
                    "number": p["number"],
                    "url": p.get("html_url"),
                    "title": p.get("title"),
                    "created_at": p.get("created_at"),
                    "age_days": (now - parse_date(p["created_at"])).days,
                    "checks": {
                        "success": sum(
                            1 for c in checks if c.get("conclusion") == "success"
                        ),
                        "failure": sum(
                            1 for c in checks if c.get("conclusion") in FAILED_CHECKS
                        ),
                        "pending": sum(
                            1 for c in checks if c.get("status") != "completed"
                        ),
                    },
                }
            out["pr"] = pr
            done = sorted(
                (r for r in self.runs(file) if r.get("status") == "completed"),
                key=lambda r: r.get("created_at") or "",
                reverse=True,
            )
            out["last_run"] = (
                {
                    "id": done[0]["id"],
                    "url": done[0].get("html_url"),
                    "conclusion": done[0].get("conclusion"),
                    "completed_at": done[0].get("updated_at"),
                }
                if done
                else None
            )
        except API_ERRORS as exc:
            out.setdefault("pr", None)
            out.setdefault("last_run", None)
            out["error"] = _describe(exc)
        return out


# ------------------------------------------------------------- dependencies


def _aged(date: datetime.date, now: datetime.datetime) -> dict:
    return {"date": date.isoformat(), "age_days": (now.date() - date).days}


def parse_peano(text: str, now: datetime.datetime) -> dict:
    """The pin of ``llvm-aie==22.0.0.2026092801+b0d37423``: nightly date and commit."""
    m = re.search(r"^llvm-aie==(\S+)$", text, re.M)
    if not m:
        raise ValueError("no llvm-aie== line")
    pin = m.group(1)
    # Version is <llvm major.minor.patch>.<YYYYMMDD><build>+<commit>.
    v = re.match(r"^\d+\.\d+\.\d+\.(\d{8})\d{2}\+([0-9a-fA-F]+)$", pin)
    if not v:
        raise ValueError(f"unrecognised llvm-aie version {pin!r}")
    date = datetime.datetime.strptime(v.group(1), "%Y%m%d").date()
    return {"pin": pin, "commit": v.group(2), **_aged(date, now)}


def parse_llvm(text: str, now: datetime.datetime) -> dict:
    """The pin of clone-llvm.sh: ``LLVM_PROJECT_COMMIT=<sha>`` and ``DATETIME=YYYYMMDDHH``."""
    sha = re.search(r"^LLVM_PROJECT_COMMIT=([0-9a-fA-F]+)\s*$", text, re.M)
    when = re.search(r"^DATETIME=(\d{8})\d{2}\s*$", text, re.M)
    if not sha or not when:
        raise ValueError("no LLVM_PROJECT_COMMIT= / DATETIME= lines")
    date = datetime.datetime.strptime(when.group(1), "%Y%m%d").date()
    return {"pin": sha.group(1)[:8], "commit": sha.group(1), **_aged(date, now)}


PARSERS = {"peano": parse_peano, "llvm": parse_llvm}


def dependencies(collector: Collector, root: Path, now: datetime.datetime) -> dict:
    out = {}
    for name, spec in DEPENDENCIES.items():
        entry = {"file": spec["file"]}
        try:
            entry.update(PARSERS[name]((root / spec["file"]).read_text(), now))
        except (OSError, ValueError) as exc:
            entry["error"] = _describe(exc)
            print(f"::warning::{spec['file']}: {entry['error']}")
        entry["bump"] = collector.bump(spec["workflow"], spec["branch"], now)
        if "error" in entry["bump"]:
            print(f"::warning::{spec['workflow']}: {entry['bump']['error']}")
        out[name] = entry
    return out


# ------------------------------------------------------------------ outputs


def collect(
    collector: Collector,
    workflows: list[dict],
    *,
    root: Path = ROOT,
    now: datetime.datetime | None = None,
    commit: str = "",
) -> dict:
    """The content of ``latest.json``; raises CollectionFailed when every workflow failed."""
    now = now or now_utc()
    entries = []
    failures = 0
    for wf in workflows:
        entry = dict(wf)
        try:
            entry.update(collector.workflow_status(wf["file"]))
        except API_ERRORS as exc:
            entry["error"] = _describe(exc)
            failures += 1
            print(f"::warning::{wf['file']}: {entry['error']}")
        entries.append(entry)
    if workflows and failures == len(workflows):
        raise CollectionFailed(f"all {failures} workflows failed; is GITHUB_TOKEN set?")
    return {
        "schema": SCHEMA,
        "repo": collector.repo,
        "server": collector.server,
        "date": iso(now),
        "commit": commit,
        "workflows": entries,
        "dependencies": dependencies(collector, root, now),
    }


def day_entry(latest: dict) -> dict:
    """One history entry: each workflow's newest completed run, in brief."""
    workflows = {}
    for wf in latest["workflows"]:
        run = wf.get("latest")
        if run:
            workflows[wf["file"]] = {
                k: run.get(k)
                for k in ("conclusion", "id", "duration_s", "queue_s", "completed_at")
            }
    return {"date": latest["date"], "workflows": workflows}


def update_history(history: dict | None, latest: dict) -> dict:
    """Append today's entry, replacing one of the same UTC day; newest last."""
    history = history or {"schema": SCHEMA, "days": []}
    if history.get("schema", 1) > SCHEMA:
        raise NewerSchema(
            f"history.json is schema {history['schema']}; this collect.py writes {SCHEMA}"
        )
    entry = day_entry(latest)
    days = [d for d in history.get("days", []) if d["date"][:10] != entry["date"][:10]]
    days.append(entry)
    days.sort(key=lambda d: d["date"])
    return {"schema": SCHEMA, "days": days[-HISTORY_DAYS:]}


def write(out: Path, latest: dict) -> None:
    out.mkdir(parents=True, exist_ok=True)
    path = out / "history.json"
    previous = json.loads(path.read_text()) if path.exists() else None
    # Refuse a newer history before touching latest.json, so the two stay a pair.
    history = update_history(previous, latest)
    (out / "latest.json").write_text(json.dumps(latest, indent=1) + "\n")
    path.write_text(json.dumps(history, separators=(",", ":")) + "\n")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n", 1)[0])
    parser.add_argument("--repo", required=True, help="owner/name")
    parser.add_argument("--workflows", type=Path, default=ROOT / ".github/workflows")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument(
        "--server", default=os.environ.get("GITHUB_SERVER_URL") or "https://github.com"
    )
    parser.add_argument(
        "--root", type=Path, default=ROOT, help="checkout holding the pin files"
    )
    parser.add_argument(
        "--now", default="", help="ISO time of the collection (default: now)"
    )
    parser.add_argument("--commit", default=os.environ.get("GITHUB_SHA", ""))
    args = parser.parse_args(argv)

    now = parse_date(args.now) if args.now else now_utc()
    collector = Collector(args.repo, server=args.server)
    workflows = discover_workflows(args.workflows, args.repo, collector.server)
    try:
        latest = collect(
            collector, workflows, root=args.root, now=now, commit=args.commit
        )
    except CollectionFailed as exc:
        print(f"::error::{exc}")
        return 1
    write(args.out, latest)
    for wf in latest["workflows"]:
        run = wf.get("latest")
        if "error" in wf:
            print(f"{wf['name']}: error ({wf['error']})")
        elif run:
            print(f"{wf['name']}: {run['conclusion']} ({run['completed_at']})")
        else:
            print(f"{wf['name']}: no runs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
