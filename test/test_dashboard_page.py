# Copyright (C) 2026 Advanced Micro Devices, Inc.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# RUN: %pytest %s

"""Exercise the maintainer dashboard's data and table code under node, without a browser."""

import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PAGE = ROOT / "utils/dashboard/index.html"


@pytest.fixture
def page():
    node = shutil.which("node")
    if not node:
        pytest.skip("node is required to test the dashboard page")
    script = PAGE.read_text().split("<script>", 1)[1].split("</script>", 1)[0]
    setup = """
const assert = require('node:assert/strict');
global.fetch = async () => ({ ok: false });
class El {
  constructor(tag) { this.tag = tag; this.children = []; this.on = {}; this.hidden = false; }
  append(...nodes) { this.children.push(...nodes); }
  replaceChildren(...nodes) { this.children = nodes; }
  addEventListener(type, f) { this.on[type] = f; }
  querySelectorAll() { return []; }
  get text() {
    return (this.textContent || '') +
      this.children.map(c => typeof c === 'string' ? c : c.text).join('');
  }
  find(pred, out = []) {
    for (const c of this.children) if (typeof c !== 'string') { if (pred(c)) out.push(c); c.find(pred, out); }
    return out;
  }
}
const byId = new Map();
global.document = {
  getElementById: id => byId.get(id) || byId.set(id, new El()).get(id),
  createElement: tag => new El(tag),
  documentElement: { getAttribute: () => null, setAttribute() {}, removeAttribute() {} },
};
global.location = { hash: '' };
const drawn = [];
global.Chart = function (_canvas, config) { drawn.push(config); };
global.window = { addEventListener: () => {}, DASHBOARD_TEST: true };
"""
    data = """
const NOW = Date.parse('2026-09-30T12:00:00Z');
const iso = ms => new Date(ms).toISOString();
const run = (id, conclusion, hoursAgo, extra) => ({
  id, url: `https://gh/run/${id}`, event: 'schedule', conclusion, status: 'completed', head_sha: `sha${id}`,
  created_at: iso(NOW - hoursAgo * HOUR - 600000), started_at: iso(NOW - hoursAgo * HOUR - 300000),
  completed_at: iso(NOW - hoursAgo * HOUR), duration_s: 1800, queue_s: 300, attempt: 1,
  jobs: [{ name: 'build', conclusion, runner: 'bench-1', labels: [], queue_s: 300, duration_s: 1800, url: `https://gh/job/${id}` }],
  ...(extra || {}),
});
const wf = (file, name, latest, extra) => ({ file, name, schedule: ['0 0 * * *'], url: `https://gh/wf/${file}`,
  latest, in_progress: null, last_success: null, since_green: null, recent: [], ...(extra || {}) });
const status = {
  schema: 1, repo: 'Xilinx/mlir-aie', server: 'https://github.com', date: iso(NOW - HOUR), commit: 'abcdef1234',
  workflows: [
    // `recent` is newest first, as collect.py writes it.
    wf('ok.yml', 'Zebra ok', run(1, 'success', 6), { recent: [{ id: 1, url: 'https://gh/run/1', conclusion: 'success', completed_at: iso(NOW - 6 * HOUR), duration_s: 1800 },
                                                                  { id: 0, url: 'https://gh/run/0', conclusion: 'failure', completed_at: iso(NOW - 30 * HOUR), duration_s: 100 }] }),
    wf('red.yml', 'Alpha red', run(2, 'failure', 5, { attempt: 2 }), { last_success: { id: 9, url: 'https://gh/run/9', head_sha: 'good', completed_at: iso(NOW - 29 * HOUR) },
                                                    since_green: { compare_url: 'https://gh/compare/good...sha2', commits: 6 } }),
    wf('old.yml', 'Mid stale', run(3, 'success', 80)),
    wf('weekly.yml', 'Weekly', run(4, 'success', 80), { schedule: ['0 3 * * 0'] }),
    wf('broken.yml', 'Broken api', null, { error: 'HTTP 500' }),
    wf('never.yml', 'Never ran', null),
  ],
  dependencies: {
    peano: { pin: 'llvm-aie==22.0.0.2026092801+b0d37423', date: '2026-09-28', age_days: 2, commit: 'b0d37423',
             bump: { workflow: 'update-peano.yml', pr: null, last_run: { id: 5, url: 'https://gh/run/5', conclusion: 'success', completed_at: iso(NOW - 40 * HOUR) } } },
    llvm: { pin: 'e4fcd128', date: '2026-08-01', age_days: 60, commit: 'e4fcd128',
            bump: { workflow: 'update-llvm.yml', pr: { number: 42, url: 'https://gh/pr/42', title: 'Update LLVM version', created_at: iso(NOW - 10 * DAY), age_days: 10, checks: { success: 3, failure: 2, pending: 0 } }, last_run: null } },
  },
};
const history = { schema: 1, days: [0, 1, 2, 3].map(d => ({ date: iso(NOW - d * DAY), workflows: { 'ok.yml': { conclusion: 'success', id: d, duration_s: 1800 + 60 * d, queue_s: 10 } } })) };
// A metric history file as publish.py writes it: runs by date, one series per name.
const hist = (metric, unit, days, series) => ({
  schema: 1, target: 't', metric, unit,
  runs: days.map((d, i) => ({ id: `r${i}`, date: iso(NOW - d * DAY), commit: { id: 'c'.repeat(10), url: 'https://gh/c' }, pmode: null, provenance: {}, url: `https://gh/run/${i}` })),
  series: Object.fromEntries(Object.entries(series).map(([k, values]) => [k, { values }])),
});
"""

    def run(checks):
        subprocess.run([node, "-e", setup + script + data + checks], check=True)

    return run


def test_schedule_intervals_and_staleness(page):
    page("""
assert.equal(intervalHours(['0 0 * * *']), 24);
assert.equal(intervalHours(['0 */4 * * *']), 4);
assert.equal(intervalHours(['17 3 * * 0']), 168);
assert.equal(intervalHours(['0 6 1,15 * *']), 720);
assert.equal(intervalHours([]), 24);
assert.equal(intervalHours(['0 0 * * *', '0 */4 * * *']), 4);
assert.equal(staleAfterMs(['0 0 * * *']), 48 * HOUR);
assert.equal(staleAfterMs(['0 */4 * * *']), 36 * HOUR);
assert.equal(isStale(NOW - 47 * HOUR, NOW, ['0 0 * * *']), false);
assert.equal(isStale(NOW - 49 * HOUR, NOW, ['0 0 * * *']), true);
assert.equal(isStale(NOW - 80 * HOUR, NOW, ['0 3 * * 0']), false);
assert.equal(isStale(NaN, NOW, ['0 0 * * *']), true);
""")


def test_workflows_sort_failing_then_broken_then_stale(page):
    page("""
const order = sortWorkflows(status.workflows, NOW).map(w => w.file);
assert.deepEqual(order, ['red.yml', 'broken.yml', 'old.yml', 'never.yml', 'weekly.yml', 'ok.yml']);
assert.deepEqual(runnerNames(status.workflows[0].latest.jobs), ['bench-1']);
assert.deepEqual(runnerNames(null), []);
""")


def test_change_over_days_needs_a_point_old_enough(page):
    page("""
const h = hist('line_pct', '%', [40, 31, 10, 0], { total: [60, 61, null, 63], 'lib': [70, 70, 70, 70] });
const pts = pointsOf(h, 'total');
assert.deepEqual(pts.map(p => p.value), [60, 61, 63]);
const c = changeOverDays(pts, 30);
assert.equal(c.before.value, 61);
assert.equal(c.delta, 2);
assert.equal(changeOverDays(pointsOf(h, 'lib'), 30).delta, 0);
const short = changeOverDays(pointsOf(hist('x', '', [5, 0], { a: [1, 2] }), 'a'), 30);
assert.equal(short.delta, null);
assert.equal(changeOverDays([], 30), null);
assert.deepEqual(pointsOf(null, 'a'), []);
""")


def test_workflow_rows_link_runs_and_name_what_is_wrong(page):
    page("""
const days = history.days;
const red = workflowRow(status.workflows[1], days, NOW);
assert.equal(red.className, 'failing');
const links = red.find(e => e.tag === 'a');
assert.ok(links.some(a => a.href === 'https://gh/run/2' && a.className.includes('failure')));
assert.ok(links.some(a => a.href === 'https://gh/compare/good...sha2' && a.text === '6 commits'));
assert.ok(links.some(a => a.href === 'https://gh/job/2' && a.text === 'build'));
assert.ok(red.text.includes('attempt 2'));
assert.ok(red.text.includes('30 min'));
const ok = workflowRow(status.workflows[0], days, NOW);
assert.ok(!ok.className);
assert.ok(ok.text.includes('green'));
// Last-14 strip: oldest to newest, each a link to its run.
const strip = ok.find(e => e.className === 'strip')[0];
assert.deepEqual(strip.children.map(a => [a.className, a.href]), [['failure', 'https://gh/run/0'], ['success', 'https://gh/run/1']]);
// The duration trend comes from the collection history.
assert.equal(drawn.length, 1);
assert.deepEqual(drawn[0].data.datasets[0].data, [33, 32, 31, 30]);
const stale = workflowRow(status.workflows[2], days, NOW);
assert.ok(stale.text.includes('stale: nothing since'));
const weekly = workflowRow(status.workflows[3], days, NOW);
assert.ok(!weekly.text.includes('stale'));
const broken = workflowRow(status.workflows[4], days, NOW);
assert.ok(broken.text.includes('status unavailable'));
assert.ok(broken.text.includes('no completed run'));
const summary = renderNightlies(status, history, NOW);
assert.deepEqual(summary.failing.map(w => w.file), ['red.yml']);
assert.deepEqual(summary.stale.map(w => w.file), ['old.yml', 'broken.yml', 'never.yml']);
assert.equal($('nightlies-rows').children.length, 6);
assert.equal($('nightlies-empty').hidden, true);
""")


def test_kernel_cards_summarize_the_run_index(page):
    page("""
const runs = [
  { id: '1', url: 'https://gh/run/1', date: iso(NOW - 30 * HOUR), commit: { id: 'abcdef12345', url: 'https://gh/c', message: 'older' }, pmode: 'turbo', published: true, n_rows: 300, failed: [], sane: true },
  { id: '2', url: 'https://gh/run/2', date: iso(NOW - 6 * HOUR), commit: { id: 'fedcba54321', url: 'https://gh/c2', message: 'newer' }, pmode: 'turbo', published: false, n_rows: 0, failed: ['t1', 't2'], sane: false,
    cases: { passed: 10, failed: 2, timed: 8, timing_failed: 0, untimed: 0 } },
];
const card = kernelCard('npu1', { schema: 1, target: 'npu1', runs }, NOW);
assert.ok(card.text.includes('measurement sanity check failed'));
assert.ok(card.text.includes('10 passed'));
assert.ok(card.text.includes('2 failing'));
assert.ok(card.text.includes('2 tests'));
assert.ok(card.text.includes('Last numbers'));
assert.ok(card.find(e => e.href === 'https://gh/run/2').length === 1);
assert.ok(card.find(e => e.href === 'https://gh/run/1').length === 1);
const stale = kernelCard('npu2', { runs: [{ ...runs[0], date: iso(NOW - 3 * DAY) }] }, NOW);
assert.ok(stale.text.includes('stale'));
assert.ok(kernelCard('npu2', null, NOW).text.includes('no results'));
assert.ok(kernelCard('npu2', { schema: 99, runs }, NOW).text.includes('newer version'));
renderKernels(new Map([['npu1', { runs }], ['npu2', null]]), NOW);
assert.equal($('kernel-cards').children.length, 2);
assert.equal($('kernels-when').className, 'when');
""")


def test_coverage_card_and_rows_color_notable_30_day_moves(page):
    page("""
const days = [45, 31, 15, 0];
const lines = hist('line_pct', '%', days, { total: [60, 60.2, 61, 61.4], 'lib': [70, 70, 69, 68], 'lib/Dialect': [80, 80, 80, 80.2], 'python': [null, null, 50, 55] });
const fns = hist('function_pct', '%', days, { total: [55, 55, 56, 57], lib: [65, 65, 65, 65] });
const counted = hist('lines', 'lines', days, { total: [40000, 40100, 40200, 40210], lib: [30000, 30000, 30000, 30000] });
renderCoverage(lines, fns, counted, NOW);
const card = $('coverage-cards').children[0];
assert.ok(card.text.includes('61.4%'));
assert.ok(card.text.includes('+1.2 pt in 30 days'));
assert.ok(card.find(e => e.className && e.className.includes('better')).length === 1);
assert.ok(card.text.includes('57.0%'));
assert.ok(card.text.includes('40,210'));
const rows = $('coverage-rows').children;
assert.deepEqual(rows.map(r => r.children[0].text), ['lib', 'lib/Dialect', 'python']);
assert.equal(rows[0].children[2].text, '-2.0 pt');
assert.equal(rows[0].children[2].className, 'num worse');
assert.equal(rows[1].children[2].text, '+0.2 pt');
assert.equal(rows[1].children[2].className, 'num');
assert.equal(rows[2].children[2].text, 'no 30-day history');
assert.equal($('coverage-empty').hidden, true);
assert.equal($('coverage-when').className, 'when');
""")


def test_coverage_without_data_says_so(page):
    page("""
renderCoverage(null, null, null, NOW);
assert.equal($('coverage-empty').hidden, false);
assert.equal($('coverage-when').text, 'not published yet');
assert.equal($('coverage-cards').children.length, 0);
""")


def test_wheel_sizes_list_wheels_before_the_total(page):
    page("""
const days = [40, 0];
const bytes = hist('bytes', 'bytes', days, { 'win_amd64/rtti_ON/cp312': [100 * 1048576, 104 * 1048576], all: [300 * 1048576, 305 * 1048576], 'manylinux_2_28_x86_64/rtti_ON/cp312': [200 * 1048576, 199 * 1048576] });
const unpacked = hist('uncompressed_bytes', 'bytes', days, { all: [900 * 1048576, 910 * 1048576] });
const files = hist('files', 'files', days, { all: [1000, 1001] });
renderSizes(bytes, unpacked, files, NOW);
const rows = $('sizes-rows').children;
assert.deepEqual(rows.map(r => r.children[0].text), ['manylinux_2_28_x86_64/rtti_ON/cp312', 'win_amd64/rtti_ON/cp312', 'all']);
assert.equal(rows[1].children[1].text, '104.0 MiB');
assert.equal(rows[1].children[2].text, '+4.0%');
assert.equal(rows[1].children[2].className, 'num worse');
assert.equal(rows[0].children[2].className, 'num');
assert.equal(rows[2].children[3].text, '910.0 MiB');
assert.equal(rows[2].children[4].text, '1,001');
assert.equal(formatBytes(512), '512 B');
assert.equal(formatBytes(2048), '2 KiB');
assert.equal($('sizes-empty').hidden, true);
""")


def test_dependency_cards_flag_old_pins_and_red_bump_prs(page):
    page("""
renderDependencies(status, NOW);
const [peano, llvm] = $('dependency-cards').children;
assert.ok(peano.text.includes('2 days old'));
assert.ok(peano.text.includes('none open'));
assert.ok(peano.find(e => e.href === 'https://gh/run/5').length === 1);
assert.ok(llvm.text.includes('60 days old'));
assert.equal(llvm.find(e => e.className === 'worse').length, 2);
assert.ok(llvm.text.includes('#42'));
assert.ok(llvm.text.includes('2 failing'));
assert.ok(llvm.find(e => e.href === 'https://github.com/Xilinx/mlir-aie/actions/workflows/update-llvm.yml').length === 1);
assert.ok(dependencyCard('x', null, '').text.includes('not collected'));
""")


def test_warnings_name_the_stale_and_the_failing(page):
    page("""
assert.deepEqual(warningsFor(null, { failing: [], stale: [] }, NOW).map(w => w.level), ['bad']);
const w = warningsFor(status, { failing: [status.workflows[1]], stale: [status.workflows[2]] }, NOW);
assert.deepEqual(w.map(x => x.level), ['warn', 'warn', 'info']);
assert.ok(w[0].text.includes('Alpha red'));
assert.ok(w[1].text.includes('Mid stale'));
assert.ok(w[2].text.includes('Broken api'));
const old = warningsFor({ ...status, date: iso(NOW - 3 * DAY) }, { failing: [], stale: [] }, NOW);
assert.equal(old[0].level, 'bad');
assert.ok(old[0].text.includes('collector itself'));
assert.equal(readable({ schema: 2 }), null);
assert.equal(repoUrl(null), 'https://github.com/Xilinx/mlir-aie');
assert.equal(repoUrl(status), 'https://github.com/Xilinx/mlir-aie');
assert.equal(formatDuration(45), '45 s');
assert.equal(formatDuration(3000), '50 min');
assert.equal(formatDuration(7200), '2.0 h');
assert.equal(ago(NOW - 3 * DAY, NOW), '3 days ago');
""")
