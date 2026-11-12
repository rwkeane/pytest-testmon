"""L5: differential validation of selective execution against full runs.

After every random step (edits and git operations across two checkouts) three runs are made
in the checkout the step touched:

1. selective: ``pytest --testmon`` against the long-lived shared database;
2. full, uninstrumented: plain ``pytest`` (outcomes only);
3. full, instrumented, into a fresh scratch database (what the dependencies are *now*).

and checked:

* the same tests exist in the selective report (selected + deselected, file-ignored included)
  as in the full run;
* every test failing in the full run was selected (no incorrect exclusion);
* every deselected test's current dependencies (scratch database) are exactly those of the
  observation the selection relied on;
* selected tests had the same outcome as in the full run.

Scale: TESTMON_DIFF_SEEDS (default "0,1,2"), TESTMON_DIFF_STEPS (default 15), and
TESTMON_DIFF_XDIST (probability of running the selective run under -n 2, default 0.2).
A failure writes a replay bundle (seed, op log, reports) and names it in the message.
"""

import json
import os
import random
import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest

from tests.differential.project import initial_files, random_edit
from tests.shared.conftest import linux_only
from tests.shared.helpers import GitRepo, clean_env, init_db, run_pytest

pytestmark = linux_only

SEEDS = [int(s) for s in os.environ.get("TESTMON_DIFF_SEEDS", "0,1,2").split(",") if s]
STEPS = int(os.environ.get("TESTMON_DIFF_STEPS", "15"))
XDIST = float(os.environ.get("TESTMON_DIFF_XDIST", "0.2"))

OUTCOME_PLUGIN = """
import json, os

_outcomes = {}

def pytest_runtest_logreport(report):
    if report.when == "call" or report.outcome != "passed":
        if _outcomes.get(report.nodeid) in (None, "passed"):
            _outcomes[report.nodeid] = report.outcome

def pytest_sessionfinish(session):
    with open(os.environ["DIFF_OUTCOMES"], "w") as f:
        json.dump(_outcomes, f)
"""


def full_outcomes(checkout, plugin_dir, out):
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-p",
            "no:cacheprovider",
            "-q",
            "-p",
            "diff_outcomes",
        ],
        cwd=checkout,
        env=clean_env({"PYTHONPATH": str(plugin_dir), "DIFF_OUTCOMES": str(out)}),
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert proc.returncode in (0, 1), proc.stdout + proc.stderr
    return json.loads(out.read_text())


def deps_hashes(db_path, ids=None):
    con = sqlite3.connect(db_path)
    try:
        if ids is None:
            rows = con.execute(
                "SELECT test_name, deps_hash FROM observation"
            ).fetchall()
            return {name: bytes(digest) for name, digest in rows}
        return {
            row[0]: bytes(row[1])
            for row in con.execute(
                f"SELECT id, deps_hash FROM observation WHERE id IN ({','.join('?' * len(ids))})",
                list(ids),
            )
        }
    finally:
        con.close()


class Session:
    def __init__(self, tmp_path, seed):
        self.tmp = tmp_path
        self.seed = seed
        self.rng = random.Random(seed)
        self.shared_db = init_db(tmp_path / "cache" / "shared.sqlite")
        self.main = GitRepo.init(tmp_path / "checkout-a", initial_files(seed))
        self.main.git("branch", "other")
        self.other = self.main.worktree(tmp_path / "checkout-b", "other")
        self.other.git("checkout", "-q", "other")
        self.checkouts = [self.main, self.other]
        self.branches = ["main", "other"]
        self.log = []
        self.plugin_dir = tmp_path / "plugin"
        self.plugin_dir.mkdir()
        (self.plugin_dir / "diff_outcomes.py").write_text(OUTCOME_PLUGIN)

    def commit_all(self, repo):
        repo.git("add", "-A")
        repo.git("commit", "-q", "--allow-empty", "-m", f"step {len(self.log)}")

    def git_op(self, repo):
        self.commit_all(repo)
        op = self.rng.choice(["branch", "switch", "cherry-pick", "rebase", "squash"])
        current = repo.git("rev-parse", "--abbrev-ref", "HEAD")
        others = [b for b in self.branches if b != current]
        target = self.rng.choice(others) if others else None
        if op == "branch":
            name = f"b{len(self.branches)}"
            repo.git("checkout", "-q", "-b", name)
            self.branches.append(name)
            return f"git branch {name}"
        if target is None:
            return "git noop"
        if op == "switch":
            ok = (
                repo.git("checkout", "-q", target, check=False) == ""
                and repo.git("rev-parse", "--abbrev-ref", "HEAD") == target
            )
            return f"git switch {target}" + ("" if ok else " (refused)")
        if op == "cherry-pick":
            sha = repo.git("rev-parse", target)
            result = subprocess.run(
                [
                    "git",
                    "cherry-pick",
                    "--allow-empty",
                    "--keep-redundant-commits",
                    sha,
                ],
                cwd=repo.path,
                capture_output=True,
                env=clean_env(),
                check=False,
            )
            if result.returncode:
                repo.git("cherry-pick", "--abort", check=False)
                return f"git cherry-pick {target} (conflict, aborted)"
            return f"git cherry-pick {target}"
        if op == "rebase":
            result = subprocess.run(
                ["git", "rebase", "-q", target],
                cwd=repo.path,
                capture_output=True,
                env=clean_env(),
                check=False,
            )
            if result.returncode:
                repo.git("rebase", "--abort", check=False)
                return f"git rebase {target} (conflict, aborted)"
            return f"git rebase {target}"
        result = subprocess.run(
            ["git", "merge", "--squash", target],
            cwd=repo.path,
            capture_output=True,
            env=clean_env(),
            check=False,
        )
        if result.returncode:
            repo.git("merge", "--abort", check=False)
            repo.git("reset", "-q", "--hard", "HEAD")
            return f"git squash {target} (conflict, aborted)"
        self.commit_all(repo)
        return f"git squash {target}"

    def step(self):
        repo = self.rng.choice(self.checkouts)
        if self.rng.random() < 0.3:
            done = self.git_op(repo)
        else:
            done = "; ".join(
                random_edit(self.rng, repo.path) for _ in range(self.rng.randint(1, 2))
            )
        self.log.append(f"{repo.path.name}: {done}")
        return repo

    def check(self, repo, index):
        xdist = ("-n", "2") if self.rng.random() < XDIST else ()
        selective = run_pytest(repo.path, self.shared_db, *xdist)
        full = full_outcomes(
            repo.path, self.plugin_dir, self.tmp / f"outcomes-{index}.json"
        )
        scratch_db = init_db(self.tmp / f"scratch-{index}" / "db.sqlite")
        scratch = run_pytest(repo.path, scratch_db, "--testmon-noselect")
        problems = []
        if selective.report is None or selective.returncode not in (0, 1):
            problems.append(f"selective run broke: {selective!r}")
            return problems, selective
        if scratch.returncode not in (0, 1):
            problems.append(f"scratch run broke: {scratch!r}")
        if set(selective.tests) != set(full):
            problems.append(
                "collected sets differ: only selective "
                f"{sorted(set(selective.tests) - set(full))}, only full "
                f"{sorted(set(full) - set(selective.tests))}"
            )
        failing = {name for name, outcome in full.items() if outcome == "failed"}
        missed = failing - selective.selected
        if missed:
            problems.append(f"INCORRECT EXCLUSION of failing tests: {sorted(missed)}")
        current = deps_hashes(scratch_db)
        used = {
            name: selective.tests[name]["observation_id"]
            for name in selective.deselected
        }
        relied_on = deps_hashes(self.shared_db, [i for i in used.values() if i])
        for name, obs_id in sorted(used.items()):
            if obs_id is None or relied_on.get(obs_id) != current.get(name):
                problems.append(
                    f"deselected {name}: observation {obs_id} != current deps"
                )
        for name in selective.selected:
            got = selective.tests[name].get("outcome")
            if got is not None and full.get(name) is not None and got != full[name]:
                problems.append(f"{name}: selective {got} vs full {full[name]}")
        return problems, selective


@pytest.mark.parametrize("seed", SEEDS)
def test_differential(tmp_path, seed):
    session = Session(tmp_path, seed)
    stats = {"selected": 0, "deselected": 0}
    for index in range(STEPS):
        repo = session.main if index == 0 else session.step()
        problems, selective = session.check(repo, index)
        if problems:
            bundle = tmp_path / "replay.json"
            bundle.write_text(
                json.dumps(
                    {
                        "seed": seed,
                        "step": index,
                        "log": session.log,
                        "problems": problems,
                        "selective_report": selective.report,
                    },
                    indent=1,
                )
            )
            pytest.fail(
                f"seed {seed} step {index}: "
                + "\n".join(problems)
                + f"\nreplay: {bundle}"
            )
        stats["selected"] += len(selective.selected)
        stats["deselected"] += len(selective.deselected)
    # the scenario must actually exercise deselection, or it proves nothing
    assert stats["deselected"] > 0, (stats, session.log)
    print(f"seed {seed}: {stats}, ops: {session.log}")


def test_generator_is_deterministic():
    assert initial_files(7) == initial_files(7)
    assert initial_files(7) != initial_files(8)
    for path, text in initial_files(3).items():
        if path.endswith(".py") and text:
            compile(text, path, "exec")
    assert Path(__file__).exists()
