"""L6: testmon-verify tests exactly the snapshot it claims to have tested."""

import json
import os
import subprocess
import sys

import pytest

from tests.shared.conftest import linux_only
from tests.shared.helpers import (
    ALL_SAMPLE_TESTS,
    SAMPLE,
    GitRepo,
    clean_env,
    run_pytest,
)
from tests.shared.test_scenarios import ADD

pytestmark = linux_only

GOOD = {
    "pkg/calc.py": """
        def add(a, b):
            return b + a


        def mul(a, b):
            return a * b
        """
}
BROKEN = {
    "pkg/calc.py": """
        def add(a, b):
            return a - b


        def mul(a, b):
            return a * b
        """
}


def verify(repo, shared_db, *args, env=None, tmpdir=None):
    report = repo.path / "report-verify.json"
    if report.exists():
        report.unlink()
    extra = {"TESTMON_SHARED_DB": str(shared_db)}
    if tmpdir is not None:
        extra["TMPDIR"] = str(tmpdir)
    extra.update(env or {})
    proc = subprocess.run(
        [sys.executable, "-m", "testmon.verify", *args, "--report", str(report)],
        cwd=repo.path,
        env=clean_env(extra),
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    data = json.loads(report.read_text()) if report.exists() else None
    return proc, data


@pytest.fixture
def recorded(repo, shared_db):
    run_pytest(repo.path, shared_db)
    return repo


def test_staged_ignores_unstaged_breakage(recorded, shared_db, tmp_path):
    recorded.write(GOOD)
    recorded.git("add", "pkg/calc.py")
    recorded.write(BROKEN)  # unstaged on top
    tmpdir = tmp_path / "tmp"
    tmpdir.mkdir()
    proc, report = verify(recorded, shared_db, "--staged", tmpdir=tmpdir)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert report["verified_tree"] == recorded.git("write-tree")
    selected = {n for n, t in report["tests"].items() if t["decision"] == "selected"}
    assert selected == ADD
    assert {report["tests"][n]["outcome"] for n in ADD} == {"passed"}
    assert list(tmpdir.iterdir()) == []  # snapshot cleaned up
    assert "testmon-verify:" in proc.stdout


def test_staged_catches_staged_breakage(recorded, shared_db, tmp_path):
    recorded.write(BROKEN)
    recorded.git("add", "pkg/calc.py")
    recorded.write(GOOD)  # the working tree would pass
    tmpdir = tmp_path / "tmp"
    tmpdir.mkdir()
    proc, report = verify(recorded, shared_db, "--staged", tmpdir=tmpdir)
    assert proc.returncode == 1, proc.stdout + proc.stderr
    failed = {n for n, t in report["tests"].items() if t.get("outcome") == "failed"}
    assert failed == ADD
    assert list(tmpdir.iterdir()) == []  # cleaned up on failure too


def test_ref(recorded, shared_db):
    recorded.write(GOOD)
    sha = recorded.commit("good add")
    recorded.write(BROKEN)  # dirty working tree must not matter
    proc, report = verify(recorded, shared_db, "--ref", sha)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert report["verified_tree"] == recorded.tree(sha)
    worktrees = recorded.git("worktree", "list").splitlines()
    assert len(worktrees) == 1, worktrees
    # what --ref recorded is reused by an in-place run of the same content
    recorded.write(GOOD)
    run = run_pytest(recorded.path, shared_db)
    assert run.selected == set()


def test_in_place_reports_dirty_tree(recorded, shared_db):
    proc, report = verify(recorded, shared_db)
    assert proc.returncode == 0
    assert report["verified_tree"] is None
    assert report["worktree"]["dirty"] is False
    recorded.write(GOOD)
    proc, report = verify(recorded, shared_db)
    assert report["worktree"]["dirty"] is True
    assert report["worktree"]["diff_hash"]
    assert report["worktree"]["git_head"] == recorded.head()


def test_snapshot_shadows_an_editable_install_of_another_checkout(tmp_path, shared_db):
    files = {("src/" + k if k.startswith("pkg/") else k): v for k, v in SAMPLE.items()}
    main = GitRepo.init(tmp_path / "main", files)
    other = main.clone(tmp_path / "other")
    other.write({"src/" + name: content for name, content in GOOD.items()})
    other.git("add", "-A")
    # PYTHONPATH -> main/src stands for an editable install made from the main checkout
    proc, report = verify(
        other, shared_db, "--staged", env={"PYTHONPATH": str(main.path / "src")}
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert report["fallback"] is None  # no foreign checkout: the snapshot's src/ wins
    selected = {n for n, t in report["tests"].items() if t["decision"] == "selected"}
    assert (
        selected == ALL_SAMPLE_TESTS
    )  # new database: everything ran, from the snapshot
    assert os.path.realpath(report["rootdir"]) != os.path.realpath(main.path)
