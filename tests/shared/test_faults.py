"""L4: interrupted runs, corrupted/incompatible databases. Never a false pass."""

import os
import signal
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

import pytest

from tests.shared.conftest import linux_only
from tests.shared.fixtures import (
    EVERY_TEST,
    MANY_TESTS,
    assert_consistent,
    big_repo,
    query,
)
from tests.shared.helpers import (
    ALL_SAMPLE_TESTS,
    assert_ran,
    popen_pytest,
    run_pytest,
)
from tests.shared.test_scenarios import MUL, MUL_CHANGED

pytestmark = linux_only

not_root = pytest.mark.skipif(os.geteuid() == 0, reason="root ignores permissions")


@pytest.fixture
def repo_many(tmp_path):
    return big_repo(tmp_path / "repo")


def statuses(shared_db):
    return [
        row[0] for row in query(shared_db, "SELECT status FROM session ORDER BY id")
    ]


class TestInterruption:
    def test_killed_between_batches(self, repo_many, shared_db):
        run = run_pytest(
            repo_many.path, shared_db, env={"TESTMON_FAULT_INJECT": "after_commit"}
        )
        assert run.returncode == 17, run
        assert_consistent(shared_db)
        (stored,) = query(shared_db, "SELECT count(*) FROM observation")[0]
        assert stored == 250  # the first batch, committed before the crash
        run = run_pytest(repo_many.path, shared_db)
        assert run.returncode == 0, run
        assert len(run.selected) == len(EVERY_TEST) - 250
        assert {run.reason(name) for name in run.selected} == {"no_observation"}
        assert_ran(run_pytest(repo_many.path, shared_db), set())
        assert statuses(shared_db)[0] == "interrupted"  # found dead by the next session

    def test_killed_inside_the_write_transaction(self, repo_many, shared_db):
        run = run_pytest(
            repo_many.path, shared_db, env={"TESTMON_FAULT_INJECT": "mid_write"}
        )
        assert run.returncode == 17, run
        assert_consistent(shared_db)
        assert query(shared_db, "SELECT count(*) FROM observation") == [(0,)]
        assert query(shared_db, "SELECT count(*) FROM dep_fp") == [(0,)]
        assert_ran(run_pytest(repo_many.path, shared_db), EVERY_TEST)

    def test_sigkill_from_outside(self, repo_many, shared_db, tmp_path):
        marker = tmp_path / "marker"
        repo_many.write(
            {
                "tests/test_zz_slow.py": f"""
                    import time
                    from pathlib import Path


                    def test_slow():
                        Path({str(marker)!r}).touch()
                        time.sleep(60)
                    """
            }
        )
        proc = popen_pytest(repo_many.path, shared_db, "-p", "no:randomly")
        deadline = time.time() + 120
        while not marker.exists():
            assert proc.poll() is None, proc.communicate()
            assert time.time() < deadline
            time.sleep(0.05)
        os.kill(proc.pid, signal.SIGKILL)
        proc.communicate(timeout=30)
        assert_consistent(shared_db)
        run = run_pytest(
            repo_many.path, shared_db, "--deselect", "tests/test_zz_slow.py::test_slow"
        )
        assert run.returncode == 0, run
        # whatever was committed is reused, everything else runs; nothing is lost silently
        (stored,) = query(shared_db, "SELECT count(*) FROM observation")[0]
        assert stored == len(EVERY_TEST)

    def test_keyboard_interrupt(self, tmp_path, shared_db):
        from tests.shared.helpers import GitRepo  # pylint: disable=import-outside-toplevel

        repo = GitRepo.init(tmp_path / "repo")
        repo.write(
            {
                "tests/test_zz_interrupt.py": """
                    import os
                    import signal


                    def test_interrupt():
                        if os.environ.get("INTERRUPT"):
                            os.kill(os.getpid(), signal.SIGINT)
                    """
            }
        )
        run = run_pytest(
            repo.path, shared_db, "-p", "no:randomly", env={"INTERRUPT": "1"}
        )
        assert run.returncode == 2, run
        assert statuses(shared_db) == ["interrupted"]
        assert_consistent(shared_db)
        run = run_pytest(repo.path, shared_db)
        # the interrupted test was not recorded (discarded), the others were
        assert "tests/test_zz_interrupt.py::test_interrupt" in run.selected
        assert run.selected <= ALL_SAMPLE_TESTS | {
            "tests/test_zz_interrupt.py::test_interrupt"
        }


def truncate(path):
    data = path.read_bytes()
    path.write_bytes(data[: len(data) // 3])


def garbage_header(path):
    with open(path, "r+b") as file:
        file.write(b"definitely not sqlite!")


def zero_page(path):
    """Zero the root page of the observation table (which selection has to read)."""
    con = sqlite3.connect(path)
    page_size = con.execute("PRAGMA page_size").fetchone()[0]
    (root,) = con.execute(
        "SELECT rootpage FROM sqlite_master WHERE name = 'observation'"
    ).fetchone()
    con.close()
    with open(path, "r+b") as file:
        file.seek((root - 1) * page_size)
        file.write(b"\0" * page_size)


def set_versions(user_version, compat):
    def corrupt(path):
        con = sqlite3.connect(path)
        con.execute(f"PRAGMA user_version = {user_version}")
        con.execute(
            "UPDATE metadata SET value = ? WHERE key = 'compat_version'", (compat,)
        )
        con.commit()
        con.close()

    return corrupt


def upstream_format(path):
    path.unlink()
    con = sqlite3.connect(path)
    con.executescript(
        "CREATE TABLE test_execution (id INTEGER PRIMARY KEY, test_name TEXT);"
        "PRAGMA user_version = 15;"
    )
    con.close()
    os.chmod(path, 0o600)


def unreadable(path):
    os.chmod(path, 0)


def group_writable_dir(path):
    os.chmod(path.parent, 0o770)


CORRUPTIONS = {
    "truncated": (truncate, {"corrupt"}),
    "garbage_header": (garbage_header, {"corrupt"}),
    "zeroed_page": (zero_page, {"corrupt"}),
    "too_new": (set_versions(5000, 5000), {"incompatible_schema"}),
    "breaking_compat": (set_versions(1001, 1001), {"incompatible_schema"}),
    "upstream_format": (upstream_format, {"foreign_db"}),
    "unreadable": (unreadable, {"open_failed", "permission"}),
    "group_writable_dir": (group_writable_dir, {"permission"}),
}


@pytest.fixture
def recorded_small(tmp_path, shared_db):
    from tests.shared.helpers import GitRepo  # pylint: disable=import-outside-toplevel

    repo = GitRepo.init(tmp_path / "repo")
    assert_ran(run_pytest(repo.path, shared_db), ALL_SAMPLE_TESTS)
    repo.write(MUL_CHANGED)
    return repo


@not_root
@pytest.mark.parametrize("kind", sorted(CORRUPTIONS))
def test_unusable_database_runs_everything(kind, recorded_small, shared_db):
    corrupt, reasons = CORRUPTIONS[kind]
    corrupt(shared_db)
    before = shared_db.read_bytes() if os.access(shared_db, os.R_OK) else None
    run = run_pytest(recorded_small.path, shared_db, "--testmon-db-fallback=run-all")
    assert run.report["fallback"] is not None, run
    assert run.report["fallback"]["reason_code"] in reasons, run.report["fallback"]
    assert_ran(run, ALL_SAMPLE_TESTS)
    assert run.outcomes() == {name: "passed" for name in ALL_SAMPLE_TESTS}
    assert "testmon: unusable" in run.output
    # the shared file is never deleted, renamed or modified by pytest
    assert shared_db.exists()
    if before is not None:
        assert shared_db.read_bytes() == before


@not_root
@pytest.mark.parametrize("kind", sorted(CORRUPTIONS))
def test_unusable_database_with_error_policy(kind, recorded_small, shared_db):
    CORRUPTIONS[kind][0](shared_db)
    before = shared_db.read_bytes() if os.access(shared_db, os.R_OK) else None
    run = run_pytest(recorded_small.path, shared_db)  # error is the default policy
    assert run.returncode == 4, run
    assert "passed" not in run.output
    assert "database unavailable" in run.output and "No tests were run" in run.output
    assert run.report["fallback"]["policy"] == "error"
    assert run.report["fallback"]["reason_code"] in CORRUPTIONS[kind][1]
    assert run.report["tests"] == {}
    if before is not None:
        assert shared_db.read_bytes() == before


@not_root
def test_read_only_directory_never_skips_affected_tests(recorded_small, shared_db):
    os.chmod(shared_db.parent, 0o500)
    try:
        run = run_pytest(
            recorded_small.path, shared_db, "--testmon-db-fallback=run-all"
        )
    finally:
        os.chmod(shared_db.parent, 0o700)
    assert run.returncode == 0, run
    # either the fallback ran everything, or reading worked and the change was selected
    assert MUL <= run.selected
    if run.report["fallback"] is None:
        assert run.selected == MUL


def test_missing_database_is_an_error_until_initialized(tmp_path):
    from tests.shared.helpers import GitRepo, init_db  # pylint: disable=import-outside-toplevel

    repo = GitRepo.init(tmp_path / "repo")
    shared_db = tmp_path / "does" / "not" / "exist" / "db.sqlite"
    run = run_pytest(repo.path, shared_db)
    assert run.returncode == 4, run
    assert "passed" not in run.output
    assert run.report["fallback"]["reason_code"] == "missing"
    assert "testmon-admin" in run.output
    assert not shared_db.parent.exists()  # nothing created as a side effect
    # the explicit opt-in still runs everything, and still creates nothing
    run = run_pytest(repo.path, shared_db, "--testmon-db-fallback=run-all")
    assert_ran(run, ALL_SAMPLE_TESTS)
    assert not shared_db.exists()
    init_db(shared_db)
    assert oct(os.stat(shared_db.parent).st_mode & 0o777) == "0o700"
    assert_ran(run_pytest(repo.path, shared_db), ALL_SAMPLE_TESTS)
    assert_ran(run_pytest(repo.path, shared_db), set())


def test_restart_persistence(repo_many, shared_db):
    assert_ran(run_pytest(repo_many.path, shared_db), EVERY_TEST)
    # a "restart": only the files on disk carry over; checkpoint the WAL into the db file
    checkpoint = subprocess.run(
        [sys.executable, "-m", "testmon.admin", "--db", str(shared_db), "checkpoint"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert checkpoint.returncode == 0, checkpoint.stdout + checkpoint.stderr
    wal = Path(str(shared_db) + "-wal")
    assert not wal.exists() or wal.stat().st_size == 0
    assert_ran(run_pytest(repo_many.path, shared_db), set())
    assert (
        MANY_TESTS
        <= run_pytest(repo_many.path, shared_db, "--testmon-noselect").selected
    )


def test_xdist_workers_follow_the_controller_fallback(recorded_small, shared_db):
    pytest.importorskip("xdist")
    garbage_header(shared_db)
    run = run_pytest(
        recorded_small.path, shared_db, "-n", "2", "--testmon-db-fallback=run-all"
    )
    assert run.report["fallback"]["reason_code"] == "corrupt", run
    assert_ran(run, ALL_SAMPLE_TESTS)  # no worker deselected anything on its own
