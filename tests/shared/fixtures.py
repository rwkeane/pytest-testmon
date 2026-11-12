"""Building blocks for the concurrency / fault / lifecycle tests."""

import os
import sqlite3
import subprocess
import sys
import textwrap
import time
from pathlib import Path

from tests.shared.helpers import ALL_SAMPLE_TESTS, GitRepo, clean_env

MANY = 300  # > TEST_BATCH_SIZE: a run writes several batches
MANY_TESTS = {f"tests/test_many.py::test_many[{i}]" for i in range(MANY)}
EVERY_TEST = ALL_SAMPLE_TESTS | MANY_TESTS

BARRIER_CONFTEST = """
    import os
    import time


    def pytest_sessionstart(session):
        barrier = os.environ.get("BARRIER_DIR")
        if not barrier or hasattr(session.config, "workerinput"):
            return
        open(os.path.join(barrier, str(os.getpid())), "w").close()
        wanted = int(os.environ["BARRIER_N"])
        deadline = time.time() + 120
        while len(os.listdir(barrier)) < wanted and time.time() < deadline:
            time.sleep(0.01)
    """

MANY_FILE = f"""
    import pytest

    from pkg.calc import add


    @pytest.mark.parametrize("i", range({MANY}))
    def test_many(i):
        assert add(i, 0) == i
    """


def big_repo(path, ini="[pytest]\n"):
    repo = GitRepo.init(path)
    repo.write(
        {
            "conftest.py": BARRIER_CONFTEST,
            "tests/test_many.py": MANY_FILE,
            "pytest.ini": ini,
        }
    )
    repo.commit("many tests")
    return repo


def connect(shared_db):
    return sqlite3.connect(str(shared_db), timeout=60)


def query(shared_db, sql, params=()):
    con = connect(shared_db)
    try:
        return con.execute(sql, params).fetchall()
    finally:
        con.close()


def assert_consistent(shared_db):
    """integrity_check ok and no observation without exactly dep_count dependencies."""
    assert query(shared_db, "PRAGMA integrity_check") == [("ok",)]
    broken = query(
        shared_db,
        "SELECT o.id FROM observation o LEFT JOIN observation_dep od "
        "ON od.observation_id = o.id GROUP BY o.id HAVING count(od.fp_id) != o.dep_count",
    )
    assert broken == []


def python(code, *args, env=None, wait=True):
    command = [sys.executable, "-c", textwrap.dedent(code), *map(str, args)]
    if not wait:
        return subprocess.Popen(
            command,
            env=clean_env(env),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
    result = subprocess.run(
        command,
        env=clean_env(env),
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout


LOCK_HOLDER = """
    import os, sqlite3, sys, time
    path, mode, ready, release = sys.argv[1:5]
    con = sqlite3.connect(path, isolation_level=None, timeout=60)
    if mode == "exclusive":
        con.execute("PRAGMA locking_mode = EXCLUSIVE")
        con.execute("BEGIN EXCLUSIVE")
    else:
        con.execute("BEGIN IMMEDIATE")
    con.execute(
        "INSERT INTO metadata (key, value) VALUES ('holder', 1) "
        "ON CONFLICT(key) DO UPDATE SET value = value + 1"
    )
    open(ready, "w").close()
    while not os.path.exists(release):
        time.sleep(0.02)
    con.execute("ROLLBACK")
    con.close()
"""


class LockHolder:
    """Another process holding the database lock until released (a stuck agent)."""

    def __init__(self, shared_db, tmp_path, mode):
        self.ready = Path(tmp_path) / f"ready-{mode}"
        self.release_file = Path(tmp_path) / f"release-{mode}"
        self.proc = python(
            LOCK_HOLDER, shared_db, mode, self.ready, self.release_file, wait=False
        )
        deadline = time.time() + 30
        while not self.ready.exists():
            assert self.proc.poll() is None, self.proc.communicate()
            assert time.time() < deadline, "lock holder did not start"
            time.sleep(0.02)

    def release(self):
        self.release_file.touch()
        out, err = self.proc.communicate(timeout=30)
        assert self.proc.returncode == 0, out + err

    def __enter__(self):
        return self

    def __exit__(self, *args):
        if self.proc.poll() is None:
            self.release()
