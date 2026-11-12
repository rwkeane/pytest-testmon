"""L4: long-running database lifecycle: GC racing writers, backups under load, growth."""

import json
import subprocess
import sys

from tests.shared.conftest import linux_only
from tests.shared.fixtures import assert_consistent, python, query
from tests.shared.helpers import ALL_SAMPLE_TESTS, GitRepo, assert_ran, run_pytest
from tests.shared.test_scenarios import ADD, ADD_CHANGED, MUL_CHANGED

pytestmark = linux_only


def admin(*args):
    result = subprocess.run(
        [sys.executable, "-m", "testmon.admin", *map(str, args)],
        capture_output=True,
        text=True,
        check=False,
    )
    return result


WRITER = """
    import json, sys, time
    from testmon import db
    from testmon.process_code import checksums_to_blob
    path, seconds = sys.argv[1], float(sys.argv[2])
    database = db.DB(path, shared=True)
    env_id, _ = database.fetch_or_create_environment("race", "p", "3")
    written = []
    deadline = time.time() + seconds
    n = 0
    while time.time() < deadline:
        deps = (
            db.DepInput("shared.py", bytes(checksums_to_blob([n % 7])), "s"),
            db.DepInput(f"own{n}.py", bytes(checksums_to_blob([n])), str(n)),
        )
        database.record_observations(env_id, [db.ObservationInput(f"t{n}", deps, False, 0.1)])
        written.append(f"t{n}")
        n += 1
    database.close()
    print(json.dumps(written))
"""

COLLECTOR = """
    import sys, time
    from testmon import db
    path, seconds = sys.argv[1], float(sys.argv[2])
    database = db.DB(path, shared=True)
    deadline = time.time() + seconds
    rounds = 0
    while time.time() < deadline:
        database.gc(aggressive=True)
        rounds += 1
    database.close()
    print(rounds)
"""


def test_gc_racing_a_writer_loses_nothing_recent(tmp_path, shared_db):
    GitRepo.init(tmp_path / "repo")
    from testmon.db import DB  # pylint: disable=import-outside-toplevel

    DB(str(shared_db), shared=True).close()
    writer = python(WRITER, shared_db, 8, wait=False)
    collector = python(COLLECTOR, shared_db, 8, wait=False)
    written, werr = writer.communicate(timeout=120)
    rounds, cerr = collector.communicate(timeout=120)
    assert writer.returncode == 0, werr
    assert collector.returncode == 0, cerr
    assert int(rounds) > 0
    written = json.loads(written)
    assert len(written) > 10
    stored = {row[0] for row in query(shared_db, "SELECT test_name FROM observation")}
    assert stored == set(written)
    assert_consistent(shared_db)


def test_backup_under_load_and_restore(tmp_path, shared_db):
    repo = GitRepo.init(tmp_path / "repo")
    assert_ran(run_pytest(repo.path, shared_db), ALL_SAMPLE_TESTS)
    writer = python(WRITER, shared_db, 4, wait=False)
    backup = tmp_path / "backup.sqlite"
    try:
        result = admin("--db", shared_db, "backup", backup)
    finally:
        _, err = writer.communicate(timeout=60)
    assert writer.returncode == 0, err
    assert result.returncode == 0, result.stdout + result.stderr
    assert admin("--db", backup, "check").returncode == 0

    repo.write(ADD_CHANGED)
    original = run_pytest(repo.path, shared_db, "--testmon-nocollect")
    restored_db = tmp_path / "restored" / "db.sqlite"
    restored_db.parent.mkdir(mode=0o700)
    restored_db.write_bytes(b"")
    restored_db.chmod(0o600)
    result = admin("--db", restored_db, "restore", backup)
    assert result.returncode == 0, result.stdout + result.stderr
    restored = run_pytest(repo.path, restored_db, "--testmon-nocollect")
    assert restored.selected == original.selected == ADD
    assert restored.deselected == original.deselected


def test_restore_over_a_corrupt_file_needs_force(tmp_path, shared_db):
    repo = GitRepo.init(tmp_path / "repo")
    assert_ran(run_pytest(repo.path, shared_db), ALL_SAMPLE_TESTS)
    backup = tmp_path / "backup.sqlite"
    assert admin("--db", shared_db, "backup", backup).returncode == 0
    with open(shared_db, "r+b") as file:
        file.write(b"garbage garbage garbage")
    assert admin("--db", shared_db, "check").returncode != 0
    assert admin("--db", shared_db, "restore", backup).returncode != 0
    result = admin("--db", shared_db, "restore", backup, "--force")
    assert result.returncode == 0, result.stdout + result.stderr
    assert admin("--db", shared_db, "check").returncode == 0
    assert_ran(run_pytest(repo.path, shared_db), set())


def test_repeated_runs_do_not_grow_the_database(tmp_path, shared_db):
    repo = GitRepo.init(tmp_path / "repo")
    run_pytest(repo.path, shared_db, "--testmon-noselect")

    def counts():
        data = json.loads(admin("--db", shared_db, "--json", "stats").stdout)["counts"]
        data.pop("session")
        return data

    baseline = counts()
    for _ in range(5):
        run_pytest(repo.path, shared_db, "--testmon-noselect")
    assert counts() == baseline
    # two alternating versions: exactly one more observation (test_mul's), not one per run
    for index in range(4):
        repo.write(MUL_CHANGED if index % 2 == 0 else {"pkg/calc.py": repo_calc()})
        run_pytest(repo.path, shared_db, "--testmon-noselect")
    after = counts()
    assert after["observation"] == baseline["observation"] + 1


def repo_calc():
    from tests.shared.helpers import SAMPLE  # pylint: disable=import-outside-toplevel

    return SAMPLE["pkg/calc.py"]


def test_explain(tmp_path, shared_db):
    repo = GitRepo.init(tmp_path / "repo")
    run_pytest(repo.path, shared_db)
    repo.write(ADD_CHANGED)
    result = admin(
        "--db",
        shared_db,
        "--rootdir",
        repo.path,
        "--json",
        "explain",
        "tests/test_calc.py::test_add",
    )
    assert result.returncode == 0, result.stderr
    data = json.loads(result.stdout)
    (obs,) = data["observations"]
    assert not obs["valid"]
    statuses = {dep["filename"]: dep["status"] for dep in obs["deps"]}
    assert statuses["pkg/calc.py"] == "mismatch"
    assert statuses["tests/test_calc.py"] == "valid"
    assert [d["deselect"] for d in data["decisions"].values()] == [False]
    text = admin(
        "--db",
        shared_db,
        "--rootdir",
        repo.path,
        "explain",
        "tests/test_calc.py::test_mul",
    )
    assert "deselect" in text.stdout and "valid" in text.stdout
