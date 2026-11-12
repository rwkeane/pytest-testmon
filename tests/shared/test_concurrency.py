"""L4: several agents and xdist workers sharing one database at the same time."""

import pytest

from tests.shared.conftest import linux_only
from tests.shared.fixtures import (
    EVERY_TEST,
    LockHolder,
    assert_consistent,
    big_repo,
    python,
    query,
)
from tests.shared.helpers import assert_ran, collect, popen_pytest, run_pytest
from tests.shared.test_scenarios import (
    ADD,
    ADD_CHANGED,
    MUL,
    MUL_CHANGED,
    SHOUT,
    SHOUT_CHANGED,
)

pytestmark = linux_only
xdist = pytest.importorskip("xdist")

REPEAT = 5


@pytest.mark.parametrize("attempt", range(REPEAT))
def test_six_agents_three_branches(tmp_path, shared_db, attempt):  # pylint: disable=unused-argument
    repo = big_repo(tmp_path / "repo")
    for branch, change in (("b1", MUL_CHANGED), ("b2", SHOUT_CHANGED)):
        repo.checkout("main")
        repo.checkout(branch, create=True)
        repo.write(change)
        repo.commit(branch)
    repo.checkout("main")
    checkouts = []
    for branch in ("main", "b1", "b2"):
        for copy in range(2):
            checkouts.append(repo.worktree(tmp_path / f"wt-{branch}-{copy}", branch))
    barrier = tmp_path / "barrier"
    barrier.mkdir()
    env = {"BARRIER_DIR": str(barrier), "BARRIER_N": str(len(checkouts))}
    procs = [
        popen_pytest(co.path, shared_db, *(("-n", "2") if i % 2 else ()), env=env)
        for i, co in enumerate(checkouts)
    ]
    runs = [collect(proc, co.path) for proc, co in zip(procs, checkouts)]
    for run in runs:
        assert run.returncode == 0, run
        assert run.selected == EVERY_TEST  # nothing was recorded when they selected
        assert run.report["counts"]["dropped_batches"] == 0, run

    assert_consistent(shared_db)
    # nothing lost: every distinct (test, dependencies) pair of the three trees is stored
    (stored,) = query(shared_db, "SELECT count(*) FROM observation")[0]
    assert stored == len(EVERY_TEST) + 2  # test_mul and test_shout have 2 versions each
    # and each checkout now reuses everything
    for checkout in checkouts:
        assert_ran(run_pytest(checkout.path, shared_db), set())


WRITER = """
    import sys, time
    from testmon import db
    from testmon.process_code import checksums_to_blob
    path, seconds = sys.argv[1], float(sys.argv[2])
    database = db.DB(path, shared=True)
    env_ids = [row[0] for row in database.con.execute("SELECT id FROM environment")]
    deadline = time.time() + seconds
    n = 0
    while time.time() < deadline:
        for env_id in env_ids:
            dep = db.DepInput("noise.py", bytes(checksums_to_blob([n])), str(n))
            database.record_observations(
                env_id, [db.ObservationInput(f"noise.py::t{n}", (dep,), False, 0.1)]
            )
        n += 1
    database.close()
    print(n)
"""


def test_xdist_workers_agree_while_others_write(tmp_path, shared_db):
    repo = big_repo(tmp_path / "repo")
    assert_ran(run_pytest(repo.path, shared_db, "-n", "4"), EVERY_TEST)
    repo.write(ADD_CHANGED)
    writer = python(WRITER, shared_db, 20, wait=False)
    try:
        for _ in range(3):
            run = run_pytest(repo.path, shared_db, "-n", "4", "--testmon-nocollect")
            assert "Different tests were collected" not in run.output, run
            assert_ran(run, ADD | {name for name in EVERY_TEST if "test_many" in name})
    finally:
        out, err = writer.communicate(timeout=60)
    assert writer.returncode == 0, err
    assert int(out) > 0
    assert_consistent(shared_db)


@pytest.fixture
def recorded(tmp_path, shared_db):
    repo = big_repo(tmp_path / "repo", ini="[pytest]\ntestmon_db_timeout = 1\n")
    repo.commit("timeout")
    assert_ran(run_pytest(repo.path, shared_db), EVERY_TEST)
    return repo


def test_lock_at_selection_runs_everything(recorded, shared_db, tmp_path):
    recorded.write(MUL_CHANGED)
    with LockHolder(shared_db, tmp_path, "exclusive"):
        run = run_pytest(recorded.path, shared_db)
    assert run.report["fallback"]["reason_code"] == "lock_timeout", run
    assert_ran(run, EVERY_TEST)
    assert {run.reason(name) for name in EVERY_TEST} == {"fallback"}


def test_lock_at_selection_with_error_policy(recorded, shared_db, tmp_path):
    with LockHolder(shared_db, tmp_path, "exclusive"):
        run = run_pytest(recorded.path, shared_db, "--testmon-db-fallback=error")
    assert run.returncode == 4, run
    assert "passed" not in run.output
    assert "lock_timeout" in run.output


def test_lock_while_writing_drops_batches_only(recorded, shared_db, tmp_path):
    recorded.write(MUL_CHANGED)  # only test_mul depends on mul()
    with LockHolder(shared_db, tmp_path, "write"):
        run = run_pytest(recorded.path, shared_db)
    assert_ran(run, MUL)
    assert run.report["counts"]["dropped_batches"] >= 1
    assert any("could not store dependency data" in w for w in run.report["warnings"])
    assert run.outcomes() == {name: "passed" for name in MUL}
    # the lost observation is simply recollected
    run = run_pytest(recorded.path, shared_db)
    assert_ran(run, MUL)
    assert {run.reason(name) for name in MUL} == {"all_invalid"}
    assert_ran(run_pytest(recorded.path, shared_db), set())
    assert_consistent(shared_db)


INCREMENT = """
    import sys
    from testmon import db
    database = db.DB(sys.argv[1], shared=True)
    for _ in range(100):
        database.increment_attributes({"counter": 1})
    database.close()
"""


def test_concurrent_counter_increments_are_not_lost(recorded, shared_db):  # pylint: disable=unused-argument
    procs = [python(INCREMENT, shared_db, wait=False) for _ in range(4)]
    for proc in procs:
        _, err = proc.communicate(timeout=120)
        assert proc.returncode == 0, err
    assert query(
        shared_db, "SELECT value FROM metadata WHERE key = 'None:counter'"
    ) == [(400,)]


def test_mul_and_shout_constants_are_the_expected_sets():
    # guard for the observation-count arithmetic in test_six_agents_three_branches
    assert MUL == {"tests/test_calc.py::test_mul"}
    assert SHOUT == {"tests/test_text.py::test_shout"}
    assert len(ADD) == 2
