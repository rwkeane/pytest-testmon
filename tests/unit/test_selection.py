import os
import tempfile

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from testmon import db as dbmod
from testmon import selection
from testmon.process_code import match_fingerprint
from testmon.selection import (
    ALL_INVALID,
    DESELECT,
    INCOMPLETE,
    MISMATCH,
    MISSING,
    VALID_FAILING,
    InvalidDep,
    compute_selection,
    conftest_chain,
    context_hash,
)
from tests.unit.dbhelpers import dep, fingerprint, module, new_db, observation, tree


@pytest.fixture
def database(tmp_path):
    database, env_id = new_db(tmp_path / "db.sqlite")
    database.env_id = env_id
    yield database
    database.close()


def select(database, files, **kwargs):
    return compute_selection(database, database.env_id, tree(files), **kwargs)


V1 = (1, 2)
V2 = (1, 3)  # g1 changed
V3 = (4, 2)  # g0 changed


class TestDecisionTable:
    def test_no_observation_means_no_decision(self, database):
        result = select(database, {"a.py": V1})
        assert result.decisions == {}

    def test_valid_passing_observation_deselects(self, database):
        database.record_observations(
            database.env_id, [observation("t", [dep("a.py", V1, [0])])]
        )
        result = select(database, {"a.py": V1})
        decision = result.decisions["t"]
        assert decision.deselect and decision.reason == DESELECT
        assert result.used_observation_ids == [decision.observation_id]

    def test_change_outside_the_executed_blocks_keeps_it_valid(self, database):
        database.record_observations(
            database.env_id, [observation("t", [dep("a.py", V1, [0])])]
        )
        assert select(database, {"a.py": V2}).decisions["t"].deselect

    def test_mismatch(self, database):
        database.record_observations(
            database.env_id, [observation("t", [dep("a.py", V1, [0])])]
        )
        decision = select(database, {"a.py": V3}).decisions["t"]
        assert not decision.deselect
        assert decision.reason == ALL_INVALID
        assert decision.invalid_dep == InvalidDep("a.py", MISMATCH)

    def test_missing_file(self, database):
        database.record_observations(
            database.env_id,
            [observation("t", [dep("a.py", V1, [0]), dep("b.py", V1, [1])])],
        )
        decision = select(database, {"a.py": V1, "b.py": None}).decisions["t"]
        assert decision.reason == ALL_INVALID
        assert decision.invalid_dep == InvalidDep("b.py", MISSING)

    def test_valid_failing_beats_valid_passing(self, database):
        database.record_observations(
            database.env_id,
            [
                observation("t", [dep("a.py", V1, [0])], failed=False),
                observation("t", [dep("a.py", V1, [0, 1])], failed=True),
            ],
        )
        decision = select(database, {"a.py": V1}).decisions["t"]
        assert not decision.deselect
        assert decision.reason == VALID_FAILING

    def test_invalid_failing_observation_does_not_force_a_run(self, database):
        database.record_observations(
            database.env_id,
            [
                observation("t", [dep("a.py", V1, [0])], failed=False),
                observation("t", [dep("a.py", V2, [1])], failed=True),  # other branch
            ],
        )
        assert select(database, {"a.py": V1}).decisions["t"].deselect

    def test_superseded_failing_observation_is_ignored(self, database):
        database.record_observations(
            database.env_id,
            [
                observation("t", [dep("a.py", V1, [0])], failed=False),
                observation("t", [dep("a.py", V1, [0, 1])], failed=True),
            ],
        )
        result = select(database, {"a.py": V1})
        (failing_id,) = result.valid_failing["t"]
        # the test passes again in this state: the failure is superseded
        database.record_observations(
            database.env_id,
            [observation("t", [dep("a.py", V1, [0])], failed=False)],
            supersede=result.valid_failing,
        )
        assert database.con.execute(
            "SELECT superseded FROM observation WHERE id = ?", (failing_id,)
        ).fetchone()[0]
        assert select(database, {"a.py": V1}).decisions["t"].deselect

    def test_abs_dependency(self, database, tmp_path):
        outside = tmp_path / "outside.py"
        outside.write_text("def g0():\n    return 1\n\n")
        name = selection.ABS_PREFIX + str(outside)
        database.record_observations(
            database.env_id,
            [observation("t", [dep(name, (1,), [0]), dep("a.py", V1, [0])])],
        )
        source_tree = tree({"a.py": V1})
        result = compute_selection(database, database.env_id, source_tree)
        assert result.decisions["t"].deselect  # read through the abs: path from disk
        outside.write_text("def g0():\n    return 2\n\n")
        source_tree = tree({"a.py": V1})
        decision = compute_selection(database, database.env_id, source_tree).decisions[
            "t"
        ]
        assert decision.invalid_dep == InvalidDep(name, MISMATCH)

    def test_wrong_dep_count_is_never_valid(self, database):
        database.record_observations(
            database.env_id, [observation("t", [dep("a.py", V1, [0])])]
        )
        database.con.execute("UPDATE observation SET dep_count = 2")
        decision = select(database, {"a.py": V1}).decisions["t"]
        assert not decision.deselect
        assert decision.invalid_dep == InvalidDep("", INCOMPLETE)
        database.con.execute("UPDATE observation SET dep_count = 1")
        database.con.execute(
            "INSERT INTO dep_fp (filename, checksums) VALUES ('extra.py', x'01000000')"
        )
        database.con.execute(
            "INSERT INTO observation_dep SELECT o.id, f.id FROM observation o, dep_fp f "
            "WHERE f.filename = 'extra.py'"
        )
        # more dependency rows than dep_count: also never valid
        assert (
            not select(database, {"a.py": V1, "extra.py": (1,)}).decisions["t"].deselect
        )

    def test_memo_of_another_parser_is_not_used(self, database):
        database.record_observations(
            database.env_id, [observation("t", [dep("a.py", V1, [0])])]
        )
        # poison the memo for another parser key: must be ignored
        database.con.execute("UPDATE fp_match SET valid = 0, parser_key = 'other:3.0'")
        result = select(database, {"a.py": V1})
        assert result.decisions["t"].deselect
        assert len(result.new_memo) == 1  # recomputed for this parser key

    def test_memo_hit_avoids_reanalysis(self, database):
        database.record_observations(
            database.env_id, [observation("t", [dep("a.py", V1, [0])])]
        )
        first = select(database, {"a.py": V2})
        assert [entry[1:] for entry in first.new_memo] == [(module(V2).fs_fsha, 1)]
        database.record_selection_side_effects(
            first.new_memo, first.used_observation_ids
        )
        second = select(database, {"a.py": V2})
        assert second.new_memo == []
        assert second.decisions == first.decisions

    def test_inconsistent_module_is_not_memoised(self, database):
        database.record_observations(
            database.env_id, [observation("t", [dep("a.py", V1, [0])])]
        )
        source_tree = tree({"a.py": V2})
        source_tree.cache[
            "a.py"
        ].source_fsha = "something else"  # changed while reading
        result = compute_selection(database, database.env_id, source_tree)
        assert not result.decisions["t"].deselect
        assert result.new_memo == []


class TestFileLevel:
    def test_conftest_chain(self):
        assert conftest_chain("test_a.py") == ["conftest.py"]
        assert conftest_chain("a/b/test_x.py") == [
            "conftest.py",
            "a/conftest.py",
            "a/b/conftest.py",
        ]

    def test_context_hash(self):
        files = {
            "tests/test_x.py": V1,
            "tests/conftest.py": V1,
            "other/conftest.py": V1,
        }
        base = context_hash(tree(files), "tests/test_x.py")
        assert base == context_hash(tree(files), "tests/test_x.py")
        assert base != context_hash(
            tree({**files, "tests/test_x.py": V2}), "tests/test_x.py"
        )
        assert base != context_hash(
            tree({**files, "tests/conftest.py": V2}), "tests/test_x.py"
        )
        assert base != context_hash(
            tree({**files, "conftest.py": V2}), "tests/test_x.py"
        )
        assert base == context_hash(
            tree({**files, "other/conftest.py": V2}), "tests/test_x.py"
        )
        assert base != context_hash(
            tree({**files, "pytest.ini": V1}), "tests/test_x.py", ["pytest.ini"]
        )
        assert context_hash(tree({"tests/test_x.py": None}), "tests/test_x.py") is None

    def test_ignorable_only_when_every_snapshot_test_is_deselectable(self, database):
        files = {"test_x.py": V1}
        database.record_observations(
            database.env_id,
            [
                observation("test_x.py::a", [dep("test_x.py", V1, [0])]),
                observation("test_x.py::b", [dep("test_x.py", V1, [1])]),
            ],
        )
        ctx = context_hash(tree(files), "test_x.py")
        database.record_collection_snapshots(
            database.env_id, [("test_x.py", ctx, ["test_x.py::a", "test_x.py::b"])]
        )
        assert select(database, files).ignorable_files == {
            "test_x.py": ["test_x.py::a", "test_x.py::b"]
        }
        # a test in the snapshot without observation: the file must be collected
        database.record_collection_snapshots(
            database.env_id, [("test_x.py", ctx, ["test_x.py::a", "test_x.py::new"])]
        )
        assert select(database, files).ignorable_files == {}
        assert select(database, files, use_snapshots=False).ignorable_files == {}


# ------------------------------------------------------------------------------------------
# Property test: the SQL engine agrees with a naive implementation of the correctness rule.
# ------------------------------------------------------------------------------------------

FILES = ("a.py", "b.py", "c.py")
TESTS = ("t0", "t1", "t2")
versions = st.tuples(st.integers(0, 2), st.integers(0, 2))
dependency = st.tuples(versions, st.sets(st.integers(0, 1), min_size=1))
observations = st.lists(
    st.tuples(
        st.sampled_from(TESTS),
        st.dictionaries(st.sampled_from(FILES), dependency, min_size=1, max_size=3),
        st.booleans(),  # failed
        st.booleans(),  # superseded (applied to the stored row)
    ),
    max_size=10,
)
current_state = st.fixed_dictionaries(
    {name: st.one_of(st.none(), versions) for name in FILES}
)


def reference_decisions(generated, current):
    stored = {}
    for test_name, deps, failed, superseded in generated:
        key = (
            test_name,
            frozenset((f, tuple(fingerprint(v, fs))) for f, (v, fs) in deps.items()),
        )
        stored[key] = (failed, superseded)  # identical observations upsert: last wins
    by_test = {}
    for (test_name, deps), (failed, superseded) in stored.items():
        valid = all(
            current[f] is not None and match_fingerprint(module(current[f]), list(fp))
            for f, fp in deps
        )
        by_test.setdefault(test_name, []).append((valid, failed, superseded))
    decisions = {}
    for test_name, rows in by_test.items():
        live_valid = [r for r in rows if r[0] and not r[2]]
        if any(r[1] for r in live_valid):
            decisions[test_name] = (False, VALID_FAILING)
        elif live_valid:
            decisions[test_name] = (True, DESELECT)
        else:
            decisions[test_name] = (False, ALL_INVALID)
    return decisions


@settings(max_examples=150, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(generated=observations, current=current_state)
def test_engine_matches_reference(generated, current):
    with tempfile.TemporaryDirectory() as tmp:
        database, env_id = new_db(os.path.join(tmp, "db.sqlite"))
        try:
            for test_name, deps, failed, superseded in generated:
                obs = observation(
                    test_name, [dep(f, v, fs) for f, (v, fs) in deps.items()], failed
                )
                database.record_observations(env_id, [obs])
                digest = dbmod.deps_hash([(d.filename, d.checksums) for d in obs.deps])
                database.con.execute(
                    "UPDATE observation SET superseded = ? WHERE test_name = ? AND deps_hash = ?",
                    (int(superseded), test_name, digest),
                )
            expected = reference_decisions(generated, current)
            first = compute_selection(database, env_id, tree(current))
            got = {name: (d.deselect, d.reason) for name, d in first.decisions.items()}
            assert got == expected
            # with the memo filled by the first selection: same answers, no reanalysis
            database.record_selection_side_effects(
                first.new_memo, first.used_observation_ids
            )
            second = compute_selection(database, env_id, tree(current))
            assert {
                n: (d.deselect, d.reason) for n, d in second.decisions.items()
            } == expected
            assert second.new_memo == []
        finally:
            database.close()
