"""L3: git scenarios across checkouts, branches and history rewrites (see the test plan)."""

import os
import sqlite3

import pytest

from tests.shared.helpers import (
    ALL_SAMPLE_TESTS,
    assert_ran,
    copy_without_git,
    observation_total,
    run_pytest,
)

ADD = {"tests/test_calc.py::test_add", "tests/test_calc.py::test_add_fixture"}
MUL = {"tests/test_calc.py::test_mul"}
SHOUT = {"tests/test_text.py::test_shout"}

MUL_CHANGED = {
    "pkg/calc.py": """
        def add(a, b):
            return a + b


        def mul(a, b):
            return b * a
        """
}
ADD_CHANGED = {
    "pkg/calc.py": """
        def add(a, b):
            return b + a


        def mul(a, b):
            return a * b
        """
}
SHOUT_CHANGED = {
    "pkg/text.py": """
        def shout(s):
            return str(s).upper()
        """
}


def observation_count(shared_db, test_name):
    con = sqlite3.connect(shared_db)
    try:
        return con.execute(
            "SELECT count(*) FROM observation WHERE test_name = ?", (test_name,)
        ).fetchone()[0]
    finally:
        con.close()


class TestCheckoutIndependence:
    def test_reuse_across_clone_worktree_and_symlink(self, repo, shared_db, tmp_path):
        assert_ran(run_pytest(repo.path, shared_db), ALL_SAMPLE_TESTS)

        clone = repo.clone(tmp_path / "elsewhere" / "clone")
        run = run_pytest(clone.path, shared_db)
        assert_ran(run, set())
        assert run.deselected == ALL_SAMPLE_TESTS
        assert {run.reason(name) for name in ALL_SAMPLE_TESTS} == {"deselect"}
        assert all(run.tests[name]["observation_id"] for name in ALL_SAMPLE_TESTS)

        worktree = repo.worktree(tmp_path / "wt" / "feature", "HEAD", branch="feature")
        assert_ran(run_pytest(worktree.path, shared_db), set())

        link = tmp_path / "link-to-clone"
        os.symlink(clone.path, link)
        run = run_pytest(link, shared_db)
        assert_ran(run, set())

        # and a change in the clone selects only what it affects, recorded for everyone
        clone.write(MUL_CHANGED)
        assert_ran(run_pytest(clone.path, shared_db), MUL)
        worktree.write(MUL_CHANGED)
        assert_ran(run_pytest(worktree.path, shared_db), set())

    def test_non_git_directory_untracked_and_staged_files(
        self, repo, shared_db, tmp_path
    ):
        assert_ran(run_pytest(repo.path, shared_db), ALL_SAMPLE_TESTS)
        plain = copy_without_git(repo.path, tmp_path / "plain")
        assert_ran(run_pytest(plain, shared_db), set())

        repo.write({"tests/test_new.py": "def test_new():\n    pass\n"})  # untracked
        run = run_pytest(repo.path, shared_db)
        assert_ran(run, {"tests/test_new.py::test_new"})
        assert run.reason("tests/test_new.py::test_new") == "no_observation"

        repo.write(ADD_CHANGED)
        repo.git("add", "pkg/calc.py")  # staged, not committed
        assert_ran(run_pytest(repo.path, shared_db), ADD)


class TestBranches:
    def test_switching_back_and_forth(self, repo, shared_db):
        assert_ran(run_pytest(repo.path, shared_db), ALL_SAMPLE_TESTS)
        repo.checkout("feat-calc", create=True)
        repo.write(MUL_CHANGED)
        repo.commit("mul")
        repo.checkout("main")
        repo.checkout("feat-text", create=True)
        repo.write(SHOUT_CHANGED)
        repo.commit("shout")

        repo.checkout("feat-calc")
        assert_ran(run_pytest(repo.path, shared_db), MUL)
        repo.checkout("main")
        assert_ran(run_pytest(repo.path, shared_db), set())  # main's records survived
        repo.checkout("feat-calc")
        assert_ran(run_pytest(repo.path, shared_db), set())  # and so did feat-calc's
        repo.checkout("feat-text")
        assert_ran(run_pytest(repo.path, shared_db), SHOUT)
        repo.checkout("main")
        assert_ran(run_pytest(repo.path, shared_db), set())
        repo.checkout("feat-text")
        assert_ran(run_pytest(repo.path, shared_db), set())

    def test_multiple_versions_of_one_test(self, repo, shared_db):
        bodies = ["a * b", "b * a", "sum([a] * b)"]
        for index, body in enumerate(bodies):
            repo.checkout("main")
            repo.checkout(f"v{index}", create=True)
            repo.write(
                {
                    "pkg/calc.py": f"""
                        def add(a, b):
                            return a + b


                        def mul(a, b):
                            return {body}
                        """
                }
            )
            repo.commit(f"mul {index}")
            run_pytest(repo.path, shared_db)
        assert observation_count(shared_db, "tests/test_calc.py::test_mul") == 3
        for index in (0, 1, 2, 1, 0):
            repo.checkout(f"v{index}")
            assert_ran(run_pytest(repo.path, shared_db), set())


class TestHistoryRewrites:
    def feature_branch(self, repo, shared_db):
        assert_ran(run_pytest(repo.path, shared_db), ALL_SAMPLE_TESTS)
        repo.checkout("feature", create=True)
        repo.write(MUL_CHANGED)
        sha = repo.commit("feature: mul")
        assert_ran(run_pytest(repo.path, shared_db), MUL)
        return sha

    def test_cherry_pick(self, repo, shared_db):
        sha = self.feature_branch(repo, shared_db)
        repo.checkout("main")
        # an unrelated commit on main: the pick gets a new parent, hence always a new SHA
        repo.write({"NOTES.txt": "main moved on\n"})
        repo.commit("main: notes")
        repo.git("cherry-pick", sha)
        assert repo.head() != sha
        assert_ran(run_pytest(repo.path, shared_db), set())

    def test_rebase_combines_records_of_both_sides(self, repo, shared_db):
        self.feature_branch(repo, shared_db)
        repo.checkout("main")
        repo.write(SHOUT_CHANGED)
        repo.commit("main: shout")
        assert_ran(run_pytest(repo.path, shared_db), SHOUT)
        repo.checkout("feature")
        repo.git("rebase", "-q", "main")
        # a tree nobody ran, but every test's dependencies were seen on one side or the other
        assert_ran(run_pytest(repo.path, shared_db), set())

    def test_rebase_onto_an_untested_helper_change(self, repo, shared_db):
        self.feature_branch(repo, shared_db)
        repo.checkout("main")
        repo.write(
            {
                "pkg/calc.py": """
                    def add(a, b):
                        return b + a


                    def mul(a, b):
                        return a * b
                    """
            }
        )
        repo.commit("main: add (never tested)")
        repo.checkout("feature")
        repo.git("rebase", "-q", "main")
        run = run_pytest(repo.path, shared_db)
        assert_ran(run, ADD)
        for name in ADD:
            assert run.tests[name]["invalid_dep"] == {
                "filename": "pkg/calc.py",
                "cause": "mismatch",
            }

    def test_squash_merge(self, repo, shared_db):
        self.feature_branch(repo, shared_db)
        repo.write(SHOUT_CHANGED)
        repo.commit("feature: shout")
        assert_ran(run_pytest(repo.path, shared_db), SHOUT)
        feature_tree = repo.tree()
        repo.checkout("main")
        repo.git("merge", "--squash", "feature")
        repo.commit("squashed")
        assert repo.tree() == feature_tree
        assert_ran(run_pytest(repo.path, shared_db), set())


class TestTestChanges:
    def test_new_modified_deleted_renamed(self, repo, shared_db):
        assert_ran(run_pytest(repo.path, shared_db), ALL_SAMPLE_TESTS)
        repo.write(
            {
                "tests/test_new.py": "def test_new():\n    pass\n",
                "tests/test_text.py": """
                    from pkg.text import shout


                    def test_shout_renamed():
                        assert shout("a") == "A"
                    """,
                "tests/test_calc.py": """
                    from pkg.calc import add, mul


                    def test_add():
                        assert add(1, 2) == 3 and add(0, 0) == 0


                    def test_add_fixture(base):
                        assert add(base, 1) == 11
                    """,
            }
        )
        run = run_pytest(repo.path, shared_db)
        assert run.reason("tests/test_new.py::test_new") == "no_observation"
        assert run.reason("tests/test_text.py::test_shout_renamed") == "no_observation"
        assert run.reason("tests/test_calc.py::test_add") == "all_invalid"
        assert "tests/test_calc.py::test_mul" not in run.tests  # deleted
        # removing a function changed test_calc.py's module block: its tests re-run
        assert run.reason("tests/test_calc.py::test_add_fixture") == "all_invalid"
        assert run.returncode == 0
        # the old names' records are kept (other branches still have those tests)
        assert observation_count(shared_db, "tests/test_text.py::test_shout") == 1
        assert observation_count(shared_db, "tests/test_calc.py::test_mul") == 1


class TestDependencyChanges:
    def test_helper_body(self, repo, shared_db):
        assert_ran(run_pytest(repo.path, shared_db), ALL_SAMPLE_TESTS)
        repo.write(ADD_CHANGED)
        run = run_pytest(repo.path, shared_db)
        assert_ran(run, ADD)
        assert run.tests["tests/test_calc.py::test_mul"]["file_ignored"] is False

    def test_module_level_code(self, repo, shared_db):
        assert_ran(run_pytest(repo.path, shared_db), ALL_SAMPLE_TESTS)
        repo.write(
            {
                "pkg/calc.py": """
                    PRECISION = 2


                    def add(a, b):
                        return a + b


                    def mul(a, b):
                        return a * b
                    """
            }
        )
        assert_ran(run_pytest(repo.path, shared_db), ADD | MUL)

    def test_conftest_fixture(self, repo, shared_db):
        assert_ran(run_pytest(repo.path, shared_db), ALL_SAMPLE_TESTS)
        repo.write(
            {
                "tests/conftest.py": """
                    import pytest


                    @pytest.fixture
                    def base():
                        return 9 + 1
                    """
            }
        )
        run = run_pytest(repo.path, shared_db)
        assert_ran(run, {"tests/test_calc.py::test_add_fixture"})
        assert run.tests["tests/test_calc.py::test_add_fixture"]["invalid_dep"] == {
            "filename": "tests/conftest.py",
            "cause": "mismatch",
        }
        # the conftest is part of every test file's collection context: nothing file-ignored
        assert run.report["ignored_files"] == []

    def test_conftest_parametrization(self, repo, shared_db):
        def conftest(values):
            return {
                "tests/conftest.py": f"""
                    import pytest


                    @pytest.fixture
                    def base():
                        return 10


                    def pytest_generate_tests(metafunc):
                        if "n" in metafunc.fixturenames:
                            metafunc.parametrize("n", {values!r})
                    """,
                "tests/test_param.py": "def test_n(n):\n    assert n > 0\n",
            }

        repo.write(conftest([1, 2]))
        assert_ran(
            run_pytest(repo.path, shared_db),
            ALL_SAMPLE_TESTS
            | {"tests/test_param.py::test_n[1]", "tests/test_param.py::test_n[2]"},
        )
        assert_ran(run_pytest(repo.path, shared_db), set())
        repo.write(conftest([1, 2, 3]))
        run = run_pytest(repo.path, shared_db)
        # the new parameter is collected even though test_param.py itself did not change
        assert_ran(run, {"tests/test_param.py::test_n[3]"})
        assert run.reason("tests/test_param.py::test_n[3]") == "no_observation"

    def test_failing_then_fixed(self, repo, shared_db):
        repo.write(
            {
                "pkg/boom.py": "def boom():\n    raise AssertionError('boom')\n",
                "tests/test_calc.py": """
                    import os

                    from pkg.boom import boom
                    from pkg.calc import add, mul


                    def test_add():
                        assert add(1, 2) == 3


                    def test_mul():
                        if os.environ.get("FAIL_MUL"):
                            boom()
                        assert mul(2, 3) == 6


                    def test_add_fixture(base):
                        assert add(base, 1) == 11
                    """,
            }
        )
        run_pytest(repo.path, shared_db)
        # environment variables are not tracked inputs (R7: agents choose flags), so a
        # selective run deselects test_mul; a full collecting run records the failure
        assert_ran(run_pytest(repo.path, shared_db, env={"FAIL_MUL": "1"}), set())
        failing = run_pytest(
            repo.path, shared_db, "--testmon-noselect", env={"FAIL_MUL": "1"}
        )
        assert failing.failed() == MUL
        run = run_pytest(repo.path, shared_db)
        # a valid failing observation forces the test to run (and it passes again)
        assert_ran(run, MUL)
        assert run.reason("tests/test_calc.py::test_mul") == "valid_failing_observation"
        # the pass superseded the failure
        assert_ran(run_pytest(repo.path, shared_db), set())


class TestEnvironments:
    def fake_dist(self, site, version):
        info = site / f"fakepkg-{version}.dist-info"
        info.mkdir(parents=True)
        (info / "METADATA").write_text(
            f"Metadata-Version: 2.1\nName: fakepkg\nVersion: {version}\n"
        )

    def test_package_change_and_revert(self, repo, shared_db, tmp_path):
        assert_ran(run_pytest(repo.path, shared_db), ALL_SAMPLE_TESTS)
        site = tmp_path / "site"
        self.fake_dist(site, "1.0")
        run = run_pytest(repo.path, shared_db, env={"PYTHONPATH": str(site)})
        assert_ran(run, ALL_SAMPLE_TESTS)
        assert run.report["environment"]["packages_changed"]
        assert_ran(
            run_pytest(repo.path, shared_db), set()
        )  # back: the old env is reused
        assert_ran(
            run_pytest(repo.path, shared_db, env={"PYTHONPATH": str(site)}), set()
        )

    def test_testmon_env_separates(self, repo, shared_db):
        assert_ran(run_pytest(repo.path, shared_db), ALL_SAMPLE_TESTS)
        assert_ran(
            run_pytest(repo.path, shared_db, "--testmon-env=gpu"), ALL_SAMPLE_TESTS
        )
        assert_ran(run_pytest(repo.path, shared_db, "--testmon-env=gpu"), set())
        assert_ran(run_pytest(repo.path, shared_db), set())


class TestForeignCheckout:
    @pytest.fixture
    def src_repo(self, tmp_path):
        from tests.shared.helpers import SAMPLE, GitRepo  # pylint: disable=import-outside-toplevel

        files = {
            ("src/" + name if name.startswith("pkg/") else name): content
            for name, content in SAMPLE.items()
        }
        files["pytest.ini"] = "[pytest]\n"
        return GitRepo.init(tmp_path / "a", files)

    def test_editable_install_from_another_checkout(
        self, src_repo, shared_db, tmp_path
    ):
        other = src_repo.clone(tmp_path / "b")
        # like an editable install of checkout a: its src/ is on the path when running in b
        foreign = {"PYTHONPATH": str(src_repo.path / "src")}
        run = run_pytest(other.path, shared_db, env=foreign)
        assert run.returncode == 4, run  # default: refuse, run nothing
        assert run.report["fallback"]["reason_code"] == "foreign_checkout"
        assert "passed" not in run.output and "another checkout" in run.output
        run = run_pytest(
            other.path, shared_db, "--testmon-db-fallback=run-all", env=foreign
        )
        assert run.report["fallback"]["reason_code"] == "foreign_checkout"
        assert_ran(run, ALL_SAMPLE_TESTS)
        assert {run.reason(name) for name in ALL_SAMPLE_TESTS} == {"fallback"}
        assert observation_total(shared_db) == 0  # nothing recorded
        # its own src first (the fix): normal operation
        run = run_pytest(
            other.path, shared_db, env={"PYTHONPATH": str(other.path / "src")}
        )
        assert run.report["fallback"] is None
        assert_ran(run, ALL_SAMPLE_TESTS)


class TestModes:
    def test_three_modes(self, repo, shared_db):
        # full suite without instrumentation
        plain = run_pytest(repo.path, shared_db, testmon=False)
        assert plain.returncode == 0 and "4 passed" in plain.output
        assert observation_total(shared_db) == 0
        # full suite with collection
        run = run_pytest(repo.path, shared_db, "--testmon-noselect")
        assert_ran(run, ALL_SAMPLE_TESTS)
        # selective with collection
        assert_ran(run_pytest(repo.path, shared_db), set())
        repo.write(ADD_CHANGED)
        assert_ran(run_pytest(repo.path, shared_db), ADD)
        # full with collection again: everything runs; deselectable ones say so
        run = run_pytest(repo.path, shared_db, "--testmon-noselect")
        assert_ran(run, ALL_SAMPLE_TESTS)
        assert run.reason("tests/test_calc.py::test_mul") == "selector_forced"
        # --testmon-nocollect selects but records nothing new
        repo.write(MUL_CHANGED)
        assert_ran(run_pytest(repo.path, shared_db, "--testmon-nocollect"), MUL)
        assert_ran(run_pytest(repo.path, shared_db, "--testmon-nocollect"), MUL)

    def test_xdist_modes(self, repo, shared_db):
        pytest.importorskip("xdist")
        assert_ran(run_pytest(repo.path, shared_db, "-n", "2"), ALL_SAMPLE_TESTS)
        assert_ran(run_pytest(repo.path, shared_db, "-n", "2"), set())
        repo.write(ADD_CHANGED)
        run = run_pytest(repo.path, shared_db, "-n", "2")
        assert_ran(run, ADD)
        assert run.deselected == MUL | SHOUT


FLAG_CONFTEST = """
    import pytest

    MODE = {mode!r}


    @pytest.fixture
    def base():
        return 10


    def pytest_addoption(parser):
        parser.addoption("--gpu", action="store_true")
        parser.addoption("--slow", action="store_true")


    def pytest_configure(config):
        config.addinivalue_line("markers", "gpu: needs --gpu")
        config.addinivalue_line("markers", "slow: needs --slow")


    def pytest_collection_modifyitems(config, items):
        keep, drop = [], []
        for item in items:
            needed = [name for name in ("gpu", "slow") if name in item.keywords]
            if all(config.getoption(name) for name in needed):
                keep.append(item)
            elif MODE == "skip":
                item.add_marker(pytest.mark.skip(reason="flag not given"))
                keep.append(item)
            else:
                drop.append(item)
        if drop:
            config.hook.pytest_deselected(items=drop)
            items[:] = keep
    """

FLAG_TESTS = {
    "pkg/hw.py": """
        def gpu_value():
            return 1


        def slow_value():
            return 2
        """,
    "tests/test_flags.py": """
        import pytest

        from pkg.hw import gpu_value, slow_value


        def test_cpu():
            pass


        @pytest.mark.gpu
        def test_gpu():
            assert gpu_value() == 1


        @pytest.mark.slow
        def test_slow():
            assert slow_value() == 2


        @pytest.mark.gpu
        @pytest.mark.slow
        def test_gpu_slow():
            assert gpu_value() + slow_value() == 3
        """,
}
CPU, GPU, SLOW, BOTH = (
    "tests/test_flags.py::test_cpu",
    "tests/test_flags.py::test_gpu",
    "tests/test_flags.py::test_slow",
    "tests/test_flags.py::test_gpu_slow",
)


def passed(run):
    return {name for name, outcome in run.outcomes().items() if outcome == "passed"}


class TestFilterFlags:
    """Conftest flags (or -m/-k) narrow the tests; testmon picks what needs running among them."""

    @pytest.mark.parametrize("mode", ["skip", "deselect"])
    def test_intersection_with_conftest_flags(self, repo, shared_db, mode):
        repo.write({"tests/conftest.py": FLAG_CONFTEST.format(mode=mode), **FLAG_TESTS})
        everything_else = ALL_SAMPLE_TESTS | {CPU}

        run = run_pytest(repo.path, shared_db)
        assert run.returncode == 0, run
        assert (
            passed(run) == everything_else
        )  # flagged tests off (skipped or deselected)

        run = run_pytest(repo.path, shared_db, "--gpu")
        assert passed(run) == {GPU}, run  # the gpu test was never recorded: it runs now
        run = run_pytest(repo.path, shared_db, "--gpu", "--slow")
        assert passed(run) == {SLOW, BOTH}, run  # the gpu-only one is already known
        run = run_pytest(repo.path, shared_db, "--gpu", "--slow")
        assert passed(run) == set(), run
        assert run.returncode == 0

        # a change used by the gpu tests: only the flags that allow them run them
        repo.write(
            {"pkg/hw.py": FLAG_TESTS["pkg/hw.py"].replace("return 1", "return 0 + 1")}
        )
        assert passed(run_pytest(repo.path, shared_db, "--slow")) == set()
        assert passed(run_pytest(repo.path, shared_db, "--gpu")) == {GPU}
        assert passed(run_pytest(repo.path, shared_db, "--gpu", "--slow")) == {BOTH}
        assert passed(run_pytest(repo.path, shared_db, "--gpu", "--slow")) == set()

    def test_intersection_with_marker_and_keyword_expressions(self, repo, shared_db):
        repo.write(
            {"tests/conftest.py": FLAG_CONFTEST.format(mode="skip"), **FLAG_TESTS}
        )
        run_pytest(repo.path, shared_db, "--gpu", "--slow")
        run = run_pytest(repo.path, shared_db, "--gpu", "--slow", "-m", "gpu")
        assert passed(run) == set() and run.returncode == 0, run
        repo.write(
            {"pkg/hw.py": FLAG_TESTS["pkg/hw.py"].replace("return 2", "return 1 + 1")}
        )
        assert passed(
            run_pytest(repo.path, shared_db, "--gpu", "--slow", "-m", "gpu")
        ) == {BOTH}
        assert passed(
            run_pytest(repo.path, shared_db, "--gpu", "--slow", "-k", "slow")
        ) == {SLOW}
        assert passed(run_pytest(repo.path, shared_db, "--gpu", "--slow")) == set()
