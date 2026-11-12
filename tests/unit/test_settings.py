"""Settings from the rootdir conftest (pytest_testmon_settings) and ini options."""

import json

import pytest

from testmon.config import SETTINGS, build_settings
from testmon.db import DB

pytest_plugins = ("pytester",)

TEST_FILE = "def test_1():\n    pass\n"


def conftest(settings, optional=True):
    decorator = "@pytest.hookimpl(optionalhook=True)\n" if optional else ""
    return f"import pytest\n\n\n{decorator}def pytest_testmon_settings(config):\n    return {settings!r}\n"


def report(pytester):
    return json.loads((pytester.path / "r.json").read_text())


@pytest.fixture
def shared(tmp_path):
    path = tmp_path / "cache" / "db.sqlite"
    DB(str(path), shared=True, create=True).close()
    return path


class TestBuildSettings:
    def test_hook_over_ini(self):
        ini = {"testmon_shared_db": "/ini", "testmon_db_timeout": "5"}.get
        settings = build_settings({"shared_db": "/hook"}, lambda name: ini(name, ""))
        assert (settings.get("shared_db"), settings.source("shared_db")) == (
            "/hook",
            "conftest",
        )
        assert (settings.get("db_timeout"), settings.source("db_timeout")) == (
            5.0,
            "ini",
        )
        assert settings.get("max_versions", 32) == 32
        assert settings.source("project") is None

    def test_types(self):
        settings = build_settings(
            {"ignore_collect": False, "max_versions": 4, "retention_days": 1.5}, None
        )
        assert settings.get("ignore_collect") is False
        assert settings.get("max_versions") == 4
        ini = {"testmon_ignore_collect": "false"}
        assert (
            build_settings({}, lambda n: ini.get(n, "")).get("ignore_collect") is False
        )

    @pytest.mark.parametrize(
        "values, message",
        [
            ({"shared_dbb": "x"}, "unknown testmon setting"),
            ({"db_timeout": "soon"}, "must be a number"),
            ({"max_versions": -1}, "must not be negative"),
            ({"ignore_collect": "maybe"}, "true or false"),
            ({"shared_db": 3}, "must be a string"),
            ({"db_fallback": "skip"}, "testmon_db_fallback must be one of"),
            (["shared_db"], "must return a dict"),
        ],
    )
    def test_invalid(self, values, message):
        with pytest.raises(ValueError, match=message):
            build_settings(values, None)

    def test_every_setting_has_an_ini_option(self):
        from testmon.config import INI_OPTIONS  # pylint: disable=import-outside-toplevel

        assert {f"testmon_{name}" for name in SETTINGS} <= set(INI_OPTIONS)


class TestConftestHook:
    def test_shared_db_from_conftest(self, pytester, shared):
        pytester.makeconftest(conftest({"shared_db": str(shared)}))
        pytester.makepyfile(test_a=TEST_FILE)
        result = pytester.runpytest_inprocess("--testmon", "--testmon-report=r.json")
        result.assert_outcomes(passed=1)
        data = report(pytester)
        assert data["db"] == {
            "path": str(shared),
            "shared": True,
            "source": "conftest",
            "new": False,
        }
        assert not (pytester.path / ".testmondata").exists()
        pytester.runpytest_inprocess("--testmon").assert_outcomes()  # reused

    def test_relative_path_is_relative_to_the_rootdir(self, pytester):
        pytester.makeconftest(conftest({"shared_db": "../cache-rel/db.sqlite"}))
        pytester.makepyfile(test_a=TEST_FILE)
        target = pytester.path.parent / "cache-rel" / "db.sqlite"
        DB(str(target), shared=True, create=True).close()
        pytester.runpytest_inprocess("--testmon", "--testmon-report=r.json")
        assert report(pytester)["db"]["path"] == str(target)

    def test_conftest_beats_ini_env_and_cli_beat_conftest(
        self, pytester, shared, tmp_path, monkeypatch
    ):
        other = tmp_path / "other" / "db.sqlite"
        DB(str(other), shared=True, create=True).close()
        pytester.makeini("[pytest]\ntestmon_shared_db = /nonexistent/ini.sqlite\n")
        pytester.makeconftest(conftest({"shared_db": str(shared)}))
        pytester.makepyfile(test_a=TEST_FILE)
        pytester.runpytest_inprocess("--testmon", "--testmon-report=r.json")
        assert report(pytester)["db"]["source"] == "conftest"
        pytester.runpytest_inprocess(
            "--testmon", "--testmon-report=r.json", f"--testmon-shared-db={other}"
        )
        assert report(pytester)["db"]["source"] == "cli"
        monkeypatch.setenv("TESTMON_SHARED_DB", str(other))
        pytester.runpytest_inprocess("--testmon", "--testmon-report=r.json")
        assert report(pytester)["db"]["source"] == "env:TESTMON_SHARED_DB"

    def test_other_settings_from_conftest(self, pytester, tmp_path):
        missing = tmp_path / "missing" / "db.sqlite"
        pytester.makeconftest(
            conftest({"shared_db": str(missing), "db_fallback": "run-all"})
        )
        pytester.makepyfile(test_a=TEST_FILE)
        result = pytester.runpytest_inprocess("--testmon", "--testmon-report=r.json")
        result.assert_outcomes(passed=1)  # run-all chosen in conftest
        assert report(pytester)["fallback"]["reason_code"] == "missing"

    @pytest.mark.parametrize(
        "settings, message",
        [
            ({"sharedb": "/x"}, "unknown testmon setting"),
            ({"db_timeout": "x"}, "must be a number"),
        ],
    )
    def test_invalid_settings_stop_pytest(self, pytester, settings, message):
        pytester.makeconftest(conftest(settings))
        pytester.makepyfile(test_a=TEST_FILE)
        result = pytester.runpytest_inprocess("--testmon")
        assert result.ret == pytest.ExitCode.USAGE_ERROR
        result.stderr.fnmatch_lines([f"*{message}*"])
        assert "passed" not in result.stdout.str()

    def test_conftest_below_the_rootdir_is_refused(self, pytester, shared):
        pytester.makeini("[pytest]\n")
        sub = pytester.mkpydir("sub")
        (sub / "conftest.py").write_text(conftest({"shared_db": str(shared)}))
        (sub / "test_a.py").write_text(TEST_FILE)
        result = pytester.runpytest_inprocess(
            "--testmon", "--testmon-db-fallback=run-all"
        )
        assert result.ret != 0
        assert "loaded too late" in result.stdout.str() + result.stderr.str()
        assert "1 passed" not in result.stdout.str()

    def test_conftest_given_as_an_argument_directory_is_early_enough(
        self, pytester, shared
    ):
        pytester.makeini("[pytest]\n")
        sub = pytester.mkpydir("sub")
        (sub / "conftest.py").write_text(conftest({"shared_db": str(shared)}))
        (sub / "test_a.py").write_text(TEST_FILE)
        result = pytester.runpytest_inprocess(
            "--testmon", "--testmon-report=r.json", "sub"
        )
        result.assert_outcomes(passed=1)
        assert report(pytester)["db"]["source"] == "conftest"

    def test_conftest_works_without_testmon(self, pytester):
        pytester.makeconftest(conftest({"shared_db": "/x"}))
        pytester.makepyfile(test_a=TEST_FILE)
        result = pytester.runpytest_inprocess("-p", "no:pytest-testmon")
        result.assert_outcomes(passed=1)

    def test_hook_not_called_when_testmon_not_requested(self, pytester):
        pytester.makeconftest(
            "import pytest\n\n\n@pytest.hookimpl(optionalhook=True)\n"
            "def pytest_testmon_settings(config):\n    raise RuntimeError('called')\n"
        )
        pytester.makepyfile(test_a=TEST_FILE)
        pytester.runpytest_inprocess().assert_outcomes(passed=1)
