import importlib.metadata
import json
import os
import sys
import types
from pathlib import Path

import jsonschema
import pytest

from testmon import common
from testmon.config import (
    FALLBACK_ERROR,
    FALLBACK_RUN_ALL,
    resolve_db_location,
    resolve_fallback,
    read_project_ini,
)
from testmon.report import load_report

pytest_plugins = ("pytester",)

SCHEMA = json.loads(
    (Path(__file__).parents[2] / "docs" / "report.schema.json").read_text(
        encoding="utf-8"
    )
)


class TestDbLocation:
    ROOT = os.path.abspath("/work/checkout")

    def test_default(self):
        location = resolve_db_location(self.ROOT, environ={})
        assert location.path == os.path.join(self.ROOT, ".testmondata")
        assert not location.shared and location.source == "default"

    def test_precedence(self):
        environ = {"TESTMON_SHARED_DB": "/env/db", "TESTMON_DATAFILE": "local.db"}
        assert (
            resolve_db_location(self.ROOT, "/cli/db", "/ini/db", environ).source
            == "cli"
        )
        assert (
            resolve_db_location(self.ROOT, None, "/ini/db", environ).source
            == "env:TESTMON_SHARED_DB"
        )
        environ.pop("TESTMON_SHARED_DB")
        location = resolve_db_location(self.ROOT, None, "/ini/db", environ)
        assert (location.source, location.shared) == ("ini", True)
        location = resolve_db_location(self.ROOT, None, "", environ)
        assert (location.source, location.shared) == ("env:TESTMON_DATAFILE", False)
        assert location.path == os.path.join(self.ROOT, "local.db")

    @pytest.mark.skipif(sys.platform == "win32", reason="expanduser uses USERPROFILE")
    def test_expansion(self, monkeypatch):
        monkeypatch.setenv("HOME", "/home/someone")
        monkeypatch.setenv("CACHE", "/var/cache")
        assert resolve_db_location(
            self.ROOT, ini_value="~/x/db", environ={}
        ).path == os.path.abspath("/home/someone/x/db")
        assert resolve_db_location(
            self.ROOT, ini_value="$CACHE/db", environ={}
        ).path == os.path.abspath("/var/cache/db")
        assert resolve_db_location(self.ROOT, ini_value="rel/db", environ={}).path == (
            os.path.join(self.ROOT, "rel", "db")
        )

    def test_fallback_policy(self):
        assert resolve_fallback() == FALLBACK_ERROR  # default: stop, never run silently
        assert resolve_fallback(None, "error") == FALLBACK_ERROR
        assert resolve_fallback("run-all", "error") == FALLBACK_RUN_ALL
        with pytest.raises(ValueError):
            resolve_fallback(None, "skip-everything")

    def test_read_project_ini(self, tmp_path):
        (tmp_path / "pyproject.toml").write_text(
            '[tool.pytest.ini_options]\ntestmon_shared_db = "~/db"\n'
        )
        assert read_project_ini(str(tmp_path)) == {"testmon_shared_db": "~/db"}
        (tmp_path / "pytest.ini").write_text("[pytest]\ntestmon_shared_db = /p\n")
        assert read_project_ini(str(tmp_path)) == {"testmon_shared_db": "/p"}


def fake_dist(directory, name, version, editable_dir=None):
    info = directory / f"{name}-{version}.dist-info"
    info.mkdir(parents=True)
    (info / "METADATA").write_text(
        f"Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n"
    )
    if editable_dir is not None:
        (info / "direct_url.json").write_text(
            json.dumps(
                {"url": Path(editable_dir).as_uri(), "dir_info": {"editable": True}}
            )
        )
    return importlib.metadata.PathDistribution(info)


class TestSystemPackages:
    def test_editable_install_of_the_project_is_left_out(self, tmp_path, monkeypatch):
        project = tmp_path / "project"
        other = tmp_path / "other"
        project.mkdir()
        other.mkdir()
        site = tmp_path / "site"
        dists = [
            fake_dist(site, "proj", "0.1.dev7+gabc", editable_dir=project),
            fake_dist(site, "otherlib", "2.0", editable_dir=other),
            fake_dist(site, "requests", "2.31.0"),
        ]
        monkeypatch.setattr(importlib.metadata, "distributions", lambda: dists)
        assert (
            common.get_system_packages(rootdir=str(project))
            == "otherlib 2.0, requests 2.31.0"
        )
        assert (
            "proj" in common.get_system_packages()
        )  # upstream behaviour without rootdir
        assert common.get_system_packages(
            ignore={"requests"}, rootdir=str(project)
        ) == ("otherlib 2.0")


class TestForeignCheckout:
    def make_checkout(self, root, layout=""):
        package = root / layout / "pkg" if layout else root / "pkg"
        package.mkdir(parents=True)
        (package / "__init__.py").write_text("")
        (root / ".git").mkdir()
        return package / "__init__.py"

    def test_imported_from_other_checkout(self, tmp_path):
        self.make_checkout(tmp_path / "a")
        other_init = self.make_checkout(tmp_path / "b")
        modules = {"pkg": types.SimpleNamespace(__file__=str(other_init))}
        assert common.find_foreign_checkout_imports(str(tmp_path / "a"), modules) == [
            ("pkg", os.path.realpath(other_init))
        ]

    def test_own_checkout_and_non_checkouts_are_fine(self, tmp_path):
        own_init = self.make_checkout(tmp_path / "a")
        assert not common.find_foreign_checkout_imports(
            str(tmp_path / "a"), {"pkg": types.SimpleNamespace(__file__=str(own_init))}
        )
        site = tmp_path / "site-packages" / "pkg"
        site.mkdir(parents=True)
        (site / "__init__.py").write_text("")
        assert not common.find_foreign_checkout_imports(
            str(tmp_path / "a"),
            {"pkg": types.SimpleNamespace(__file__=str(site / "__init__.py"))},
        )

    def test_src_layout_found_without_import(self, tmp_path, monkeypatch):
        self.make_checkout(tmp_path / "a", layout="src")
        other_init = self.make_checkout(tmp_path / "b", layout="src")
        monkeypatch.syspath_prepend(str(tmp_path / "b" / "src"))
        assert common.find_foreign_checkout_imports(str(tmp_path / "a"), {}) == [
            ("pkg", os.path.realpath(other_init))
        ]
        assert "pkg" not in sys.modules or sys.modules["pkg"].__file__ != str(
            other_init
        )


class TestReport:
    def test_report_validates_and_is_consistent(self, pytester):
        pytester.makepyfile(
            lib="def f():\n    return 1\n",
            test_a="from lib import f\n\ndef test_1():\n    assert f() == 1\n\n"
            "def test_2():\n    pass\n",
            test_b="def test_3():\n    pass\n",
        )
        pytester.runpytest_inprocess("--testmon").assert_outcomes(passed=3)
        pytester.makepyfile(lib="def f():\n    return 1 + 0\n")
        pytester.runpytest_inprocess(
            "--testmon", "--testmon-report=r.json"
        ).assert_outcomes(passed=1)
        report = load_report(str(pytester.path / "r.json"))
        jsonschema.validate(report, SCHEMA)
        tests = report["tests"]
        assert tests["test_a.py::test_1"]["reason"] == "all_invalid"
        assert tests["test_a.py::test_1"]["invalid_dep"] == {
            "filename": "lib.py",
            "cause": "mismatch",
        }
        assert tests["test_a.py::test_2"] == {
            "decision": "deselected",
            "reason": "deselect",
            "observation_id": tests["test_a.py::test_2"]["observation_id"],
            "invalid_dep": None,
            "file_ignored": False,
        }
        assert tests["test_b.py::test_3"]["file_ignored"] is True
        counts = report["counts"]
        assert counts["collected"] == counts["selected"] + counts["deselected"] == 3
        assert (counts["selected"], counts["deselected"], counts["file_ignored"]) == (
            1,
            2,
            1,
        )
        assert report["ignored_files"] == ["test_b.py"]
        assert report["reasons"] == {"all_invalid": 1, "deselect": 2}

    def test_error_report(self, pytester):
        pytester.makepyfile(test_a="def test_1():\n    pass\n")
        bad = pytester.path / "bad.sqlite"
        bad.write_bytes(b"this is not a database" * 100)
        os.chmod(bad, 0o600)
        result = pytester.runpytest_inprocess(
            "--testmon", f"--testmon-shared-db={bad}", "--testmon-report=r.json"
        )
        assert result.ret == 4
        assert "passed" not in result.stdout.str()  # stopped before running anything
        report = load_report(str(pytester.path / "r.json"))
        jsonschema.validate(report, SCHEMA)
        assert report["fallback"] == {
            "reason_code": "corrupt",
            "detail": report["fallback"]["detail"],
            "policy": "error",
        }
        assert report["tests"] == {} and report["exitstatus"] == 4

    def test_fallback_report(self, pytester):
        pytester.makepyfile(test_a="def test_1():\n    pass\n")
        bad = pytester.path / "bad.sqlite"
        bad.write_bytes(b"this is not a database" * 100)
        os.chmod(bad, 0o600)
        result = pytester.runpytest_inprocess(
            "--testmon",
            f"--testmon-shared-db={bad}",
            "--testmon-report=r.json",
            "--testmon-db-fallback=run-all",
        )
        result.assert_outcomes(passed=1)
        report = load_report(str(pytester.path / "r.json"))
        jsonschema.validate(report, SCHEMA)
        assert report["fallback"]["reason_code"] == "corrupt"
        assert report["tests"]["test_a.py::test_1"]["reason"] == "fallback"
        assert report["mode"] == {
            "select": False,
            "collect": False,
            "message": report["mode"]["message"],
        }
