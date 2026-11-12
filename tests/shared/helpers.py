"""Real git repositories and pytest subprocesses for end-to-end tests of the shared database.

Every pytest run here is a separate process (like an agent), uses the shared database given
explicitly and writes a JSON report that the tests assert on.
"""

import json
import os
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path

PYTHON = sys.executable

GITIGNORE = "__pycache__/\n*.pyc\n.testmondata*\nreport*.json\n.tmontmp/\n"

SAMPLE = {
    "pytest.ini": "[pytest]\n",
    "pkg/__init__.py": "",
    "pkg/calc.py": """
        def add(a, b):
            return a + b


        def mul(a, b):
            return a * b
        """,
    "pkg/text.py": """
        def shout(s):
            return s.upper()
        """,
    "tests/__init__.py": "",
    "tests/conftest.py": """
        import pytest


        @pytest.fixture
        def base():
            return 10
        """,
    "tests/test_calc.py": """
        from pkg.calc import add, mul


        def test_add():
            assert add(1, 2) == 3


        def test_mul():
            assert mul(2, 3) == 6


        def test_add_fixture(base):
            assert add(base, 1) == 11
        """,
    "tests/test_text.py": """
        from pkg.text import shout


        def test_shout():
            assert shout("a") == "A"
        """,
}

ALL_SAMPLE_TESTS = {
    "tests/test_calc.py::test_add",
    "tests/test_calc.py::test_mul",
    "tests/test_calc.py::test_add_fixture",
    "tests/test_text.py::test_shout",
}


def clean_env(extra=None):
    """The environment of an independent agent: no pytest/coverage/testmon leftovers."""
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(
            ("TESTMON_", "PYTEST_", "COV_CORE", "COVERAGE", "PYTHONPATH")
        )
    }
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env.update(extra or {})
    return env


class GitRepo:
    def __init__(self, path):
        self.path = Path(path)

    @classmethod
    def init(cls, path, files=None):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        repo = cls(path)
        repo.git("init", "-q", "-b", "main")
        repo.configure_identity()
        repo.write({".gitignore": GITIGNORE})
        repo.write(SAMPLE if files is None else files)
        repo.commit("initial")
        return repo

    def configure_identity(self):
        self.git("config", "user.name", "testmon tests")
        self.git("config", "user.email", "tests@example.invalid")
        self.git("config", "commit.gpgsign", "false")

    def git(self, *args, check=True):
        result = subprocess.run(
            ["git", *args],
            cwd=self.path,
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=False,
            env=clean_env({"GIT_EDITOR": "true"}),
        )
        if check and result.returncode:
            raise AssertionError(
                f"git {' '.join(args)} failed:\n{result.stdout}{result.stderr}"
            )
        return result.stdout.strip()

    def write(self, files):
        for name, content in files.items():
            target = self.path / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(textwrap.dedent(content).lstrip("\n"), encoding="utf-8")

    def remove(self, *names):
        for name in names:
            (self.path / name).unlink()

    def commit(self, message):
        self.git("add", "-A")
        self.git("commit", "-q", "--allow-empty", "-m", message)
        return self.head()

    def head(self):
        return self.git("rev-parse", "HEAD")

    def tree(self, ref="HEAD"):
        return self.git("rev-parse", f"{ref}^{{tree}}")

    def checkout(self, ref, create=False):
        if create:
            self.git("checkout", "-q", "-b", ref)
        else:
            self.git("checkout", "-q", ref)

    def worktree(self, path, ref, branch=None):
        if branch:
            self.git("worktree", "add", "-q", "-b", branch, str(path), ref)
        else:
            self.git("worktree", "add", "-q", "--detach", str(path), ref)
        return GitRepo(path)

    def clone(self, path):
        subprocess.run(
            ["git", "clone", "-q", str(self.path), str(path)],
            check=True,
            capture_output=True,
            env=clean_env(),
        )
        repo = GitRepo(path)
        repo.configure_identity()
        return repo


class Run:
    def __init__(self, proc, report):
        self.proc = proc
        self.returncode = proc.returncode
        self.output = proc.stdout + proc.stderr
        self.report = report

    def __repr__(self):
        return f"<Run rc={self.returncode}>\n{self.output}"

    @property
    def tests(self):
        return self.report["tests"]

    @property
    def selected(self):
        return {
            name for name, test in self.tests.items() if test["decision"] == "selected"
        }

    @property
    def deselected(self):
        return {
            name
            for name, test in self.tests.items()
            if test["decision"] == "deselected"
        }

    def reason(self, name):
        return self.tests[name]["reason"]

    def outcomes(self):
        return {
            name: t.get("outcome") for name, t in self.tests.items() if "outcome" in t
        }

    def failed(self):
        return {
            name for name, outcome in self.outcomes().items() if outcome == "failed"
        }


def run_pytest(checkout, shared_db, *args, env=None, timeout=300, testmon=True):
    """pytest --testmon in checkout (a separate process), returning the parsed report."""
    checkout = Path(checkout)
    report_path = checkout / "report.json"
    if report_path.exists():
        report_path.unlink()
    command = [PYTHON, "-m", "pytest", "-p", "no:cacheprovider", "-q"]
    if testmon:
        command += ["--testmon", f"--testmon-report={report_path}"]
    command += list(args)
    extra = {"TESTMON_SHARED_DB": str(shared_db)} if shared_db else {}
    extra.update(env or {})
    proc = subprocess.run(
        command,
        cwd=checkout,
        env=clean_env(extra),
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    report = None
    if report_path.exists():
        report = json.loads(report_path.read_text(encoding="utf-8"))
    return Run(proc, report)


def popen_pytest(checkout, shared_db, *args, env=None):
    """Start pytest --testmon in the background; finish with collect(popen, checkout)."""
    checkout = Path(checkout)
    report_path = checkout / "report.json"
    if report_path.exists():
        report_path.unlink()
    extra = {"TESTMON_SHARED_DB": str(shared_db)}
    extra.update(env or {})
    return subprocess.Popen(
        [
            PYTHON,
            "-m",
            "pytest",
            "-p",
            "no:cacheprovider",
            "-q",
            "--testmon",
            f"--testmon-report={report_path}",
            *args,
        ],
        cwd=checkout,
        env=clean_env(extra),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def collect(popen, checkout, timeout=300):
    stdout, stderr = popen.communicate(timeout=timeout)
    proc = subprocess.CompletedProcess(popen.args, popen.returncode, stdout, stderr)
    report_path = Path(checkout) / "report.json"
    report = json.loads(report_path.read_text()) if report_path.exists() else None
    return Run(proc, report)


def copy_without_git(src, dest):
    shutil.copytree(src, dest, ignore=shutil.ignore_patterns(".git", "report.json"))
    return Path(dest)


def assert_ran(run, expected):
    """The run executed exactly `expected` tests (and the report agrees with pytest)."""
    assert run.report is not None, run
    assert run.selected == set(expected), (sorted(run.selected), run)
    assert run.returncode == 0, run
