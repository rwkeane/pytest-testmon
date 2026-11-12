"""torch.compile: code reached only through compiled functions is a dependency too.

Coverage never sees compiled code execute (see testmon/dynamo.py), so without dynamo tracking
an edit inside a compiled function would leave its tests deselected. Every scenario below edits
code that only runs compiled and checks that exactly the tests using it run again.

Needs torch (skipped otherwise). Each run is a separate pytest process.
"""

import importlib.metadata
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]

MODEL = """
    import torch

    import helpers


    def inner(x):
        return x * 2 + 1


    def outer(x):
        return helpers.helper(inner(x)).sum()


    def before_break(x):
        return x + 1


    def after_break(x):
        return x * 3


    def with_break(x):
        y = before_break(x)
        torch._dynamo.graph_break()
        return after_break(y).sum()


    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = 2.0

        def forward(self, x):
            return helpers.helper(x * self.scale).sum()


    def unrelated(value):
        return value + 1
    """

HELPERS = """
    def helper(y):
        return y + 3
    """

CONFTEST = """
    import pytest
    import torch

    import model


    @pytest.fixture(scope="session")
    def compiled_model():
        return torch.compile(model.Model(), backend="eager")
    """

TESTS = """
    import torch

    import model

    outer = torch.compile(model.outer, backend="eager")
    with_break = torch.compile(model.with_break, backend="eager")


    def test_outer():
        assert outer(torch.ones(3)).item() == 18.0


    def test_outer_again():
        # the same compiled function as test_outer: a cache hit unless dynamo is reset
        assert outer(torch.ones(3)).item() == 18.0


    def test_break():
        assert with_break(torch.ones(3)).item() == 18.0


    def test_module(compiled_model):
        assert compiled_model(torch.ones(3)).item() == 15.0


    def test_module_again(compiled_model):
        assert compiled_model(torch.ones(3)).item() == 15.0


    def test_eager_only():
        assert model.unrelated(1) == 2
    """

OUTER = {"test_outer", "test_outer_again"}
MODULE = {"test_module", "test_module_again"}
EVERYTHING = OUTER | MODULE | {"test_break", "test_eager_only"}


def plugin_args():
    names = {ep.name for ep in importlib.metadata.entry_points(group="pytest11")}
    return [] if "pytest-testmon" in names else ["-p", "testmon.pytest_testmon"]


def python_path():
    import coverage  # pylint: disable=import-outside-toplevel

    import testmon  # pylint: disable=import-outside-toplevel

    paths = [
        str(Path(testmon.__file__).parents[1]),
        str(Path(coverage.__file__).parents[1]),
    ]
    return os.pathsep.join(dict.fromkeys(paths))


class Project:
    def __init__(self, path):
        self.path = path
        self.write(
            {
                "pytest.ini": "[pytest]\n",
                "model.py": MODEL,
                "helpers.py": HELPERS,
                "conftest.py": CONFTEST,
                "test_compiled.py": TESTS,
            }
        )

    def write(self, files):
        for name, text in files.items():
            (self.path / name).write_text(textwrap.dedent(text).lstrip("\n"))

    def replace(self, name, old, new):
        target = self.path / name
        text = target.read_text()
        assert old in text, (name, old)
        target.write_text(text.replace(old, new))

    def run(self, *args, expect=0):
        report = self.path / "report.json"
        if report.exists():
            report.unlink()
        env = {
            key: value
            for key, value in os.environ.items()
            if not key.startswith(("TESTMON_", "PYTEST_", "COV_CORE", "COVERAGE"))
        }
        env["PYTHONPATH"] = python_path()
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        proc = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                *plugin_args(),
                "-p",
                "no:cacheprovider",
                "-q",
                "--testmon",
                f"--testmon-report={report}",
                *args,
            ],
            cwd=self.path,
            env=env,
            capture_output=True,
            text=True,
            timeout=600,
            check=False,
        )
        output = proc.stdout + proc.stderr
        assert proc.returncode == expect, output
        if not report.exists():
            return None, output
        data = json.loads(report.read_text())
        ran = {
            name.split("::")[-1]
            for name, test in data["tests"].items()
            if test.get("outcome") == "passed"
        }
        return ran, output


@pytest.fixture
def project(tmp_path):
    return Project(tmp_path)


def test_coverage_alone_misses_compiled_code(tmp_path):
    """Why this module exists: the body of a compiled function never shows up as executed."""
    from coverage import Coverage  # pylint: disable=import-outside-toplevel

    (tmp_path / "compiled_probe.py").write_text(
        "import torch\n\n\ndef body(x):\n    return x * 2 + 1\n\n\n"
        "compiled = torch.compile(body, backend='eager')\n"
    )
    sys.path.insert(0, str(tmp_path))
    try:
        cov = Coverage(include=[str(tmp_path / "*")], config_file=False)
        cov.start()
        import compiled_probe  # pylint: disable=import-outside-toplevel,import-error

        compiled_probe.compiled(torch.ones(2))
        cov.stop()
        assert 5 not in cov.get_data().lines(compiled_probe.__file__)
    finally:
        sys.path.remove(str(tmp_path))
        sys.modules.pop("compiled_probe", None)
        torch._dynamo.reset()  # pylint: disable=protected-access


@pytest.mark.parametrize("xdist", [False, True], ids=["single", "xdist"])
def test_edits_in_compiled_code_rerun_exactly_their_tests(project, xdist):
    if xdist:
        pytest.importorskip("xdist")
    args = ("-n", "2") if xdist else ()
    ran, output = project.run(*args)
    assert ran == EVERYTHING, output
    ran, output = project.run(*args)
    assert ran == set(), output

    # inside a function that runs only inlined into a compiled one
    project.replace("model.py", "return x * 2 + 1", "return x * 2 + 1 + 0")
    ran, output = project.run(*args)
    assert ran == OUTER, (
        output
    )  # both tests, although the second one hit dynamo's cache

    # a helper in another module, inlined by two different compiled entry points
    project.replace("helpers.py", "return y + 3", "return y + 2 + 1")
    ran, output = project.run(*args)
    assert ran == OUTER | MODULE, output

    # code after a graph break (dynamo's resume function)
    project.replace("model.py", "return x * 3", "return x * 2 + x")
    ran, output = project.run(*args)
    assert ran == {"test_break"}, output

    # nn.Module.forward of a model compiled once in a session fixture
    project.replace("model.py", "x * self.scale", "self.scale * x")
    ran, output = project.run(*args)
    assert ran == MODULE, output

    # eager code is unaffected by all of this
    project.replace("model.py", "return value + 1", "return 1 + value")
    ran, output = project.run(*args)
    assert ran == {"test_eager_only"}, output
    ran, output = project.run(*args)
    assert ran == set(), output


def test_torch_imported_during_a_test(project):
    project.write(
        {
            "conftest.py": "",
            "test_compiled.py": """
                def test_late_import():
                    import torch

                    import model

                    compiled = torch.compile(model.outer, backend="eager")
                    assert compiled(torch.ones(3)).item() == 18.0
                """,
        }
    )
    ran, output = project.run()
    assert ran == {"test_late_import"}, output
    project.replace("model.py", "return x * 2 + 1", "return x * 2 + 1 + 0")
    ran, output = project.run()
    assert ran == {"test_late_import"}, output


def test_unsupported_torch_is_an_error(project):
    project.write(
        {
            "conftest.py": textwrap.dedent(CONFTEST)
            + textwrap.dedent(
                """
                import torch._guards

                _original = torch._guards.TracingContext.__init__


                def _without_traced_code(self, *args, **kwargs):
                    _original(self, *args, **kwargs)
                    del self.traced_code


                torch._guards.TracingContext.__init__ = _without_traced_code
                """
            )
        }
    )
    _, output = project.run("-p", "no:randomly", expect=3)
    assert "cannot record what torch.compile traced" in output


def test_many_tests_across_xdist_workers(project):
    """Many tests share compiled functions; workers pick them up in any order."""
    pytest.importorskip("xdist")
    tests = ["import pytest", "import torch", "", "import model", ""]
    tests.append('outer = torch.compile(model.outer, backend="eager")')
    tests.append('with_break = torch.compile(model.with_break, backend="eager")')
    for index in range(24):
        function = "outer" if index % 2 == 0 else "with_break"
        tests += [
            "",
            "",
            f"def test_{function}_{index}():",
            f"    assert {function}(torch.ones(3)).item() == 18.0",
        ]
    project.write({"conftest.py": "", "test_compiled.py": "\n".join(tests) + "\n"})
    outer_tests = {f"test_outer_{i}" for i in range(0, 24, 2)}
    break_tests = {f"test_with_break_{i}" for i in range(1, 24, 2)}

    ran, output = project.run("-n", "4")
    assert ran == outer_tests | break_tests, output
    assert project.run("-n", "4")[0] == set()
    project.replace("model.py", "return x * 3", "return x * 2 + x")
    ran, output = project.run("-n", "4")
    assert ran == break_tests, output
    project.replace("helpers.py", "return y + 3", "return y + 2 + 1")
    ran, output = project.run("-n", "4")
    assert ran == outer_tests, output
    assert project.run("-n", "4")[0] == set()
