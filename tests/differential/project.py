"""A seeded, deterministic sample project and random edits to it (text level).

Everything is derived from the files themselves (canonical formatting, edited with regular
expressions), so git operations can freely rearrange history: there is no separate model to
keep in sync. Tests assert ``value % 5 != 0`` so that edits flip outcomes now and then.
"""

import random
import re

MODULES = 4
FUNCTIONS = 3
TEST_FILES = 3


def render_module(rng, index):
    lines = []
    lower = list(range(index))
    if lower:
        lines.append("from proj import " + ", ".join(f"m{j}" for j in lower))
        lines.append("")
    lines.append("VERSION = 0")
    for func in range(FUNCTIONS):
        lines += ["", ""]
        body = f"x + {rng.randint(1, 9)}"
        if lower and rng.random() < 0.6:
            callee = f"m{rng.choice(lower)}.f{rng.randrange(FUNCTIONS)}"
            body += f" + {callee}(x)"
        lines.append(f"def f{func}(x):")
        lines.append(f"    return {body}")
    return "\n".join(lines) + "\n"


def random_call(rng):
    return f"m{rng.randrange(MODULES)}.f{rng.randrange(FUNCTIONS)}"


def render_test(rng, name):
    kind = rng.random()
    first, second = random_call(rng), random_call(rng)
    if kind < 0.3:
        values = sorted(rng.sample(range(1, 7), 2))
        return (
            f'@pytest.mark.parametrize("v", {values!r})\n'
            f"def {name}(v):\n"
            f"    assert ({first}(v) + {second}(v)) % 5 != 0\n"
        )
    if kind < 0.5:
        return f"def {name}(k):\n    assert ({first}(k) * 2 + k) % 5 != 0\n"
    if kind < 0.65:
        # a branch: which callee runs depends on the value (still deterministic)
        return (
            f"def {name}():\n"
            f"    value = {first}(1)\n"
            f"    if value % 2:\n"
            f"        value += {second}(2)\n"
            f"    assert value % 5 != 0\n"
        )
    return f"def {name}():\n    assert {first}({rng.randint(0, 3)}) % 5 != 0\n"


def render_test_file(rng, index):
    header = "import pytest\n\nfrom proj import " + ", ".join(
        f"m{j}" for j in range(MODULES)
    )
    tests = [render_test(rng, f"test_{index}_{n}") for n in range(3)]
    return header + "\n\n\n" + "\n\n".join(tests)


CONFTEST = """import pytest

from proj import m0


@pytest.fixture
def k():
    return 1 + m0.f0(0)
"""


def initial_files(seed):
    rng = random.Random(seed)
    files = {
        "pytest.ini": "[pytest]\n",
        "proj/__init__.py": "",
        "tests/__init__.py": "",
        "tests/conftest.py": CONFTEST,
    }
    for index in range(MODULES):
        files[f"proj/m{index}.py"] = render_module(rng, index)
    for index in range(TEST_FILES):
        files[f"tests/test_t{index}.py"] = render_test_file(rng, index)
    return files


# ------------------------------------------------------------------------------- edits
# Each edit takes (rng, path) of the checkout and returns a description or None.


def _read(path):
    return path.read_text(encoding="utf-8")


def _write(path, text):
    path.write_text(text, encoding="utf-8")


def _module(rng, root):
    return root / "proj" / f"m{rng.randrange(MODULES)}.py"


def _test_file(rng, root):
    return root / "tests" / f"test_t{rng.randrange(TEST_FILES)}.py"


RETURN = re.compile(r"(    return x \+ )(\(?)(\d+)(\)?)")


def change_constant(rng, root):
    path = _module(rng, root)
    text = _read(path)
    matches = list(RETURN.finditer(text))
    match = rng.choice(matches)
    new = f"{match.group(1)}{match.group(2)}{rng.randint(1, 9)}{match.group(4)}"
    _write(path, text[: match.start()] + new + text[match.end() :])
    return f"change_constant {path.name}"


def reformat_body(rng, root):
    """Same value, different code: the block changes but outcomes don't."""
    path = _module(rng, root)
    text = _read(path)
    match = rng.choice(list(RETURN.finditer(text)))
    number = match.group(3)
    new = match.group(1) + (number if match.group(2) else f"({number})")
    _write(path, text[: match.start()] + new + text[match.end() :])
    return f"reformat_body {path.name}"


def module_level(rng, root):
    path = _module(rng, root)
    text = _read(path)
    text = re.sub(
        r"VERSION = (\d+)", lambda m: f"VERSION = {int(m.group(1)) + 1}", text
    )
    _write(path, text)
    return f"module_level {path.name}"


def add_test(rng, root):
    path = _test_file(rng, root)
    name = f"test_new_{rng.randrange(10**6)}"
    _write(path, _read(path).rstrip("\n") + "\n\n\n" + render_test(rng, name))
    return f"add_test {path.name}::{name}"


TEST_DEF = re.compile(r"^(?:@pytest\.mark\.parametrize\(.*\)\n)?def (test_\w+)\(", re.M)


def remove_test(rng, root):
    path = _test_file(rng, root)
    text = _read(path)
    starts = [m.start() for m in TEST_DEF.finditer(text)]
    if len(starts) < 2:
        return None
    index = rng.randrange(len(starts))
    end = starts[index + 1] if index + 1 < len(starts) else len(text)
    _write(path, (text[: starts[index]] + text[end:]).rstrip("\n") + "\n")
    return f"remove_test {path.name}"


def rename_test(rng, root):
    path = _test_file(rng, root)
    text = _read(path)
    names = TEST_DEF.findall(text)
    old = rng.choice(names)
    new = f"{old}_r{rng.randrange(100)}"
    _write(path, re.sub(rf"\bdef {old}\(", f"def {new}(", text))
    return f"rename_test {old} -> {new}"


def change_params(rng, root):
    path = _test_file(rng, root)
    text = _read(path)
    if "parametrize" not in text:
        return None
    values = sorted(rng.sample(range(1, 7), rng.randint(1, 3)))
    text = re.sub(
        r'parametrize\("v", \[[^\]]*\]\)', f'parametrize("v", {values!r})', text, 1
    )
    _write(path, text)
    return f"change_params {path.name} {values}"


def change_fixture(rng, root):
    path = root / "tests" / "conftest.py"
    text = re.sub(
        r"return (\d+) \+ m0", f"return {rng.randint(0, 4)} + m0", _read(path)
    )
    _write(path, text)
    return "change_fixture"


EDITS = [
    (change_constant, 4),
    (reformat_body, 3),
    (module_level, 1),
    (add_test, 2),
    (remove_test, 1),
    (rename_test, 1),
    (change_params, 1),
    (change_fixture, 1),
]


def random_edit(rng, root):
    edits, weights = zip(*EDITS)
    for _ in range(5):
        done = rng.choices(edits, weights)[0](rng, root)
        if done:
            return done
    return "no-op"
