"""Settings: where the database lives, what to do when it can't be used, and the rest.

Settings come from the ``pytest_testmon_settings`` hook (implemented in the rootdir's
conftest.py, returning a dict) or else from ``testmon_<name>`` pytest ini options. See
SETTINGS for the names.

Precedence for the database location (first match wins):

1. ``--testmon-shared-db=PATH``                      (shared mode)
2. ``TESTMON_SHARED_DB`` environment var             (shared mode)
3. ``shared_db`` from the conftest hook              (shared mode)
4. ``testmon_shared_db`` ini option                  (shared mode)
5. ``TESTMON_DATAFILE`` environment var              (private mode, upstream-compatible)
6. ``<rootdir>/.testmondata``                        (private mode, upstream default)

The fork-specific names keep an unforked testmon that inherits the environment away from the
shared file. Paths expand ``~`` and ``$VARS``; relative paths are relative to the rootdir.
"""

import configparser
import os
from dataclasses import dataclass

LOCAL_DB_FILENAME = ".testmondata"
SHARED_DB_ENV = "TESTMON_SHARED_DB"
LOCAL_DB_ENV = "TESTMON_DATAFILE"
REPORT_ENV = "TESTMON_REPORT"
VERIFIED_TREE_ENV = "TESTMON_VERIFIED_TREE"

FALLBACK_RUN_ALL = "run-all"
FALLBACK_ERROR = "error"
FALLBACK_POLICIES = (FALLBACK_RUN_ALL, FALLBACK_ERROR)

# name -> (type, help). Each is settable from the conftest hook (as `name`) or as the ini
# option `testmon_<name>`. Every ini option defaults to "" (= not set; testmon's default).
SETTINGS = {
    "shared_db": (str, "path of the shared testmon database"),
    "db_fallback": (str, "when the database is unusable: error (default) or run-all"),
    "db_timeout": (float, "seconds to wait for a database lock (default 30)"),
    "project": (str, "namespace for this project's data in a shared database"),
    "retention_days": (float, "days an unused observation is kept (default 60)"),
    "max_versions": (int, "observations kept per test and environment (default 32)"),
    "gc_interval_hours": (float, "hours between automatic cleanups (default 6)"),
    "ignore_collect": (
        bool,
        "skip collecting test files whose tests are all deselected (default true)",
    ),
}

# ini options registered by the plugin: name -> (help, type, default)
INI_OPTIONS = {
    f"testmon_{name}": (text, None, "") for name, (_, text) in SETTINGS.items()
}
INI_OPTIONS["testmon_verify_pythonpath"] = (
    "source roots put on PYTHONPATH by testmon-verify (default: src if it exists)",
    "args",
    [],
)

HOOK_SOURCE = "conftest"


def _coerce(name, value, source):
    kind = SETTINGS[name][0]
    where = f"testmon setting {name!r} (from {source})"
    if kind is bool:
        if isinstance(value, bool):
            return value
        text = str(value).strip().lower()
        if text in ("1", "true", "yes", "on"):
            return True
        if text in ("0", "false", "no", "off"):
            return False
        raise ValueError(f"{where} must be true or false, not {value!r}")
    if kind in (int, float):
        if isinstance(value, bool):
            raise ValueError(f"{where} must be a number, not {value!r}")
        try:
            number_value = kind(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{where} must be a number, not {value!r}") from exc
        if number_value < 0:
            raise ValueError(f"{where} must not be negative")
        return number_value
    if not isinstance(value, (str, os.PathLike)):
        raise ValueError(f"{where} must be a string, not {type(value).__name__}")
    return os.fspath(value)


@dataclass(frozen=True)
class Settings:
    """Resolved settings: {name: (value, source)}; unset names fall back to defaults."""

    values: dict

    def get(self, name, default=None):
        entry = self.values.get(name)
        return default if entry is None else entry[0]

    def source(self, name):
        entry = self.values.get(name)
        return entry[1] if entry else None


def build_settings(hook_values=None, ini_getter=None):
    """Merge conftest hook values over ini values; raise ValueError on anything invalid."""
    if hook_values is None:
        hook_values = {}
    if not isinstance(hook_values, dict):
        raise ValueError(
            "pytest_testmon_settings must return a dict, "
            f"not {type(hook_values).__name__}"
        )
    unknown = sorted(set(hook_values) - set(SETTINGS))
    if unknown:
        raise ValueError(
            f"unknown testmon setting(s) {', '.join(map(repr, unknown))} returned by "
            f"pytest_testmon_settings; known: {', '.join(sorted(SETTINGS))}"
        )
    values = {}
    for name in SETTINGS:
        if hook_values.get(name) is not None:
            values[name] = (_coerce(name, hook_values[name], HOOK_SOURCE), HOOK_SOURCE)
            continue
        raw = ini_getter(f"testmon_{name}") if ini_getter else None
        if raw not in (None, "", []):
            values[name] = (_coerce(name, raw, "ini"), "ini")
    if "db_fallback" in values:
        resolve_fallback(None, values["db_fallback"][0])  # validates
    return Settings(values)


@dataclass(frozen=True)
class DbLocation:
    path: str
    shared: bool
    source: str


def expand_path(value, rootdir):
    path = os.path.expandvars(os.path.expanduser(str(value).strip()))
    if not os.path.isabs(path):
        path = os.path.join(rootdir or os.getcwd(), path)
    return os.path.abspath(path)


def resolve_db_location(
    rootdir, cli_value=None, ini_value=None, environ=None, source="ini"
):
    """ini_value is the configured shared_db (source: "ini" or "conftest")."""
    environ = os.environ if environ is None else environ
    if cli_value:
        return DbLocation(expand_path(cli_value, rootdir), True, "cli")
    if environ.get(SHARED_DB_ENV):
        return DbLocation(
            expand_path(environ[SHARED_DB_ENV], rootdir), True, f"env:{SHARED_DB_ENV}"
        )
    if ini_value:
        return DbLocation(expand_path(ini_value, rootdir), True, source)
    if environ.get(LOCAL_DB_ENV):
        return DbLocation(
            os.path.abspath(os.path.join(rootdir or "", environ[LOCAL_DB_ENV])),
            False,
            f"env:{LOCAL_DB_ENV}",
        )
    return DbLocation(
        os.path.abspath(os.path.join(rootdir or "", LOCAL_DB_FILENAME)),
        False,
        "default",
    )


def resolve_fallback(cli_value=None, ini_value=None):
    """error (default): stop before running tests; run-all: an explicit opt-in."""
    value = (cli_value or ini_value or FALLBACK_ERROR).strip()
    if value not in FALLBACK_POLICIES:
        raise ValueError(
            f"testmon_db_fallback must be one of {', '.join(FALLBACK_POLICIES)}, not {value!r}"
        )
    return value


def number(value, default, kind=float):
    if value in (None, ""):
        return default
    try:
        return kind(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"expected a number, got {value!r}") from exc


def read_project_ini(rootdir):
    """The pytest ini options of a project, for tools that run outside pytest.

    Looks at the same files as pytest, in its order: pytest.ini, pyproject.toml
    ([tool.pytest.ini_options]), tox.ini ([pytest]), setup.cfg ([tool:pytest]).
    """
    rootdir = rootdir or os.getcwd()
    candidates = (
        ("pytest.ini", "pytest"),
        ("pyproject.toml", None),
        ("tox.ini", "pytest"),
        ("setup.cfg", "tool:pytest"),
    )
    for filename, section in candidates:
        path = os.path.join(rootdir, filename)
        if not os.path.isfile(path):
            continue
        if section is None:
            import tomllib  # pylint: disable=import-outside-toplevel

            with open(path, "rb") as file:
                data = tomllib.load(file)
            options = data.get("tool", {}).get("pytest", {}).get("ini_options")
            if options is not None:
                return {key: value for key, value in options.items()}
            continue
        parser = configparser.ConfigParser(interpolation=None)
        parser.read(path, encoding="utf-8")
        if parser.has_section(section):
            return dict(parser.items(section))
        if filename == "pytest.ini":
            return {}
    return {}
