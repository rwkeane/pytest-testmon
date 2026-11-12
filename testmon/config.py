"""Where the database lives and what to do when it can't be used.

Precedence for the database location (first match wins):

1. ``--testmon-shared-db PATH``            (shared mode)
2. ``TESTMON_SHARED_DB`` environment var   (shared mode)
3. ``testmon_shared_db`` ini option        (shared mode; commit it in the project's config)
4. ``TESTMON_DATAFILE`` environment var    (private mode, upstream-compatible)
5. ``<rootdir>/.testmondata``              (private mode, upstream default)

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

# ini options registered by the plugin: name -> (help, type, default)
INI_OPTIONS = {
    "testmon_shared_db": ("path of the shared testmon database", None, ""),
    "testmon_db_fallback": (
        "what to do when the database is unusable: error (default) or run-all",
        None,
        "",
    ),
    "testmon_db_timeout": (
        "seconds to wait for a database lock (default 30)",
        None,
        "",
    ),
    "testmon_project": (
        "namespace for this project's data in a shared database",
        None,
        "",
    ),
    "testmon_retention_days": (
        "days an unused observation is kept (default 60)",
        None,
        "",
    ),
    "testmon_max_versions": (
        "observations kept per test and environment (default 32)",
        None,
        "",
    ),
    "testmon_gc_interval_hours": (
        "hours between automatic cleanups (default 6)",
        None,
        "",
    ),
    "testmon_ignore_collect": (
        "skip collecting test files whose tests are all deselected (default true)",
        "bool",
        True,
    ),
    "testmon_verify_pythonpath": (
        "source roots put on PYTHONPATH by testmon-verify (default: src if it exists)",
        "args",
        [],
    ),
}


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


def resolve_db_location(rootdir, cli_value=None, ini_value=None, environ=None):
    environ = os.environ if environ is None else environ
    if cli_value:
        return DbLocation(expand_path(cli_value, rootdir), True, "cli")
    if environ.get(SHARED_DB_ENV):
        return DbLocation(
            expand_path(environ[SHARED_DB_ENV], rootdir), True, f"env:{SHARED_DB_ENV}"
        )
    if ini_value:
        return DbLocation(expand_path(ini_value, rootdir), True, "ini")
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
