# -*- coding: utf-8 -*-
"""
Main module of testmon pytest plugin.
"""

import os
import sqlite3
import time

from collections import defaultdict
from datetime import date, timedelta

from pathlib import Path
import pytest

from _pytest.config import ExitCode, Config
from _pytest.terminal import TerminalReporter

from testmon.configure import TmConf

from testmon.testmon_core import (
    TestmonCollector,
    eval_environment,
    TestmonData,
    home_file,
    TestmonException,
    get_test_execution_class_name,
    get_test_execution_module_name,
    cached_relpath,
)
from testmon import configure
from testmon import db
from testmon import report as tm_report
from testmon.common import (
    get_logger,
    get_system_packages,
    find_foreign_checkout_imports,
)
from testmon.config import (
    FALLBACK_ERROR,
    FALLBACK_POLICIES,
    INI_OPTIONS,
    REPORT_ENV,
    build_settings,
    resolve_db_location,
    resolve_fallback,
)

SURVEY_NOTIFICATION_INTERVAL = timedelta(days=28)

logger = get_logger(__name__)


class TestmonHookspecs:
    @pytest.hookspec(firstresult=True)
    def pytest_testmon_settings(self, config):
        """Return testmon settings as a dict, e.g. ``{"shared_db": "~/.cache/x.sqlite"}``.

        Implement it in the rootdir's conftest.py (it is called before collection, so a
        conftest.py further down is too late; testmon stops with an error then). Decorate it
        with ``@pytest.hookimpl(optionalhook=True)`` so the conftest still loads where testmon
        isn't installed. Names: see testmon.config.SETTINGS. They take precedence over the
        ``testmon_<name>`` ini options; command line and environment take precedence over both.
        """


def pytest_addhooks(pluginmanager):
    pluginmanager.add_hookspecs(TestmonHookspecs)


def load_settings(config):
    """Resolve settings once: conftest hook over ini options. Invalid -> UsageError."""
    try:
        hook_values = config.hook.pytest_testmon_settings(config=config)
        return build_settings(hook_values, config.getini)
    except ValueError as exc:
        raise pytest.UsageError(f"testmon: {exc}") from exc


class TestmonLateSettingsGuard:
    """A conftest loaded after configuration that sets testmon settings would be ignored."""

    def __init__(self, pluginmanager):
        # pytest_plugin_registered is historic: it is replayed for the plugins (and initial
        # conftests) registered before this guard; those were already taken into account
        self._before = {id(plugin) for plugin in pluginmanager.get_plugins()}

    @pytest.hookimpl
    def pytest_plugin_registered(self, plugin, manager):  # pylint: disable=unused-argument
        if id(plugin) not in self._before and hasattr(
            plugin, "pytest_testmon_settings"
        ):
            where = getattr(plugin, "__file__", repr(plugin))
            raise pytest.UsageError(
                f"testmon: pytest_testmon_settings in {where} is loaded too late to take "
                "effect; move it to the conftest.py in the rootdir"
            )


def pytest_addoption(parser):
    group = parser.getgroup(
        "automatically select tests affected by changes (pytest-testmon)"
    )

    group.addoption(
        "--testmon",
        action="store_true",
        dest="testmon",
        help=(
            "Select tests affected by changes (based on previously collected data) "
            "and collect + write new data (.testmondata file). "
            "Either collection or selection might be deactivated "
            "(sometimes automatically). See below."
        ),
    )

    group.addoption(
        "--testmon-noselect",
        action="store_true",
        dest="testmon_noselect",
        help=(
            "Reorder and prioritize the tests most likely to fail first, but don't deselect anything. "
            "Forced if you use -m, -k, -l, -lf, test_file.py::test_name"
        ),
    )

    group.addoption(
        "--testmon-nocollect",
        action="store_true",
        dest="testmon_nocollect",
        help=(
            "Run testmon but deactivate the collection and writing of testmon data. "
            "Forced if you run under debugger or coverage."
        ),
    )

    group.addoption(
        "--testmon-forceselect",
        action="store_true",
        dest="testmon_forceselect",
        help=(
            "Run testmon and select only tests affected by changes "
            "and satisfying pytest selectors at the same time."
        ),
    )

    group.addoption(
        "--no-testmon",
        action="store_true",
        dest="no-testmon",
        help=(
            "Turn off (even if activated from config by default).\n"
            "Forced if neither read nor write is possible "
            "(debugger plus test selector)."
        ),
    )

    group.addoption(
        "--testmon-env",
        action="store",
        type=str,
        dest="environment_expression",
        default="",
        help=(
            "This allows you to have separate coverage data within one"
            " .testmondata file, e.g. when using the same source"
            " code serving different endpoints or Django settings."
        ),
    )
    group.addoption(
        "--testmon-singleprocess",
        action="store_false",
        dest="testmon_singleprocess",
        help="Don't track subprocesses",
    )
    group.addoption(
        "--tmnet",
        action="store_true",
        dest="tmnet",
        help="Not supported by this fork (testmon.net cloud storage).",
    )
    group.addoption(
        "--testmon-shared-db",
        action="store",
        dest="testmon_shared_db",
        default=None,
        metavar="PATH",
        help=(
            "Use the shared database at PATH (overrides TESTMON_SHARED_DB and the "
            "testmon_shared_db ini option)."
        ),
    )
    group.addoption(
        "--testmon-db-fallback",
        action="store",
        dest="testmon_db_fallback",
        default=None,
        choices=FALLBACK_POLICIES,
        help=(
            "When the database can't be used: error (exit before running any test; the "
            "default) or run-all (run every test, collect nothing)."
        ),
    )
    group.addoption(
        "--testmon-report",
        action="store",
        dest="testmon_report",
        default=None,
        metavar="PATH",
        help="Write a JSON report of the selection decisions to PATH (or $TESTMON_REPORT).",
    )

    parser.addini("environment_expression", "environment expression", default="")
    parser.addini(
        "testmon_ignore_dependencies",
        "ignore dependencies",
        type="args",
        default=[],
    )
    parser.addini("tmnet_url", "URL of the testmon.net api server (unsupported).")
    parser.addini("tmnet_api_key", "testmon api key (unsupported)")
    for name, (help_text, ini_type, default) in INI_OPTIONS.items():
        if ini_type:
            parser.addini(name, help_text, type=ini_type, default=default)
        else:
            parser.addini(name, help_text, default=default)


def testmon_options(config):
    result = []
    for label in [
        "testmon",
        "no-testmon",
        "environment_expression",
        "testmon_singleprocess",
    ]:
        if config.getoption(label):
            result.append(label.replace("testmon_", ""))
    return result


def context_files(config):
    """The pytest config file, when it lives in the rootdir: it influences collection."""
    inipath = getattr(config, "inipath", None)
    if not inipath:
        return ()
    rel = os.path.relpath(str(inipath), str(config.rootpath)).replace(os.sep, "/")
    return () if rel.startswith("..") else (rel,)


def snapshot_allowed(config):
    """Collection is complete per file unless tests were picked individually or by --lf."""
    file_or_dir = config.getoption("file_or_dir", None) or []
    if any("::" in str(arg) for arg in file_or_dir):
        return False
    if getattr(config.option, "lf", False):
        return False
    return True


def gc_policy(config):
    settings = config.testmon_settings
    return {
        "interval_hours": settings.get(
            "gc_interval_hours", db.DEFAULT_GC_INTERVAL_HOURS
        ),
        "retention_days": settings.get("retention_days", db.DEFAULT_RETENTION_DAYS),
        "max_versions": settings.get("max_versions", db.DEFAULT_MAX_VERSIONS),
    }


def init_testmon_data(config: Config):
    environment = config.getoption("environment_expression") or eval_environment(
        config.getini("environment_expression")
    )
    rootdir = str(config.rootpath)
    running_as = get_running_as(config)
    timings = config.testmon_timings

    if running_as == "worker" and hasattr(config, "workerinput"):
        workerinput = config.workerinput
        if workerinput.get("testmon_selection") is not None:
            # the controller decided; the worker needs no database at all
            config.testmon_data = TestmonData.for_worker(
                rootdir=rootdir,
                exec_id=workerinput.get("testmon_exec_id"),
                system_packages_change=workerinput.get(
                    "testmon_system_packages_change"
                ),
                environment=environment,
                selection_payload=workerinput["testmon_selection"],
            )
            return

    settings = config.testmon_settings
    location = resolve_db_location(
        rootdir,
        cli_value=config.getoption("testmon_shared_db"),
        ini_value=settings.get("shared_db"),
        source=settings.source("shared_db") or "ini",
    )
    config.testmon_db_location = location
    timeout = settings.get("db_timeout", db.DEFAULT_TIMEOUT)
    use_snapshots = settings.get("ignore_collect", True)
    project = settings.get("project", "")
    start = time.perf_counter()
    if running_as == "worker":
        # no selection from the controller (e.g. testmon inactive there): decide locally
        testmon_data = TestmonData.for_worker(
            rootdir=rootdir,
            exec_id=config.workerinput.get("testmon_exec_id"),
            environment=environment,
            db_location=location,
        )
        config.testmon_data = testmon_data
        if testmon_data.exec_id is None:
            testmon_data._init_for_local_run(  # pylint: disable=protected-access
                environment,
                get_system_packages(
                    ignore=config.getini("testmon_ignore_dependencies"), rootdir=rootdir
                ),
                None,
                project,
            )
        testmon_data.determine_stable(context_files(config), use_snapshots)
        return

    ignore_dependencies = config.getini("testmon_ignore_dependencies")
    system_packages = get_system_packages(ignore=ignore_dependencies, rootdir=rootdir)
    testmon_data: TestmonData = TestmonData.for_local_run(
        rootdir=rootdir,
        environment=environment,
        system_packages=system_packages,
        project=project,
        db_location=location,
        timeout=timeout,
    )
    config.testmon_data = testmon_data
    timings["db_open"] = time.perf_counter() - start
    testmon_data.determine_stable(context_files(config), use_snapshots)
    tm_conf = config.testmon_config
    testmon_data.begin_session(select=tm_conf.select, collect=tm_conf.collect)


def get_running_as(config):
    if hasattr(config, "workerinput"):
        return "worker"

    if getattr(config.option, "dist", "no") == "no":
        return "single"

    return "controller"


def register_plugins(config, should_select, should_collect, cov_plugin):
    running_as = get_running_as(config)
    if should_select or should_collect:
        config.pluginmanager.register(
            TestmonSelect(config, config.testmon_data), "TestmonSelect"
        )

    if should_collect:
        config.pluginmanager.register(
            TestmonCollect(
                TestmonCollector(
                    str(config.rootpath),
                    testmon_labels=testmon_options(config),
                    cov_plugin=cov_plugin,
                ),
                config.testmon_data,
                running_as=running_as,
            ),
            "TestmonCollect",
        )
    if (should_select or should_collect) and running_as != "worker":
        config.pluginmanager.register(
            TestmonSession(config, config.testmon_data), "TestmonSession"
        )
    if (should_select or should_collect) and config.pluginmanager.hasplugin("xdist"):
        config.pluginmanager.register(
            TestmonXdistSync(sync=should_collect), "TestmonXdistSync"
        )


def fallback_message(error):
    return (
        f"testmon: unusable ({error.reason_code}: {error.detail}) "
        "- running all tests, not collecting (testmon_db_fallback = run-all)"
    )


def handle_db_failure(config, error: db.DbError):
    """Apply the fallback policy: never let a database problem skip a test silently."""
    testmon_data = getattr(config, "testmon_data", None)
    if testmon_data is not None:
        testmon_data.close_connection()
    config.testmon_data = None
    try:
        policy = resolve_fallback(
            config.getoption("testmon_db_fallback"),
            config.testmon_settings.get("db_fallback"),
        )
    except ValueError as exc:
        raise pytest.UsageError(str(exc)) from exc
    config.testmon_fallback = {
        "reason_code": error.reason_code,
        "detail": error.detail,
        "policy": policy,
    }
    if policy == FALLBACK_ERROR:
        write_early_report(config)
        pytest.exit(
            f"testmon: database unavailable ({error.reason_code}: {error.detail}). "
            "No tests were run. Fix the database, or run without --testmon "
            "(testmon_db_fallback = run-all runs everything instead).",
            returncode=ExitCode.USAGE_ERROR,
        )
    config.testmon_config = TmConf(fallback_message(error), False, False)
    logger.warning(fallback_message(error))


def write_early_report(config):
    """The report of a session stopped in pytest_configure (nothing ran)."""
    reporter = config.pluginmanager.get_plugin("TestmonReporter")
    if reporter is None:
        return
    try:
        tm_report.write_report(
            reporter.path,
            tm_report.build_report(
                config, [], {}, ExitCode.USAGE_ERROR, reporter.started_at
            ),
        )
    except OSError as exc:
        logger.warning(
            "testmon: could not write the report to %s: %s", reporter.path, exc
        )


def pytest_configure(config):
    coverage_stack = None
    try:
        from tmnet.testmon_core import (  # pylint: disable=import-outside-toplevel
            Testmon as UberTestmon,
        )

        coverage_stack = UberTestmon.coverage_stack
    except ImportError:
        pass

    config.testmon_fallback = None
    config.testmon_timings = {}
    config.testmon_db_location = None
    if config.getoption("tmnet", False):
        pytest.exit(
            "--tmnet (testmon.net) is not supported by this fork of pytest-testmon; "
            "use a shared SQLite database instead (testmon_shared_db, see docs/shared-db.md)",
            returncode=ExitCode.USAGE_ERROR,
        )

    cov_plugin = None
    cov_plugin = config.pluginmanager.get_plugin("_cov")

    tm_conf = configure.header_collect_select(
        config, coverage_stack, cov_plugin=cov_plugin
    )
    running_as = get_running_as(config)
    if running_as == "worker" and config.workerinput.get("testmon_disabled"):
        tm_conf = TmConf(tm_conf.message, False, False)
    config.testmon_config: TmConf = tm_conf
    config.testmon_settings = build_settings()  # defaults until testmon is requested
    if tm_conf.message is not None:  # testmon was mentioned (even if switched off)
        config.testmon_settings = load_settings(config)
        config.pluginmanager.register(
            TestmonLateSettingsGuard(config.pluginmanager), "TestmonLateSettingsGuard"
        )
    report_path = config.getoption("testmon_report") or os.environ.get(REPORT_ENV)
    if report_path and running_as != "worker" and tm_conf.message is not None:
        config.pluginmanager.register(
            TestmonReporter(config, report_path), "TestmonReporter"
        )
    if tm_conf.select or tm_conf.collect:
        if config.pluginmanager.has_plugin(
            "TestmonSelect"
        ) or config.pluginmanager.has_plugin("TestmonCollect"):
            pytest.exit("You cannot use two instances of testmon plugin.")
        if running_as != "worker":
            foreign = find_foreign_checkout_imports(str(config.rootpath))
            if foreign:
                names = ", ".join(f"{name} -> {origin}" for name, origin in foreign)
                handle_db_failure(
                    config,
                    db.DbError(
                        "foreign_checkout",
                        "packages of this checkout are imported from another checkout "
                        f"({names}); dependency data would describe the wrong code",
                    ),
                )
                return
        try:
            init_testmon_data(config)
            register_plugins(config, tm_conf.select, tm_conf.collect, cov_plugin)
        except TestmonException as error:
            pytest.exit(str(error))
        except db.DbError as error:
            handle_db_failure(config, error)
        except sqlite3.Error as error:
            handle_db_failure(config, db.DbError.from_sqlite(error))


def pytest_report_header(config):
    tm_conf = config.testmon_config
    testmon_data = getattr(config, "testmon_data", None)

    if (tm_conf.collect or tm_conf.select) and testmon_data is not None:
        unstable_files = getattr(testmon_data, "unstable_files", set())
        stable_files = getattr(testmon_data, "stable_files", set())
        environment = testmon_data.environment

        tm_conf.message += changed_message(
            config,
            environment,
            testmon_data.system_packages_change,
            tm_conf.select,
            stable_files,
            unstable_files,
        )
        location = getattr(config, "testmon_db_location", None)
        if location is not None and location.shared:
            reused = len(testmon_data.deselected_tests)
            tm_conf.message += (
                f"\ntestmon: shared db {location.path} "
                f"(deselectable tests: {reused}, source: {location.source})"
            )
        if get_running_as(config) != "worker":
            tm_conf.message += survey_notification(testmon_data)
    return tm_conf.message


def survey_notification(testmon_data):
    try:
        last_notification_date = testmon_data.db.fetch_attribute(
            "last_survey_notification_date"
        )
        if last_notification_date:
            last_notification_date = date.fromisoformat(last_notification_date)
            if date.today() - last_notification_date < SURVEY_NOTIFICATION_INTERVAL:
                return ""
        testmon_data.db.write_attribute(
            "last_survey_notification_date", date.today().isoformat()
        )
    except db.DbError:
        return ""
    return (
        "\nHelp shape testmon! Tell us what saves you time "
        "and what we can improve: https://testmon.org/survey"
    )


def changed_message(
    config,
    environment,
    packages_change,
    should_select,
    stable_files,
    unstable_files,
):
    message = ""
    if should_select:
        changed_files_msg = ", ".join(sorted(unstable_files))
        if changed_files_msg == "" or len(changed_files_msg) > 100:
            changed_files_msg = str(len(config.testmon_data.unstable_files))

        if config.testmon_data.new_db:
            message += "new DB, "
        else:
            message += (
                "The packages installed in your Python environment have been changed. "
                "All tests have to be re-executed. "
                if packages_change
                else f"changed files: {changed_files_msg}, unchanged files: {len(stable_files)}, "
            )
    if config.testmon_data.environment:
        message += f"environment: {environment}"
    return message


def pytest_unconfigure(config):
    testmon_data = getattr(config, "testmon_data", None)
    if testmon_data is not None:
        testmon_data.close_connection()


class TestmonCollect:
    def __init__(
        self, testmon, testmon_data: TestmonData, running_as="single", cov_plugin=None
    ):
        self.testmon_data: TestmonData = testmon_data
        self.testmon: TestmonCollector = testmon
        self._running_as = running_as

        self.reports = defaultdict(lambda: {})
        self.raw_test_names = []
        self.cov_plugin = cov_plugin
        self._sessionstarttime = time.time()

    @pytest.hookimpl(tryfirst=True, hookwrapper=True)
    def pytest_pycollect_makeitem(self, collector, name, obj):  # pylint: disable=unused-argument
        makeitem_result = yield
        items = makeitem_result.get_result() or []
        try:
            self.raw_test_names.extend(
                [item.nodeid for item in items if isinstance(item, pytest.Item)]
            )
        except TypeError:  # 'Class' object is not iterable
            pass

    @pytest.hookimpl(tryfirst=True)
    def pytest_collection_modifyitems(self, session, config, items):  # pylint: disable=unused-argument
        complete = not session.testsfailed and snapshot_allowed(config)
        names = set(self.raw_test_names) | {item.nodeid for item in items}
        if self._running_as == "worker":
            workeroutput = getattr(config, "workeroutput", None)
            if workeroutput is not None:
                workeroutput["testmon_collected"] = sorted(names) if complete else None
            return
        if complete:
            config.testmon_data.record_collection(names, context_files(config))

    @pytest.hookimpl(hookwrapper=True)
    def pytest_runtest_protocol(self, item, nextitem):  # pylint: disable=unused-argument
        self.testmon.start_testmon(item.nodeid, nextitem.nodeid if nextitem else None)
        result = yield
        if result.excinfo and issubclass(result.excinfo[0], BaseException):
            self.testmon.discard_current()

    @pytest.hookimpl(hookwrapper=True)
    def pytest_runtest_makereport(self, item, call):  # pylint: disable=unused-argument
        result = yield

        if call.when == "teardown":
            report = result.get_result()
            report.nodes_files_lines = self.testmon.get_batch_coverage_data()
            result.force_result(
                report
            )  # under xdist, report is serialized on the worker and sent to the controller

    @pytest.hookimpl
    def pytest_runtest_logreport(self, report):
        if self._running_as == "worker":
            return

        self.reports[report.nodeid][report.when] = report
        if report.when == "teardown" and hasattr(report, "nodes_files_lines"):
            if report.nodes_files_lines:
                test_executions_fingerprints = self.testmon_data.get_tests_fingerprints(
                    report.nodes_files_lines, self.reports
                )
                self.testmon_data.save_test_execution_file_fps(
                    test_executions_fingerprints
                )

    def pytest_keyboard_interrupt(self, excinfo):  # pylint: disable=unused-argument
        if self._running_as == "single":
            nodes_files_lines = self.testmon.get_batch_coverage_data()

            test_executions_fingerprints = self.testmon_data.get_tests_fingerprints(
                nodes_files_lines, self.reports
            )
            self.testmon_data.save_test_execution_file_fps(test_executions_fingerprints)
            self.testmon.close()

    def pytest_sessionfinish(self, session):  # pylint: disable=unused-argument
        self.testmon.close()


class TestmonSession:
    """Run statistics, the session row and opportunistic maintenance (single/controller)."""

    def __init__(self, config, testmon_data: TestmonData):
        self.config = config
        self.testmon_data = testmon_data
        self.durations = defaultdict(float)
        self.interrupted = False

    @pytest.hookimpl
    def pytest_runtest_logreport(self, report):
        self.durations[report.nodeid] += report.duration or 0.0

    def pytest_keyboard_interrupt(self, excinfo):  # pylint: disable=unused-argument
        self.interrupted = True

    def deselected_names(self):
        run_info = self.testmon_data.run_info
        names = set(run_info.get("deselected") or ())
        for filename in run_info.get("ignored_files") or ():
            names.update(self.testmon_data.ignorable_files.get(filename, ()))
        return names

    def pytest_sessionfinish(self, session, exitstatus):  # pylint: disable=unused-argument
        select = self.config.testmon_config.select
        recorded = self.testmon_data._durations or {}  # pylint: disable=protected-access
        ran = set(self.durations)
        if select:
            saved = self.deselected_names() - ran
        else:  # potential savings: tests that ran although they were deselectable
            saved = set(self.testmon_data.deselected_tests) & ran
        saved_time = sum(recorded.get(name) or 0 for name in saved)
        run_time = sum(self.durations.values())
        stats = {
            "n_selected": len(ran),
            "n_deselected": len(saved) if select else 0,
            "n_saved": len(saved),
            "n_all": len(ran | saved),
            "saved_time": saved_time,
            "all_time": run_time + (saved_time if select else 0),
        }
        interrupted = self.interrupted or exitstatus == ExitCode.INTERRUPTED
        self.testmon_data.finish_session(
            stats,
            select,
            status="interrupted" if interrupted else "finished",
            gc_policy=gc_policy(self.config),
        )


class TestmonXdistSync:
    def __init__(self, sync=True):
        self.sync = sync
        self._collected_recorded = False

    @pytest.hookimpl(optionalhook=True)
    def pytest_testnodedown(self, node, error):  # pylint: disable=unused-argument
        testmon_data = getattr(node.config, "testmon_data", None)
        workeroutput = getattr(node, "workeroutput", None) or {}
        if testmon_data is None or self._collected_recorded:
            return
        if "testmon_deselected" not in workeroutput:
            return
        self._collected_recorded = True
        run_info = testmon_data.run_info
        run_info["deselected"] = workeroutput.get("testmon_deselected") or []
        run_info["ignored_files"] = workeroutput.get("testmon_ignored_files") or []
        collected = workeroutput.get("testmon_collected")
        if self.sync and collected:
            testmon_data.record_collection(collected, context_files(node.config))


@pytest.hookimpl(optionalhook=True)
def pytest_configure_node(node):
    """
    Tell every xdist worker what the controller decided, before the worker configures.

    Either the controller's selection (every worker must collect exactly the same tests;
    letting each worker query a shared database that other agents keep writing to could
    make them disagree, and the worker then needs no database connection at all) or that
    testmon is off on the controller (not requested, database unusable, foreign checkout),
    in which case the workers must not use it either.

    Note: called on the controller for each worker node; node.config is the controller's.
    """
    if get_running_as(node.config) != "controller" or not hasattr(node, "workerinput"):
        return
    testmon_data = getattr(node.config, "testmon_data", None)
    tm_conf = getattr(node.config, "testmon_config", None)
    if (
        testmon_data is not None
        and tm_conf is not None
        and (tm_conf.select or tm_conf.collect)
    ):
        node.workerinput["testmon_exec_id"] = testmon_data.exec_id
        node.workerinput["testmon_system_packages_change"] = (
            testmon_data.system_packages_change
        )
        node.workerinput["testmon_selection"] = testmon_data.selection_payload()
    else:
        node.workerinput["testmon_disabled"] = True


def did_fail(reports):
    return reports["failed"]


def get_failing(all_test_executions):
    failing_files, failing_tests = set(), {}
    for test_name, result in all_test_executions.items():
        if did_fail(all_test_executions[test_name]):
            failing_files.add(home_file(test_name))
            failing_tests[test_name] = result
    return failing_files, failing_tests


def sort_items_by_duration(items, avg_durations) -> None:
    items.sort(key=lambda item: avg_durations[item.nodeid])
    items.sort(
        key=lambda item: avg_durations[get_test_execution_class_name(item.nodeid)]
    )
    items.sort(
        key=lambda item: avg_durations[get_test_execution_module_name(item.nodeid)]
    )


def format_time_saved(seconds):
    if not seconds:
        seconds = 0
    if seconds >= 3600:
        return f"{int(seconds / 3600)}h {int((seconds % 3600) / 60)}m"
    return f"{int(seconds / 60)}m {int((seconds % 60) % 60)}s"


class TestmonSelect:
    def __init__(self, config, testmon_data: TestmonData):
        self.testmon_data: TestmonData = testmon_data
        self.config = config
        # decided by the selection engine (or received from the xdist controller): tests with
        # a valid passing observation and no valid failing one; files whose every test is so
        self.deselected_files = set(testmon_data.deselected_files)
        self.deselected_tests = set(testmon_data.deselected_tests)
        self.ignored_files = []
        self._interrupted = False

    def pytest_ignore_collect(self, collection_path: Path, config):
        strpath = cached_relpath(str(collection_path), str(config.rootpath))
        if strpath in self.deselected_files and self.config.testmon_config.select:
            self.ignored_files.append(strpath)
            self.testmon_data.run_info["ignored_files"] = self.ignored_files
            return True
        return None

    @pytest.hookimpl(trylast=True)
    def pytest_collection_modifyitems(self, session, config, items):  # pylint: disable=unused-argument
        selected = []
        deselected = []
        for item in items:
            if item.nodeid in self.deselected_tests:
                deselected.append(item)
            else:
                selected.append(item)

        avg_durations = self.testmon_data.avg_durations
        sort_items_by_duration(selected, avg_durations)

        if self.config.testmon_config.select:
            items[:] = selected
            session.config.hook.pytest_deselected(
                items=([FakeItemFromTestmon(session.config)] * len(deselected))
            )
            deselected_names = [item.nodeid for item in deselected]
        else:
            sort_items_by_duration(deselected, avg_durations)
            items[:] = selected + deselected
            deselected_names = []
        self.testmon_data.run_info["deselected"] = deselected_names
        workeroutput = getattr(config, "workeroutput", None)
        if workeroutput is not None:
            workeroutput["testmon_deselected"] = deselected_names
            workeroutput["testmon_ignored_files"] = list(self.ignored_files)

    @pytest.hookimpl(trylast=True)
    def pytest_sessionfinish(self, session, exitstatus):
        run_info = self.testmon_data.run_info
        if exitstatus == ExitCode.NO_TESTS_COLLECTED and (
            run_info.get("deselected") or run_info.get("ignored_files")
        ):
            session.exitstatus = ExitCode.OK

    @pytest.hookimpl(trylast=True)
    def pytest_terminal_summary(self):
        if self._interrupted:
            return

        verbose = self.config.option.verbose
        if verbose >= 1 and self.testmon_data.selection is not None:
            write_reason_summary(self.config, self.testmon_data)

        if not verbose >= 2:
            return

        (
            run_saved_time,
            run_all_time,
            run_saved_tests,
            run_all_tests,
            total_saved_time,
            total_all_time,
            total_saved_tests,
            total_tests_all,
        ) = self.testmon_data.fetch_saving_stats(self.config.testmon_config.select)

        terminal_reporter = TerminalReporter(self.config)
        potential_or_not = ""
        if not self.config.testmon_config.select:
            potential_or_not = "Potential t"
        else:
            potential_or_not = "T"
        terminal_reporter.section(
            f"{potential_or_not}estmon savings (deselected/no testmon)",
            "=",
            **{"blue": True},
        )

        try:
            tests_all_ratio = f"{100.0 * total_saved_tests / total_tests_all:.0f}"
        except ZeroDivisionError:
            tests_all_ratio = "0"
        try:
            tests_current_ratio = f"{100.0 * run_saved_tests / run_all_tests:.0f}"
        except ZeroDivisionError:
            tests_current_ratio = "0"
        msg = f"this run: {run_saved_tests}/{run_all_tests} ({tests_current_ratio}%) tests, "
        msg += format_time_saved(run_saved_time) + "/" + format_time_saved(run_all_time)
        msg += f", all runs: {total_saved_tests}/{total_tests_all} ({tests_all_ratio}%) tests, "
        msg += (
            format_time_saved(total_saved_time)
            + "/"
            + format_time_saved(total_all_time)
        )
        terminal_reporter.write_line(msg)

    def pytest_keyboard_interrupt(self, excinfo):  # pylint: disable=unused-argument
        self._interrupted = True


def write_reason_summary(config, testmon_data):
    counts = defaultdict(int)
    for decision in testmon_data.selection.decisions.values():
        counts[decision.reason] += 1
    if not counts:
        return
    summary = ", ".join(f"{reason}={count}" for reason, count in sorted(counts.items()))
    reporter = config.pluginmanager.get_plugin("terminalreporter")
    if reporter is not None:
        reporter.write_line(f"testmon decisions for known tests: {summary}")


class TestmonReporter:
    """Writes the --testmon-report JSON at the end of the session (single/controller)."""

    def __init__(self, config, path):
        self.config = config
        self.path = path
        self.selected = None
        self.outcomes = {}
        self.started_at = time.time()

    @pytest.hookimpl(hookwrapper=True)
    def pytest_collection_modifyitems(self, session, config, items):  # pylint: disable=unused-argument
        yield
        self.selected = [item.nodeid for item in items]

    @pytest.hookimpl(optionalhook=True)
    def pytest_xdist_node_collection_finished(self, node, ids):  # pylint: disable=unused-argument
        if self.selected is None:
            self.selected = list(ids)

    @pytest.hookimpl
    def pytest_runtest_logreport(self, report):
        if report.when == "call" or report.outcome != "passed":
            previous = self.outcomes.get(report.nodeid)
            if previous in (None, "passed"):
                self.outcomes[report.nodeid] = report.outcome

    @pytest.hookimpl(trylast=True)
    def pytest_sessionfinish(self, session, exitstatus):
        testmon_data = getattr(self.config, "testmon_data", None)
        try:
            report = tm_report.build_report(
                self.config,
                self.selected or [],
                self.outcomes,
                exitstatus,
                self.started_at,
                run_stats=getattr(testmon_data, "run_stats", None),
            )
            tm_report.write_report(self.path, report)
        except OSError as exc:
            logger.warning(
                "testmon: could not write the report to %s: %s", self.path, exc
            )


class FakeItemFromTestmon:  # pylint: disable=too-few-public-methods
    def __init__(self, config):
        self.config = config
