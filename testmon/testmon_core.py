import hashlib
import os
import random
import sys
import sysconfig
import textwrap
import time
from functools import lru_cache
from collections import defaultdict

from typing import TypeVar

try:
    from pytest_cov.plugin import CovPlugin
except ImportError:
    pass
import pytest
from coverage import Coverage, CoverageData
from coverage.exceptions import CoverageException

from testmon import db
from testmon import selection
from testmon.dynamo import DynamoTracker, DynamoUnsupported
from testmon import TESTMON_VERSION as TM_CLIENT_VERSION
from testmon.common import (
    get_logger,
    get_system_packages,
    drop_patch_version,
    git_current_head,
)
from testmon.config import resolve_db_location

from testmon.process_code import (
    match_fingerprint,
    create_fingerprint,
    methods_to_checksums,  # noqa: F401  (re-exported for tests)
    get_files_shas,
    get_source_sha,
    read_source_sha,
    Module,
)

from testmon.common import DepsNOutcomes, TestExecutions

T = TypeVar("T")

TEST_BATCH_SIZE = 250

CHECKUMS_ARRAY_TYPE = "I"
DB_FILENAME = ".testmondata"

logger = get_logger(__name__)


def get_data_file_path():
    return os.environ.get("TESTMON_DATAFILE", DB_FILENAME)


def home_file(test_execution_name):
    return test_execution_name.split("::", 1)[0]


def is_python_file(file_path):
    return file_path[-3:] == ".py"


class TestmonException(Exception):
    pass


class SourceTree:
    """
    - reads files from file_system and caches them in memory
      (if the file was once read, let's use the on memory
      copy for further operations. Also caches the mtime.)
    - store rootdir and convert between relative and absolute paths
    - implement the check if the file is changed based on mtime, fsha.
    - implement the ckeck if the node is stable based on fingerprint
    - mockability (unit tests without filesystem, check if the file is changed
      based on mtime, fsha)
    """

    def __init__(self, rootdir, packages=None):
        self.rootdir = rootdir
        self.packages = packages
        self.cache: dict = {}

    def get_file(self, filename):
        if filename not in self.cache:
            if filename.startswith(selection.ABS_PREFIX):
                path = selection.strip_abs(filename)
                code, fsha = read_source_sha(path)
                directory, name = "", path
            else:
                path = os.path.join(self.rootdir, filename)
                if filename not in get_files_shas(self.rootdir) and not os.path.exists(
                    path
                ):
                    self.cache[filename] = None
                    return None
                code, fsha = get_source_sha(directory=self.rootdir, filename=filename)
                directory, name = self.rootdir, filename
            if fsha:
                base = os.path.basename(name)
                try:
                    fs_mtime = os.path.getmtime(path)
                    self.cache[filename] = Module(
                        source_code=code,
                        mtime=fs_mtime,
                        ext=base.rsplit(".", 1)[1] if "." in base else "",
                        fs_fsha=fsha,
                        filename=name,
                        rootdir=directory,
                    )
                except FileNotFoundError:
                    self.cache[filename] = None
            else:
                self.cache[filename] = None
        return self.cache[filename]


def check_mtime(file_system: SourceTree, record):
    absfilename = os.path.join(file_system.rootdir, record["filename"])

    cache_module = file_system.cache.get(record["filename"], None)
    try:
        fs_mtime = cache_module.mtime if cache_module else os.path.getmtime(absfilename)
    except OSError:
        return False
    return record["mtime"] == fs_mtime


def check_fsha(file_system, record):
    cache_module = file_system.get_file(record["filename"])
    fs_fsha = cache_module.fs_fsha if cache_module else None

    return record["fsha"] == fs_fsha


def check_fingerprint(disk, record):  # filename name method_fshas id failed
    file = record[0]
    fingerprint = record[2]

    module = disk.get_file(file)
    return module and match_fingerprint(module, fingerprint)


def split_filter(disk, function, records: [T]) -> ([T], [T]):
    first = []
    second = []
    for record in records:
        if function(disk, record):
            first.append(record)
        else:
            second.append(record)
    return first, second


@lru_cache(maxsize=1000)
def should_include(cov, filename):
    return cov._should_trace(str(filename), None).trace


def collect_mhashes(source_tree, new_changed_file_data):
    files_mhashes = {}
    for filename in new_changed_file_data:
        module = source_tree.get_file(filename)
        files_mhashes[filename] = module.method_checksums if module else None
    return files_mhashes


class TestmonData:  # pylint: disable=too-many-instance-attributes,too-many-public-methods
    """The plugin's view of the database for one pytest process.

    Selection state (stable/unstable tests, deselectable tests and files) comes from
    testmon.selection; observations are written through testmon.db. On an xdist worker the
    controller's selection arrives through workerinput and no database is opened.
    """

    __test__ = False

    def __init__(
        self,
        rootdir,
        database=None,
        readonly=False,
        db_location=None,
        timeout=db.DEFAULT_TIMEOUT,
        open_db=True,
    ):
        self.rootdir = rootdir
        self.source_tree = SourceTree(rootdir=self.rootdir)
        get_files_shas.cache_clear()  # a long-lived process may run several sessions
        self.db_location = None
        if database is not None:
            self.db = database  # pylint: disable=invalid-name
        elif not open_db:
            self.db = None
        else:
            self.db_location = db_location or resolve_db_location(rootdir)
            self.db = db.DB(
                self.db_location.path,
                readonly=readonly,
                shared=self.db_location.shared,
                timeout=timeout,
            )

        # Initialize instance variables (will be set by init methods)
        self.environment = None
        self.exec_id = None
        self.system_packages_change = None
        self.files_of_interest = None
        self.all_files = {}
        self.unstable_test_names = None
        self.unstable_files = None
        self.stable_test_names = None
        self.stable_files = None
        self.failing_tests = None
        self.selection: selection.SelectionResult = None
        self.deselected_tests = set()
        self.deselected_files = set()
        self.ignorable_files = {}
        self._durations = None
        self.session_id = None
        self.dropped_batches = 0
        self.write_time = 0.0
        self.gc_time = 0.0
        self.warnings = []
        self.run_info = {"selected": None, "deselected": [], "ignored_files": []}
        self.run_stats = None

    # ------------------------------------------------------------------ construction

    @classmethod
    def for_local_run(  # pylint: disable=too-many-arguments
        cls,
        rootdir,
        database=None,
        environment=None,
        system_packages=None,
        python_version=None,
        project="",
        db_location=None,
        timeout=db.DEFAULT_TIMEOUT,
    ):
        instance = cls(
            rootdir,
            database=database,
            readonly=False,
            db_location=db_location,
            timeout=timeout,
        )
        try:
            instance._init_for_local_run(
                environment, system_packages, python_version, project
            )
        except BaseException:
            instance.close_connection()
            raise
        return instance

    def _init_for_local_run(
        self,
        environment=None,
        system_packages=None,
        python_version=None,
        project="",
    ):
        """Internal method to initialize for local run."""
        self.environment = environment if environment else "default"

        if system_packages is None:
            system_packages = get_system_packages()
        system_packages = drop_patch_version(system_packages)

        if not python_version:
            python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"

        result = self.db.initiate_execution(
            self.environment,
            system_packages,
            python_version,
            {"tm_client_version": TM_CLIENT_VERSION, "ci": os.environ.get("CI")},
            project=project or "",
        )
        self.exec_id = result["exec_id"]
        self.system_packages_change = result["packages_changed"]
        self.files_of_interest = result["filenames"]

    @classmethod
    def for_worker(  # pylint: disable=too-many-arguments
        cls,
        rootdir,
        exec_id,
        database=None,
        system_packages_change=None,
        files_of_interest=None,
        environment=None,
        selection_payload=None,
        db_location=None,
    ):
        if selection_payload is not None:
            instance = cls(rootdir, database=database, open_db=False)
            instance.apply_selection_payload(selection_payload)
        else:
            instance = cls(
                rootdir, database=database, readonly=True, db_location=db_location
            )
        instance._init_for_worker(
            exec_id, system_packages_change, files_of_interest, environment
        )
        return instance

    def _init_for_worker(
        self,
        exec_id,
        system_packages_change=None,
        files_of_interest=None,
        environment=None,
    ):
        """Internal method to initialize for worker run."""
        self.exec_id = exec_id
        self.system_packages_change = system_packages_change
        self.files_of_interest = files_of_interest
        if environment:
            self.environment = environment
        elif not hasattr(self, "environment") or self.environment is None:
            self.environment = "default"

    # ------------------------------------------------------------------ xdist transfer

    def selection_payload(self):
        """What a worker needs to make the controller's selection, as plain data."""
        return {
            "deselected_tests": sorted(self.deselected_tests),
            "deselected_files": sorted(self.deselected_files),
            "durations": dict(self._durations or {}),
            "stable_test_names": sorted(self.stable_test_names or ()),
        }

    def apply_selection_payload(self, payload):
        self.deselected_tests = set(payload["deselected_tests"])
        self.deselected_files = set(payload["deselected_files"])
        self._durations = dict(payload["durations"])
        self.stable_test_names = set(payload.get("stable_test_names", ()))
        self.unstable_test_names = set()
        self.failing_tests = []
        self.all_files = set()
        self.stable_files = set()
        self.unstable_files = set()

    # ------------------------------------------------------------------ connection

    @property
    def new_db(self):
        if isinstance(self.db, db.DB):
            return self.db.file_created
        return False

    def close_connection(self):
        """Release the sqlite connection. Idempotent.

        Tests that inspect the data after a run reopen it (see tests/lib4tests.py).
        """
        database, self.db = self.db, None
        if database is not None:
            database.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close_connection()

    def warn(self, message):
        if message not in self.warnings:
            self.warnings.append(message)
            logger.warning(message)

    # ------------------------------------------------------------------ queries

    @property
    def all_tests(self):
        if self.db is None:
            return {}
        return self.db.all_test_executions(self.exec_id)

    @property
    def avg_durations(self) -> dict:
        if self._durations is not None:
            durations_by_test = self._durations
        else:
            durations_by_test = {
                name: (report or {}).get("duration")
                for name, report in self.all_tests.items()
            }
        stats = defaultdict(lambda: {"test_execution": 0, "sum_duration": 0})
        for test_execution_id, duration in durations_by_test.items():
            class_name = get_test_execution_class_name(test_execution_id)
            module_name = get_test_execution_module_name(test_execution_id)

            stats[test_execution_id]["test_execution"] += 1
            stats[test_execution_id]["sum_duration"] = duration or 0
            if class_name:
                stats[class_name]["test_execution"] += 1
                stats[class_name]["sum_duration"] += stats[test_execution_id][
                    "sum_duration"
                ]
            stats[module_name]["test_execution"] += 1
            stats[module_name]["sum_duration"] += stats[test_execution_id][
                "sum_duration"
            ]

        durations = defaultdict(lambda: 0)
        for key, stat in stats.items():
            durations[key] = stat["sum_duration"] / stat["test_execution"]

        return durations

    # ------------------------------------------------------------------ selection

    def determine_stable(self, context_files=(), use_snapshots=True):
        result = selection.compute_selection(
            self.db,
            self.exec_id,
            self.source_tree,
            context_files=context_files,
            use_snapshots=use_snapshots,
        )
        self.apply_selection(result)
        try:
            self.db.record_selection_side_effects(
                result.new_memo, result.used_observation_ids
            )
        except db.DbError as exc:
            self.warn(f"testmon: could not store selection cache ({exc})")

    def apply_selection(self, result: selection.SelectionResult):
        self.selection = result
        self.files_of_interest = result.filenames
        self.all_files = set(result.filenames)
        self.unstable_test_names = result.unstable
        self.failing_tests = sorted(result.failing)
        self.stable_test_names = result.stable
        self.unstable_files = {home_file(name) for name in self.unstable_test_names}
        self.stable_files = set(self.all_files) - self.unstable_files
        self.deselected_tests = result.deselectable
        self.ignorable_files = result.ignorable_files
        self.deselected_files = set(result.ignorable_files)
        self._durations = dict(result.durations)

    # ------------------------------------------------------------------ recording

    def get_tests_fingerprints(self, nodes_files_lines, reports) -> TestExecutions:
        """Observations of the tests in nodes_files_lines.

        A test is left out (it will simply run again next time) when one of the files it
        executed is gone or changed while the session ran: its observation would be
        incomplete or would pair a fingerprint with the wrong content.
        """
        test_executions_fingerprints = {}
        for context in nodes_files_lines:
            if was_skipped(reports[context]):
                # a skip (marker, skipif, pytest.skip(), a conftest flag turning the test
                # off) says nothing about what the test depends on; recording it as a pass
                # would later deselect the test when it is enabled and never ran
                continue
            deps = []
            complete = True
            for filename, covered in nodes_files_lines[context].items():
                if filename.startswith(selection.ABS_PREFIX):
                    path = selection.strip_abs(filename)
                else:
                    path = os.path.join(self.rootdir, filename)
                module = (
                    self.source_tree.get_file(filename)
                    if os.path.exists(path)
                    else None
                )
                if module is None:
                    complete = False
                    break
                fingerprint = create_fingerprint(module, covered)
                if not module.consistent:
                    complete = False
                    break
                deps.append(
                    {
                        "filename": filename,
                        "fsha": module.fs_fsha,
                        "method_checksums": fingerprint,
                    }
                )
            if not complete or not deps:
                self.warn(
                    "testmon: some files changed or disappeared while tests ran; "
                    "the affected tests were not recorded and will run again"
                )
                continue

            deps_n_outcomes: DepsNOutcomes = {"deps": deps}
            deps_n_outcomes.update(process_result(reports[context]))
            deps_n_outcomes["forced"] = context in (self.stable_test_names or ()) and (
                context not in (self.failing_tests or ())
            )
            test_executions_fingerprints[context] = deps_n_outcomes
        return test_executions_fingerprints

    def save_test_execution_file_fps(self, test_executions_fingerprints):
        """Store one batch; a database failure loses the batch, never the test results."""
        if not test_executions_fingerprints:
            return
        start = time.perf_counter()
        try:
            self.db.insert_test_file_fps(
                test_executions_fingerprints,
                self.exec_id,
                supersede=self.selection.valid_failing if self.selection else None,
            )
        except db.DbError as exc:
            self.dropped_batches += 1
            self.warn(
                f"testmon: could not store dependency data ({exc}); "
                "the affected tests will run again next time"
            )
        finally:
            self.write_time += time.perf_counter() - start

    def record_collection(self, nodeids, context_files=()):
        """Remember which tests each collected test file produced (for file-level skipping)."""
        by_file = defaultdict(set)
        for nodeid in nodeids:
            by_file[home_file(nodeid)].add(nodeid)
        snapshots = []
        for filename, names in by_file.items():
            if not is_python_file(filename):
                continue
            ctx = selection.context_hash(self.source_tree, filename, context_files)
            if ctx:
                snapshots.append((filename, ctx, sorted(names)))
        try:
            self.db.record_collection_snapshots(self.exec_id, snapshots)
        except db.DbError as exc:
            self.warn(f"testmon: could not store the collection snapshot ({exc})")

    def sync_db_fs_tests(self, retain, context_files=()):
        """Upstream name. Nothing is deleted any more: other branches' tests stay."""
        self.record_collection(retain, context_files)

    # ------------------------------------------------------------------ session

    def begin_session(self, select, collect):
        try:
            self.session_id = self.db.start_session(
                self.exec_id,
                select,
                collect,
                rootdir_hash=hashlib.sha256(
                    os.path.realpath(self.rootdir or ".").encode()
                ).hexdigest()[:16],
                git_head=git_current_head(self.rootdir or None),
            )
        except db.DbError as exc:
            self.warn(f"testmon: could not record the session ({exc})")

    def finish_session(self, stats, select, status="finished", gc_policy=None):
        """Store run statistics and do opportunistic maintenance; failures only warn."""
        self.run_stats = stats
        prefix = "" if select else "potential_"
        try:
            self.db.increment_attributes(
                {
                    f"{prefix}time_saved": stats["saved_time"],
                    f"{prefix}time_all": stats["all_time"],
                    f"{prefix}tests_saved": stats["n_saved"],
                    f"{prefix}tests_all": stats["n_all"],
                },
                exec_id=None,
            )
            if self.session_id:
                self.db.finish_session(
                    self.session_id,
                    status=status,
                    n_collected=stats["n_all"],
                    n_selected=stats["n_selected"],
                    n_deselected=stats["n_deselected"],
                    saved_time=stats["saved_time"],
                    all_time=stats["all_time"],
                    dropped_batches=self.dropped_batches,
                )
        except db.DbError as exc:
            self.warn(f"testmon: could not store run statistics ({exc})")
        start = time.perf_counter()
        try:
            self.db.maybe_gc(**(gc_policy or {}))
        except db.DbError as exc:
            self.warn(f"testmon: database cleanup failed ({exc})")
        finally:
            self.gc_time = time.perf_counter() - start

    def fetch_saving_stats(self, select):
        stats = self.run_stats or {
            "saved_time": 0,
            "all_time": 0,
            "n_saved": 0,
            "n_all": 0,
        }
        totals = self.db.fetch_saving_totals(select)
        return (
            stats["saved_time"],
            stats["all_time"],
            stats["n_saved"],
            stats["n_all"],
            *totals,
        )


def get_new_mtimes(filesystem, hits):
    """hits: [(filename, _, _, fingerprint_id)]"""
    try:
        for hit in hits:
            module = filesystem.get_file(hit[0])
            if module:
                yield module.mtime, module.fs_fsha, hit[3]
    except KeyError:
        for hit in hits:
            module = filesystem.get_file(hit["filename"])
            if module:
                yield module.mtime, module.fs_fsha, hit["fingerprint_id"]


def get_test_execution_class_name(node_id):
    if len(node_id.split("::")) > 2:
        return node_id.split("::")[1]
    return None


def get_test_execution_module_name(node_id):
    return node_id.split("::")[0]


@lru_cache(1000)
def cached_relpath(path, basepath):
    return os.path.relpath(path, basepath).replace(os.sep, "/")


@lru_cache(4000)
def cached_dep_name(path, real_rootdir):
    """Dependency name of a measured file (coverage reports realpaths)."""
    return selection.realpath_relative(path, real_rootdir)


# coverage 7.10+ defaults to the sys.monitoring core on CPython 3.14+. That core
# doesn't support dynamic contexts: sys.monitoring disables a line event once the
# line has been recorded, so a line executed by several tests of one batch is
# attributed to the first test only and the other tests lose the dependency.
# testmon relies on contexts_by_lineno, so it needs the C tracer.
COVERAGE_CORE = "ctrace"


def force_context_capable_core(cov: Coverage):
    try:
        cov.set_option("run:core", COVERAGE_CORE)
    except CoverageException:
        # coverage < 7.10 has no core option and never defaults to sys.monitoring.
        pass


class TestmonCollector:
    coverage_stack: [Coverage] = []

    def __init__(
        self, rootdir, testmon_labels=None, cov_plugin=None
    ):  # TODO remove cov_plugin
        try:
            from testmon.testmon_core import (  # pylint: disable=import-outside-toplevel
                Testmon as UberTestmon,
            )

            TestmonCollector.coverage_stack: [Coverage] = UberTestmon.coverage_stack
        except ImportError:
            pass
        if testmon_labels is None:
            testmon_labels = {"singleprocess"}
        self.rootdir = rootdir
        self.real_rootdir = os.path.realpath(rootdir)
        self.testmon_labels = testmon_labels
        self.cov: Coverage = None
        self.sub_cov_file = None
        self.cov_plugin: CovPlugin = cov_plugin
        self._test_name = None
        self._next_test_name = None
        self.batched_test_names = set()
        self.check_stack = []
        self.is_started = False
        self._interrupted_at = None
        # code compiled by torch.compile never executes as Python lines: see testmon/dynamo.py
        self.dynamo = DynamoTracker()
        self._dynamo_lines = {}  # test name -> {filename: lines traced by dynamo}
        self._dynamo_failed = False

    def start_cov(self):
        if not self.cov._started:
            TestmonCollector.coverage_stack.append(self.cov)
            self.cov.start()

    def stop_cov(self):
        if self.cov is None:
            return
        assert self.cov in TestmonCollector.coverage_stack
        if TestmonCollector.coverage_stack:
            while TestmonCollector.coverage_stack[-1] != self.cov:
                cov = TestmonCollector.coverage_stack.pop()
                cov.stop()
        if self.cov._started:
            self.cov.stop()
            TestmonCollector.coverage_stack.pop()
        if TestmonCollector.coverage_stack:
            TestmonCollector.coverage_stack[-1].start()

    def setup_coverage(self, subprocess=False):
        params = {
            "include": [os.path.join(self.rootdir, "*")],
            "omit": {
                os.path.join(value, "*")
                for key, value in sysconfig.get_paths().items()
                if key.endswith("lib")
            },
        }
        if self.cov_plugin and self.cov_plugin._started:
            cov = self.cov_plugin.cov_controller.cov
            TestmonCollector.coverage_stack.append(cov)
            if cov.config.source:
                params["include"] = list(
                    set(
                        [os.path.join(self.rootdir, "*")]
                        + [
                            os.path.join(os.path.abspath(source), "*")
                            for source in cov.config.source
                        ]
                    )
                )
            elif cov.config.run_include:
                params["include"] = list(
                    set(cov.config.run_include + params["include"])
                )
            # params["omit"] = cov.config.run_omit
            if cov.config.branch:
                raise TestmonException(
                    "testmon doesn't support simultaneous run with pytest-cov when "
                    "branch coverage is on. Please disable branch coverage."
                )
        if subprocess:
            self.setup_subprocess(params["include"])
        self.cov = Coverage(data_file=self.sub_cov_file, config_file=False, **params)
        self.cov._warn_no_data = False
        force_context_capable_core(self.cov)
        if TestmonCollector.coverage_stack:
            TestmonCollector.coverage_stack[-1].stop()

        self.start_cov()

    def start_testmon(self, test_name, next_test_name=None):
        self._next_test_name = next_test_name

        self.batched_test_names.add(test_name)
        if self.cov is None:
            self.setup_coverage()
            self._dynamo_call(self.dynamo.start)

        self._dynamo_call(self.dynamo.begin_test)
        self.start_cov()
        self._test_name = test_name
        self.cov.switch_context(test_name)
        self.check_stack = TestmonCollector.coverage_stack.copy()

    def discard_current(self):
        self._interrupted_at = self._test_name

    def _dynamo_call(self, function):
        if self._dynamo_failed:
            return None
        try:
            return function()
        except DynamoUnsupported as exc:
            # dependencies can't be complete any more: record nothing from now on (the
            # interrupt handler would otherwise save the batch in progress) and stop
            self._dynamo_failed = True
            pytest.exit(f"testmon: {exc}", returncode=3)
        return None

    def harvest_dynamo(self):
        """Attribute what dynamo traced during the current test (coverage can't see it).

        Called from pytest_runtest_teardown (after fixture teardown), where an unsupported
        torch can stop the session cleanly; get_batch_coverage_data calls it again (no-op).
        """
        traced = self._dynamo_call(self.dynamo.harvest)
        if not traced or not self._test_name or self.cov is None:
            return
        files = self._dynamo_lines.setdefault(self._test_name, {})
        for filename, lines in traced.items():
            if should_include(self.cov, filename):
                files.setdefault(filename, set()).update(lines)

    def get_batch_coverage_data(self):
        self.harvest_dynamo()
        if self._dynamo_failed:
            return {}  # never store dependencies that may lack compiled code
        if self.check_stack != TestmonCollector.coverage_stack:
            pytest.exit(
                (
                    "Exiting pytest!!!! This test corrupts Testmon.coverage_stack:"
                    f" {self._test_name} {self.check_stack},"
                    f" {TestmonCollector.coverage_stack}"
                ),
                returncode=3,
            )

        nodes_files_lines = {}

        if self.cov and (
            len(self.batched_test_names) >= TEST_BATCH_SIZE
            or self._next_test_name is None
            or self._interrupted_at
        ):
            self.cov.stop()
            nodes_files_lines, lines_data = self.get_nodes_files_lines(
                dont_include=self._interrupted_at
            )

            if (
                len(TestmonCollector.coverage_stack) > 1
                and TestmonCollector.coverage_stack[-1] == self.cov
            ):
                filtered_lines_data = {
                    file: data
                    for file, data in lines_data.items()
                    if should_include(TestmonCollector.coverage_stack[-2], file)
                }
                TestmonCollector.coverage_stack[-2].get_data().add_lines(
                    filtered_lines_data
                )

            self.cov.erase()
            self.cov.start()
            self.batched_test_names = set()
        return nodes_files_lines

    def get_nodes_files_lines(self, dont_include):
        cov_data: CoverageData = self.cov.get_data()
        files = cov_data.measured_files()
        nodes_files_lines = {}
        files_lines = {}
        for file in files:
            relfilename = cached_dep_name(file, self.real_rootdir)

            contexts_by_lineno = cov_data.contexts_by_lineno(file)

            for lineno, contexts in contexts_by_lineno.items():
                for context in contexts:
                    nodes_files_lines.setdefault(context, {}).setdefault(
                        relfilename, set()
                    ).add(lineno)
                    files_lines.setdefault(file, set()).add(lineno)
        dynamo_lines, self._dynamo_lines = self._dynamo_lines, {}
        for context, traced in dynamo_lines.items():
            for file, lines in traced.items():
                relfilename = cached_dep_name(file, self.real_rootdir)
                nodes_files_lines.setdefault(context, {}).setdefault(
                    relfilename, set()
                ).update(lines)
        nodes_files_lines.pop(dont_include, None)
        self.batched_test_names.discard(dont_include)
        nodes_files_lines.pop("", None)
        for test_name in self.batched_test_names:
            if home_file(test_name) not in nodes_files_lines.setdefault(test_name, {}):
                nodes_files_lines[test_name].setdefault(home_file(test_name), {1})
        return nodes_files_lines, files_lines

    def close(self):
        self.dynamo.stop()
        if self.cov is None:
            return
        assert self.cov in TestmonCollector.coverage_stack
        if TestmonCollector.coverage_stack:
            while TestmonCollector.coverage_stack[-1] != self.cov:
                cov = TestmonCollector.coverage_stack.pop()
                cov.stop()
        if self.cov._started:
            self.cov.stop()
            TestmonCollector.coverage_stack.pop()
        if self.sub_cov_file:
            os.remove(self.sub_cov_file + "_rc")
            os.environ.pop("COVERAGE_CORE", None)
        os.environ.pop("COVERAGE_PROCESS_START", None)
        self.cov = None
        if TestmonCollector.coverage_stack:
            TestmonCollector.coverage_stack[-1].start()

    def setup_subprocess(self, includes):
        # This can be redone to use functionality from pytest-cov
        if not os.path.exists(".tmontmp"):
            os.makedirs(".tmontmp")
        self.sub_cov_file = os.path.abspath(
            ".tmontmp/.testmoncoverage" + str(random.randint(0, 1000000))
        )
        with open(self.sub_cov_file + "_rc", "w", encoding="utf-8") as subprocess_rc:
            rc_content = textwrap.dedent(
                """\
                    [run]
                    data_file = {}
                    include = {}
                    parallel=True
                    """
            ).format(
                self.sub_cov_file,
                "\n ".join(includes),
            )
            subprocess_rc.write(rc_content)
        os.environ["COVERAGE_PROCESS_START"] = self.sub_cov_file + "_rc"
        os.environ["COVERAGE_CORE"] = COVERAGE_CORE


def eval_environment(environment, **kwargs):
    if not environment:
        return ""

    def md5(string):
        return hashlib.md5(string.encode()).hexdigest()

    eval_globals = {"os": os, "sys": sys, "hashlib": hashlib, "md5": md5}
    eval_globals.update(kwargs)

    try:
        return str(eval(environment, eval_globals))
    except Exception as error:  # pylint: disable=broad-except
        return repr(error)


def was_skipped(result):
    """Skipped in any phase (an expected failure, xfail, did run: it is recorded)."""
    return any(
        report.skipped and not hasattr(report, "wasxfail") for report in result.values()
    )


def process_result(result):
    failed = any(r.outcome == "failed" for r in result.values())
    duration = sum(value.duration for value in result.values())
    return {"failed": failed, "duration": duration}
