"""Machine-readable explanation of a testmon run (``--testmon-report PATH``).

The schema is versioned (``schema_version``) and checked in as docs/report.schema.json.
Every collected test gets a decision (selected/deselected) and the reason for it, so an agent
can tell why an individual test ran or was skipped without reading the terminal output.
"""

import json
import os
import tempfile
import time

from testmon import TESTMON_VERSION
from testmon import selection as sel
from testmon.common import git_worktree_state
from testmon.config import VERIFIED_TREE_ENV

REPORT_SCHEMA_VERSION = 1


def _test_entry(decision, outcome, decision_obj=None, reason=None, file_ignored=False):
    entry = {
        "decision": decision,
        "reason": reason,
        "observation_id": None,
        "invalid_dep": None,
        "file_ignored": file_ignored,
    }
    if decision_obj is not None:
        entry["observation_id"] = decision_obj.observation_id
        if decision_obj.invalid_dep is not None and decision == "selected":
            entry["invalid_dep"] = decision_obj.invalid_dep.as_dict()
    if outcome is not None:
        entry["outcome"] = outcome
    return entry


def selected_reason(decision_obj, select_active, fallback):
    if fallback:
        return sel.FALLBACK
    if decision_obj is None:
        return sel.NO_OBSERVATION
    if decision_obj.deselect:
        # it was deselectable, but selection was off (-k, --testmon-noselect, ...)
        return sel.DESELECT if select_active else sel.SELECTOR_FORCED
    return decision_obj.reason


def build_report(  # pylint: disable=too-many-locals,too-many-arguments
    config, selected, outcomes, exitstatus, started_at, run_stats=None
):
    testmon_data = getattr(config, "testmon_data", None)
    tm_conf = getattr(config, "testmon_config", None)
    fallback = getattr(config, "testmon_fallback", None)
    location = getattr(config, "testmon_db_location", None)
    timings = dict(getattr(config, "testmon_timings", {}) or {})
    select_active = bool(tm_conf and tm_conf.select) and not fallback
    selection_result = getattr(testmon_data, "selection", None)
    decisions = selection_result.decisions if selection_result else {}
    run_info = getattr(testmon_data, "run_info", None) or {}

    tests = {}
    for nodeid in selected or ():
        decision_obj = decisions.get(nodeid)
        tests[nodeid] = _test_entry(
            "selected",
            outcomes.get(nodeid),
            decision_obj,
            selected_reason(decision_obj, select_active, fallback),
        )
    for nodeid in run_info.get("deselected") or ():
        tests[nodeid] = _test_entry(
            "deselected", None, decisions.get(nodeid), sel.DESELECT
        )
    ignored_files = sorted(run_info.get("ignored_files") or ())
    ignorable = getattr(testmon_data, "ignorable_files", None) or {}
    for filename in ignored_files:
        for nodeid in ignorable.get(filename, ()):
            tests[nodeid] = _test_entry(
                "deselected",
                None,
                decisions.get(nodeid),
                sel.DESELECT,
                file_ignored=True,
            )

    reason_counts = {}
    for entry in tests.values():
        reason_counts[entry["reason"]] = reason_counts.get(entry["reason"], 0) + 1
    n_deselected = sum(
        1 for entry in tests.values() if entry["decision"] == "deselected"
    )
    rootdir = str(config.rootpath)
    timings["total"] = time.time() - started_at
    if testmon_data is not None:
        timings["writes"] = testmon_data.write_time
        timings["gc"] = testmon_data.gc_time
        if selection_result is not None:
            timings["selection"] = selection_result.elapsed

    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "testmon_version": TESTMON_VERSION,
        "rootdir": rootdir,
        "exitstatus": int(exitstatus),
        "db": {
            "path": location.path if location else None,
            "shared": bool(location.shared) if location else False,
            "source": location.source if location else None,
            "new": bool(testmon_data.new_db) if testmon_data is not None else False,
        },
        "mode": {
            "select": bool(tm_conf.select) if tm_conf else False,
            "collect": bool(tm_conf.collect) if tm_conf else False,
            "message": (tm_conf.message if tm_conf else None),
        },
        "fallback": fallback,
        "environment": {
            "id": getattr(testmon_data, "exec_id", None),
            "name": getattr(testmon_data, "environment", None),
            "packages_changed": bool(
                getattr(testmon_data, "system_packages_change", False)
            ),
        },
        "session_id": getattr(testmon_data, "session_id", None),
        "worktree": git_worktree_state(rootdir),
        "verified_tree": os.environ.get(VERIFIED_TREE_ENV) or None,
        "timings": timings,
        "counts": {
            "collected": len(tests),
            "selected": len(tests) - n_deselected,
            "deselected": n_deselected,
            "file_ignored": len(ignored_files),
            "observations_reused": sum(
                1
                for entry in tests.values()
                if entry["decision"] == "deselected" and entry["observation_id"]
            ),
            "observations_known": selection_result.observation_count
            if selection_result
            else 0,
            "dropped_batches": getattr(testmon_data, "dropped_batches", 0),
        },
        "reasons": reason_counts,
        "run_stats": run_stats,
        "ignored_files": ignored_files,
        "tests": tests,
        "warnings": list(getattr(testmon_data, "warnings", []) or []),
    }


def write_report(path, report):
    """Write atomically (temp file + rename) so readers never see a partial report."""
    path = os.path.abspath(path)
    directory = os.path.dirname(path)
    os.makedirs(directory, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".testmon-report-", dir=directory)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as file:
            json.dump(report, file, indent=1, sort_keys=True)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise


def load_report(path):
    with open(path, encoding="utf-8") as file:
        return json.load(file)
