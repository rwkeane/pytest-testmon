"""Decide which tests can be deselected, from all stored observations of each test.

Correctness rule (a generalisation of upstream testmon's): an observation is *valid* when
every file it depends on exists and still contains every code block the test executed
(``match_fingerprint``). Then the test would execute exactly the same code again, wherever
the observation came from. A test is deselected iff it has a valid passing observation and
no valid failing one (superseded failures excepted). Anything else runs.

compute_selection() either returns a complete result or raises; a partial result is never
produced. The decision rule itself is the pure function decide(), so it can be checked
against a naive reference implementation (tests/unit/test_selection_property.py).
"""

import hashlib
import os
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

from testmon import db as dbmod
from testmon.process_code import blob_to_checksums, match_fingerprint

# Decision reasons (stable strings: they are part of the JSON report schema)
DESELECT = "deselect"
NO_OBSERVATION = "no_observation"
ALL_INVALID = "all_invalid"
VALID_FAILING = "valid_failing_observation"
SELECTOR_FORCED = "selector_forced"
FALLBACK = "fallback"

MISSING = "missing"
MISMATCH = "mismatch"
INCOMPLETE = "incomplete"

ABS_PREFIX = "abs:"


def home_file(test_name):
    return test_name.split("::", 1)[0]


@dataclass(frozen=True)
class InvalidDep:
    filename: str
    cause: str  # missing | mismatch | incomplete

    def as_dict(self):
        return {"filename": self.filename, "cause": self.cause}


@dataclass
class Decision:
    test_name: str
    deselect: bool
    reason: str
    observation_id: Optional[int] = None
    invalid_dep: Optional[InvalidDep] = None

    def as_dict(self):
        return {
            "reason": self.reason,
            "observation_id": self.observation_id,
            "invalid_dep": self.invalid_dep.as_dict() if self.invalid_dep else None,
        }


@dataclass(frozen=True)
class ObservationRow:
    id: int
    test_name: str
    failed: bool
    duration: Optional[float]
    superseded: bool
    last_seen_at: float


@dataclass
class SelectionResult:  # pylint: disable=too-many-instance-attributes
    decisions: Dict[str, Decision] = field(default_factory=dict)
    durations: Dict[str, float] = field(default_factory=dict)
    valid_failing: Dict[str, List[int]] = field(default_factory=dict)
    filenames: List[str] = field(default_factory=list)
    used_observation_ids: List[int] = field(default_factory=list)
    new_memo: List[tuple] = field(default_factory=list)
    ignorable_files: Dict[str, List[str]] = field(default_factory=dict)
    observation_count: int = 0
    elapsed: float = 0.0

    @property
    def deselectable(self) -> Set[str]:
        return {name for name, d in self.decisions.items() if d.deselect}

    @property
    def failing(self) -> Set[str]:
        return {name for name, d in self.decisions.items() if d.reason == VALID_FAILING}

    @property
    def stable(self) -> Set[str]:
        """Tests with at least one valid observation (upstream's "stable")."""
        return self.deselectable | self.failing

    @property
    def unstable(self) -> Set[str]:
        return set(self.decisions) - self.stable


def newest(rows):
    return max(rows, key=lambda row: (row.last_seen_at, row.id))


def decide(test_name, rows, valid_ids, invalid_dep_of=None):
    """The decision rule for one test with stored observations ``rows``."""
    live = [row for row in rows if not row.superseded]
    valid = [row for row in live if row.id in valid_ids]
    failing = [row for row in valid if row.failed]
    passing = [row for row in valid if not row.failed]
    if failing:
        return Decision(test_name, False, VALID_FAILING, newest(failing).id)
    if passing:
        return Decision(test_name, True, DESELECT, newest(passing).id)
    invalid = [row for row in rows if row.id not in valid_ids]
    reference = newest(invalid) if invalid else newest(rows)
    dep = invalid_dep_of(reference.id) if (invalid_dep_of and invalid) else None
    return Decision(test_name, False, ALL_INVALID, reference.id, dep)


def strip_abs(filename):
    return filename[len(ABS_PREFIX) :] if filename.startswith(ABS_PREFIX) else filename


def fingerprint_matches(module, checksums_blob):
    if module is None or not module.consistent:
        return False
    try:
        return match_fingerprint(module, blob_to_checksums(checksums_blob))
    except (OSError, AttributeError, ValueError):
        return False


def conftest_chain(filename):
    """conftest.py paths that apply to filename (relative to rootdir, outermost first)."""
    parts = filename.split("/")[:-1]
    chain = ["conftest.py"]
    for depth in range(1, len(parts) + 1):
        chain.append("/".join(parts[:depth]) + "/conftest.py")
    return chain


def context_hash(source_tree, filename, extra_files=()):
    """Identity of what determines the tests collected from filename.

    The file itself, every conftest.py on its path and the pytest config file(s). None when
    filename doesn't exist (or isn't consistent), i.e. no snapshot may be trusted.
    """
    module = source_tree.get_file(filename)
    if module is None:
        return None
    hsh = hashlib.sha256()
    hsh.update(f"{filename}\0{module.fs_fsha}\0".encode("utf-8", "surrogateescape"))
    for other in list(conftest_chain(filename)) + list(extra_files):
        other_module = source_tree.get_file(other)
        fsha = other_module.fs_fsha if other_module else "-"
        hsh.update(f"{other}\0{fsha}\0".encode("utf-8", "surrogateescape"))
    return hsh.hexdigest()


def compute_selection(  # pylint: disable=too-many-locals,too-many-statements
    database, env_id, source_tree, pkey=None, context_files=(), use_snapshots=True
):
    """Read everything in one snapshot and decide every known test of the environment."""
    start = time.perf_counter()
    pkey = pkey or dbmod.parser_key()
    result = SelectionResult()
    with database.read_snapshot() as con:
        fps = con.execute(
            "SELECT id, filename, checksums FROM dep_fp WHERE id IN ("
            " SELECT od.fp_id FROM observation_dep od JOIN observation o "
            " ON o.id = od.observation_id WHERE o.env_id = ?)",
            (env_id,),
        ).fetchall()
        filenames = sorted({row[1] for row in fps})
        current = {}
        for filename in filenames:
            module = source_tree.get_file(filename)
            current[filename] = module.fs_fsha if module else None
        con.execute("DELETE FROM temp.cur_file")
        con.executemany(
            "INSERT INTO temp.cur_file (filename, fsha) VALUES (?, ?)",
            [(name, fsha) for name, fsha in current.items() if fsha],
        )
        memo = {
            row[0]: row[1]
            for row in con.execute(
                "SELECT m.fp_id, m.valid FROM fp_match m JOIN dep_fp f ON f.id = m.fp_id "
                "JOIN temp.cur_file c ON c.filename = f.filename AND c.fsha = m.fsha "
                "WHERE m.parser_key = ?",
                (pkey,),
            )
        }
        valid_fp_ids = []
        invalid_cause = {}
        for fp_id, filename, blob in fps:
            fsha = current[filename]
            if fsha is None:
                invalid_cause[fp_id] = MISSING
                continue
            valid = memo.get(fp_id)
            if valid is None:
                module = source_tree.get_file(filename)
                valid = 1 if fingerprint_matches(module, blob) else 0
                if module is not None and module.consistent:
                    result.new_memo.append((fp_id, fsha, valid))
            if valid:
                valid_fp_ids.append(fp_id)
            else:
                invalid_cause[fp_id] = MISMATCH

        con.execute("DELETE FROM temp.valid_fp")
        con.executemany(
            "INSERT INTO temp.valid_fp (id) VALUES (?)", [(i,) for i in valid_fp_ids]
        )
        rows_by_test = defaultdict(list)
        for row in con.execute(
            "SELECT id, test_name, failed, duration, superseded, last_seen_at "
            "FROM observation WHERE env_id = ?",
            (env_id,),
        ):
            rows_by_test[row[1]].append(
                ObservationRow(
                    row[0], row[1], bool(row[2]), row[3], bool(row[4]), row[5]
                )
            )
        # every dependency present and valid; dep_count guards against partial data
        valid_obs = {
            row[0]
            for row in con.execute(
                "SELECT o.id FROM observation o "
                "JOIN observation_dep od ON od.observation_id = o.id "
                "LEFT JOIN temp.valid_fp v ON v.id = od.fp_id "
                "WHERE o.env_id = ? GROUP BY o.id "
                "HAVING count(*) = o.dep_count AND count(v.id) = o.dep_count",
                (env_id,),
            )
        }

        # explanations: the first invalid dependency of the newest invalid observation
        unexplained = []
        for test_name, rows in rows_by_test.items():
            live_valid = [r for r in rows if not r.superseded and r.id in valid_obs]
            if not live_valid:
                invalid = [r for r in rows if r.id not in valid_obs]
                if invalid:
                    unexplained.append(newest(invalid).id)
        explanation = {}
        if unexplained:
            con.execute("DELETE FROM temp.obs_ids")
            con.executemany(
                "INSERT INTO temp.obs_ids (id) VALUES (?)", [(i,) for i in unexplained]
            )
            for obs_id, fp_id, filename in con.execute(
                "SELECT od.observation_id, f.id, f.filename FROM observation_dep od "
                "JOIN dep_fp f ON f.id = od.fp_id "
                "WHERE od.observation_id IN (SELECT id FROM temp.obs_ids) "
                "ORDER BY od.observation_id, f.filename"
            ):
                if obs_id not in explanation and fp_id in invalid_cause:
                    explanation[obs_id] = InvalidDep(filename, invalid_cause[fp_id])

        def invalid_dep_of(obs_id):
            return explanation.get(obs_id, InvalidDep("", INCOMPLETE))

        for test_name, rows in rows_by_test.items():
            decision = decide(test_name, rows, valid_obs, invalid_dep_of)
            result.decisions[test_name] = decision
            result.durations[test_name] = newest(rows).duration or 0
            if decision.deselect:
                result.used_observation_ids.append(decision.observation_id)
            failing_ids = [
                r.id
                for r in rows
                if r.failed and not r.superseded and r.id in valid_obs
            ]
            if failing_ids:
                result.valid_failing[test_name] = failing_ids
        result.observation_count = sum(len(rows) for rows in rows_by_test.values())
        result.filenames = filenames

        if use_snapshots:
            result.ignorable_files = _ignorable_files(
                con, env_id, source_tree, result, context_files
            )
    result.elapsed = time.perf_counter() - start
    return result


def _ignorable_files(con, env_id, source_tree, result, context_files):
    """Test files whose every test (per a snapshot of the same context) is deselectable."""
    deselectable = result.deselectable
    candidates = sorted({home_file(name) for name in deselectable})
    keys = {}
    for filename in candidates:
        if filename.startswith(ABS_PREFIX) or not filename.endswith(".py"):
            continue
        ctx = context_hash(source_tree, filename, context_files)
        if ctx:
            keys[filename] = ctx
    if not keys:
        return {}
    con.execute("DELETE FROM temp.snap_keys")
    con.executemany(
        "INSERT INTO temp.snap_keys (filename, context_hash) VALUES (?, ?)",
        keys.items(),
    )
    import json  # pylint: disable=import-outside-toplevel

    ignorable = {}
    for filename, names_json in con.execute(
        "SELECT s.filename, s.test_names FROM collection_snapshot s JOIN temp.snap_keys k "
        "ON k.filename = s.filename AND k.context_hash = s.context_hash WHERE s.env_id = ?",
        (env_id,),
    ):
        names = json.loads(names_json)
        if names and all(name in deselectable for name in names):
            ignorable[filename] = names
    return ignorable


def realpath_relative(path, real_rootdir):
    """Repo-relative '/'-separated name for a measured file, or 'abs:<realpath>' outside."""
    real = os.path.realpath(path)
    rel = os.path.relpath(real, real_rootdir)
    if rel == os.pardir or rel.startswith(os.pardir + os.sep) or os.path.isabs(rel):
        return ABS_PREFIX + real.replace(os.sep, "/")
    return rel.replace(os.sep, "/")
