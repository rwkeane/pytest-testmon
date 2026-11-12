"""SQLite storage for testmon: shared between checkouts and processes, multi-version.

One database can serve every checkout, branch and agent on a machine. Each test can have
many *observations* (the set of fingerprinted code blocks one execution covered, plus its
outcome); an observation is reused wherever its fingerprints still match the source,
regardless of the path, branch or commit that produced it. See docs/shared-db.md.

Concurrency rules followed throughout this module:

* connections are never shared between processes; transactions are explicit
  (``isolation_level=None``), writes always start with ``BEGIN IMMEDIATE`` so a read lock is
  never upgraded (which would fail with SQLITE_BUSY without consulting the busy handler);
* a write transaction contains only database work; everything else is prepared before it;
* row ids are never cached across transactions (garbage collection may delete rows between).
"""

import hashlib
import json
import os
import random
import sqlite3
import stat
import sys
import time
from collections import namedtuple
from contextlib import contextmanager

from testmon.process_code import checksums_to_blob

DATA_VERSION = 1000
SCHEMA_VERSION = (
    DATA_VERSION  # documentation alias; code reads DATA_VERSION (tests patch it)
)
# The oldest schema version a client may support and still use a database at DATA_VERSION.
COMPAT_VERSION = 1000
# "TMON"; distinguishes this fork's databases from upstream testmon's (application_id 0).
APPLICATION_ID = 0x544D4F4E
# Bump when the block/fingerprint algorithm changes: fp_match memo rows are keyed by it.
FINGERPRINT_ALGO_VERSION = 1

DEFAULT_TIMEOUT = 30.0
WRITE_ATTEMPTS = 3

DAY = 24 * 3600.0
DEFAULT_RETENTION_DAYS = 60
DEFAULT_MAX_VERSIONS = 32
DEFAULT_GC_INTERVAL_HOURS = 6
SUPERSEDED_RETENTION_DAYS = 7
MEMO_RETENTION_DAYS = 30
SESSION_RETENTION_DAYS = 30
SNAPSHOTS_PER_FILE = 8
GC_CHUNK = 5000
LAST_USED_GRANULARITY = DAY

# version -> (migrate(connection), compat_version). Applied in order, each in an EXCLUSIVE
# transaction. compat_version is the oldest client schema that can still use the result:
# an additive migration keeps the previous value, a breaking one raises it to its version.
MIGRATIONS = {}

_clock = time.time


def now():
    """Wall-clock time; tests replace ``testmon.db._clock`` to drive retention."""
    return _clock()


def parser_key():
    """Identifies how block checksums were computed (they depend on the ast of the Python)."""
    return (
        f"{FINGERPRINT_ALGO_VERSION}:{sys.version_info.major}.{sys.version_info.minor}"
    )


class TestmonDbException(Exception):
    __test__ = False


class DbError(TestmonDbException):
    """A database problem, classified so callers can apply the fallback policy and report it.

    reason_code is one of: missing, open_failed, permission, corrupt, incompatible_schema,
    foreign_db,
    lock_timeout, io, db_error.
    """

    def __init__(self, reason_code, detail):
        super().__init__(f"{reason_code}: {detail}")
        self.reason_code = reason_code
        self.detail = detail

    @classmethod
    def from_sqlite(cls, exc):
        return cls(classify_sqlite_error(exc), str(exc))


def classify_sqlite_error(exc):
    message = str(exc).lower()
    if "locked" in message or "busy" in message:
        return "lock_timeout"
    if (
        "malformed" in message
        or "not a database" in message
        or "disk image" in message
        or "file is encrypted" in message
    ):
        return "corrupt"
    if "readonly" in message or "read-only" in message or "permission" in message:
        return "permission"
    if "unable to open" in message:
        return "open_failed"
    if (
        "disk i/o" in message
        or "disk is full" in message
        or "database or disk is full" in message
    ):
        return "io"
    return "db_error"


def _is_busy(exc):
    return classify_sqlite_error(exc) == "lock_timeout"


_commits = {"observations": 0}


def _fault(point):
    """Fault injection for tests: TESTMON_FAULT_INJECT=point[:raise|:exit][,point...]."""
    spec = os.environ.get("TESTMON_FAULT_INJECT")
    if not spec:
        return
    for item in spec.split(","):
        name, _, action = item.strip().partition(":")
        if name != point:
            continue
        if action == "raise":
            raise sqlite3.OperationalError(f"injected fault at {point}")
        sys.stderr.flush()
        os._exit(17)  # pylint: disable=protected-access


# --------------------------------------------------------------------------------------
# Location security (shared databases only)
# --------------------------------------------------------------------------------------


def secure_location(path, create=False):
    """Verify (and with create=True, create) a private location for a shared database.

    Refuses (DbError "permission") a directory or file owned by another user, or one that is
    group/world-writable; tightens group/world-readable files that we own. A database that
    does not exist is DbError "missing" unless create=True: a shared database is created
    deliberately (testmon-admin init), never as a side effect of a test run.
    """
    if not create and not (os.path.exists(path) and os.path.getsize(path) > 0):
        raise DbError(
            "missing",
            f"{path} does not exist; create it once with `testmon-admin --db {path} init`, "
            "or run pytest without --testmon",
        )
    if os.name != "posix":
        return
    directory = os.path.dirname(path) or "."
    try:
        if create:
            os.makedirs(directory, mode=0o700, exist_ok=True)
    except OSError as exc:
        raise DbError("open_failed", f"cannot create {directory}: {exc}") from exc
    uid = os.getuid()
    st = os.stat(directory)
    if st.st_uid != uid:
        raise DbError(
            "permission",
            f"directory {directory} is owned by uid {st.st_uid}, not by this user ({uid})",
        )
    if st.st_mode & 0o022:
        raise DbError(
            "permission",
            f"directory {directory} is group- or world-writable "
            f"(mode {stat.S_IMODE(st.st_mode):o}); use a private directory (chmod 700)",
        )
    for candidate in (path, path + "-wal", path + "-shm"):
        try:
            st = os.stat(candidate)
        except FileNotFoundError:
            continue
        if st.st_uid != uid:
            raise DbError(
                "permission",
                f"{candidate} is owned by uid {st.st_uid}, not by this user ({uid})",
            )
        if st.st_mode & 0o022:
            raise DbError(
                "permission",
                f"{candidate} is group- or world-writable (mode {stat.S_IMODE(st.st_mode):o})",
            )
        if st.st_mode & 0o044:
            os.chmod(candidate, 0o600)
    if not os.path.exists(path):
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600)
            os.close(fd)
        except FileExistsError:
            pass
        except OSError as exc:
            raise DbError("open_failed", f"cannot create {path}: {exc}") from exc


# --------------------------------------------------------------------------------------
# Schema
# --------------------------------------------------------------------------------------

SCHEMA_STATEMENTS = (
    # value is untyped on purpose: counters are stored as numbers, other data as JSON text
    "CREATE TABLE metadata (key TEXT PRIMARY KEY, value)",
    """CREATE TABLE environment (
        id INTEGER PRIMARY KEY,
        project TEXT NOT NULL,
        name TEXT NOT NULL,
        system_packages TEXT NOT NULL,
        python_version TEXT NOT NULL,
        created_at REAL NOT NULL,
        last_used_at REAL NOT NULL,
        UNIQUE (project, name, system_packages, python_version))""",
    """CREATE TABLE dep_fp (
        id INTEGER PRIMARY KEY,
        filename TEXT NOT NULL,
        checksums BLOB NOT NULL,
        UNIQUE (filename, checksums))""",
    """CREATE TABLE observation (
        id INTEGER PRIMARY KEY,
        env_id INTEGER NOT NULL REFERENCES environment(id) ON DELETE CASCADE,
        test_name TEXT NOT NULL,
        deps_hash BLOB NOT NULL,
        dep_count INTEGER NOT NULL CHECK (dep_count > 0),
        failed INTEGER NOT NULL,
        duration REAL,
        superseded INTEGER NOT NULL DEFAULT 0,
        created_at REAL NOT NULL,
        last_seen_at REAL NOT NULL,
        last_used_at REAL NOT NULL,
        UNIQUE (env_id, test_name, deps_hash))""",
    """CREATE TABLE observation_dep (
        observation_id INTEGER NOT NULL REFERENCES observation(id) ON DELETE CASCADE,
        fp_id INTEGER NOT NULL REFERENCES dep_fp(id),
        PRIMARY KEY (observation_id, fp_id)) WITHOUT ROWID""",
    "CREATE INDEX observation_dep_fp ON observation_dep (fp_id)",
    # memo of the pure function "fingerprint fp_id is contained in the blocks of content fsha"
    """CREATE TABLE fp_match (
        fp_id INTEGER NOT NULL REFERENCES dep_fp(id) ON DELETE CASCADE,
        fsha TEXT NOT NULL,
        parser_key TEXT NOT NULL,
        valid INTEGER NOT NULL,
        checked_at REAL NOT NULL,
        PRIMARY KEY (fp_id, fsha, parser_key)) WITHOUT ROWID""",
    """CREATE TABLE collection_snapshot (
        env_id INTEGER NOT NULL REFERENCES environment(id) ON DELETE CASCADE,
        filename TEXT NOT NULL,
        context_hash TEXT NOT NULL,
        test_names TEXT NOT NULL,
        recorded_at REAL NOT NULL,
        PRIMARY KEY (env_id, filename, context_hash)) WITHOUT ROWID""",
    """CREATE TABLE session (
        id INTEGER PRIMARY KEY,
        env_id INTEGER REFERENCES environment(id) ON DELETE SET NULL,
        started_at REAL NOT NULL,
        finished_at REAL,
        pid INTEGER,
        rootdir_hash TEXT,
        git_head TEXT,
        select_mode INTEGER,
        collect_mode INTEGER,
        n_collected INTEGER,
        n_selected INTEGER,
        n_deselected INTEGER,
        saved_time REAL,
        all_time REAL,
        dropped_batches INTEGER NOT NULL DEFAULT 0,
        status TEXT NOT NULL,
        report TEXT)""",
)

TEMP_STATEMENTS = (
    "CREATE TEMP TABLE IF NOT EXISTS tmp_fp "
    "(filename TEXT, checksums BLOB, PRIMARY KEY (filename, checksums))",
    "CREATE TEMP TABLE IF NOT EXISTS cur_file (filename TEXT PRIMARY KEY, fsha TEXT)",
    "CREATE TEMP TABLE IF NOT EXISTS valid_fp (id INTEGER PRIMARY KEY)",
    "CREATE TEMP TABLE IF NOT EXISTS obs_ids (id INTEGER PRIMARY KEY)",
    "CREATE TEMP TABLE IF NOT EXISTS snap_keys "
    "(filename TEXT, context_hash TEXT, PRIMARY KEY (filename, context_hash))",
)

DepInput = namedtuple("DepInput", "filename checksums fsha")
"""One dependency of an observation: filename, checksums blob (bytes), content fsha or None."""

ObservationInput = namedtuple("ObservationInput", "test_name deps failed duration")


def deps_hash(deps):
    """Order-independent identity of a dependency set: [(filename, checksums_blob), ...]."""
    hsh = hashlib.sha256()
    for filename, blob in sorted(deps):
        encoded = filename.encode("utf-8", "surrogateescape")
        hsh.update(len(encoded).to_bytes(4, "little"))
        hsh.update(encoded)
        hsh.update(len(blob).to_bytes(4, "little"))
        hsh.update(blob)
    return hsh.digest()


def _remove_database_files(path):
    for candidate in (path, path + "-wal", path + "-shm"):
        try:
            os.remove(candidate)
        except FileNotFoundError:
            pass


class DB:  # pylint: disable=too-many-public-methods
    """A connection to a testmon database (one per process, never shared)."""

    def __init__(
        self,
        datafile,
        readonly=False,
        shared=False,
        timeout=DEFAULT_TIMEOUT,
        create=False,
    ):
        self.datafile = os.fspath(datafile)
        self._create = create
        self._readonly = readonly
        self.shared = shared
        self.timeout = timeout
        self.con = None
        self.file_created = False
        try:
            self._open()
        except sqlite3.Error as exc:
            self._close_quietly()
            raise DbError.from_sqlite(exc) from exc
        except BaseException:
            self._close_quietly()
            raise

    # ---------------------------------------------------------------- connection lifecycle

    def _connect(self):
        if self._readonly:
            from urllib.parse import quote  # pylint: disable=import-outside-toplevel

            return sqlite3.connect(
                f"file:{quote(os.path.abspath(self.datafile))}?mode=ro",
                uri=True,
                timeout=self.timeout,
                isolation_level=None,
            )
        return sqlite3.connect(
            self.datafile, timeout=self.timeout, isolation_level=None
        )

    def _configure(self, con):
        """Read-only settings; safe on a file we may still refuse to touch."""
        con.row_factory = sqlite3.Row
        con.execute(f"PRAGMA busy_timeout = {int(self.timeout * 1000)}")
        con.execute("PRAGMA foreign_keys = ON")
        for statement in TEMP_STATEMENTS:
            con.execute(statement)

    def _configure_for_writing(self, con):
        """Settings that modify the file (journal mode in the header): only on our files."""
        if self._readonly:
            return
        if not con.execute("SELECT count(*) FROM sqlite_master").fetchone()[0]:
            # must precede the first table; ignored afterwards
            con.execute("PRAGMA auto_vacuum = INCREMENTAL")
        mode = con.execute("PRAGMA journal_mode = WAL").fetchone()[0]
        if str(mode).lower() != "wal" and self.shared:
            raise DbError("db_error", f"could not enable WAL mode (got {mode})")
        con.execute("PRAGMA synchronous = NORMAL")
        con.execute("PRAGMA journal_size_limit = 67108864")

    def _open(self):
        if self.shared:
            if self._readonly:
                secure_location(self.datafile, create=False)  # existence only
            else:
                secure_location(self.datafile, create=self._create)
        self.con = self._connect()
        self._configure(self.con)
        # inspect before changing anything: a foreign file must be left exactly as it is
        app_id = self.con.execute("PRAGMA application_id").fetchone()[0]
        version = self.con.execute("PRAGMA user_version").fetchone()[0]
        has_tables = self.con.execute(
            "SELECT count(*) FROM sqlite_master WHERE type = 'table'"
        ).fetchone()[0]
        if not has_tables and app_id == 0 and version == 0:
            if self._readonly:
                raise DbError("open_failed", f"{self.datafile} is empty")
            self._configure_for_writing(self.con)
            self.file_created = self._initialize()
        elif app_id != APPLICATION_ID:
            if self.shared or self._readonly:
                raise DbError(
                    "foreign_db",
                    f"{self.datafile} is not a database of this testmon fork "
                    f"(application_id={app_id}, user_version={version}); refusing to modify it",
                )
            # a private .testmondata in upstream's (or another) format: recreate, like upstream
            self._recreate()
        else:
            self._configure_for_writing(self.con)
        self._check_schema()

    def _recreate(self):
        self.con.close()
        _remove_database_files(self.datafile)
        self.con = self._connect()
        self._configure(self.con)
        self._configure_for_writing(self.con)
        self.file_created = self._initialize()

    def _initialize(self):
        def body(con):
            if con.execute(
                "SELECT count(*) FROM sqlite_master WHERE type = 'table'"
            ).fetchone()[0]:
                return (
                    False  # another process initialized it while we waited for the lock
                )
            for statement in SCHEMA_STATEMENTS:
                con.execute(statement)
            con.execute(f"PRAGMA application_id = {APPLICATION_ID}")
            con.execute(f"PRAGMA user_version = {DATA_VERSION}")
            con.execute(
                "INSERT INTO metadata (key, value) VALUES ('compat_version', ?)",
                (COMPAT_VERSION,),
            )
            con.execute(
                "INSERT INTO metadata (key, value) VALUES ('created_at', ?)", (now(),)
            )
            return True

        return self._write(body, begin="EXCLUSIVE")

    def _check_schema(self):
        version = self.con.execute("PRAGMA user_version").fetchone()[0]
        if version == DATA_VERSION:
            return
        if version > DATA_VERSION:
            compat = self._compat_version(default=version)
            if compat > DATA_VERSION:
                raise DbError(
                    "incompatible_schema",
                    f"database schema {version} requires a testmon supporting schema "
                    f">= {compat}; this testmon supports {DATA_VERSION}",
                )
            return  # newer, but only additively: usable
        if self._readonly:
            raise DbError(
                "incompatible_schema",
                f"database schema {version} is older than {DATA_VERSION} (read-only)",
            )
        if not self._migration_path_exists(version):
            if self.shared:
                raise DbError(
                    "incompatible_schema",
                    f"database schema {version} cannot be migrated to {DATA_VERSION}; "
                    "refusing to modify the shared database",
                )
            # private database without a migration path: recreate it, like upstream
            self._recreate()
            return
        self.migrate()

    @staticmethod
    def _migration_path_exists(version):
        reachable = version
        for step in sorted(MIGRATIONS):
            if reachable < step <= DATA_VERSION:
                reachable = step
        return reachable == DATA_VERSION

    def _compat_version(self, default):
        row = self.con.execute(
            "SELECT value FROM metadata WHERE key = 'compat_version'"
        ).fetchone()
        try:
            return int(row[0]) if row else default
        except (TypeError, ValueError):
            return default

    def migrate(self):
        """Apply pending MIGRATIONS; safe to call concurrently from several processes."""

        def body(con):
            current = con.execute("PRAGMA user_version").fetchone()[0]
            applied = []
            for version in sorted(MIGRATIONS):
                if current < version <= DATA_VERSION:
                    migrate_fn, compat = MIGRATIONS[version]
                    migrate_fn(con)
                    con.execute(f"PRAGMA user_version = {int(version)}")
                    old_compat = self._compat_version(default=COMPAT_VERSION)
                    con.execute(
                        "INSERT INTO metadata (key, value) VALUES ('compat_version', ?) "
                        "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                        (max(old_compat, compat),),
                    )
                    con.execute(
                        "INSERT INTO metadata (key, value) VALUES ('migrations_applied', 1) "
                        "ON CONFLICT(key) DO UPDATE SET value = value + 1"
                    )
                    current = version
                    applied.append(version)
            return applied

        return self._write(body, begin="EXCLUSIVE")

    def version_compatibility(self):
        return DATA_VERSION

    def _close_quietly(self):
        if self.con is not None:
            try:
                self.con.close()
            except sqlite3.Error:
                pass
            self.con = None

    def close(self):
        self._close_quietly()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    # ---------------------------------------------------------------- transactions

    def _write(self, body, begin="IMMEDIATE"):
        """Run body(con) in a write transaction, retrying the whole body when busy."""
        if self._readonly:
            raise DbError("permission", "the connection is read-only")
        for attempt in range(1, WRITE_ATTEMPTS + 1):
            try:
                self.con.execute(f"BEGIN {begin}")
                try:
                    result = body(self.con)
                    self.con.execute("COMMIT")
                    return result
                except BaseException:
                    if self.con.in_transaction:
                        try:
                            self.con.execute("ROLLBACK")
                        except sqlite3.Error:
                            pass
                    raise
            except sqlite3.OperationalError as exc:
                if _is_busy(exc) and attempt < WRITE_ATTEMPTS:
                    time.sleep(random.uniform(0.05, 0.25) * attempt)
                    continue
                raise DbError.from_sqlite(exc) from exc
            except sqlite3.Error as exc:
                raise DbError.from_sqlite(exc) from exc
        raise AssertionError("unreachable")  # pragma: no cover

    def write(self, body):
        """Public form of _write for callers outside this module (admin, tests)."""
        return self._write(body)

    @contextmanager
    def read_snapshot(self):
        """A consistent read view (WAL snapshot); temp tables may be written inside it."""
        try:
            self.con.execute("BEGIN DEFERRED")
        except sqlite3.Error as exc:
            raise DbError.from_sqlite(exc) from exc
        try:
            yield self.con
        except sqlite3.Error as exc:
            raise DbError.from_sqlite(exc) from exc
        finally:
            # never let the cleanup replace the real error (sqlite may already have rolled
            # back, e.g. after detecting corruption)
            if self.con is not None and self.con.in_transaction:
                try:
                    self.con.execute("COMMIT")
                except sqlite3.Error:
                    try:
                        self.con.execute("ROLLBACK")
                    except sqlite3.Error:
                        pass

    def _read(self, sql, params=()):
        try:
            return self.con.execute(sql, params).fetchall()
        except sqlite3.Error as exc:
            raise DbError.from_sqlite(exc) from exc

    # ---------------------------------------------------------------- metadata

    def write_attribute(self, attribute, data, exec_id=None):
        key = f"{exec_id}:{attribute}"
        self._write(
            lambda con: con.execute(
                "INSERT INTO metadata (key, value) VALUES (?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (key, json.dumps(data)),
            )
        )

    def fetch_attribute(self, attribute, default=None, exec_id=None):
        rows = self._read(
            "SELECT value FROM metadata WHERE key = ?", (f"{exec_id}:{attribute}",)
        )
        if not rows:
            return default
        value = rows[0][0]
        if isinstance(value, (int, float)) or value is None:
            return value
        try:
            return json.loads(value)
        except (TypeError, ValueError):
            return value

    def increment_attributes(self, attributes_to_increment, exec_id=None):
        """Atomic counter increments (one statement each, one transaction)."""
        rows = [
            (f"{exec_id}:{name}", value or 0)
            for name, value in attributes_to_increment.items()
        ]

        def body(con):
            con.executemany(
                "INSERT INTO metadata (key, value) VALUES (?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value = coalesce(value, 0) + excluded.value",
                rows,
            )

        self._write(body)

    # ---------------------------------------------------------------- environments

    def fetch_or_create_environment(
        self, environment_name, system_packages, python_version, project=""
    ):
        """Return (env_id, packages_changed). Environments are never deleted here.

        packages_changed is True when this exact environment is new but the same
        (project, name) was seen with other packages/Python (upstream's "packages changed").
        """
        key = (
            project or "",
            environment_name,
            system_packages or "",
            python_version or "",
        )
        rows = self._read(
            "SELECT id, last_used_at FROM environment WHERE project = ? AND name = ? "
            "AND system_packages = ? AND python_version = ?",
            key,
        )
        if rows and rows[0]["last_used_at"] > now() - LAST_USED_GRANULARITY:
            return rows[0]["id"], False

        def body(con):
            timestamp = now()
            row = con.execute(
                "SELECT id FROM environment WHERE project = ? AND name = ? "
                "AND system_packages = ? AND python_version = ?",
                key,
            ).fetchone()
            if row:
                con.execute(
                    "UPDATE environment SET last_used_at = ? WHERE id = ?",
                    (timestamp, row[0]),
                )
                return row[0], False
            other = con.execute(
                "SELECT 1 FROM environment WHERE project = ? AND name = ? LIMIT 1",
                key[:2],
            ).fetchone()
            cursor = con.execute(
                "INSERT INTO environment (project, name, system_packages, python_version, "
                "created_at, last_used_at) VALUES (?, ?, ?, ?, ?, ?)",
                key + (timestamp, timestamp),
            )
            return cursor.lastrowid, bool(other)

        return self._write(body)

    def initiate_execution(  # pylint: disable=unused-argument
        self,
        environment_name,
        system_packages,
        python_version,
        execution_metadata=None,
        project="",
    ):
        exec_id, packages_changed = self.fetch_or_create_environment(
            environment_name, system_packages, python_version, project=project
        )
        return {
            "exec_id": exec_id,
            "filenames": None,
            "packages_changed": packages_changed,
        }

    # ---------------------------------------------------------------- sessions

    def start_session(self, env_id, select, collect, rootdir_hash=None, git_head=None):
        def body(con):
            cursor = con.execute(
                "INSERT INTO session (env_id, started_at, pid, rootdir_hash, git_head, "
                "select_mode, collect_mode, status) VALUES (?, ?, ?, ?, ?, ?, ?, 'running')",
                (
                    env_id,
                    now(),
                    os.getpid(),
                    rootdir_hash,
                    git_head,
                    int(select),
                    int(collect),
                ),
            )
            return cursor.lastrowid

        return self._write(body)

    def finish_session(self, session_id, status="finished", **counters):
        columns = {
            key: counters[key]
            for key in (
                "n_collected",
                "n_selected",
                "n_deselected",
                "saved_time",
                "all_time",
                "dropped_batches",
                "report",
            )
            if key in counters
        }
        assignments = ", ".join(f"{column} = ?" for column in columns)

        def body(con):
            con.execute(
                f"UPDATE session SET finished_at = ?, status = ?"
                f"{', ' + assignments if assignments else ''} WHERE id = ?",
                (now(), status, *columns.values(), session_id),
            )

        self._write(body)

    def mark_stale_sessions(self):
        """Mark sessions of processes that died without finishing (kill -9) as interrupted."""
        rows = self._read("SELECT id, pid FROM session WHERE status = 'running'")
        dead = []
        for row in rows:
            if row["pid"] == os.getpid():
                continue
            try:
                os.kill(row["pid"], 0)
            except ProcessLookupError:
                dead.append((row["id"],))
            except (PermissionError, OSError, TypeError):
                continue
        if dead:
            self._write(
                lambda con: con.executemany(
                    "UPDATE session SET status = 'interrupted' "
                    "WHERE id = ? AND status = 'running'",
                    dead,
                )
            )
        return len(dead)

    # ---------------------------------------------------------------- observations: write

    def record_observations(self, env_id, observations, supersede=None):
        """Store a batch of observations atomically; returns the number stored.

        supersede maps test_name -> ids of failing observations that were valid at the start
        of this session; when the test now passes they are marked superseded.
        """
        prepared = []
        for obs in observations:
            deps = {}
            for dep in obs.deps:
                deps[(dep.filename, bytes(dep.checksums))] = dep.fsha
            if not deps:
                continue
            prepared.append((obs, deps, deps_hash(deps.keys())))
        if not prepared:
            return 0
        pairs = sorted({key for _, deps, _ in prepared for key in deps})
        pkey = parser_key()

        def body(con):
            timestamp = now()
            con.execute("DELETE FROM temp.tmp_fp")
            con.executemany(
                "INSERT OR IGNORE INTO temp.tmp_fp (filename, checksums) VALUES (?, ?)",
                pairs,
            )
            con.execute(
                "INSERT OR IGNORE INTO dep_fp (filename, checksums) "
                "SELECT filename, checksums FROM temp.tmp_fp"
            )
            fp_ids = {
                (row[1], bytes(row[2])): row[0]
                for row in con.execute(
                    "SELECT f.id, f.filename, f.checksums FROM dep_fp f JOIN temp.tmp_fp t "
                    "ON t.filename = f.filename AND t.checksums = f.checksums"
                )
            }
            for obs, deps, digest in prepared:
                con.execute(
                    "INSERT INTO observation (env_id, test_name, deps_hash, dep_count, failed, "
                    "duration, superseded, created_at, last_seen_at, last_used_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, 0, ?, ?, ?) "
                    "ON CONFLICT(env_id, test_name, deps_hash) DO UPDATE SET "
                    "failed = excluded.failed, duration = excluded.duration, superseded = 0, "
                    "last_seen_at = excluded.last_seen_at",
                    (
                        env_id,
                        obs.test_name,
                        digest,
                        len(deps),
                        1 if obs.failed else 0,
                        obs.duration,
                        timestamp,
                        timestamp,
                        timestamp,
                    ),
                )
                obs_id = con.execute(
                    "SELECT id FROM observation WHERE env_id = ? AND test_name = ? "
                    "AND deps_hash = ?",
                    (env_id, obs.test_name, digest),
                ).fetchone()[0]
                con.executemany(
                    "INSERT OR IGNORE INTO observation_dep (observation_id, fp_id) VALUES (?, ?)",
                    [(obs_id, fp_ids[key]) for key in deps],
                )
                con.executemany(
                    "INSERT OR IGNORE INTO fp_match (fp_id, fsha, parser_key, valid, checked_at) "
                    "VALUES (?, ?, ?, 1, ?)",
                    [
                        (fp_ids[key], fsha, pkey, timestamp)
                        for key, fsha in deps.items()
                        if fsha
                    ],
                )
                if not obs.failed and supersede:
                    candidates = [
                        (candidate,)
                        for candidate in supersede.get(obs.test_name, ())
                        if candidate != obs_id
                    ]
                    con.executemany(
                        "UPDATE observation SET superseded = 1 WHERE id = ? AND failed = 1",
                        candidates,
                    )
            _fault("mid_write")
            return len(prepared)

        stored = self._write(body)
        _commits["observations"] += 1
        _fault("after_commit")
        return stored

    def insert_test_file_fps(self, tests_deps_n_outcomes, exec_id=None, supersede=None):
        """Upstream-format adapter: {test: {"deps": [{filename, method_checksums, fsha}], ...}}."""
        assert exec_id
        observations = []
        for test_name, deps_n_outcomes in tests_deps_n_outcomes.items():
            deps = tuple(
                DepInput(
                    record["filename"],
                    checksums_to_blob(record["method_checksums"]),
                    record.get("fsha"),
                )
                for record in deps_n_outcomes["deps"]
            )
            observations.append(
                ObservationInput(
                    test_name,
                    deps,
                    bool(deps_n_outcomes.get("failed")),
                    deps_n_outcomes.get("duration"),
                )
            )
        return self.record_observations(exec_id, observations, supersede=supersede)

    def record_selection_side_effects(self, new_memo, used_observation_ids):
        """Persist fp_match results computed during selection and bump last_used_at."""
        if not new_memo and not used_observation_ids:
            return
        pkey = parser_key()

        def body(con):
            timestamp = now()
            con.executemany(
                "INSERT OR IGNORE INTO fp_match (fp_id, fsha, parser_key, valid, checked_at) "
                "SELECT ?, ?, ?, ?, ? WHERE EXISTS (SELECT 1 FROM dep_fp WHERE id = ?)",
                [
                    (fp_id, fsha, pkey, valid, timestamp, fp_id)
                    for fp_id, fsha, valid in new_memo
                ],
            )
            con.executemany(
                "UPDATE observation SET last_used_at = ? WHERE id = ? AND last_used_at < ?",
                [
                    (timestamp, obs_id, timestamp - LAST_USED_GRANULARITY)
                    for obs_id in used_observation_ids
                ],
            )

        self._write(body)

    # ---------------------------------------------------------------- collection snapshots

    def record_collection_snapshots(self, env_id, snapshots):
        """snapshots: [(filename, context_hash, [test names])]"""
        if not snapshots:
            return

        def body(con):
            timestamp = now()
            con.executemany(
                "INSERT INTO collection_snapshot (env_id, filename, context_hash, test_names, "
                "recorded_at) VALUES (?, ?, ?, ?, ?) "
                "ON CONFLICT(env_id, filename, context_hash) DO UPDATE SET "
                "test_names = excluded.test_names, recorded_at = excluded.recorded_at",
                [
                    (
                        env_id,
                        filename,
                        context,
                        json.dumps(sorted(set(names))),
                        timestamp,
                    )
                    for filename, context, names in snapshots
                ],
            )

        self._write(body)

    # ---------------------------------------------------------------- reads (compat API)

    def all_test_executions(self, exec_id):
        """Newest observation per test: {test_name: {duration, failed, forced}}."""
        rows = self._read(
            """
            SELECT test_name, duration, failed FROM (
                SELECT test_name, duration, failed, ROW_NUMBER() OVER (
                    PARTITION BY test_name ORDER BY last_seen_at DESC, id DESC) AS rn
                FROM observation WHERE env_id = ?)
            WHERE rn = 1
            """,
            (exec_id,),
        )
        return {
            row["test_name"]: {
                "duration": row["duration"],
                "failed": row["failed"],
                "forced": None,
            }
            for row in rows
        }

    def filenames(self, exec_id):
        rows = self._read(
            """
            SELECT DISTINCT f.filename FROM dep_fp f
            JOIN observation_dep od ON od.fp_id = f.id
            JOIN observation o ON o.id = od.observation_id
            WHERE o.env_id = ? ORDER BY f.filename
            """,
            (exec_id,),
        )
        return [row[0] for row in rows]

    def all_filenames(self):
        return [row[0] for row in self._read("SELECT DISTINCT filename FROM dep_fp")]

    def observations_of(self, test_name, exec_id=None):
        """Observations of one test (all environments unless exec_id), newest first."""
        sql = (
            "SELECT o.*, e.name AS env_name FROM observation o "
            "JOIN environment e ON e.id = o.env_id WHERE o.test_name = ?"
        )
        params = [test_name]
        if exec_id is not None:
            sql += " AND o.env_id = ?"
            params.append(exec_id)
        return self._read(sql + " ORDER BY o.last_seen_at DESC, o.id DESC", params)

    def observation_deps(self, observation_id):
        return self._read(
            "SELECT f.id, f.filename, f.checksums FROM observation_dep od "
            "JOIN dep_fp f ON f.id = od.fp_id WHERE od.observation_id = ? ORDER BY f.filename",
            (observation_id,),
        )

    # ---------------------------------------------------------------- session end / stats

    def finish_execution(self, exec_id=None, duration=None, select=True):  # pylint: disable=unused-argument
        """Upstream entry point: opportunistic maintenance at the end of a session."""
        self.maybe_gc()

    def fetch_saving_totals(self, select):
        prefix = "" if select else "potential_"
        return tuple(
            self.fetch_attribute(attribute=f"{prefix}{name}", default=0)
            for name in ("time_saved", "time_all", "tests_saved", "tests_all")
        )

    # ---------------------------------------------------------------- maintenance

    def maybe_gc(self, interval_hours=DEFAULT_GC_INTERVAL_HOURS, **policy):
        """Run gc() if nobody did in the last interval_hours; returns its stats or None."""
        key = "None:last_gc_at"

        def claim(con):
            row = con.execute(
                "SELECT value FROM metadata WHERE key = ?", (key,)
            ).fetchone()
            timestamp = now()
            if (
                row
                and row[0] is not None
                and float(row[0]) > timestamp - interval_hours * 3600
            ):
                return False
            con.execute(
                "INSERT INTO metadata (key, value) VALUES (?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (key, timestamp),
            )
            return True

        if not self._write(claim):
            return None
        return self.gc(**policy)

    def _delete_chunked(self, select_ids_sql, delete_table, params=()):
        deleted = 0
        while True:

            def body(con):
                ids = [
                    row[0]
                    for row in con.execute(f"{select_ids_sql} LIMIT {GC_CHUNK}", params)
                ]
                if ids:
                    con.executemany(
                        f"DELETE FROM {delete_table} WHERE id = ?", [(i,) for i in ids]
                    )
                return len(ids)

            count = self._write(body)
            deleted += count
            if count < GC_CHUNK:
                return deleted

    def gc(
        self,
        retention_days=DEFAULT_RETENTION_DAYS,
        max_versions=DEFAULT_MAX_VERSIONS,
        aggressive=False,
    ):
        """Bounded-growth cleanup in short chunked transactions. Never touches recent data."""
        timestamp = now()
        cutoff = timestamp - retention_days * DAY
        stats = {}
        stats["expired_observations"] = self._delete_chunked(
            "SELECT id FROM observation WHERE max(last_seen_at, last_used_at) < ?",
            "observation",
            (cutoff,),
        )
        stats["superseded_observations"] = self._delete_chunked(
            "SELECT id FROM observation WHERE superseded = 1 AND last_seen_at < ?",
            "observation",
            (timestamp - (0 if aggressive else SUPERSEDED_RETENTION_DAYS) * DAY,),
        )
        stats["excess_versions"] = self._delete_chunked(
            "SELECT id FROM (SELECT id, ROW_NUMBER() OVER (PARTITION BY env_id, test_name "
            "ORDER BY max(last_seen_at, last_used_at) DESC, id DESC) AS rn FROM observation) "
            "WHERE rn > ?",
            "observation",
            (max_versions,),
        )
        stats["orphan_fingerprints"] = self._delete_chunked(
            "SELECT id FROM dep_fp f WHERE NOT EXISTS "
            "(SELECT 1 FROM observation_dep od WHERE od.fp_id = f.id)",
            "dep_fp",
        )

        def small_tables(con):
            result = {}
            result["memo_rows"] = con.execute(
                "DELETE FROM fp_match WHERE checked_at < ?",
                (timestamp - (0 if aggressive else MEMO_RETENTION_DAYS) * DAY,),
            ).rowcount
            result["snapshots"] = con.execute(
                "DELETE FROM collection_snapshot WHERE recorded_at < ? OR "
                "(env_id, filename, context_hash) IN ("
                " SELECT env_id, filename, context_hash FROM (SELECT env_id, filename, "
                " context_hash, ROW_NUMBER() OVER (PARTITION BY env_id, filename "
                " ORDER BY recorded_at DESC) AS rn FROM collection_snapshot) WHERE rn > ?)",
                (cutoff, SNAPSHOTS_PER_FILE),
            ).rowcount
            result["environments"] = con.execute(
                "DELETE FROM environment WHERE last_used_at < ? AND NOT EXISTS "
                "(SELECT 1 FROM observation o WHERE o.env_id = environment.id)",
                (cutoff,),
            ).rowcount
            result["sessions"] = con.execute(
                "DELETE FROM session WHERE started_at < ?",
                (timestamp - SESSION_RETENTION_DAYS * DAY,),
            ).rowcount
            return result

        stats.update(self._write(small_tables))
        try:
            stats["interrupted_sessions"] = self.mark_stale_sessions()
        except DbError:
            pass
        try:
            self.con.execute("PRAGMA wal_checkpoint(PASSIVE)")
            self.con.execute("PRAGMA incremental_vacuum(2000)")
            self.con.execute("PRAGMA optimize")
        except sqlite3.Error:
            pass  # maintenance only
        return stats

    def checkpoint(self, mode="TRUNCATE"):
        return tuple(self.con.execute(f"PRAGMA wal_checkpoint({mode})").fetchone())

    def integrity_check(self):
        return [row[0] for row in self._read("PRAGMA integrity_check")]

    def backup(self, dest_path):
        """Consistent online copy (sqlite backup API) to dest_path (private permissions)."""
        dest = sqlite3.connect(dest_path)
        try:
            self.con.backup(dest)
        finally:
            dest.close()
        if os.name == "posix":
            os.chmod(dest_path, 0o600)

    def restore_from(self, src_path):
        """Replace this database's content with src_path's (writers wait on the lock)."""
        src = sqlite3.connect(f"file:{os.path.abspath(src_path)}?mode=ro", uri=True)
        try:
            src.backup(self.con)
        finally:
            src.close()

    def table_counts(self):
        return {
            table: self._read(f"SELECT count(*) FROM {table}")[0][0]
            for table in (
                "environment",
                "observation",
                "observation_dep",
                "dep_fp",
                "fp_match",
                "collection_snapshot",
                "session",
            )
        }
