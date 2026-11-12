import os
import sqlite3
import subprocess
import sys
import textwrap

import pytest

from testmon import db as dbmod
from testmon.db import DB, DbError
from tests.unit.dbhelpers import dep, new_db, observation

posix_only = pytest.mark.skipif(os.name != "posix", reason="POSIX permissions")
not_root = pytest.mark.skipif(
    os.name == "posix" and os.geteuid() == 0, reason="root ignores file permissions"
)

V1 = (1, 2)
V2 = (1, 3)


@pytest.fixture
def clock(monkeypatch):
    class Clock:
        now = 1_000_000.0

        def advance(self, days):
            self.now += days * dbmod.DAY

    fake = Clock()
    monkeypatch.setattr(dbmod, "_clock", lambda: fake.now)
    return fake


@pytest.fixture
def database(tmp_path, clock):  # pylint: disable=unused-argument
    database, env_id = new_db(tmp_path / "db.sqlite")
    database.env_id = env_id
    yield database
    database.close()


def count(database, table, where="", params=()):
    return database.con.execute(
        f"SELECT count(*) FROM {table} {where}", params
    ).fetchone()[0]


class TestWritePath:
    def test_deps_hash_is_order_independent(self):
        pairs = [("a.py", b"\x01"), ("b.py", b"\x02")]
        assert dbmod.deps_hash(pairs) == dbmod.deps_hash(list(reversed(pairs)))
        assert dbmod.deps_hash(pairs) != dbmod.deps_hash([("a.py", b"\x01")])
        # no ambiguity between filename and blob boundaries
        assert dbmod.deps_hash([("ab", b"c")]) != dbmod.deps_hash([("a", b"bc")])

    def test_identical_observations_dedupe(self, database, clock):
        obs = observation("t", [dep("a.py", V1, [0]), dep("b.py", V1, [1])])
        database.record_observations(database.env_id, [obs])
        created = database.con.execute(
            "SELECT created_at, last_seen_at FROM observation"
        ).fetchone()
        clock.advance(1)
        database.record_observations(database.env_id, [obs])
        assert count(database, "observation") == 1
        assert count(database, "observation_dep") == 2
        assert count(database, "dep_fp") == 2
        row = database.con.execute(
            "SELECT created_at, last_seen_at FROM observation"
        ).fetchone()
        assert row[0] == created[0]
        assert row[1] == created[1] + dbmod.DAY

    def test_new_outcome_on_identical_deps_updates_and_unsupersedes(self, database):
        deps = [dep("a.py", V1, [0])]
        database.record_observations(
            database.env_id, [observation("t", deps, failed=True)]
        )
        database.con.execute("UPDATE observation SET superseded = 1")
        database.record_observations(
            database.env_id, [observation("t", deps, failed=True)]
        )
        assert database.con.execute(
            "SELECT failed, superseded FROM observation"
        ).fetchone()[:] == (
            1,
            0,
        )
        database.record_observations(
            database.env_id, [observation("t", deps, failed=False)]
        )
        assert database.con.execute("SELECT failed FROM observation").fetchone()[0] == 0

    def test_supersede_touches_only_given_failing_observations(self, database):
        database.record_observations(
            database.env_id,
            [
                observation("t", [dep("a.py", V1, [0, 1])], failed=True),
                observation("t", [dep("b.py", V1, [0])], failed=True),
                observation("u", [dep("a.py", V1, [0, 1])], failed=True),
            ],
        )
        ids = {
            (row[0], row[1]): row[2]
            for row in database.con.execute(
                "SELECT o.test_name, f.filename, o.id FROM observation o "
                "JOIN observation_dep od ON od.observation_id = o.id "
                "JOIN dep_fp f ON f.id = od.fp_id"
            )
        }
        database.record_observations(
            database.env_id,
            [observation("t", [dep("a.py", V1, [0])], failed=False)],
            supersede={"t": [ids[("t", "a.py")]], "u": [ids[("u", "a.py")]]},
        )
        superseded = {
            row[0]
            for row in database.con.execute(
                "SELECT id FROM observation WHERE superseded"
            )
        }
        assert superseded == {ids[("t", "a.py")]}

    def test_failing_run_does_not_supersede(self, database):
        database.record_observations(
            database.env_id, [observation("t", [dep("a.py", V1, [0, 1])], failed=True)]
        )
        (failing,) = database.con.execute("SELECT id FROM observation").fetchone()
        database.record_observations(
            database.env_id,
            [observation("t", [dep("a.py", V1, [0])], failed=True)],
            supersede={"t": [failing]},
        )
        assert count(database, "observation", "WHERE superseded = 1") == 0

    def test_batch_after_gc_deleted_its_fingerprints(self, database, clock):
        obs = observation("t", [dep("a.py", V1, [0])])
        database.record_observations(database.env_id, [obs])
        clock.advance(dbmod.DEFAULT_RETENTION_DAYS + 1)
        database.gc()
        assert count(database, "dep_fp") == 0
        # a new session (which refreshes its environment) records the same dependency:
        # a new fingerprint row, nothing cached from before
        env_id, _ = database.fetch_or_create_environment("default", "pkgs", "3.12")
        database.record_observations(env_id, [obs])
        assert count(database, "observation") == 1
        assert count(database, "observation_dep") == 1

    def test_batch_is_atomic(self, database, monkeypatch):
        monkeypatch.setenv("TESTMON_FAULT_INJECT", "mid_write:raise")
        with pytest.raises(DbError):
            database.record_observations(
                database.env_id,
                [
                    observation("t", [dep("a.py", V1, [0])]),
                    observation("u", [dep("b.py", V1, [0])]),
                ],
            )
        for table in ("observation", "observation_dep", "dep_fp", "fp_match"):
            assert count(database, table) == 0, table

    def test_empty_observation_is_not_stored(self, database):
        assert (
            database.record_observations(database.env_id, [observation("t", [])]) == 0
        )
        assert count(database, "observation") == 0

    def test_memo_seeded_with_recording_content(self, database):
        d = dep("a.py", V1, [0])
        database.record_observations(database.env_id, [observation("t", [d])])
        assert database.con.execute(
            "SELECT fsha, valid, parser_key FROM fp_match"
        ).fetchone()[:] == (
            d.fsha,
            1,
            dbmod.parser_key(),
        )


class TestConnection:
    def test_pragmas(self, database):
        con = database.con
        assert con.isolation_level is None
        assert con.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
        assert con.execute("PRAGMA synchronous").fetchone()[0] == 1  # NORMAL
        assert con.execute("PRAGMA foreign_keys").fetchone()[0] == 1
        assert con.execute("PRAGMA auto_vacuum").fetchone()[0] == 2  # INCREMENTAL
        assert (
            con.execute("PRAGMA application_id").fetchone()[0] == dbmod.APPLICATION_ID
        )

    def test_writes_begin_immediate_reads_deferred(self, database):
        statements = []
        database.con.set_trace_callback(statements.append)
        database.record_observations(
            database.env_id, [observation("t", [dep("a.py", V1, [0])])]
        )
        database.increment_attributes({"x": 1})
        with database.read_snapshot():
            pass
        database.con.set_trace_callback(None)
        begins = [
            s.strip() for s in statements if s.strip().upper().startswith("BEGIN")
        ]
        assert begins == ["BEGIN IMMEDIATE", "BEGIN IMMEDIATE", "BEGIN DEFERRED"]

    def test_retry_until_success(self, database, monkeypatch):
        monkeypatch.setattr(dbmod.time, "sleep", lambda seconds: None)
        attempts = []

        def body(con):
            attempts.append(1)
            if len(attempts) < dbmod.WRITE_ATTEMPTS:
                raise sqlite3.OperationalError("database is locked")
            return "done"

        assert database.write(body) == "done"
        assert len(attempts) == dbmod.WRITE_ATTEMPTS

    def test_retry_gives_up_with_typed_error(self, database, monkeypatch):
        monkeypatch.setattr(dbmod.time, "sleep", lambda seconds: None)
        attempts = []

        def body(con):
            attempts.append(1)
            raise sqlite3.OperationalError("database is locked")

        with pytest.raises(DbError) as excinfo:
            database.write(body)
        assert excinfo.value.reason_code == "lock_timeout"
        assert len(attempts) == dbmod.WRITE_ATTEMPTS
        assert not database.con.in_transaction

    def test_other_errors_are_not_retried(self, database):
        attempts = []

        def body(con):
            attempts.append(1)
            raise sqlite3.IntegrityError("constraint")

        with pytest.raises(DbError):
            database.write(body)
        assert len(attempts) == 1

    @pytest.mark.parametrize(
        "message, code",
        [
            ("database is locked", "lock_timeout"),
            ("database table is locked", "lock_timeout"),
            ("file is not a database", "corrupt"),
            ("database disk image is malformed", "corrupt"),
            ("attempt to write a readonly database", "permission"),
            ("unable to open database file", "open_failed"),
            ("disk I/O error", "io"),
            ("no such table: x", "db_error"),
        ],
    )
    def test_classification(self, message, code):
        assert dbmod.classify_sqlite_error(sqlite3.OperationalError(message)) == code

    def test_attribute_increments_are_single_statements(self, database):
        database.write_attribute("n", 0)
        database.increment_attributes({"n": 2, "m": 3})
        database.increment_attributes({"n": 0.5, "m": None})
        assert database.fetch_attribute("n") == 2.5
        assert database.fetch_attribute("m") == 3
        database.write_attribute("d", {"a": [1]})
        assert database.fetch_attribute("d") == {"a": [1]}


@posix_only
@not_root
class TestPermissions:
    def test_creates_private_directory_and_file(self, tmp_path):
        path = tmp_path / "new" / "dir" / "db.sqlite"
        with DB(str(path), shared=True):
            pass
        assert oct(os.stat(path.parent).st_mode & 0o777) == "0o700"
        assert oct(os.stat(path).st_mode & 0o777) == "0o600"

    def test_group_writable_directory_refused(self, tmp_path):
        directory = tmp_path / "shared"
        directory.mkdir()
        os.chmod(directory, 0o770)
        with pytest.raises(DbError) as excinfo:
            DB(str(directory / "db.sqlite"), shared=True)
        assert excinfo.value.reason_code == "permission"
        assert not (directory / "db.sqlite").exists()

    def test_world_writable_file_refused(self, tmp_path):
        directory = tmp_path / "shared"
        directory.mkdir(mode=0o700)
        path = directory / "db.sqlite"
        with DB(str(path), shared=True):
            pass
        os.chmod(path, 0o606)
        with pytest.raises(DbError) as excinfo:
            DB(str(path), shared=True)
        assert excinfo.value.reason_code == "permission"

    def test_readable_file_is_tightened(self, tmp_path):
        directory = tmp_path / "shared"
        directory.mkdir(mode=0o700)
        path = directory / "db.sqlite"
        with DB(str(path), shared=True):
            pass
        os.chmod(path, 0o644)
        with DB(str(path), shared=True):
            pass
        assert oct(os.stat(path).st_mode & 0o777) == "0o600"

    def test_symlink_to_own_file_allowed(self, tmp_path):
        real_dir = tmp_path / "real"
        real_dir.mkdir(mode=0o700)
        with DB(str(real_dir / "db.sqlite"), shared=True):
            pass
        link_dir = tmp_path / "link"
        link_dir.mkdir(mode=0o700)
        os.symlink(real_dir / "db.sqlite", link_dir / "db.sqlite")
        with DB(str(link_dir / "db.sqlite"), shared=True) as database:
            assert database.con.execute("PRAGMA application_id").fetchone()[0]

    def test_private_database_is_not_checked(self, tmp_path):
        directory = tmp_path / "proj"
        directory.mkdir()
        os.chmod(directory, 0o777)
        with DB(str(directory / ".testmondata")):
            pass


def upstream_v15(path):
    con = sqlite3.connect(path)
    con.executescript(
        "CREATE TABLE test_execution (id INTEGER PRIMARY KEY, test_name TEXT);"
        "INSERT INTO test_execution (test_name) VALUES ('x');"
        "PRAGMA user_version = 15;"
    )
    con.close()


class TestSchema:
    def test_upstream_database_at_shared_path_is_refused_and_kept(self, tmp_path):
        path = tmp_path / "s" / "db.sqlite"
        path.parent.mkdir(mode=0o700)
        upstream_v15(str(path))
        before = path.read_bytes()
        with pytest.raises(DbError) as excinfo:
            DB(str(path), shared=True)
        assert excinfo.value.reason_code == "foreign_db"
        assert path.read_bytes() == before

    def test_private_upstream_database_is_recreated(self, tmp_path):
        path = tmp_path / ".testmondata"
        upstream_v15(str(path))
        with DB(str(path)) as database:
            assert database.file_created
            assert (
                database.con.execute("PRAGMA user_version").fetchone()[0]
                == dbmod.DATA_VERSION
            )

    def test_newer_incompatible_schema_is_refused(self, tmp_path):
        path = tmp_path / "s" / "db.sqlite"
        with DB(str(path), shared=True) as database:
            database.con.execute("PRAGMA user_version = 5000")
            database.con.execute(
                "UPDATE metadata SET value = 5000 WHERE key = 'compat_version'"
            )
        with pytest.raises(DbError) as excinfo:
            DB(str(path), shared=True)
        assert excinfo.value.reason_code == "incompatible_schema"

    def test_newer_compatible_schema_is_used(self, tmp_path):
        path = tmp_path / "s" / "db.sqlite"
        with DB(str(path), shared=True) as database:
            database.con.execute("PRAGMA user_version = 1001")
        with DB(str(path), shared=True) as database:
            assert database.con.execute("PRAGMA user_version").fetchone()[0] == 1001

    def test_additive_migration(self, tmp_path, monkeypatch):
        path = str(tmp_path / "s" / "db.sqlite")
        database, env_id = new_db(path, shared=True)
        database.record_observations(env_id, [observation("t", [dep("a.py", V1, [0])])])
        database.close()

        def add_column(con):
            con.execute("ALTER TABLE observation ADD COLUMN note TEXT")

        monkeypatch.setattr(dbmod, "MIGRATIONS", {1001: (add_column, 1000)})
        monkeypatch.setattr(dbmod, "DATA_VERSION", 1001)
        with DB(path, shared=True) as database:
            assert database.con.execute("PRAGMA user_version").fetchone()[0] == 1001
            assert database.con.execute(
                "SELECT test_name, note FROM observation"
            ).fetchall()[0][:] == (
                "t",
                None,
            )
        # an older client (pinned on another branch) keeps working
        monkeypatch.setattr(dbmod, "DATA_VERSION", 1000)
        with DB(path, shared=True) as database:
            assert database.all_test_executions(env_id)

    def test_breaking_migration_locks_out_older_clients(self, tmp_path, monkeypatch):
        path = str(tmp_path / "s" / "db.sqlite")
        DB(path, shared=True).close()
        monkeypatch.setattr(dbmod, "MIGRATIONS", {1001: (lambda con: None, 1001)})
        monkeypatch.setattr(dbmod, "DATA_VERSION", 1001)
        DB(path, shared=True).close()
        monkeypatch.setattr(dbmod, "DATA_VERSION", 1000)
        with pytest.raises(DbError) as excinfo:
            DB(path, shared=True)
        assert excinfo.value.reason_code == "incompatible_schema"

    def test_no_migration_path(self, tmp_path, monkeypatch):
        shared = str(tmp_path / "s" / "db.sqlite")
        private = str(tmp_path / ".testmondata")
        DB(shared, shared=True).close()
        DB(private).close()
        monkeypatch.setattr(dbmod, "DATA_VERSION", 1002)
        with pytest.raises(DbError) as excinfo:
            DB(shared, shared=True)
        assert excinfo.value.reason_code == "incompatible_schema"
        with DB(private) as database:
            assert database.file_created

    def test_concurrent_migration_applies_once(self, tmp_path):
        path = str(tmp_path / "s" / "db.sqlite")
        DB(path, shared=True).close()
        script = tmp_path / "migrate.py"
        script.write_text(
            textwrap.dedent(
                f"""
                import sys, time
                from testmon import db
                def slow(con):
                    time.sleep(0.3)
                    con.execute("CREATE TABLE IF NOT EXISTS extra (x)")
                db.MIGRATIONS = {{1001: (slow, 1000)}}
                db.DATA_VERSION = 1001
                with db.DB({path!r}, shared=True) as database:
                    assert database.con.execute("PRAGMA user_version").fetchone()[0] == 1001
                """
            )
        )
        procs = [
            subprocess.Popen([sys.executable, str(script)], stderr=subprocess.PIPE)
            for _ in range(3)
        ]
        for proc in procs:
            _, err = proc.communicate(timeout=60)
            assert proc.returncode == 0, err.decode()
        with sqlite3.connect(path) as con:
            assert (
                con.execute(
                    "SELECT value FROM metadata WHERE key = 'migrations_applied'"
                ).fetchone()[0]
                == 1
            )
        con.close()


class TestEnvironments:
    def test_package_change_keeps_the_old_environment(self, database):
        first, changed = database.fetch_or_create_environment("e", "a 1.0", "3.12")
        assert not changed
        database.record_observations(first, [observation("t", [dep("a.py", V1, [0])])])
        second, changed = database.fetch_or_create_environment("e", "a 1.1", "3.12")
        assert changed and second != first
        back, changed = database.fetch_or_create_environment("e", "a 1.0", "3.12")
        assert back == first and not changed
        assert database.all_test_executions(first)

    def test_project_namespaces(self, database):
        a, _ = database.fetch_or_create_environment("default", "p", "3.12", project="a")
        b, _ = database.fetch_or_create_environment("default", "p", "3.12", project="b")
        assert a != b


class TestGc:
    def test_retention(self, database, clock):
        database.record_observations(
            database.env_id, [observation("old", [dep("a.py", V1, [0])])]
        )
        clock.advance(30)
        database.record_observations(
            database.env_id, [observation("new", [dep("b.py", V1, [0])])]
        )
        clock.advance(31)
        stats = database.gc(retention_days=60)
        assert stats["expired_observations"] == 1
        assert set(database.all_test_executions(database.env_id)) == {"new"}
        assert stats["orphan_fingerprints"] == 1

    def test_recent_use_keeps_an_observation(self, database, clock):
        database.record_observations(
            database.env_id, [observation("t", [dep("a.py", V1, [0])])]
        )
        (obs_id,) = database.con.execute("SELECT id FROM observation").fetchone()
        clock.advance(59)
        database.record_selection_side_effects([], [obs_id])
        clock.advance(59)
        database.gc(retention_days=60)
        assert count(database, "observation") == 1

    def test_max_versions_keeps_newest(self, database, clock):
        for value in range(10):
            database.record_observations(
                database.env_id, [observation("t", [dep("a.py", (value, 0), [0])])]
            )
            clock.advance(0.01)
        database.gc(max_versions=4)
        assert count(database, "observation") == 4
        newest = {dep("a.py", (value, 0), [0]).checksums for value in range(6, 10)}
        kept = {
            bytes(row[0])
            for row in database.con.execute(
                "SELECT f.checksums FROM dep_fp f JOIN observation_dep od ON od.fp_id = f.id"
            )
        }
        assert kept == newest
        assert count(database, "dep_fp") == 4

    def test_superseded_dropped_after_a_week(self, database, clock):
        database.record_observations(
            database.env_id, [observation("t", [dep("a.py", V1, [0])], failed=True)]
        )
        database.con.execute("UPDATE observation SET superseded = 1")
        clock.advance(6)
        database.gc()
        assert count(database, "observation") == 1
        clock.advance(2)
        database.gc()
        assert count(database, "observation") == 0

    def test_memo_snapshots_environments_sessions(self, database, clock):
        database.record_observations(
            database.env_id, [observation("t", [dep("a.py", V1, [0])])]
        )
        database.record_collection_snapshots(
            database.env_id, [("test_a.py", "ctx", ["x"])]
        )
        database.start_session(database.env_id, True, True)
        unused_env, _ = database.fetch_or_create_environment("unused", "p", "3.12")
        clock.advance(31)
        database.gc(retention_days=60)
        assert count(database, "fp_match") == 0  # memo older than 30 days
        assert count(database, "session") == 0
        assert count(database, "observation") == 1
        clock.advance(30)
        database.gc(retention_days=60)
        assert count(database, "collection_snapshot") == 0
        assert count(database, "environment", "WHERE id = ?", (unused_env,)) == 0

    def test_snapshots_per_file_are_bounded(self, database, clock):
        for i in range(dbmod.SNAPSHOTS_PER_FILE + 3):
            database.record_collection_snapshots(
                database.env_id, [("f.py", f"ctx{i}", ["x"])]
            )
            clock.advance(0.01)
        database.gc()
        assert count(database, "collection_snapshot") == dbmod.SNAPSHOTS_PER_FILE

    def test_gc_slot_is_claimed_once(self, database, clock):
        assert database.maybe_gc(interval_hours=6) is not None
        assert database.maybe_gc(interval_hours=6) is None
        clock.advance(0.3)
        assert database.maybe_gc(interval_hours=6) is not None

    def test_stale_sessions_are_marked_interrupted(self, database):
        proc = subprocess.Popen([sys.executable, "-c", "pass"])
        proc.wait()
        session = database.start_session(database.env_id, True, True)
        database.con.execute(
            "UPDATE session SET pid = ? WHERE id = ?", (proc.pid, session)
        )
        mine = database.start_session(database.env_id, True, True)
        assert database.mark_stale_sessions() == 1
        statuses = dict(
            database.con.execute("SELECT id, status FROM session").fetchall()
        )
        assert statuses == {session: "interrupted", mine: "running"}


class TestBoundedGrowth:
    def test_identical_runs_do_not_grow(self, database, clock):
        batch = [observation(f"t{i}", [dep("a.py", V1, [i % 2])]) for i in range(20)]
        database.record_observations(database.env_id, batch)
        counts = database.table_counts()
        for _ in range(50):
            clock.advance(0.01)
            database.record_observations(database.env_id, batch)
        assert database.table_counts() == counts

    def test_alternating_branches_are_bounded(self, database, clock):
        branches = [(1, 1), (2, 1), (3, 1)]
        for run in range(50):
            values = branches[run % 3]
            database.record_observations(
                database.env_id,
                [observation(f"t{i}", [dep("a.py", values, [0])]) for i in range(10)],
            )
            clock.advance(0.01)
        assert count(database, "observation") == 30
        assert count(database, "dep_fp") == 3


class TestBackup:
    def test_backup_and_restore(self, database, tmp_path):
        database.record_observations(
            database.env_id, [observation("t", [dep("a.py", V1, [0])])]
        )
        backup = tmp_path / "backup.sqlite"
        database.backup(str(backup))
        if os.name == "posix":
            assert oct(os.stat(backup).st_mode & 0o777) == "0o600"
        database.con.execute("DELETE FROM observation")
        database.restore_from(str(backup))
        assert set(database.all_test_executions(database.env_id)) == {"t"}
        assert database.integrity_check() == ["ok"]
