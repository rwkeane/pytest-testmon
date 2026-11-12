"""testmon-admin: inspect and maintain a testmon database outside pytest.

    testmon-admin [--db PATH | --rootdir DIR] [--json] COMMAND

    stats                      row counts, file sizes, sessions
    check                      PRAGMA integrity_check (exit 1 when not ok)
    gc [--aggressive] [--retention-days N] [--max-versions N]
    checkpoint                 WAL checkpoint (TRUNCATE)
    backup DEST                consistent online copy (safe while tests run)
    restore SRC [--force]      replace the database content with SRC
    explain NODEID [--env NAME]
                               every observation of a test and, against the source in
                               --rootdir, which dependency makes it (in)valid
    migrate                    apply pending schema migrations

Without --db the database is found like pytest finds it: TESTMON_SHARED_DB, then the
testmon_shared_db option in --rootdir's pytest config, then TESTMON_DATAFILE / .testmondata.
"""

import argparse
import json
import os
import shutil
import sqlite3
import sys
import tempfile

from testmon import db as dbmod
from testmon import selection
from testmon.config import read_project_ini, resolve_db_location
from testmon.process_code import blob_to_checksums
from testmon.testmon_core import SourceTree


def resolve(args):
    rootdir = os.path.abspath(args.rootdir)
    if args.db:
        return os.path.abspath(os.path.expanduser(args.db)), True
    ini = read_project_ini(rootdir)
    location = resolve_db_location(rootdir, ini_value=ini.get("testmon_shared_db"))
    return location.path, location.shared


def open_db(args):
    path, shared = resolve(args)
    if not os.path.exists(path):
        raise SystemExit(f"testmon-admin: no database at {path}")
    return dbmod.DB(path, shared=shared)


def emit(args, data, text=None):
    if args.json:
        print(json.dumps(data, indent=1, sort_keys=True, default=str))
    else:
        print(text if text is not None else json.dumps(data, indent=1, default=str))


def cmd_stats(args):
    with open_db(args) as database:
        counts = database.table_counts()
        sizes = {
            suffix or "db": os.path.getsize(database.datafile + suffix)
            for suffix in ("", "-wal", "-shm")
            if os.path.exists(database.datafile + suffix)
        }
        sessions = [
            dict(row)
            for row in database.con.execute(
                "SELECT status, count(*) AS n FROM session GROUP BY status"
            )
        ]
        data = {
            "path": database.datafile,
            "schema_version": database.con.execute("PRAGMA user_version").fetchone()[0],
            "counts": counts,
            "bytes": sizes,
            "sessions": sessions,
        }
    lines = [f"{data['path']} (schema {data['schema_version']})"]
    lines += [f"  {table}: {count}" for table, count in counts.items()]
    lines += [f"  size {name}: {size} bytes" for name, size in sizes.items()]
    emit(args, data, "\n".join(lines))
    return 0


def cmd_check(args):
    try:
        with open_db(args) as database:
            result = database.integrity_check()
    except dbmod.DbError as exc:
        emit(args, {"ok": False, "error": str(exc)}, f"not ok: {exc}")
        return 1
    ok = result == ["ok"]
    emit(args, {"ok": ok, "integrity_check": result}, "ok" if ok else "\n".join(result))
    return 0 if ok else 1


def cmd_gc(args):
    with open_db(args) as database:
        stats = database.gc(
            retention_days=args.retention_days,
            max_versions=args.max_versions,
            aggressive=args.aggressive,
        )
        database.checkpoint("TRUNCATE")
    emit(args, stats, "\n".join(f"{key}: {value}" for key, value in stats.items()))
    return 0


def cmd_checkpoint(args):
    with open_db(args) as database:
        busy, log, checkpointed = database.checkpoint("TRUNCATE")
    emit(
        args,
        {"busy": busy, "log_frames": log, "checkpointed": checkpointed},
        f"busy={busy} log={log} checkpointed={checkpointed}",
    )
    return 0 if not busy else 1


def cmd_backup(args):
    with open_db(args) as database:
        database.backup(args.dest)
    with dbmod.DB(args.dest, readonly=True) as copy:
        result = copy.integrity_check()
    ok = result == ["ok"]
    emit(
        args,
        {"dest": args.dest, "ok": ok},
        f"backup written to {args.dest}: {result[0]}",
    )
    return 0 if ok else 1


def cmd_restore(args):
    path, _ = resolve(args)
    with dbmod.DB(args.src, readonly=True) as source:
        if source.integrity_check() != ["ok"]:
            raise SystemExit(f"testmon-admin: {args.src} fails integrity_check")
    try:
        target = sqlite3.connect(path, timeout=60)
        try:
            source = sqlite3.connect(
                f"file:{os.path.abspath(args.src)}?mode=ro", uri=True
            )
            try:
                source.backup(target)
            finally:
                source.close()
        finally:
            target.close()
    except sqlite3.DatabaseError as exc:
        if not args.force:
            raise SystemExit(
                f"testmon-admin: cannot restore into {path} ({exc}); stop every pytest "
                "process using it and re-run with --force to replace the file"
            ) from exc
        directory = os.path.dirname(os.path.abspath(path))
        fd, tmp = tempfile.mkstemp(dir=directory, prefix=".testmon-restore-")
        os.close(fd)
        shutil.copyfile(args.src, tmp)
        os.chmod(tmp, 0o600)
        for suffix in ("-wal", "-shm"):
            if os.path.exists(path + suffix):
                os.remove(path + suffix)
        os.replace(tmp, path)
    emit(args, {"restored": path, "from": args.src}, f"restored {path} from {args.src}")
    return 0


def explain(database, nodeid, rootdir, env_name=None):
    """Every observation of nodeid, each dependency checked against the source in rootdir."""
    tree = SourceTree(rootdir)
    env_ids = None
    if env_name is not None:
        env_ids = {
            row[0]
            for row in database.con.execute(
                "SELECT id FROM environment WHERE name = ?", (env_name,)
            )
        }
    observations = []
    for row in database.observations_of(nodeid):
        if env_ids is not None and row["env_id"] not in env_ids:
            continue
        deps = []
        valid = True
        for dep in database.observation_deps(row["id"]):
            module = tree.get_file(dep["filename"])
            if module is None:
                status = selection.MISSING
            elif selection.fingerprint_matches(module, dep["checksums"]):
                status = "valid"
            else:
                status = selection.MISMATCH
            valid = valid and status == "valid"
            deps.append(
                {
                    "filename": dep["filename"],
                    "status": status,
                    "blocks": len(blob_to_checksums(dep["checksums"])),
                }
            )
        observations.append(
            {
                "id": row["id"],
                "env_id": row["env_id"],
                "env_name": row["env_name"],
                "failed": bool(row["failed"]),
                "superseded": bool(row["superseded"]),
                "valid": valid and len(deps) == row["dep_count"],
                "last_seen_at": row["last_seen_at"],
                "deps": deps,
            }
        )
    by_env = {}
    for obs in observations:
        by_env.setdefault(obs["env_id"], []).append(obs)
    decisions = {}
    for env_id, rows in by_env.items():
        obs_rows = [
            selection.ObservationRow(
                o["id"], nodeid, o["failed"], None, o["superseded"], o["last_seen_at"]
            )
            for o in rows
        ]
        decision = selection.decide(
            nodeid, obs_rows, {o["id"] for o in rows if o["valid"]}, None
        )
        decisions[env_id] = {"deselect": decision.deselect, **decision.as_dict()}
    return {"test": nodeid, "observations": observations, "decisions": decisions}


def cmd_explain(args):
    with open_db(args) as database:
        data = explain(database, args.nodeid, os.path.abspath(args.rootdir), args.env)
    lines = [f"{data['test']}: {len(data['observations'])} observation(s)"]
    for env_id, decision in data["decisions"].items():
        verdict = "deselect" if decision["deselect"] else "run"
        lines.append(f"  env {env_id}: {verdict} ({decision['reason']})")
    for obs in data["observations"]:
        state = "valid" if obs["valid"] else "invalid"
        flags = (" failed" if obs["failed"] else "") + (
            " superseded" if obs["superseded"] else ""
        )
        lines.append(f"  observation {obs['id']} env {obs['env_name']}: {state}{flags}")
        for dep in obs["deps"]:
            lines.append(f"      {dep['status']:9} {dep['filename']}")
    emit(args, data, "\n".join(lines))
    return 0 if data["observations"] else 1


def cmd_migrate(args):
    with open_db(args) as database:
        applied = database.migrate()
    emit(args, {"applied": applied}, f"applied migrations: {applied or 'none'}")
    return 0


def build_parser():
    parser = argparse.ArgumentParser(
        prog="testmon-admin", description=__doc__.split("\n")[0]
    )
    parser.add_argument("--db", help="database path")
    parser.add_argument("--rootdir", default=".", help="project/checkout directory")
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("stats").set_defaults(func=cmd_stats)
    sub.add_parser("check").set_defaults(func=cmd_check)
    gc_parser = sub.add_parser("gc")
    gc_parser.add_argument("--aggressive", action="store_true")
    gc_parser.add_argument(
        "--retention-days", type=float, default=dbmod.DEFAULT_RETENTION_DAYS
    )
    gc_parser.add_argument(
        "--max-versions", type=int, default=dbmod.DEFAULT_MAX_VERSIONS
    )
    gc_parser.set_defaults(func=cmd_gc)
    sub.add_parser("checkpoint").set_defaults(func=cmd_checkpoint)
    backup = sub.add_parser("backup")
    backup.add_argument("dest")
    backup.set_defaults(func=cmd_backup)
    restore = sub.add_parser("restore")
    restore.add_argument("src")
    restore.add_argument("--force", action="store_true")
    restore.set_defaults(func=cmd_restore)
    explain_parser = sub.add_parser("explain")
    explain_parser.add_argument("nodeid")
    explain_parser.add_argument("--env", default=None)
    explain_parser.set_defaults(func=cmd_explain)
    sub.add_parser("migrate").set_defaults(func=cmd_migrate)
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except dbmod.DbError as exc:
        print(f"testmon-admin: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
