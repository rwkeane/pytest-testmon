"""testmon-verify: run affected tests against an exact source snapshot (git hooks, agents).

    testmon-verify --staged [-- PYTEST ARGS]   the index (what `git commit` would record)
    testmon-verify --ref SHA [-- PYTEST ARGS]  a commit (pre-push)
    testmon-verify [-- PYTEST ARGS]            the working tree in place (may be dirty)

--staged and --ref materialize the snapshot in a temporary directory and run
``python -m pytest --testmon`` there with the current interpreter, so unstaged edits can't
leak into the result. The shared database is found through the snapshot's own pytest config
(testmon_shared_db), so selection reuses everything other checkouts recorded. The report
(--report PATH, default: printed summary only) carries ``verified_tree``: the git tree that
was actually tested. In-place runs report ``worktree.dirty`` instead.

Source roots of the snapshot (default: ``src`` when it exists; ini option
testmon_verify_pythonpath or --pythonpath) are put first on PYTHONPATH so that an editable
install of the project pointing at another checkout can't shadow the snapshot's code.
"""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile

from testmon.config import REPORT_ENV, VERIFIED_TREE_ENV, read_project_ini
from testmon.report import load_report


def git(args, cwd, **kwargs):
    return subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        **kwargs,
    ).stdout.strip()


def toplevel(cwd):
    return git(["rev-parse", "--show-toplevel"], cwd)


def pythonpath_for(tree, explicit):
    roots = list(explicit or ())
    if not roots:
        configured = read_project_ini(tree).get("testmon_verify_pythonpath")
        if isinstance(configured, str):
            roots = configured.split()
        elif configured:
            roots = list(configured)
    if not roots and os.path.isdir(os.path.join(tree, "src")):
        roots = ["src"]
    if not roots:
        roots = ["."]
    return [os.path.join(tree, root) for root in roots]


def run_pytest(tree, pytest_args, report_path, verified_tree, pythonpath):
    env = dict(os.environ)
    env[REPORT_ENV] = report_path
    if verified_tree:
        env[VERIFIED_TREE_ENV] = verified_tree
    else:
        env.pop(VERIFIED_TREE_ENV, None)
    if pythonpath:
        env["PYTHONPATH"] = os.pathsep.join(
            pythonpath + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else [])
        )
    command = [sys.executable, "-m", "pytest", "--testmon", *pytest_args]
    return subprocess.run(command, cwd=tree, env=env, check=False).returncode


def materialize_index(repo, dest):
    git(["checkout-index", "--all", f"--prefix={dest.rstrip(os.sep)}{os.sep}"], repo)
    return git(["write-tree"], repo)


def summarize(report_path):
    try:
        report = load_report(report_path)
    except (OSError, ValueError):
        return "testmon-verify: no report produced"
    counts = report["counts"]
    where = report.get("verified_tree") or (
        "dirty working tree"
        if (report.get("worktree") or {}).get("dirty")
        else "worktree"
    )
    fallback = report.get("fallback")
    note = f", fallback: {fallback['reason_code']}" if fallback else ""
    return (
        f"testmon-verify: {where}: {counts['selected']} selected, "
        f"{counts['deselected']} deselected of {counts['collected']}{note}"
    )


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    pytest_args = []
    if "--" in argv:
        index = argv.index("--")
        argv, pytest_args = argv[:index], argv[index + 1 :]
    parser = argparse.ArgumentParser(prog="testmon-verify")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--staged", action="store_true", help="test the git index")
    mode.add_argument("--ref", help="test this commit")
    parser.add_argument("--report", help="also keep the JSON report at this path")
    parser.add_argument(
        "--pythonpath",
        action="append",
        help="source root inside the snapshot (repeatable)",
    )
    parser.add_argument(
        "--keep", action="store_true", help="keep the temporary snapshot"
    )
    args = parser.parse_args(argv)

    repo = toplevel(os.getcwd())
    workdir = tempfile.mkdtemp(prefix="testmon-verify-")
    report_path = (
        os.path.abspath(args.report)
        if args.report
        else os.path.join(workdir, "report.json")
    )
    tree_dir = os.path.join(workdir, "tree")
    added_worktree = False
    try:
        if args.staged:
            verified_tree = materialize_index(repo, tree_dir)
        elif args.ref:
            sha = git(["rev-parse", "--verify", f"{args.ref}^{{commit}}"], repo)
            git(["worktree", "add", "--detach", tree_dir, sha], repo)
            added_worktree = True
            verified_tree = git(["rev-parse", f"{sha}^{{tree}}"], repo)
        else:
            tree_dir, verified_tree = os.getcwd(), None
        pythonpath = (
            pythonpath_for(tree_dir, args.pythonpath)
            if (args.staged or args.ref or args.pythonpath)
            else []
        )
        code = run_pytest(tree_dir, pytest_args, report_path, verified_tree, pythonpath)
        print(summarize(report_path))
        return code
    finally:
        if added_worktree and not args.keep:
            subprocess.run(
                ["git", "worktree", "remove", "--force", tree_dir],
                cwd=repo,
                check=False,
                capture_output=True,
            )
            subprocess.run(["git", "worktree", "prune"], cwd=repo, check=False)
        if not args.keep:
            shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
