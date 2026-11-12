"""Benchmark this fork against unmodified upstream testmon on identical workloads.

    python benchmarks/bench.py --upstream-src PATH [--modules 300 --tests 3000] [--out f.json]

--upstream-src is a checkout of upstream pytest-testmon (e.g. `git worktree add /tmp/up
main`); it is put first on PYTHONPATH for the upstream runs so that its `testmon` package is
imported instead of the fork's. Both use the current interpreter and installed packages.

Workloads (each in a fresh git repository generated from the same seed):

    cold          first instrumented run (empty database)
    no_change     second run, nothing changed
    one_file      one source module changed
    branch_20     20 versions of a hot module recorded, then back to the first version
    agents_4      4 worktrees at once, cold, sharing the database (fork: shared db;
                  upstream: each its own .testmondata) and then a no-change run each
    xdist_4       cold and no-change under -n 4

Reported per run: wall time, tests run, and for the fork the selection time and database
size. Numbers depend on the machine; compare the two columns, not absolute values.
"""

import argparse
import json
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

PYTHON = sys.executable


def generate(root, modules, tests, seed=0):
    rng = random.Random(seed)
    (root / "proj").mkdir(parents=True)
    (root / "proj" / "__init__.py").write_text("")
    for index in range(modules):
        lines = [
            f"def f{j}(x):\n    return x + {rng.randint(1, 9)}\n\n" for j in range(5)
        ]
        (root / "proj" / f"m{index}.py").write_text("\n".join(lines))
    (root / "tests").mkdir()
    (root / "tests" / "__init__.py").write_text("")
    per_file = 30
    for file_index in range(max(1, tests // per_file)):
        body = ["from proj import " + ", ".join(f"m{i}" for i in range(modules)) + "\n"]
        for test_index in range(per_file):
            a, b = rng.randrange(modules), rng.randrange(modules)
            body.append(
                f"\ndef test_{file_index}_{test_index}():\n"
                f"    assert m{a}.f{rng.randrange(5)}(1) + m{b}.f{rng.randrange(5)}(2) > 0\n"
            )
        (root / "tests" / f"test_{file_index}.py").write_text("".join(body))
    (root / "pytest.ini").write_text("[pytest]\n")
    (root / ".gitignore").write_text("__pycache__/\n.testmondata*\nreport.json\n")
    git(root, "init", "-q", "-b", "main")
    git(root, "config", "user.email", "bench@example.invalid")
    git(root, "config", "user.name", "bench")
    commit(root, "initial")


def git(root, *args):
    subprocess.run(["git", *args], cwd=root, check=True, capture_output=True)


def commit(root, message):
    git(root, "add", "-A")
    git(root, "commit", "-q", "--allow-empty", "-m", message)


def env_for(flavor, upstream_src, shared_db):
    env = {
        k: v for k, v in os.environ.items() if not k.startswith(("TESTMON_", "PYTEST_"))
    }
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    if flavor == "upstream":
        env["PYTHONPATH"] = str(upstream_src)
    else:
        env.pop("PYTHONPATH", None)
        if shared_db:
            env["TESTMON_SHARED_DB"] = str(shared_db)
    return env


def init_db(shared_db):
    if shared_db and not Path(shared_db).exists():
        subprocess.run(
            [PYTHON, "-m", "testmon.admin", "--db", str(shared_db), "init"],
            check=True,
            capture_output=True,
        )


def run(root, flavor, upstream_src, shared_db=None, extra=(), wait=True):
    command = [
        PYTHON,
        "-m",
        "pytest",
        "-q",
        "-p",
        "no:cacheprovider",
        "--testmon",
        *extra,
    ]
    report = root / "report.json"
    if flavor == "fork":
        command.append(f"--testmon-report={report}")
        init_db(shared_db)
    start = time.perf_counter()
    proc = subprocess.Popen(
        command,
        cwd=root,
        env=env_for(flavor, upstream_src, shared_db),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    if not wait:
        return proc, start
    return finish(proc, start, root, flavor)


def finish(proc, start, root, flavor):
    out, _ = proc.communicate()
    elapsed = time.perf_counter() - start
    if proc.returncode not in (0, 5):
        raise SystemExit(f"pytest failed in {root} ({flavor}):\n{out[-3000:]}")
    match = re.search(r"(\d+) passed", out)
    result = {"wall": round(elapsed, 3), "ran": int(match.group(1)) if match else 0}
    report = root / "report.json"
    if flavor == "fork" and report.exists():
        data = json.loads(report.read_text())
        result["selection"] = round(data["timings"].get("selection", 0), 4)
        result["db_open"] = round(data["timings"].get("db_open", 0), 4)
        result["writes"] = round(data["timings"].get("writes", 0), 4)
    return result


def db_size(path):
    total = 0
    for suffix in ("", "-wal", "-shm"):
        candidate = Path(str(path) + suffix)
        if candidate.exists():
            total += candidate.stat().st_size
    return total


def change_module(root, index, value):
    path = root / "proj" / f"m{index}.py"
    text = path.read_text()
    path.write_text(re.sub(r"return x \+ \d+", f"return x + {value}", text, count=1))


def workloads(flavor, base, upstream_src, args):
    results = {}
    root = base / flavor / "repo"
    generate(root, args.modules, args.tests)
    shared = base / flavor / "cache" / "db.sqlite" if flavor == "fork" else None
    results["cold"] = run(root, flavor, upstream_src, shared)
    results["no_change"] = run(root, flavor, upstream_src, shared)
    change_module(root, 0, 42)
    results["one_file"] = run(root, flavor, upstream_src, shared)

    # 20 versions of a hot module, then back to the first one
    hot = 1
    for version in range(20):
        change_module(root, hot, 100 + version)
        run(root, flavor, upstream_src, shared)
    change_module(root, hot, 100)
    results["branch_20"] = run(root, flavor, upstream_src, shared)
    results["db_bytes"] = db_size(shared if shared else root / ".testmondata")

    # four agents at once in separate worktrees
    commit(root, "state")
    trees = []
    for index in range(4):
        tree = base / flavor / f"agent{index}"
        git(root, "worktree", "add", "-q", "--detach", str(tree), "HEAD")
        trees.append(tree)
    agents_db = (
        base / flavor / "cache-agents" / "db.sqlite" if flavor == "fork" else None
    )
    start = time.perf_counter()
    started = [run(tree, flavor, upstream_src, agents_db, wait=False) for tree in trees]
    cold = [finish(proc, t0, tree, flavor) for (proc, t0), tree in zip(started, trees)]
    results["agents_4_cold_wall"] = round(time.perf_counter() - start, 3)
    results["agents_4_cold"] = cold
    results["agents_4_no_change"] = [
        run(tree, flavor, upstream_src, agents_db) for tree in trees
    ]

    xroot = base / flavor / "xdist"
    generate(xroot, args.modules, args.tests)
    xshared = base / flavor / "cache-x" / "db.sqlite" if flavor == "fork" else None
    results["xdist_4_cold"] = run(xroot, flavor, upstream_src, xshared, ("-n", "4"))
    results["xdist_4_no_change"] = run(
        xroot, flavor, upstream_src, xshared, ("-n", "4")
    )
    return results


def table(fork, upstream):
    rows = [
        "| workload | upstream wall s (ran) | fork wall s (ran) | fork selection s |",
        "|---|---|---|---|",
    ]
    for key in (
        "cold",
        "no_change",
        "one_file",
        "branch_20",
        "xdist_4_cold",
        "xdist_4_no_change",
    ):
        up, fk = upstream[key], fork[key]
        rows.append(
            f"| {key} | {up['wall']} ({up['ran']}) | {fk['wall']} ({fk['ran']}) | "
            f"{fk.get('selection', '-')} |"
        )
    up_agents = sum(r["ran"] for r in upstream["agents_4_no_change"])
    fk_agents = sum(r["ran"] for r in fork["agents_4_no_change"])
    rows.append(
        f"| agents_4 cold (wall, all 4) | {upstream['agents_4_cold_wall']} | "
        f"{fork['agents_4_cold_wall']} | |"
    )
    rows.append(
        f"| agents_4 no_change (tests run, all 4) | {up_agents} | {fk_agents} | |"
    )
    rows.append(f"| database bytes | {upstream['db_bytes']} | {fork['db_bytes']} | |")
    return "\n".join(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--upstream-src", required=True)
    parser.add_argument("--modules", type=int, default=300)
    parser.add_argument("--tests", type=int, default=3000)
    parser.add_argument("--out")
    parser.add_argument("--keep", action="store_true")
    args = parser.parse_args()
    base = Path(tempfile.mkdtemp(prefix="testmon-bench-"))
    try:
        upstream = workloads("upstream", base, Path(args.upstream_src).resolve(), args)
        fork = workloads("fork", base, None, args)
    finally:
        if not args.keep:
            shutil.rmtree(base, ignore_errors=True)
    data = {
        "modules": args.modules,
        "tests": args.tests,
        "upstream": upstream,
        "fork": fork,
    }
    if args.out:
        Path(args.out).write_text(json.dumps(data, indent=1))
    print(table(fork, upstream))


if __name__ == "__main__":
    main()
