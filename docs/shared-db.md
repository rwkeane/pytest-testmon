# Shared, multi-version testmon database

This fork of pytest-testmon lets every checkout, branch and agent on a machine share one
SQLite database. A test's dependency information is collected once and reused wherever it
still matches the source, whatever the branch, commit or filesystem path. Nothing needs to be
synced, downloaded, committed or managed by hand.

## Setup (once per project)

Configure the location once, in the project. No command-line flag is needed. Either put it in
the conftest.py at the repository root (it's Python, so the value can be computed):

```python
# conftest.py (rootdir)
import pytest


@pytest.hookimpl(optionalhook=True)  # keeps the conftest loadable where testmon isn't installed
def pytest_testmon_settings(config):
    return {"shared_db": "~/.cache/testmon/equicloud.sqlite"}
```

or in the pytest configuration, for example `pyproject.toml`:

```toml
[tool.pytest.ini_options]
testmon_shared_db = "~/.cache/testmon/equicloud.sqlite"
```

Then create the database once per machine:

```console
$ testmon-admin --db ~/.cache/testmon/equicloud.sqlite init
```

From then on `pytest --testmon` in any clone, worktree or branch uses that file. Nothing has to
be done per checkout or per branch. The directory is created with mode 0700 and the database
with 0600.

pytest never creates a shared database itself. If the configured file doesn't exist (wrong
path, a new machine, deleted), `pytest --testmon` stops with an error that shows the
`testmon-admin ... init` command. A typo can't silently start an empty database.

### Settings

| conftest key | ini option | default |
|---|---|---|
| `shared_db` | `testmon_shared_db` | (private `.testmondata`) |
| `db_fallback` | `testmon_db_fallback` | `error` (or `run-all`) |
| `db_timeout` | `testmon_db_timeout` | 30 (seconds to wait for a lock) |
| `project` | `testmon_project` | "" (namespace inside a database shared by projects) |
| `retention_days` | `testmon_retention_days` | 60 |
| `max_versions` | `testmon_max_versions` | 32 |
| `gc_interval_hours` | `testmon_gc_interval_hours` | 6 |
| `ignore_collect` | `testmon_ignore_collect` | true |

The conftest value wins over the ini option. An unknown key or a wrong type stops pytest
with a usage error. The hook is read before collection, so it must live in the rootdir's
conftest.py (or in a directory given to pytest on the command line). If a conftest that loads
later implements it, testmon stops with an error instead of ignoring it. `testmon-admin`
doesn't import conftests, so give it the path with `--db` (its `--rootdir` lookup only reads
the pytest ini). `testmon-verify` runs pytest, so it sees the conftest normally.

Where the database is looked up (first match wins):

| Source | Mode |
|---|---|
| `--testmon-shared-db=PATH` | shared |
| `TESTMON_SHARED_DB=PATH` | shared |
| `shared_db` from `pytest_testmon_settings` (conftest) | shared |
| `testmon_shared_db = PATH` (ini) | shared |
| `TESTMON_DATAFILE=PATH` | private (as upstream) |
| `<rootdir>/.testmondata` | private (as upstream) |

`~` and `$VARS` are expanded, and relative paths are relative to the rootdir.

> On the command line, write `--testmon-shared-db=PATH` and `--testmon-report=PATH` with
> `=`. pytest picks its rootdir before plugins load. A separate value would be taken for a
> path argument, which moves the rootdir once the file exists.

## How selection works

Every execution of a test is stored as an **observation**: the code blocks it executed in
each file (testmon's usual block fingerprints), plus whether it failed. A test has as many
observations as distinct dependency sets have been seen. Identical ones are deduplicated, so
repeating a run doesn't grow the database.

An observation is **valid** when every file it used exists and still contains every block
the test executed. A valid observation means the test would execute the same code again.

* A test is **deselected** when it has a valid passing observation and no valid failing one.
* A test **runs** when it has no observation (`no_observation`) or no valid one
  (`all_invalid`, and the report names the first file that differs). It also runs when a
  valid observation is a failure (`valid_failing_observation`). When it then passes in that
  same state, the failure is marked superseded.

Branch names, commit SHAs, mtimes and checkout paths play no part, which has these consequences:

* Switching A → B → A reruns nothing that was already run on A and B.
* Cherry-picks, rebases and squash merges rerun nothing whose dependencies were already seen
  on either side. A rebase onto a change nobody tested reruns exactly the affected tests.
* A clone at a different path, a worktree, a symlinked checkout and a plain copy without
  `.git` all reuse each other's records.

Whole test files are skipped at collection when every test that file produced last time
(recorded per file content, conftest.py files on its path and pytest config) is deselected.
Otherwise the file is collected and individual tests are deselected.

xdist workers get the controller's selection (they open no database), so all workers collect
the same tests even while other agents write.

## torch.compile

Testmon learns a test's dependencies by watching which lines run (coverage). Code compiled
by `torch.compile` never runs line by line: dynamo reads the function and then runs the
compiled result, so coverage alone would not see the function's body at all, and an edit
inside it would not rerun its tests. This fork closes that gap.

For every compiled graph, dynamo records what it traced: the compiled function, everything
inlined into it (other modules included), and the code after a graph break. The compiled code
calls its graph on every execution, the first one and every later cache hit. Testmon wraps
that call while recording, so each test is credited with the traced code of every graph it
ran, including graphs compiled earlier by another test. An edit inside a compiled function
then reruns exactly the tests that ran it.

* **Compilation is untouched.** Nothing is reset between tests: cache hits stay cache hits,
  with the same number of compilations as without testmon. The only exception is a single
  `torch._dynamo.reset()` before the first test, and only if graphs were already compiled
  before testmon could watch, for example during collection. The per-call cost is one
  extra Python call per graph execution. Selection-only runs (`--testmon-nocollect`) add
  nothing.
* **Frames that produce no graph** aren't compiled by dynamo at all. They run as plain
  Python, which coverage sees.
* **An unsupported torch is an error.** If a torch release stops working this way
  (a graph's `traced_code` missing, or more graphs compiled than testmon saw), pytest stops
  with exit code 3 and `cannot record what torch.compile traced`, and nothing from that
  batch is stored. Verified on torch 2.10 and 2.14.

Out of scope: TorchScript (`torch.jit.script`), `torch.export`, and generated or native
kernels (Triton, CUDA, C++).

## Modes (unchanged flags)

| Goal | Command |
|---|---|
| Select affected tests and collect dependencies | `pytest --testmon` |
| Full suite, collecting dependencies | `pytest --testmon --testmon-noselect` |
| Full suite, no instrumentation | `pytest` (or `--no-testmon`) |
| Select, don't record | `pytest --testmon --testmon-nocollect` |
| Separate configuration (e.g. GPU, settings) | `--testmon-env=NAME` |

**Filters intersect.** `-k`, `-m` and conftest flags that skip or deselect tests (`--gpu`,
`--slow`, ...) only narrow the collected tests, and testmon then runs those among them that
need running. Records are shared between filters: a test that ran under `--gpu` is
deselected under `--gpu --slow` if its code hasn't changed. Skipped tests are never
recorded (a skip says nothing about dependencies), so a test that was skipped because its
flag was off runs the first time the flag is given. This differs from upstream, where `-k`/`-m`
switched selection off. Explicit `file::test` arguments and `--lf` still run what you named.
Flags that change *how* tests behave (a backend, a device, settings) need their own
`--testmon-env=NAME` instead.

## When the database can't be used

`pytest --testmon` fails loudly rather than degrading silently. If the database is missing,
unreadable, corrupted, locked past the timeout, foreign (upstream format) or incompatible,
pytest exits with **code 4 before running any test**:

```text
testmon: database unavailable (missing: ... does not exist; create it once with
`testmon-admin --db ... init`, or run pytest without --testmon). No tests were run. ...
```

It's then up to you (or the agent) what to run: fix the database, or drop `--testmon` and run
the full suite or specific tests. The report (`--testmon-report`) is still written, with
`fallback.reason_code` (missing, open_failed, permission, corrupt, incompatible_schema,
foreign_db, lock_timeout, io, db_error, foreign_checkout) and `fallback.policy`.

To run everything instead when testmon can't be used, opt in with
`testmon_db_fallback = run-all` (or `--testmon-db-fallback=run-all`). Every test then runs,
nothing is recorded, and the header and report say why.

If storing an observation fails mid-run (for example a lock timeout after retries), only that
batch is lost and those tests will run again next time. Test outcomes are never affected.
pytest prints a warning, and the report's `counts.dropped_batches` and `warnings` say so.

pytest never deletes, renames or modifies a shared database it refuses to use. A private
`.testmondata` in upstream's format is recreated, as upstream does.

**Foreign checkout guard.** The same policy applies when packages of this checkout are
imported from another checkout, typically through an editable install made from a different
clone. Dependency information would then be recorded for the wrong code.


## Explaining decisions

`pytest --testmon --testmon-report=report.json` (or `TESTMON_REPORT=...`) writes a
JSON report. Its schema is `docs/report.schema.json`, versioned by `schema_version`. It contains:

* `tests`: for each collected test, `decision` (selected/deselected), `reason`,
  `observation_id`, `invalid_dep` (`{filename, cause: missing|mismatch|incomplete}`),
  `file_ignored` and `outcome`.
* `counts`: collected, selected, deselected, file_ignored, observations_reused, dropped_batches.
* `fallback`, `timings` (db_open, selection, writes, gc, total), `worktree` (git_head,
  dirty, diff_hash), `verified_tree`, `warnings`.

`pytest -v` adds a one-line summary of decision reasons. For one test across every stored
version:

```console
$ testmon-admin --rootdir . explain tests/test_calc.py::test_add
```

## Git hooks and snapshots

`testmon-verify` runs the affected tests against an exact snapshot, so unstaged edits can't
leak into a pre-commit result:

```console
$ testmon-verify --staged           # the index, as `git commit` will record it
$ testmon-verify --ref HEAD          # a commit (pre-push)
$ testmon-verify -- -x -q            # in place; arguments after -- go to pytest
```

The snapshot is materialized in a temporary directory and tested with the current
interpreter. Its source roots go first on `PYTHONPATH` (`src` if present, or ini
`testmon_verify_pythonpath`). The report's `verified_tree` is the git tree that was
tested. In-place runs report `worktree.dirty` instead. A `.pre-commit-config.yaml` entry:

```yaml
- repo: local
  hooks:
    - id: testmon
      name: affected tests (testmon)
      entry: testmon-verify --staged
      language: system
      pass_filenames: false
      stages: [pre-commit]
```

## Maintenance

None is required routinely:

* Schema upgrades run automatically. Additive migrations keep older fork versions (pinned
  on older branches) working.
* Cleanup runs at most every `testmon_gc_interval_hours` (default 6), in short transactions,
  at the end of some session:
  * Observations unused for `testmon_retention_days` (default 60) are dropped, as are
    versions beyond `testmon_max_versions` per test (default 32) and superseded failures
    after 7 days.
  * Unreferenced fingerprints and old cache rows are dropped.
* Restarts change nothing. The database is an ordinary WAL-mode SQLite file (synchronous=NORMAL).

`testmon-admin` (use `--db PATH`, or `--rootdir` to find it like pytest does):

| Command | Purpose |
|---|---|
| `init` | create the database (once per machine) |
| `stats` | row counts, sizes, sessions |
| `check` | `PRAGMA integrity_check` (exit 1 if not ok) |
| `gc [--aggressive]` | clean up now |
| `checkpoint` | fold the WAL into the database file |
| `backup DEST` | consistent online copy, safe while tests run |
| `restore SRC [--force]` | restore; `--force` replaces a file too damaged to restore into (stop agents first) |
| `explain NODEID` | see above |
| `migrate` | apply pending migrations |

A nightly backup, for example from cron:

```cron
15 3 * * * testmon-admin --db ~/.cache/testmon/equicloud.sqlite backup ~/.cache/testmon/equicloud.bak
```

The database is a cache. If it is ever lost or corrupted, delete it (with no pytest running),
run `testmon-admin init` again, and pay one round of re-collection. Nothing else is lost.

## Limitations (shared with upstream unless noted)

What is not tracked: code executed at collection time (except that conftest.py and pytest
config changes force re-collection of the files they govern), non-Python inputs (data files,
environment variables, CLI flags of custom conftest options), native code, and anything
outside the rootdir that coverage doesn't measure. Use `--testmon-env` for
configuration-specific runs, and full runs (`--testmon-noselect`) when such inputs change.

SQLite stores no checksums: silent bit-level corruption that still reads back is not
detected. `testmon-admin check` finds structural damage.

The database must be on a local filesystem (WAL needs shared memory), not NFS.

## Testing and benchmarks

* `tox` runs the upstream suite (also against a shared database: the `shared` flavor) and
  the unit tests, including a property test of the engine against a reference
  implementation.
* `tox -e scenarios-py311` runs git scenarios, concurrent agents, faults and lifecycle.
* `tox -e differential-py311` runs differential validation of selective against full runs.
  Soak it with `TESTMON_DIFF_SEEDS=0,1,...,19 TESTMON_DIFF_STEPS=25`.
* `python benchmarks/bench.py --upstream-src <upstream checkout>` compares against upstream
  testmon on the same workloads.

### Benchmark results

One run of `benchmarks/bench.py` (300 modules, 3,000 tests, Linux, Python 3.12) against
upstream 2.3.0 on identical workloads. Wall seconds per pytest process, with tests run in
parentheses:

| workload | upstream | fork | fork selection |
|---|---|---|---|
| cold (empty database) | 15.01 (3000) | 11.44 (3000) | 0.0002 |
| no change | 0.80 (0) | 1.11 (0) | 0.074 |
| one module changed | 1.57 (5) | 1.23 (5) | 0.049 |
| back to a version seen 20 versions ago | 1.54 (4) | 0.88 (0) | 0.072 |
| `-n 4` cold | 12.45 (3000) | 10.81 (3000) | |
| `-n 4` no change | 2.00 (0) | 2.13 (0) | 0.085 |
| 4 agents at once, cold (wall for all 4) | 18.75 | 15.29 | |
| database size, bytes | 1,241,088 | 1,982,464 | |

* Multi-version storage pays off when returning to earlier source states: upstream reran the
  dependents after 20 intermediate versions, the fork ran nothing.
* Shared storage pays off across checkouts. The scenario tests show 0 tests run in a new
  clone or worktree after another checkout ran them; upstream reruns everything there.
* Cold runs were faster here (dependency writes are batched in fewer, shorter
  transactions).
* A no-change run cost about 0.3 s more in this measurement. Selection itself is 0.07 s.
  The rest is per-session bookkeeping, plus `--testmon-report`, which the fork runs were
  given and the upstream runs weren't (it shells out to git for the worktree state).
* The database is about 1.6× larger: it keeps 21 versions of the hot module's dependents
  plus the content-match cache.
