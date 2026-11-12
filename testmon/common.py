import importlib.metadata
import importlib.util
import json
import logging
import os
import re
import subprocess
import sys
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Dict, List, TypedDict


def editable_location(dist):
    """The source directory of an editable (PEP 660 / develop) install, else None."""
    metadata_dir = getattr(dist, "_path", None)
    if metadata_dir is not None:
        # cheap pre-check: most distributions have no direct_url.json at all
        try:
            if not os.path.exists(os.path.join(str(metadata_dir), "direct_url.json")):
                return None
        except (OSError, TypeError):
            pass
    try:
        text = dist.read_text("direct_url.json")
    except (OSError, ValueError):
        return None
    if not text:
        return None
    try:
        data = json.loads(text)
    except ValueError:
        return None
    if not (data.get("dir_info") or {}).get("editable"):
        return None
    url = data.get("url", "")
    if not url.startswith("file:"):
        return None
    return os.path.realpath(
        urllib.request.url2pathname(urllib.parse.urlparse(url).path)
    )


def _inside(path, real_root):
    return path == real_root or path.startswith(real_root.rstrip(os.sep) + os.sep)


def get_system_packages_raw(rootdir=None):
    """(name, version) of installed distributions.

    Distributions installed in editable mode from inside rootdir (the project under test)
    are left out: coverage tracks their source already, and their version often changes
    with every commit (setuptools-scm, hatch-vcs), which would invalidate everything.
    """
    real_root = os.path.realpath(rootdir) if rootdir else None
    for dist in importlib.metadata.distributions():
        if real_root:
            location = editable_location(dist)
            if location and _inside(location, real_root):
                continue
        yield dist.metadata["Name"], dist.version


class FileFp(TypedDict):
    filename: str
    method_checksums: List[int] = None
    mtime: float = None  # optimization helper, not really a part of the data structure fundamentally
    fsha: int = None  # optimization helper, not really a part of the data structure fundamentally
    fingerprint_id: int = None  # optimization helper,


TestName = str

TestFileFps = Dict[TestName, List[FileFp]]

Duration = float
Failed = bool


class DepsNOutcomes(TypedDict):
    deps: List[FileFp]
    failed: Failed
    duration: Duration
    forced: bool = None


TestExecutions = Dict[TestName, DepsNOutcomes]


def dummy():
    pass


def get_logger(name):
    formatter = logging.Formatter("%(levelname)s: %(message)s")
    handler = logging.StreamHandler()
    handler.setFormatter(formatter)

    # Configure the logger
    tm_logger = logging.getLogger(name)
    tm_logger.setLevel(logging.INFO)
    tm_logger.addHandler(handler)
    return tm_logger


logger = get_logger(__name__)


def get_system_packages(ignore=None, rootdir=None):
    if not ignore:
        ignore = set(("pytest-testmon", "pytest-testmon"))
    raw = get_system_packages_raw(rootdir) if rootdir else get_system_packages_raw()
    return ", ".join(
        sorted(
            {
                f"{package} {version}"
                for (package, version) in raw
                if package not in ignore and package != "UNKNOWN" and version != "0.0.0"
            }
        )
    )


def drop_patch_version(system_packages):
    return re.sub(
        r"\b([\w_-]+\s\d+\.\d+)\.\w+\b",  # extract (Package M.N).P / drop .patch
        r"\1",
        system_packages,
    )


#
# .git utilities
#
def git_path(start_path=None):  # parent dirs only
    """The git directory of the checkout containing start_path (worktrees included)."""
    start_path = Path(start_path or os.getcwd()).resolve()
    current_path = start_path
    while current_path != current_path.parent:  # '/'.parent == '/'
        path = current_path / ".git"
        if path.is_dir():
            return str(path)
        if path.is_file():  # a worktree or submodule: "gitdir: <path>"
            try:
                content = path.read_text(encoding="utf8").strip()
            except OSError:
                return None
            if content.startswith("gitdir:"):
                gitdir = Path(content[len("gitdir:") :].strip())
                if not gitdir.is_absolute():
                    gitdir = current_path / gitdir
                return str(gitdir.resolve())
            return None
        current_path = current_path.parent
    return None


def git_current_branch(path=None):
    path = git_path(path)
    if not path:
        return None
    git_head_file = os.path.join(path, "HEAD")
    try:
        with open(git_head_file, "r", encoding="utf8") as head_file:
            head_content = head_file.read().strip()
        if head_content.startswith("ref:"):
            return head_content.split("/")[-1]  # e.g. ref: refs/heads/master
    except FileNotFoundError:
        pass
    return None


def _git(args, cwd):
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=cwd or None,
            capture_output=True,
            text=True,
            encoding="utf-8",
            check=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout


def git_current_head(path=None):
    start = path
    path = git_path(path)
    if not path:
        return None
    current_branch = git_current_branch(path)
    if current_branch:
        git_branch_file = os.path.join(path, "refs", "heads", current_branch)
        try:
            with open(git_branch_file, "r", encoding="utf8") as branch_file:
                return branch_file.read().strip()
        except FileNotFoundError:
            pass
    # detached HEAD, packed refs, worktrees: ask git
    output = _git(["rev-parse", "HEAD"], start or os.getcwd())
    return output.strip() if output else None


def git_worktree_state(rootdir):
    """{git_head, dirty, diff_hash} of the checkout at rootdir, or None outside git."""
    head = _git(["rev-parse", "HEAD"], rootdir)
    if head is None:
        return None
    import hashlib  # pylint: disable=import-outside-toplevel

    status = _git(["status", "--porcelain", "--untracked-files=all"], rootdir) or ""
    diff_hash = None
    if status.strip():
        hsh = hashlib.sha256()
        hsh.update((_git(["diff", "HEAD", "--binary"], rootdir) or "").encode())
        hsh.update(status.encode())
        diff_hash = hsh.hexdigest()
    return {
        "git_head": head.strip(),
        "dirty": bool(status.strip()),
        "diff_hash": diff_hash,
    }


def find_foreign_checkout_imports(rootdir, modules=None):
    """Packages of this checkout that resolve to another checkout of the same layout.

    Typical cause: an editable install of the project made from a different clone or
    worktree. Tests would then run (and testmon would fingerprint) the other checkout's
    code, so dependency information recorded here would be incomplete. Returns
    [(package, origin)].

    A top-level package directory of rootdir (or rootdir/src) counts as foreign when the
    module already imported under that name, or for src layouts the module Python would
    import, lives at the same relative path below a *different* git checkout.
    """
    modules = sys.modules if modules is None else modules
    real_root = os.path.realpath(rootdir)
    found = []
    for base in (real_root, os.path.join(real_root, "src")):
        if not os.path.isdir(base):
            continue
        try:
            entries = list(os.scandir(base))
        except OSError:
            continue
        for entry in entries:
            if not (
                entry.is_dir()
                and entry.name.isidentifier()
                and os.path.isfile(os.path.join(entry.path, "__init__.py"))
            ):
                continue
            rel = os.path.relpath(os.path.join(entry.path, "__init__.py"), real_root)
            origin = None
            module = modules.get(entry.name)
            if module is not None:
                origin = getattr(module, "__file__", None)
            elif base != real_root:  # pytest never puts src/ on sys.path by itself
                try:
                    spec = importlib.util.find_spec(entry.name)
                except (ImportError, ValueError):
                    spec = None
                origin = spec.origin if spec else None
            if not origin:
                continue
            real_origin = os.path.realpath(origin)
            if _inside(real_origin, real_root) or not real_origin.endswith(
                os.sep + rel
            ):
                continue
            other_root = real_origin[: -len(rel) - 1]
            if os.path.exists(os.path.join(other_root, ".git")):
                found.append((entry.name, real_origin))
    return found
