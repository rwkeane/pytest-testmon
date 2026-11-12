"""Small synthetic sources and observations for unit tests of the storage/selection layer.

A "version" of a file is a tuple of ints: the return values of functions g0, g1, ...
Changing one value changes only that function's block; the module-level block stays the
same as long as the number of functions does.
"""

from testmon import db as dbmod
from testmon.process_code import Module, checksums_to_blob, create_fingerprint
from testmon.testmon_core import SourceTree


def source(values):
    return "".join(
        f"def g{i}():\n    return {value}\n\n" for i, value in enumerate(values)
    )


def module(values):
    return Module(source_code=source(values), ext="py")


def body_line(function_index):
    return 3 * function_index + 2


def fingerprint(values, functions):
    lines = {1} | {body_line(i) for i in functions}
    return create_fingerprint(module(values), lines)


def dep(filename, values, functions):
    return dbmod.DepInput(
        filename,
        bytes(checksums_to_blob(fingerprint(values, functions))),
        module(values).fs_fsha,
    )


def observation(test_name, deps, failed=False, duration=0.1):
    return dbmod.ObservationInput(test_name, tuple(deps), failed, duration)


def tree(files):
    """SourceTree whose files are {filename: values or None (missing)}."""
    source_tree = SourceTree("/nonexistent-rootdir")
    for filename, values in files.items():
        source_tree.cache[filename] = None if values is None else module(values)
    return source_tree


def new_db(path, shared=False):
    database = dbmod.DB(str(path), shared=shared)
    env_id, _ = database.fetch_or_create_environment("default", "pkgs", "3.12")
    return database, env_id
