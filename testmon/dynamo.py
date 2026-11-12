"""Dependencies of code compiled by ``torch.compile`` (TorchDynamo).

Coverage only sees lines that Python itself executes. Dynamo compiles a function by reading
its bytecode and from then on runs the compiled result, so the function's body never shows up
as executed and testmon would not know a test depends on it.

What dynamo traced for a compiled graph is in its ``TracingContext.traced_code``: the compiled
function, everything inlined into it and, after a graph break, the resume function. Dynamo
installs each compiled graph in the frame's globals (``OutputGraph.install_global_unsafe``,
names ``__compiled_fn...``) and the compiled bytecode calls it on *every* execution, the first
one and every later cache hit. Wrapping that installed function records, per test, the traced
code of every graph the test ran, without touching dynamo's caches: compilation behaves
exactly as without testmon.

Frames that produce no graph are not compiled at all; dynamo runs them as plain Python, which
coverage sees.

If a torch release stops installing graphs this way, more graphs get compiled
(``counters["stats"]["unique_graphs"]``) than were wrapped: that, or a TracingContext without
``traced_code``, is an error, never a silent gap. Verified on torch 2.10 and 2.14.
"""

import importlib.abc
import importlib.util
import sys

OUTPUT_GRAPH = "torch._dynamo.output_graph"
COMPILED_FN_PREFIX = "__compiled_fn"


class DynamoUnsupported(Exception):
    pass


def _torch_version():
    return getattr(sys.modules.get("torch"), "__version__", "?")


class _PostImportFinder(importlib.abc.MetaPathFinder):
    """Calls back right after torch._dynamo.output_graph is executed (if imported later)."""

    def __init__(self, callback):
        self.callback = callback

    def find_spec(self, fullname, path, target=None):  # pylint: disable=unused-argument
        if fullname != OUTPUT_GRAPH:
            return None
        sys.meta_path.remove(self)
        try:
            spec = importlib.util.find_spec(fullname)
        finally:
            sys.meta_path.insert(0, self)
        if spec is None or spec.loader is None:
            return spec
        loader = spec.loader
        original = loader.exec_module
        callback = self.callback

        def exec_module(module):
            original(module)
            callback(module)

        loader.exec_module = exec_module
        return spec


class DynamoTracker:
    """Collects, per test, the code traced for every compiled graph the test executed."""

    def __init__(self):
        self._ran = {}  # id(traced_code list) -> list, graphs executed since begin_test()
        self._lines = {}  # id(list) -> (list, {filename: lines}), conversion cache
        self._patched = None  # (OutputGraph, original install_global_unsafe)
        self._finder = None
        self._wrapped = 0
        self._broken = None
        self._graphs_seen = None  # unique_graphs counter at the last check
        self._wrapped_seen = 0
        self._reset_once = False  # graphs were compiled before the hook was installed

    # ------------------------------------------------------------------ lifecycle

    def start(self):
        module = sys.modules.get(OUTPUT_GRAPH)
        if module is not None:
            self._patch(module)
        else:
            self._finder = _PostImportFinder(self._patch)
            sys.meta_path.insert(0, self._finder)

    def stop(self):
        if self._finder is not None and self._finder in sys.meta_path:
            sys.meta_path.remove(self._finder)
        self._finder = None
        if self._patched is not None:
            cls, original = self._patched
            cls.install_global_unsafe = original
            self._patched = None
        self._ran = {}
        self._lines = {}

    def _patch(self, module):
        if self._patched is not None:
            return
        cls = getattr(module, "OutputGraph", None)
        original = getattr(cls, "install_global_unsafe", None)
        if original is None:
            raise DynamoUnsupported(
                f"torch {_torch_version()} has no OutputGraph.install_global_unsafe; "
                "testmon cannot record what torch.compile traced"
            )
        tracker = self

        def install_global_unsafe(self_, name, value):
            if (
                isinstance(name, str)
                and name.startswith(COMPILED_FN_PREFIX)
                and callable(value)
            ):
                value = tracker._wrap(self_, value)
            return original(self_, name, value)

        cls.install_global_unsafe = install_global_unsafe
        self._patched = (cls, original)
        self._graphs_seen = self._unique_graphs()
        # graphs compiled before now (e.g. during collection) aren't wrapped: a later cache
        # hit on one would go unrecorded, so empty dynamo's caches once, before the first test
        self._reset_once = bool(self._graphs_seen)
        if self._finder is not None and self._finder in sys.meta_path:
            sys.meta_path.remove(self._finder)
        self._finder = None

    def _wrap(self, output_graph, compiled_fn):
        self._wrapped += 1
        traced = getattr(
            getattr(output_graph, "tracing_context", None), "traced_code", None
        )
        if not isinstance(traced, list):
            self._broken = (
                f"torch {_torch_version()}: the TracingContext of a compiled graph has no "
                "traced_code list; testmon cannot record what torch.compile traced"
            )
            return compiled_fn
        ran = self._ran
        key = id(traced)

        def compiled_graph(*args, **kwargs):
            ran[key] = traced  # this graph runs (first call or cache hit) in this test
            return compiled_fn(*args, **kwargs)

        import torch._dynamo  # pylint: disable=import-outside-toplevel

        # like dynamo's own graph functions: never traced by dynamo itself
        return torch._dynamo.disable(compiled_graph)

    @staticmethod
    def _unique_graphs():
        utils = sys.modules.get("torch._dynamo.utils")
        counters = getattr(utils, "counters", None)
        if counters is None:
            return None
        return counters["stats"]["unique_graphs"]

    def _check_all_graphs_wrapped(self):
        graphs = self._unique_graphs()
        if graphs is None or self._graphs_seen is None:
            self._graphs_seen = graphs
            self._wrapped_seen = self._wrapped
            return
        # torch._dynamo.reset() clears the counters: count from zero again
        new_graphs = (
            graphs - self._graphs_seen if graphs >= self._graphs_seen else graphs
        )
        new_wrapped = self._wrapped - self._wrapped_seen
        self._graphs_seen, self._wrapped_seen = graphs, self._wrapped
        if new_graphs > new_wrapped:
            raise DynamoUnsupported(
                f"torch {_torch_version()} compiled {new_graphs} graph(s) of which testmon "
                f"saw {new_wrapped} installed; testmon cannot record what torch.compile "
                "traced with this torch"
            )

    # ------------------------------------------------------------------ per test

    def begin_test(self):
        """Start recording the graphs the next test runs.

        Dynamo's caches are left alone, except once if graphs had been compiled before the
        hook was installed.
        """
        if self._reset_once:
            self._reset_once = False
            import torch._dynamo  # pylint: disable=import-outside-toplevel

            torch._dynamo.reset()
            self._graphs_seen = self._unique_graphs()
            self._wrapped_seen = self._wrapped
        self._ran.clear()

    def harvest(self):
        """{filename: lines} traced for the graphs run since begin_test(); clears the record."""
        if self._broken:
            raise DynamoUnsupported(self._broken)
        if self._patched is not None:
            self._check_all_graphs_wrapped()
        ran = list(self._ran.values())
        self._ran.clear()
        traced = {}
        for codes in ran:
            cached = self._lines.get(id(codes))
            if cached is None or cached[0] is not codes:
                files = {}
                for code in codes:
                    lines = {line for _, _, line in code.co_lines() if line}
                    if lines:
                        files.setdefault(code.co_filename, set()).update(lines)
                cached = (codes, files)
                self._lines[id(codes)] = cached
            for filename, lines in cached[1].items():
                traced.setdefault(filename, set()).update(lines)
        return traced
