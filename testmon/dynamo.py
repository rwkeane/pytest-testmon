"""Dependencies of code compiled by ``torch.compile`` (TorchDynamo).

Coverage only sees lines that Python itself executes. Dynamo compiles a function by reading
its bytecode and from then on runs the compiled result, so the function's body never shows up
as executed and testmon would not know the test depends on it. Dynamo does record every code
object it traced (the compiled function, everything it inlined, the resume functions after a
graph break) in ``TracingContext.traced_code``; this module collects those for each test.

Two rules make that complete:

* every TracingContext created while a test runs belongs to that test (contexts are recorded
  from the moment ``torch._guards`` is imported, even mid-test);
* dynamo's caches are reset before each test, so a test that runs compiled code compiles it,
  and therefore traces it, itself. Without the reset a cache hit would record nothing.

A torch whose TracingContext has no ``traced_code`` cannot be supported: that is an error,
never a silent gap.
"""

import importlib.abc
import importlib.util
import sys

GUARDS = "torch._guards"


class DynamoUnsupported(Exception):
    pass


class _PostImportFinder(importlib.abc.MetaPathFinder):
    """Calls back right after torch._guards is executed (if it is imported later)."""

    def __init__(self, callback):
        self.callback = callback

    def find_spec(self, fullname, path, target=None):  # pylint: disable=unused-argument
        if fullname != GUARDS:
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
    """Collects the code torch.compile traced, per test."""

    def __init__(self):
        self._lists = []  # the traced_code lists of contexts created since begin_test()
        self._patched = None  # (TracingContext, original __init__)
        self._finder = None

    # ------------------------------------------------------------------ lifecycle

    def start(self):
        module = sys.modules.get(GUARDS)
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
            cls.__init__ = original
            self._patched = None
        self._lists = []

    def _patch(self, module):
        if self._patched is not None:
            return
        cls = getattr(module, "TracingContext", None)
        if cls is None:
            raise DynamoUnsupported(
                f"torch._guards has no TracingContext (torch "
                f"{getattr(sys.modules.get('torch'), '__version__', '?')}); testmon cannot "
                "record what torch.compile traced"
            )
        original = cls.__init__
        lists = self

        def __init__(self_, *args, **kwargs):
            original(self_, *args, **kwargs)
            traced = getattr(self_, "traced_code", None)
            if not isinstance(traced, list):
                lists._lists.append(None)  # reported by harvest()
            else:
                lists._lists.append(traced)

        cls.__init__ = __init__
        self._patched = (cls, original)
        if self._finder is not None and self._finder in sys.meta_path:
            sys.meta_path.remove(self._finder)
        self._finder = None

    # ------------------------------------------------------------------ per test

    def begin_test(self):
        """Forget earlier contexts and empty dynamo's caches (only if dynamo is loaded)."""
        self._lists = []
        dynamo = sys.modules.get("torch._dynamo")
        if dynamo is not None and hasattr(dynamo, "reset"):
            dynamo.reset()

    def harvest(self):
        """{filename: lines} traced by dynamo since begin_test(); clears the record."""
        lists, self._lists = self._lists, []
        traced = {}
        for codes in lists:
            if codes is None:
                raise DynamoUnsupported(
                    "torch's TracingContext has no traced_code list (torch "
                    f"{getattr(sys.modules.get('torch'), '__version__', '?')}); testmon "
                    "cannot record what torch.compile traced"
                )
            for code in codes:
                lines = {line for _, _, line in code.co_lines() if line}
                if lines:
                    traced.setdefault(code.co_filename, set()).update(lines)
        return traced
