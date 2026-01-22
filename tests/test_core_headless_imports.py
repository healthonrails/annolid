import importlib.abc
import sys


class _BlocklistFinder(importlib.abc.MetaPathFinder):
    def __init__(self, prefixes: tuple[str, ...]) -> None:
        self._prefixes = prefixes

    def find_spec(self, fullname: str, path, target=None):  # noqa: ANN001
        for prefix in self._prefixes:
            if fullname == prefix or fullname.startswith(prefix + "."):
                raise ImportError(f"Blocked import: {fullname}")
        return None


def test_core_imports_do_not_require_qt():
    blocker = _BlocklistFinder(("qtpy", "PyQt5", "PySide2", "PySide6"))
    sys.meta_path.insert(0, blocker)
    try:
        import annolid.core.media.video  # noqa: F401
        import annolid.core.media.audio  # noqa: F401
        import annolid.core.behavior.spec  # noqa: F401
        import annolid.core.types  # noqa: F401
        import annolid.core.output.validate  # noqa: F401
    finally:
        sys.meta_path.remove(blocker)
