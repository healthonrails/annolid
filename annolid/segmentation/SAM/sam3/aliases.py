from __future__ import annotations

import sys
import types
from pathlib import Path


def ensure_sam3_aliases() -> None:
    """
    Register a minimal alias under the `sam3` namespace so absolute imports
    inside the vendored SAM3 code (e.g. `sam3.model.decoder`) resolve without
    installing SAM3 as a separate package or mutating sys.path globally.
    """
    sam3_root = Path(__file__).resolve().parent / "sam3"

    # Ensure the root package exists and points at the vendored tree.
    pkg = sys.modules.get("sam3")
    if pkg is None:
        pkg = types.ModuleType("sam3")
        pkg.__path__ = [str(sam3_root)]
        pkg.__file__ = str(sam3_root / "__init__.py")
        sys.modules["sam3"] = pkg
    else:
        # Make sure our path is available if a namespace package already exists.
        if not hasattr(pkg, "__path__"):
            pkg.__path__ = []
        if str(sam3_root) not in pkg.__path__:
            pkg.__path__.append(str(sam3_root))

    # Drop placeholder submodules that may have been registered without actual code
    # so Python can import the real packages from the vendored tree.
    for sub in ("model", "eval", "train", "perflib", "sam", "logger", "utils"):
        name = f"sam3.{sub}"
        existing = sys.modules.get(name)
        if existing is not None and getattr(existing, "__file__", None) is None:
            sys.modules.pop(name, None)
