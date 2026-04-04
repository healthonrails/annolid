from __future__ import annotations

import os
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
        existing_paths = [str(p) for p in list(getattr(pkg, "__path__", []) or [])]
        mixing_allowed = str(
            os.getenv("ANNOLID_SAM3_ALLOW_NAMESPACE_PATH_MIX", "0")
        ).strip().lower() in {"1", "true", "yes", "on"}
        if existing_paths and str(sam3_root) not in existing_paths and not mixing_allowed:
            # Avoid silently mixing vendored and externally installed SAM3 packages.
            # In this case we keep the existing import root unchanged.
            return
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
