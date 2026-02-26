from __future__ import annotations

from importlib import metadata

__version__ = "1.6.1"


def get_version() -> str:
    """Return installed package version when available, else fallback."""
    try:
        return metadata.version("annolid")
    except Exception:
        return __version__
