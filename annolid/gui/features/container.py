"""Shared dependency container for GUI feature setup."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class GuiFeatureDeps:
    """Injected dependencies consumed by GUI feature setup modules."""

    window: Any
    status_message: Callable[[str, int], None]
