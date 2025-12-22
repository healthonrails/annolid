from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict

_SETTINGS_DIR = Path.home() / ".annolid"
_SETTINGS_FILE = _SETTINGS_DIR / "tts_settings.json"

_DEFAULT_SETTINGS: Dict[str, Any] = {
    "voice": "af_sarah",
    "speed": 1.0,
    "lang": "en-us",
}


def _normalise_settings(settings: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in _DEFAULT_SETTINGS.items():
        settings.setdefault(key, value)
    return settings


def load_tts_settings() -> Dict[str, Any]:
    """Load TTS settings from disk, falling back to defaults."""
    if not _SETTINGS_FILE.exists():
        return deepcopy(_DEFAULT_SETTINGS)

    try:
        with _SETTINGS_FILE.open("r", encoding="utf-8") as fh:
            persisted = json.load(fh)
    except (json.JSONDecodeError, OSError):
        return deepcopy(_DEFAULT_SETTINGS)

    merged = deepcopy(_DEFAULT_SETTINGS)
    if isinstance(persisted, dict):
        merged.update(persisted)
    return _normalise_settings(merged)


def save_tts_settings(settings: Dict[str, Any]) -> None:
    """Persist TTS settings to disk."""
    _SETTINGS_DIR.mkdir(parents=True, exist_ok=True)
    with _SETTINGS_FILE.open("w", encoding="utf-8") as fh:
        json.dump(settings, fh, indent=2)


def default_tts_settings() -> Dict[str, Any]:
    """Return a copy of the default TTS settings."""
    return deepcopy(_DEFAULT_SETTINGS)
