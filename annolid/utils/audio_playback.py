"""Utilities for safely playing audio within Annolid.

Most of the project only needs best-effort playback for previews or text to
speech.  On Linux deployments without ALSA output devices (e.g. servers,
containers, or WSL) attempting to open the default PortAudio backend crashes
with ``PaAlsaStreamComponent_BeginPolling``.  This module centralises the
allow/deny logic so the rest of the codebase can simply call
``play_audio_buffer`` and it will either play or log why it was skipped.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional
import os
import sys

from annolid.utils.logger import logger

try:
    import sounddevice as sd
except Exception as exc:  # pragma: no cover - handled gracefully
    sd = None
    _IMPORT_ERROR = str(exc)
else:
    _IMPORT_ERROR = ""

_AUDIO_AVAILABLE: Optional[bool] = None
_DISABLE_VALUES = {"1", "true", "yes", "on"}


def _audio_disabled_by_env() -> bool:
    env_value = os.getenv("ANNOLID_DISABLE_AUDIO", "").strip().lower()
    if env_value in _DISABLE_VALUES:
        logger.info(
            "Audio playback disabled because ANNOLID_DISABLE_AUDIO=%s", env_value
        )
        return True
    return False


def _linux_has_audio_device() -> bool:
    if not sys.platform.startswith("linux"):
        return True
    snd_path = Path("/dev/snd")
    if not snd_path.exists():
        logger.info(
            "Audio playback disabled: /dev/snd does not exist on this Linux system."
        )
        return False
    # Some cloud hosts provide /dev/snd but no usable output nodes.
    if not any(snd_path.iterdir()):
        logger.info(
            "Audio playback disabled: no ALSA character devices were found under /dev/snd."
        )
        return False
    return True


def audio_playback_available() -> bool:
    """Returns True when sounddevice can play audio without crashing."""
    global _AUDIO_AVAILABLE
    if _AUDIO_AVAILABLE is not None:
        return _AUDIO_AVAILABLE

    if _audio_disabled_by_env():
        _AUDIO_AVAILABLE = False
        return False

    if not _linux_has_audio_device():
        _AUDIO_AVAILABLE = False
        return False

    if sd is None:
        if _IMPORT_ERROR:
            logger.warning(
                "Audio playback disabled: sounddevice could not be imported (%s).",
                _IMPORT_ERROR,
            )
        _AUDIO_AVAILABLE = False
        return False

    try:
        devices = sd.query_devices()
    except Exception as exc:  # pragma: no cover
        logger.warning(
            "Audio playback disabled: sounddevice could not enumerate devices (%s).",
            exc,
        )
        _AUDIO_AVAILABLE = False
        return False

    _AUDIO_AVAILABLE = any(
        device.get("max_output_channels", 0) > 0 for device in devices
    )
    if not _AUDIO_AVAILABLE:
        logger.info(
            "Audio playback disabled: PortAudio found no usable output devices."
        )
    return _AUDIO_AVAILABLE


def play_audio_buffer(samples, sample_rate: int, *, blocking: bool = False) -> bool:
    """
    Attempts to play ``samples`` using sounddevice.

    Returns True when playback started and False when audio was skipped because
    no device is available or an error occurred.
    """
    if samples is None or getattr(samples, "size", 0) == 0:
        return False
    if sample_rate is None or sample_rate <= 0:
        logger.debug(
            "Audio playback skipped: invalid sample rate %s", sample_rate)
        return False
    if not audio_playback_available():
        logger.info(
            "Skipping audio playback because no usable audio device was detected."
        )
        return False
    try:
        sd.play(samples, sample_rate, blocking=blocking)
        return True
    except Exception as exc:  # pragma: no cover
        logger.warning("Audio playback failed and was skipped: %s", exc)
        return False


def stop_audio_playback() -> None:
    """Stops any active sounddevice playback, ignoring failures."""
    if sd is None:
        return
    try:
        sd.stop()
    except Exception:  # pragma: no cover
        pass
