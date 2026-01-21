from __future__ import annotations

import os
import re
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from annolid.utils.logger import logger

DEFAULT_VOICE = "alba"
POCKET_TTS_INSTALL_HINT = "pip install pocket-tts"

# -----------------------------
# Globals (model + caches)
# -----------------------------
_POCKET_MODEL: Optional[Any] = None
_MODEL_LOCK = threading.Lock()

# Voice-state cache (prompt -> state)
_VOICE_STATE_CACHE: Dict[str, Dict[str, Any]] = {}
_VOICE_STATE_LOCK = threading.Lock()

# Serialize generation for thread-safety (Pocket-TTS state & torch internals)
_GENERATION_LOCK = threading.Lock()


# -----------------------------
# Text cleanup for TTS
# -----------------------------
# Include more hyphen-like chars seen in PDF extraction:
# - ASCII hyphen-minus: -
# - hyphen: ‐ (U+2010)
# - non-breaking hyphen: - (U+2011)
# - figure dash: ‒ (U+2012)
# - en dash: – (U+2013)
# - em dash: — (U+2014)
# - minus sign: − (U+2212)  (occasionally appears)
_HYPHEN_CHARS = r"\-‐-‒–—−"

# 1) Join hyphenation across line breaks: "gen-\n   eral" -> "general"
_HYPHEN_LINEBREAK_RE = re.compile(
    rf"(\w)[{_HYPHEN_CHARS}]\s*(?:\r?\n)+\s*(\w)",
    re.UNICODE,
)

# Soft hyphen (discretionary hyphen)
_SOFT_HYPHEN_RE = re.compile("\u00ad")

# 2) Join hyphenation that survived flattening: "gen-   eral" -> "general"
# Conservative: require the right side to start lowercase (common in hyphenated wraps).
_HYPHEN_SPACES_RE = re.compile(
    rf"(\b[A-Za-z]{{2,}})[{_HYPHEN_CHARS}]\s+([a-z][A-Za-z]{{1,}}\b)"
)

_WHITESPACE_RE = re.compile(r"\s+")


def _clean_text_for_tts(text: str) -> str:
    """
    Normalize text for TTS:
    - Remove soft hyphens (U+00AD)
    - Join PDF line-wrapped hyphenations:
        "gen-\\n  eral" -> "general"
        "sup-\\r\\nplying" -> "supplying"
    - Flatten remaining newlines -> spaces
    - Join residual hyphen+spaces splits:
        "au-  dio" -> "audio"
    - Normalize whitespace
    """
    text = (text or "").strip()
    if not text:
        return ""

    # Remove discretionary hyphen artifacts.
    text = _SOFT_HYPHEN_RE.sub("", text)

    # Join hyphenation across any newline sequence.
    # Run repeatedly in case of multiple consecutive hyphen wraps.
    prev = None
    while prev != text:
        prev = text
        text = _HYPHEN_LINEBREAK_RE.sub(r"\1\2", text)

    # Flatten remaining newlines to spaces.
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace("\n", " ")

    # Join splits that became "gen-   eral" after flattening.
    # Repeat to handle multiple occurrences.
    prev = None
    while prev != text:
        prev = text
        text = _HYPHEN_SPACES_RE.sub(r"\1\2", text)

    # Final whitespace normalization.
    text = _WHITESPACE_RE.sub(" ", text).strip()
    return text


# -----------------------------
# Voice normalization + discovery
# -----------------------------


def _normalize_voice_spec(voice: str) -> str:
    """Normalize the voice identifier to a consistent cache key."""
    voice = (voice or "").strip()
    if not voice:
        return DEFAULT_VOICE

    prefixes = ("http://", "https://", "hf://", "ftp://")
    if voice.startswith(prefixes):
        return voice

    # Treat as local path if it looks like a path.
    if (
        voice.startswith(("~", "."))
        or os.path.sep in voice
        or (os.path.altsep and os.path.altsep in voice)
    ):
        return str(Path(voice).expanduser())

    return voice


def get_available_voices() -> List[str]:
    """Return the built-in voice catalog provided by pocket-tts (if installed)."""
    try:
        from pocket_tts.utils.utils import PREDEFINED_VOICES
    except Exception:
        return []
    return list(PREDEFINED_VOICES)


# -----------------------------
# Model + state caching
# -----------------------------
def _load_model() -> Optional[Any]:
    """Lazily import and initialise the Pocket TTS model."""
    global _POCKET_MODEL
    if _POCKET_MODEL is not None:
        return _POCKET_MODEL

    with _MODEL_LOCK:
        if _POCKET_MODEL is not None:
            return _POCKET_MODEL

        try:
            from pocket_tts import TTSModel
        except Exception as exc:
            logger.warning(
                "Pocket TTS is optional. Install it via '%s' (%s)",
                POCKET_TTS_INSTALL_HINT,
                exc,
            )
            return None

        try:
            model = TTSModel.load_model()
        except Exception as exc:
            logger.warning("Pocket TTS model load failed: %s", exc)
            return None

        model.eval()
        model.to("cpu")
        _POCKET_MODEL = model
        return _POCKET_MODEL


def _get_voice_state(model: Any, voice: str) -> Optional[Dict[str, Any]]:
    """Return or build a cached voice state for the requested prompt."""
    key = _normalize_voice_spec(voice)

    with _VOICE_STATE_LOCK:
        cached = _VOICE_STATE_CACHE.get(key)
        if cached is not None:
            return cached

    # Build outside the lock to avoid blocking other reads.
    try:
        state = model.get_state_for_audio_prompt(key, truncate=True)
    except Exception as exc:
        logger.warning("Pocket voice state failed for %s: %s", key, exc)
        return None

    with _VOICE_STATE_LOCK:
        # Double-check in case another thread populated while we computed.
        _VOICE_STATE_CACHE.setdefault(key, state)
        return _VOICE_STATE_CACHE[key]


def prewarm(voice: Optional[str] = None) -> None:
    """Ensure the model (and optional voice) is loaded."""
    model = _load_model()
    if model is None:
        return
    if voice:
        _get_voice_state(model, voice)


# -----------------------------
# Public API
# -----------------------------
def text_to_speech(
    text: str,
    voice: Optional[str] = None,
    *,
    frames_after_eos: Optional[int] = None,
) -> Optional[Tuple[np.ndarray, int]]:
    """
    Generate PCM float32 samples for `text` using Pocket TTS + selected voice.

    Fixes PDF-style hyphenation splits like:
      "experi-\\nments" -> "experiments"
    """
    cleaned = _clean_text_for_tts(text)
    if not cleaned:
        return None

    model = _load_model()
    if model is None:
        return None

    voice_spec = voice or DEFAULT_VOICE
    voice_state = _get_voice_state(model, voice_spec)
    if voice_state is None:
        return None

    try:
        with _GENERATION_LOCK:
            audio_tensor = model.generate_audio(
                voice_state,
                cleaned,
                frames_after_eos=frames_after_eos,
                copy_state=True,
            )
    except Exception as exc:
        logger.warning("Pocket TTS generation failed for voice %s: %s", voice_spec, exc)
        return None

    if audio_tensor is None or getattr(audio_tensor, "numel", lambda: 0)() == 0:
        return None

    samples = audio_tensor.detach()
    if getattr(samples, "ndim", 0) > 1 and samples.shape[0] == 1:
        samples = samples[0]

    samples_np = np.asarray(samples.cpu(), dtype=np.float32).reshape(-1)
    return samples_np, int(model.sample_rate)
