from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from annolid.utils.logger import logger

DEFAULT_VOICE = "alba"
POCKET_TTS_INSTALL_HINT = "pip install pocket-tts"

_POCKET_MODEL: Optional[Any] = None
_MODEL_LOCK = threading.Lock()
_VOICE_STATE_LOCK = threading.Lock()
_GENERATION_LOCK = threading.Lock()
_VOICE_STATE_CACHE: Dict[str, Dict[str, Any]] = {}


def _normalize_voice_spec(voice: str) -> str:
    """Normalize the voice identifier to a consistent cache key."""
    voice = (voice or "").strip()
    if not voice:
        return DEFAULT_VOICE
    prefixes = ("http://", "https://", "hf://", "ftp://")
    if voice.startswith(prefixes):
        return voice
    if voice.startswith(("~", ".")) or os.path.sep in voice or (
        os.path.altsep and os.path.altsep in voice
    ):
        return str(Path(voice).expanduser())
    return voice


def get_available_voices() -> List[str]:
    """Return the built-in voice catalog provided by pocket-tts."""
    try:
        from pocket_tts.utils.utils import PREDEFINED_VOICES
    except Exception:
        return []
    return list(PREDEFINED_VOICES)


def _load_model() -> Optional[Any]:
    """Lazily import and initialise the Pocket TTS model."""
    global _POCKET_MODEL
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
        try:
            state = model.get_state_for_audio_prompt(key, truncate=True)
        except Exception as exc:
            logger.warning("Pocket voice state failed for %s: %s", key, exc)
            return None
        _VOICE_STATE_CACHE[key] = state
    return state


def prewarm(voice: Optional[str] = None) -> None:
    """Ensure the model (and optional voice) is loaded in the background."""
    model = _load_model()
    if model is None or voice is None:
        return
    _get_voice_state(model, voice)


def text_to_speech(
    text: str, voice: Optional[str] = None, *, frames_after_eos: Optional[int] = None
) -> Optional[Tuple[np.ndarray, int]]:
    """Generate PCM samples for ``text`` using Pocket TTS and the selected voice."""
    text = (text or "").strip()
    if not text:
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
                text,
                frames_after_eos=frames_after_eos,
                copy_state=True,
            )
    except Exception as exc:
        logger.warning(
            "Pocket TTS generation failed for voice %s: %s", voice_spec, exc)
        return None
    if audio_tensor is None or audio_tensor.numel() == 0:
        return None
    samples = audio_tensor.detach()
    if samples.ndim > 1 and samples.shape[0] == 1:
        samples = samples[0]
    samples = np.asarray(samples.cpu(), dtype=np.float32).reshape(-1)
    return samples, int(model.sample_rate)
