from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np

from annolid.utils.logger import logger


def _adjust_pocket_speed(samples: np.ndarray, speed: float) -> np.ndarray:
    if speed <= 0 or abs(speed - 1.0) < 1e-3:
        return samples
    try:
        from scipy.signal import resample
    except Exception:
        return samples
    target_len = max(1, int(round(len(samples) / speed)))
    try:
        resampled = resample(samples, target_len)
    except Exception as exc:
        logger.warning("Pocket TTS speed adjustment failed: %s", exc)
        return samples
    return np.asarray(resampled, dtype=np.float32)


def _synthesize_kokoro(text: str, settings: Dict[str, object]) -> Optional[Tuple[np.ndarray, int]]:
    from annolid.agents.kokoro_tts import text_to_speech

    return text_to_speech(
        text,
        voice=str(settings.get("voice", "af_sarah")),
        speed=float(settings.get("speed", 1.0)),
        lang=str(settings.get("lang", "en-us")),
    )


def _synthesize_pocket(
    text: str, settings: Dict[str, object]
) -> Optional[Tuple[np.ndarray, int]]:
    try:
        from annolid.agents.pocket_tts import (
            DEFAULT_VOICE as POCKET_DEFAULT_VOICE,
            POCKET_TTS_INSTALL_HINT,
            text_to_speech,
        )
    except Exception as exc:
        logger.warning(
            "Pocket TTS playback dependencies missing; install via '%s' (%s)",
            POCKET_TTS_INSTALL_HINT,
            exc,
        )
        return None

    voice = str(settings.get("pocket_voice", POCKET_DEFAULT_VOICE)).strip() or POCKET_DEFAULT_VOICE
    pocket_prompt = str(settings.get("pocket_prompt_path", "")).strip()
    voice_spec = pocket_prompt if pocket_prompt else voice
    audio = text_to_speech(text, voice_spec)
    if not audio:
        return None
    samples, sample_rate = audio
    try:
        speed = float(settings.get("pocket_speed", 1.0))
    except Exception:
        speed = 1.0
    samples = _adjust_pocket_speed(samples, speed)
    if samples.size == 0:
        return None
    return samples, sample_rate


def _synthesize_chatterbox(
    text: str, settings: Dict[str, object]
) -> Optional[Tuple[np.ndarray, int]]:
    from annolid.agents.chatterbox_tts import text_to_speech

    return text_to_speech(
        text,
        voice_wav_path=str(settings.get("chatterbox_voice_path", "")),
        dtype=str(settings.get("chatterbox_dtype", "fp32")),
        max_new_tokens=int(settings.get("chatterbox_max_new_tokens", 1024)),
        repetition_penalty=float(settings.get(
            "chatterbox_repetition_penalty", 1.2)),
        apply_watermark=bool(settings.get(
            "chatterbox_apply_watermark", False)),
    )


def _synthesize_gtts(text: str, settings: Dict[str, object]) -> Optional[Tuple[np.ndarray, int]]:
    try:
        from gtts import gTTS
        from pydub import AudioSegment
        from io import BytesIO
    except Exception as exc:
        logger.warning(f"gTTS playback dependencies missing: {exc}")
        return None

    lang = str(settings.get("lang", "en-us")).lower()
    gtts_lang = lang.split("-")[0] if lang else "en"
    try:
        tts = gTTS(text=text, lang=gtts_lang)
        buf = BytesIO()
        tts.write_to_fp(buf)
        buf.seek(0)
        audio = AudioSegment.from_file(buf, format="mp3")
        samples = np.array(audio.get_array_of_samples())
        if samples.size == 0:
            return None
        if audio.channels == 2:
            samples = samples.reshape((-1, 2))
        return samples, int(audio.frame_rate)
    except Exception as exc:
        logger.warning(f"gTTS synthesis failed: {exc}")
        return None


def synthesize_tts(
    text: str, tts_settings: Optional[Dict[str, object]] = None
) -> Optional[Tuple[np.ndarray, int]]:
    """
    Synthesise audio for ``text`` using the configured TTS engine.

    Supported engines: auto, kokoro, pocket, chatterbox, gtts.
    """
    settings: Dict[str, object] = dict(tts_settings or {})
    engine = str(settings.get("engine", "auto") or "auto").strip().lower()
    text = (text or "").strip()
    if not text:
        return None

    if engine in {"", "auto"}:
        engines = ("kokoro", "pocket", "chatterbox", "gtts")
    else:
        engines = (engine,)

    for candidate in engines:
        try:
            if candidate == "kokoro":
                audio = _synthesize_kokoro(text, settings)
            elif candidate == "pocket":
                audio = _synthesize_pocket(text, settings)
            elif candidate == "chatterbox":
                audio = _synthesize_chatterbox(text, settings)
            elif candidate == "gtts":
                audio = _synthesize_gtts(text, settings)
            else:
                logger.warning(f"Unknown TTS engine '{candidate}'.")
                return None
        except Exception as exc:
            logger.warning(f"TTS engine '{candidate}' failed: {exc}")
            audio = None
        if audio:
            return audio

    return None
