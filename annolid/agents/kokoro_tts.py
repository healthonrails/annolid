import os
from functools import lru_cache
from pathlib import Path
from typing import Iterable, List, Optional, Tuple
import warnings

import gdown

from annolid.utils.audio_playback import play_audio_buffer
from annolid.utils.logger import logger

# --- Configuration ---
BASE_V1_URL = (
    "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0"
)
ZH_V1_URL = (
    "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.1"
)
ZH_CONFIG_URL = "https://huggingface.co/hexgrad/Kokoro-82M-v1.1-zh/raw/main/config.json"

BASE_V1_MODEL = "kokoro-v1.0.onnx"
BASE_V1_MODEL_FP16 = "kokoro-v1.0.fp16.onnx"
BASE_V1_MODEL_FP16_GPU = "kokoro-v1.0.fp16-gpu.onnx"
BASE_V1_MODEL_INT8 = "kokoro-v1.0.int8.onnx"
BASE_V1_VOICES = "voices-v1.0.bin"

ZH_MODEL_FILENAME = "kokoro-v1.1-zh.onnx"
ZH_VOICES_FILENAME = "voices-v1.1-zh.bin"
ZH_CONFIG_FILENAME = "config.json"


def _pack(
    model: str,
    voices: str,
    base_url: str,
    config: Optional[str] = None,
    config_url: Optional[str] = None,
) -> dict:
    return {
        "model": model,
        "voices": voices,
        "config": config,
        "model_url": f"{base_url}/{model}",
        "voices_url": f"{base_url}/{voices}",
        "config_url": config_url,
    }


PACKS = {
    # Base packs (v1.0)
    "base": _pack(BASE_V1_MODEL, BASE_V1_VOICES, BASE_V1_URL),
    "base-fp16": _pack(BASE_V1_MODEL_FP16, BASE_V1_VOICES, BASE_V1_URL),
    "base-fp16-gpu": _pack(BASE_V1_MODEL_FP16_GPU, BASE_V1_VOICES, BASE_V1_URL),
    "base-int8": _pack(BASE_V1_MODEL_INT8, BASE_V1_VOICES, BASE_V1_URL),
    # Japanese currently reuses the base pack but can be overridden later.
    "ja": _pack(BASE_V1_MODEL, BASE_V1_VOICES, BASE_V1_URL),
    "zh": _pack(
        ZH_MODEL_FILENAME,
        ZH_VOICES_FILENAME,
        ZH_V1_URL,
        config=ZH_CONFIG_FILENAME,
        config_url=ZH_CONFIG_URL,
    ),
}
PACKS["default"] = PACKS["base"]

DEFAULT_VOICE = "af_sarah"
DEFAULT_LANG = "en-us"
DEFAULT_SPEED = 1.0
MAX_PHONEMES = 500

SUGGESTED_VOICES = [
    "af_sarah",
    "af_bella",
    "af_eva",
    "af_sky",
    "af_heart",
    "am_adam",
    "am_michael",
    "bf_emma",
    "bf_isabella",
    "bm_george",
    "bm_lewis",
    "zf_001",
    "jf_alpha",
]
SUGGESTED_LANGUAGES = [
    "en-us",
    "en-gb",
    "es",
    "fr",
    "de",
    "it",
    "pt-br",
    "ru",
    "ja",
    "ko",
    "zh",
    "hi",
    "en",
    "pt",
]

_BASE_VARIANT = os.getenv("ANNOLID_KOKORO_VARIANT", "f32").strip().lower()
_BASE_PACK_KEY = {
    "f32": "base",
    "fp16": "base-fp16",
    "fp16-gpu": "base-fp16-gpu",
    "int8": "base-int8",
}.get(_BASE_VARIANT, "base")

_LANG_ALIASES = {
    "": "en-us",
    "en": "en-us",
    "a": "en-us",
    "us": "en-us",
    "en-us": "en-us",
    "en-gb": "en-gb",
    "b": "en-gb",
    "uk": "en-gb",
    "fr": "fr-fr",
    "fr-fr": "fr-fr",
    "ja": "ja",
    "j": "ja",
    "ko": "ko",
    "zh": "cmn",
    "z": "cmn",
    "zh-cn": "cmn",
    "zh-hans": "cmn",
    "zh-hant": "cmn",
    "cmn": "cmn",
}

_LANG_PREFIX_MAP = {
    "fr": "fr-fr",
    "ja": "ja",
    "ko": "ko",
    "zh": "cmn",
}

# Lazy import cache for kokoro_onnx to avoid loading on app startup.
_KOKORO_AVAILABLE: Optional[bool] = None
_KOKORO_CLS = None
_MISAKI_CACHE: dict[str, object] = {}
_JIEBA_SILENCED = False

# --- Helper Functions ---


def _kokoro_cache_dir() -> Path:
    override = os.getenv("ANNOLID_KOKORO_CACHE_DIR")
    base = Path(override) if override else Path.home() / ".annolid" / "kokoro"
    return base.expanduser()


def _kokoro_class():
    """Lazy importer for kokoro_onnx.Kokoro to avoid loading on startup."""
    global _KOKORO_AVAILABLE, _KOKORO_CLS
    if _KOKORO_AVAILABLE is False:
        return None
    if _KOKORO_CLS is not None:
        return _KOKORO_CLS
    try:
        from kokoro_onnx import Kokoro as _K

        _KOKORO_CLS = _K
        _KOKORO_AVAILABLE = True
        return _KOKORO_CLS
    except Exception:
        _KOKORO_AVAILABLE = False
        _KOKORO_CLS = None
        return None


def _get_cached_misaki(lang: str):
    """Return a cached Misaki G2P instance for zh/ja, or None on failure."""
    key = lang.lower().strip()
    if key in _MISAKI_CACHE:
        return _MISAKI_CACHE[key]
    _silence_jieba_logs()
    try:
        if key == "zh":
            from misaki import zh as misaki_zh  # type: ignore

            inst = misaki_zh.ZHG2P(version="1.1")
        elif key == "ja":
            from misaki import ja as misaki_ja  # type: ignore

            inst = misaki_ja.JAG2P()
        else:
            inst = None
        _MISAKI_CACHE[key] = inst
        return inst
    except Exception as exc:
        logger.warning("Failed to init %s G2P (%s); caching None.", key, exc)
        _MISAKI_CACHE[key] = None
        return None


def _silence_jieba_logs() -> None:
    """Prevent jieba from spamming stderr/debug output (especially on macOS)."""
    global _JIEBA_SILENCED
    if _JIEBA_SILENCED:
        return
    try:
        os.environ.setdefault("JIEBA_LOG_LEVEL", "ERROR")
        warnings.filterwarnings(
            "ignore",
            message=".*words count mismatch.*",
        )
        import jieba  # type: ignore
        import logging as _logging

        jieba.setLogLevel(_logging.ERROR)
        _JIEBA_SILENCED = True
    except Exception:
        # If jieba is not installed or setting log level fails, ignore silently.
        _JIEBA_SILENCED = True


def download_file(url: str, filepath: Path) -> None:
    """Downloads a file from a URL to a specified filepath using gdown."""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    try:
        # quiet=False shows progress bar, fuzzy=True handles redirects
        gdown.download(url, str(filepath), quiet=False, fuzzy=True)
        logger.info("Downloaded to '%s'", filepath)
    except Exception as e:
        logger.warning("Error downloading from %s: %s", url, e)
        raise  # Re-raise the exception to be caught in text_to_speech


def _ensure_file(filepath: Path, url: Optional[str], label: str) -> None:
    if filepath.exists():
        logger.debug("%s file found in cache: '%s'", label, filepath)
        return
    if not url:
        logger.warning(
            "%s path '%s' requested but no URL provided; skipping.", label, filepath
        )
        return
    logger.info("%s file '%s' not found in cache. Downloading...", label, filepath.name)
    download_file(url, filepath)
    logger.info("%s file downloaded to '%s'", label, filepath)


def _select_pack(lang: Optional[str], voice: Optional[str]) -> str:
    normalized_lang = _normalize_lang(lang or "")
    voice = (voice or "").strip().lower()
    if normalized_lang == "ja" or voice.startswith("j"):
        return "ja"
    if normalized_lang == "cmn" or voice.startswith("z"):
        return "zh"
    return _BASE_PACK_KEY if _BASE_PACK_KEY in PACKS else "base"


def _resolve_pack(lang: Optional[str], voice: Optional[str]) -> Tuple[str, dict]:
    pack_key = _select_pack(lang, voice)
    pack = (
        PACKS.get(pack_key)
        or PACKS.get(_BASE_PACK_KEY)
        or PACKS.get("base")
        or PACKS["default"]
    )
    return pack_key, pack


def ensure_files_exist(
    cache_dir: Optional[Path] = None,
    lang: Optional[str] = None,
    voice: Optional[str] = None,
) -> Tuple[Path, Path, Optional[Path], str]:
    """Ensures model, voices, and optional config exist in the cache, downloads if not."""
    cache_dir = cache_dir or _kokoro_cache_dir()
    pack_key, pack = _resolve_pack(lang, voice)

    model_filepath = cache_dir / pack["model"]
    voices_filepath = cache_dir / pack["voices"]
    config_path: Optional[Path] = cache_dir / pack["config"] if pack["config"] else None

    _ensure_file(model_filepath, pack.get("model_url"), "Model")
    _ensure_file(voices_filepath, pack.get("voices_url"), "Voices")
    if config_path is not None:
        _ensure_file(config_path, pack.get("config_url"), "Config")

    return model_filepath, voices_filepath, config_path, pack_key


@lru_cache(maxsize=4)
def _load_kokoro(model_path: str, voices_path: str, config_path: Optional[str]):
    cls = _kokoro_class()
    if cls is None:
        raise RuntimeError("kokoro_onnx is not available.")
    # Prefer vocab_config when supported (helps zh model); fall back otherwise.
    if config_path:
        try:
            return cls(model_path, voices_path, vocab_config=config_path)
        except TypeError:
            logger.warning(
                "Installed kokoro_onnx does not support 'vocab_config'; falling back to default initialisation."
            )
    return cls(model_path, voices_path)


def _extract_sorted_strings(values: Iterable[object]) -> List[str]:
    return sorted({str(value).strip() for value in values if str(value).strip()})


def get_available_voices(
    cache_dir: Optional[Path] = None,
    lang: Optional[str] = None,
    voice: Optional[str] = None,
) -> List[str]:
    """Return available voices if Kokoro is ready, else a suggested list."""
    if _kokoro_class() is None:
        return list(SUGGESTED_VOICES)
    try:
        cache_dir = cache_dir or _kokoro_cache_dir()
        pack_key, pack = _resolve_pack(lang, voice)
        model_path = cache_dir / pack["model"]
        voices_path = cache_dir / pack["voices"]
        if not model_path.exists() or not voices_path.exists():
            return list(SUGGESTED_VOICES)
        model_path, voices_path, config_path, _ = ensure_files_exist(
            cache_dir, lang=lang, voice=voice
        )
        kokoro = _load_kokoro(
            str(model_path),
            str(voices_path),
            str(config_path) if config_path else None,
        )
        voices = getattr(kokoro, "voices", None)
        if isinstance(voices, dict):
            return _extract_sorted_strings(voices.keys())
        if isinstance(voices, (list, tuple, set)):
            return _extract_sorted_strings(voices)
    except Exception:
        return list(SUGGESTED_VOICES)
    return list(SUGGESTED_VOICES)


def get_suggested_languages() -> List[str]:
    """Return a suggested list of language codes."""
    return sorted({code.strip() for code in SUGGESTED_LANGUAGES if code.strip()})


def _normalize_lang(lang: str) -> str:
    """Map user-friendly language codes to what Kokoro accepts."""
    raw = (lang or "").strip().lower()
    if raw in _LANG_ALIASES:
        return _LANG_ALIASES[raw]
    for prefix, target in _LANG_PREFIX_MAP.items():
        if raw.startswith(prefix):
            return target
    return raw


def _maybe_g2p(text: str, lang: str):
    """Convert text to phonemes when supported (currently zh via misaki)."""
    lang_lower = lang.lower()
    if lang_lower.startswith("zh") or lang_lower in {"z", "cmn"}:
        converter = _get_cached_misaki("zh")
        if converter:
            try:
                phonemes, _ = converter(text)
                return phonemes, True
            except Exception as exc:
                logger.warning("Chinese G2P failed (%s); using raw text instead.", exc)
    if lang_lower.startswith("ja") or lang_lower == "j":
        converter = _get_cached_misaki("ja")
        if converter:
            try:
                phonemes, _ = converter(text)
                return phonemes, True
            except Exception as exc:
                logger.warning("Japanese G2P failed (%s); using raw text instead.", exc)
    return text, False


def _cap_phonemes(phonemes, max_len: int = MAX_PHONEMES):
    """Trim phoneme sequences to a safe length for kokoro_onnx."""
    try:
        if isinstance(phonemes, str):
            parts = phonemes.split()
            if len(parts) > max_len:
                logger.warning(
                    "Phoneme string too long (%s); truncating to %s.",
                    len(parts),
                    max_len,
                )
                return " ".join(parts[:max_len])
            return phonemes
        if hasattr(phonemes, "__len__") and not isinstance(phonemes, (bytes,)):
            if len(phonemes) > max_len:
                logger.warning(
                    "Phoneme sequence too long (%s); truncating to %s.",
                    len(phonemes),
                    max_len,
                )
                return phonemes[:max_len]
    except Exception:
        return phonemes
    return phonemes


def _resolve_voice(
    requested: Optional[str],
    cache_dir: Optional[Path],
    lang: Optional[str],
) -> Tuple[str, List[str]]:
    """Pick a valid Kokoro voice, falling back when unavailable."""
    voices = get_available_voices(cache_dir, lang=lang, voice=requested)
    requested = (requested or "").strip()
    if requested and requested in voices:
        return requested, voices
    if DEFAULT_VOICE in voices:
        fallback = DEFAULT_VOICE
    elif voices:
        fallback = voices[0]
    else:
        fallback = DEFAULT_VOICE
    if requested and requested != fallback:
        logger.warning(
            "Voice '%s' not found. Falling back to '%s'.", requested, fallback
        )
    return fallback, voices


def _create_audio(
    kokoro, text: str, voice: str, speed: float, lang: str, is_phonemes: bool
):
    """Call Kokoro.create with or without is_phonemes depending on support."""
    supports = getattr(kokoro, "_annolid_supports_is_phonemes", None)
    if supports is None:
        supports = False
        try:
            import inspect

            sig = inspect.signature(kokoro.create)
            supports = "is_phonemes" in sig.parameters
        except Exception:
            supports = False
        try:
            setattr(kokoro, "_annolid_supports_is_phonemes", supports)
        except Exception:
            pass
    if supports:
        return kokoro.create(
            text, voice=voice, speed=speed, lang=lang, is_phonemes=is_phonemes
        )
    return kokoro.create(text, voice=voice, speed=speed, lang=lang)


def _extract_kokoro_voice_list(kokoro) -> List[str]:
    voices = getattr(kokoro, "voices", None)
    if isinstance(voices, dict):
        return _extract_sorted_strings(voices.keys())
    if isinstance(voices, (list, tuple, set)):
        return _extract_sorted_strings(voices)
    return list(SUGGESTED_VOICES)


@lru_cache(maxsize=8)
def _cached_kokoro_voice_list(
    model_path: str, voices_path: str, config_path: Optional[str]
) -> Tuple[str, ...]:
    kokoro = _load_kokoro(model_path, voices_path, config_path)
    return tuple(_extract_kokoro_voice_list(kokoro))


def _resolve_voice_from_list(requested: str, voices: List[str]) -> str:
    requested = (requested or "").strip()
    if requested and requested in voices:
        return requested
    if DEFAULT_VOICE in voices:
        fallback = DEFAULT_VOICE
    elif voices:
        fallback = voices[0]
    else:
        fallback = DEFAULT_VOICE
    if requested and requested != fallback:
        logger.warning(
            "Voice '%s' not found. Falling back to '%s'.", requested, fallback
        )
    return fallback


def text_to_speech(
    text,
    voice: str = DEFAULT_VOICE,
    speed: float = DEFAULT_SPEED,
    lang: str = DEFAULT_LANG,
    output_path: Optional[str] = None,
    cache_dir: Optional[Path] = None,
):
    """
    Converts text to speech using Kokoro-ONNX.

    Args:
        text (str): The text to synthesize.
        voice (str): The voice to use (default: "af_sarah").
        speed (float): The speech speed (default: 1.0).
        lang (str): The language (default: "en-us").
        output_path (str, optional): Path to save the audio file. If None, audio is not saved.
        cache_dir (Path, optional): Directory to store model and voices files.

    Returns:
        tuple: (samples, sample_rate) - NumPy array of audio samples and the sample rate.
               Returns None if there is an error.
    """
    try:
        if _kokoro_class() is None:
            logger.warning("Kokoro is not available. Install kokoro-onnx to use TTS.")
            return None

        import re

        # Guard against extreme inputs that can overflow ONNX kernels.
        cleaned = re.sub(r"\s+", " ", (text or "")).strip()
        max_chars = 800
        if len(cleaned) > max_chars:
            logger.warning(
                "Input text too long for Kokoro (%s chars); truncating to %s chars.",
                len(cleaned),
                max_chars,
            )
            cleaned = cleaned[:max_chars]

        normalized_lang = _normalize_lang(lang)
        cache_dir = cache_dir or _kokoro_cache_dir()
        model_path, voices_path, config_path, pack_key = ensure_files_exist(
            cache_dir, lang=normalized_lang, voice=voice
        )

        config_str = str(config_path) if config_path else None
        try:
            kokoro = _load_kokoro(
                str(model_path),
                str(voices_path),
                config_str,
            )
        except TypeError as exc:
            if pack_key != "default":
                logger.warning(
                    "Kokoro pack '%s' not supported by this kokoro_onnx (%s). Retrying without vocab_config.",
                    pack_key,
                    exc,
                )
                kokoro = _load_kokoro(str(model_path), str(voices_path), None)
            else:
                raise

        available_voices = list(
            _cached_kokoro_voice_list(str(model_path), str(voices_path), config_str)
        )
        resolved_voice = _resolve_voice_from_list(voice, available_voices)
        g2p_text, is_phonemes = _maybe_g2p(cleaned, normalized_lang)
        if is_phonemes:
            g2p_text = _cap_phonemes(g2p_text)
        else:
            if isinstance(g2p_text, str) and len(g2p_text) > max_chars:
                logger.warning(
                    f"Phoneme string too long ({len(g2p_text)}); truncating to {max_chars}."
                )
                g2p_text = g2p_text[:max_chars]
        samples, sample_rate = _create_audio(
            kokoro, g2p_text, resolved_voice, speed, normalized_lang, is_phonemes
        )

        if output_path:
            import soundfile as sf

            output_path = str(output_path)
            sf.write(output_path, samples, sample_rate)
            logger.info("Audio saved to '%s'", output_path)
        else:
            logger.debug("Audio generated but not saved.")

        return samples, sample_rate

    except Exception as e:
        logger.warning("An error occurred: %s", e)
        return None


def play_audio(samples, sample_rate):
    """Plays audio data using sounddevice."""
    if samples is not None and sample_rate is not None and samples.size > 0:
        logger.debug("Playing audio...")
        if play_audio_buffer(samples, sample_rate, blocking=True):
            logger.debug("Audio playback finished.")
        else:
            logger.info(
                "Audio playback skipped because no usable audio device was detected."
            )
    else:
        logger.info("No audio data to play or audio data is empty.")


# --- Main execution (Example usage) ---
if __name__ == "__main__":
    input_text = "We can rewrite the integrand using polynomial long division, because the degree of the numerator is greater than or equal to the degree of the denominator."

    # Set output_path to None to not save to file
    audio_data = text_to_speech(
        input_text, voice="af_sarah", speed=1.1, output_path=None
    )

    if audio_data:
        samples, sample_rate = audio_data
        print("\nText-to-speech conversion successful!")
        play_audio(samples, sample_rate)  # Play the audio using sounddevice
    else:
        print("\nText-to-speech conversion failed.")
