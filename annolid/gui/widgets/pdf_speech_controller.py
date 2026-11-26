"""Background speech synthesis task and TTS snapshot helper for PdfViewerWidget."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
from typing import TYPE_CHECKING, Dict, Optional

from qtpy import QtCore

from annolid.gui.widgets.pdf_viewer_bridge import _SpeakToken
from annolid.services.speech_highlighting import (
    analyze_audio_speech_segments,
    chunk_speech_text,
    normalize_speech_text,
)
from annolid.utils.audio_playback import play_audio_buffer
from annolid.utils.logger import logger
from annolid.utils.tts_settings import default_tts_settings, load_tts_settings

if TYPE_CHECKING:  # pragma: no cover
    from annolid.gui.widgets.pdf_viewer_impl import PdfViewerWidget


def build_tts_settings_snapshot() -> Dict[str, object]:
    """Load and merge persisted TTS settings with defaults."""
    settings = load_tts_settings()
    defaults = default_tts_settings()
    return {
        "engine": settings.get("engine", defaults.get("engine", "auto")),
        "voice": settings.get("voice", defaults["voice"]),
        "pocket_voice": settings.get(
            "pocket_voice", defaults.get("pocket_voice", "alba")
        ),
        "pocket_prompt_path": settings.get(
            "pocket_prompt_path", defaults.get("pocket_prompt_path", "")
        ),
        "pocket_speed": settings.get("pocket_speed", defaults.get("pocket_speed", 1.0)),
        "lang": settings.get("lang", defaults["lang"]),
        "speed": settings.get("speed", defaults["speed"]),
        "chatterbox_voice_path": settings.get(
            "chatterbox_voice_path", defaults.get("chatterbox_voice_path", "")
        ),
        "chatterbox_dtype": settings.get(
            "chatterbox_dtype", defaults.get("chatterbox_dtype", "fp32")
        ),
        "chatterbox_max_new_tokens": settings.get(
            "chatterbox_max_new_tokens",
            defaults.get("chatterbox_max_new_tokens", 1024),
        ),
        "chatterbox_repetition_penalty": settings.get(
            "chatterbox_repetition_penalty",
            defaults.get("chatterbox_repetition_penalty", 1.2),
        ),
        "chatterbox_apply_watermark": settings.get(
            "chatterbox_apply_watermark",
            defaults.get("chatterbox_apply_watermark", False),
        ),
    }


class _SpeakTextTask(QtCore.QRunnable):
    """Background task to convert text chunks to speech and synchronize highlights."""

    def __init__(
        self,
        widget: "PdfViewerWidget",
        text: str,
        tts_settings: Dict[str, object],
        chunks: Optional[list[str]] = None,
        token: Optional[_SpeakToken] = None,
    ) -> None:
        super().__init__()
        self.widget = widget
        self.text = text
        self.tts_settings = tts_settings
        self.chunks = chunks
        self.token = token

    def run(self) -> None:  # pragma: no cover - involves audio
        session_id = int(getattr(self.token, "session_id", 0) or 0)
        try:
            text = normalize_speech_text(self.text)
            if not text:
                return
            engine = (
                str(self.tts_settings.get("engine", "kokoro") or "kokoro")
                .strip()
                .lower()
            )
            max_chars = 800 if engine == "chatterbox" else 420
            if self.chunks:
                raw_chunks = [normalize_speech_text(c) for c in self.chunks]
                indexed_chunks = [
                    (orig_idx, chunk)
                    for orig_idx, chunk in enumerate(raw_chunks)
                    if chunk
                ]
            else:
                split = self._chunk_text(text, max_chars=max_chars)
                indexed_chunks = list(enumerate(split))
            if not indexed_chunks:
                return

            def cancelled() -> bool:
                return bool(self.token is not None and self.token.cancelled)

            def notify_chunk(
                orig_index: int,
                duration_ms: int,
                speech_segments: list[tuple[int, int]],
            ) -> None:
                if cancelled():
                    return
                segments_json = (
                    json.dumps([[int(s), int(e)] for s, e in speech_segments])
                    if speech_segments
                    else "[]"
                )
                QtCore.QMetaObject.invokeMethod(
                    self.widget,
                    "_on_speak_chunk_aligned",
                    QtCore.Qt.QueuedConnection,
                    QtCore.Q_ARG(int, session_id),
                    QtCore.Q_ARG(int, int(orig_index)),
                    QtCore.Q_ARG(int, int(duration_ms)),
                    QtCore.Q_ARG(str, segments_json),
                )

            def notify_error(message: str) -> None:
                if cancelled():
                    return
                QtCore.QMetaObject.invokeMethod(
                    self.widget,
                    "_on_speak_error",
                    QtCore.Qt.QueuedConnection,
                    QtCore.Q_ARG(int, session_id),
                    QtCore.Q_ARG(str, str(message)),
                )

            try:
                from annolid.agents.tts_router import synthesize_tts

                def synthesize(
                    chunk: str,
                ) -> tuple[object, int, int, list[tuple[int, int]]]:
                    audio_data = synthesize_tts(chunk, self.tts_settings)
                    if not audio_data:
                        raise RuntimeError("No audio returned by TTS engine")
                    samples, sample_rate = audio_data
                    duration_ms = 0
                    try:
                        duration_ms = int(
                            round((len(samples) / float(sample_rate)) * 1000)
                        )
                    except Exception:
                        duration_ms = 0
                    speech_segments: list[tuple[int, int]] = []
                    if duration_ms > 0:
                        try:
                            speech_segments = analyze_audio_speech_segments(
                                samples, int(sample_rate)
                            )
                        except Exception:
                            speech_segments = []
                    return (
                        samples,
                        int(sample_rate),
                        int(duration_ms),
                        speech_segments,
                    )

                with ThreadPoolExecutor(max_workers=1) as executor:
                    current_future = executor.submit(synthesize, indexed_chunks[0][1])
                    for pos, (orig_idx, _chunk_text) in enumerate(indexed_chunks):
                        if cancelled():
                            return
                        samples, sample_rate, duration_ms, speech_segments = (
                            current_future.result()
                        )
                        next_future = None
                        if pos + 1 < len(indexed_chunks):
                            next_future = executor.submit(
                                synthesize, indexed_chunks[pos + 1][1]
                            )
                        notify_chunk(orig_idx, duration_ms, speech_segments)
                        if cancelled():
                            return
                        play_audio_buffer(samples, sample_rate, blocking=True)
                        if next_future is None:
                            break
                        current_future = next_future
            except Exception as exc:
                if not cancelled():
                    logger.warning("PDF speech playback failed: %s", exc)
                    notify_error(str(exc))
                return
        finally:
            QtCore.QMetaObject.invokeMethod(
                self.widget,
                "_on_speak_finished_for_session",
                QtCore.Qt.QueuedConnection,
                QtCore.Q_ARG(int, session_id),
            )

    @staticmethod
    def _chunk_text(text: str, max_chars: int = 420) -> list[str]:
        return chunk_speech_text(text, max_chars=max_chars)


__all__ = ["_SpeakTextTask", "build_tts_settings_snapshot"]
