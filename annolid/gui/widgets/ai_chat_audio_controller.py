from __future__ import annotations

import os
import tempfile
import threading
import time
from typing import Callable, Dict

import numpy as np
from qtpy import QtCore
from qtpy.QtWidgets import QLabel, QPlainTextEdit, QPushButton

from annolid.utils.tts_settings import default_tts_settings, load_tts_settings


class _AudioUiBridge(QtCore.QObject):
    statusText = QtCore.Signal(str)
    talkButtonText = QtCore.Signal(str)
    promptText = QtCore.Signal(str)


class ChatAudioController:
    """Owns voice input (ASR) and speech output (TTS) for AI chat UI."""

    def __init__(
        self,
        *,
        status_label: QLabel,
        talk_button: QPushButton,
        prompt_text_edit: QPlainTextEdit,
        get_last_assistant_text: Callable[[], str],
    ) -> None:
        self._status_label = status_label
        self._talk_button = talk_button
        self._prompt_text_edit = prompt_text_edit
        self._get_last_assistant_text = get_last_assistant_text
        self._is_recording = False
        self._asr_pipeline = None
        self._asr_lock = threading.Lock()
        self._ui_bridge = _AudioUiBridge()
        self._ui_bridge.statusText.connect(self._status_label.setText)
        self._ui_bridge.talkButtonText.connect(self._talk_button.setText)
        self._ui_bridge.promptText.connect(self._prompt_text_edit.setPlainText)

    @property
    def is_recording(self) -> bool:
        return self._is_recording

    def read_last_reply_async(self) -> None:
        text = self._get_last_assistant_text()
        if not text:
            self._set_status_text("No assistant reply to read.")
            return
        self.speak_text_async(text)

    def speak_text_async(self, text: str) -> None:
        value = str(text or "").strip()
        if not value:
            self._set_status_text("No text to read.")
            return

        def _run_tts() -> None:
            try:
                from annolid.utils.audio_playback import play_audio_buffer
                from annolid.agents.tts_router import synthesize_tts

                audio_data = synthesize_tts(value, self._tts_settings_snapshot())
                if not audio_data:
                    raise RuntimeError("No audio generated.")
                samples, sample_rate = audio_data
                played = play_audio_buffer(samples, sample_rate, blocking=True)
                if not played:
                    raise RuntimeError("No usable audio device found.")
                self._set_status_text("Reply read aloud.")
            except Exception as exc:
                self._set_status_text(f"Read failed: {exc}")

        threading.Thread(target=_run_tts, daemon=True).start()

    def toggle_recording(self) -> None:
        if not self._is_recording:
            self._is_recording = True
            self._set_talk_button_text("Stop")
            self._set_status_text("Listening…")
            threading.Thread(target=self._record_voice, daemon=True).start()
        else:
            self._is_recording = False
            self._set_talk_button_text("Talk")
            self._set_status_text("Processing speech…")

    def _record_voice(self) -> None:
        try:
            import sounddevice as sd
            import soundfile as sf
        except ImportError:
            self._set_status_text(
                "Audio recording deps missing. Install: pip install sounddevice soundfile"
            )
            self._set_talk_button_text("Talk")
            self._is_recording = False
            return

        sample_rate = 16000
        channels = 1
        audio_chunks: list[np.ndarray] = []

        def _audio_callback(indata, frames, stream_time, status) -> None:
            del frames, stream_time
            if status:
                return
            audio_chunks.append(indata.copy())

        try:
            with sd.InputStream(
                samplerate=sample_rate,
                channels=channels,
                dtype="float32",
                callback=_audio_callback,
            ):
                while self._is_recording:
                    time.sleep(0.1)
        except Exception as exc:
            self._set_status_text(f"Mic capture failed: {exc}")
            self._set_talk_button_text("Talk")
            self._is_recording = False
            return

        final_text = ""
        if audio_chunks:
            audio_data = np.concatenate(audio_chunks, axis=0).reshape(-1)
            if np.abs(audio_data).max(initial=0.0) >= 1e-4:
                fd, audio_path = tempfile.mkstemp(prefix="annolid_talk_", suffix=".wav")
                os.close(fd)
                try:
                    sf.write(audio_path, audio_data, sample_rate)
                    final_text = self._transcribe_with_whisper_tiny(audio_path)
                except Exception as exc:
                    self._set_status_text(f"Transcription failed: {exc}")
                    self._set_talk_button_text("Talk")
                    self._is_recording = False
                    try:
                        if os.path.exists(audio_path):
                            os.remove(audio_path)
                    except OSError:
                        pass
                    return
                finally:
                    try:
                        if os.path.exists(audio_path):
                            os.remove(audio_path)
                    except OSError:
                        pass

        if final_text:
            self._set_prompt_text(final_text)
            self._set_status_text("Speech captured. Review and send.")
        else:
            self._set_status_text("No speech captured.")
        self._set_talk_button_text("Talk")
        self._is_recording = False

    def _get_asr_pipeline(self):
        with self._asr_lock:
            if self._asr_pipeline is not None:
                return self._asr_pipeline
            try:
                import torch
                from transformers import pipeline
            except ImportError as exc:
                raise RuntimeError(
                    "ASR deps missing. Install: pip install transformers torch"
                ) from exc
            device = 0 if torch.cuda.is_available() else -1
            self._asr_pipeline = pipeline(
                "automatic-speech-recognition",
                model="openai/whisper-tiny",
                device=device,
            )
            return self._asr_pipeline

    def _transcribe_with_whisper_tiny(self, audio_path: str) -> str:
        asr = self._get_asr_pipeline()
        result = asr(
            audio_path,
            return_timestamps=False,
            generate_kwargs={"task": "transcribe"},
        )
        if isinstance(result, dict):
            return str(result.get("text", "")).strip()
        return str(result).strip()

    @staticmethod
    def _tts_settings_snapshot() -> Dict[str, object]:
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
            "pocket_speed": settings.get(
                "pocket_speed", defaults.get("pocket_speed", 1.0)
            ),
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

    def _set_status_text(self, text: str) -> None:
        self._ui_bridge.statusText.emit(str(text or ""))

    def _set_talk_button_text(self, text: str) -> None:
        self._ui_bridge.talkButtonText.emit(str(text or ""))

    def _set_prompt_text(self, text: str) -> None:
        self._ui_bridge.promptText.emit(str(text or ""))
