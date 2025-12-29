from __future__ import annotations

import datetime
import os
from pathlib import Path
from typing import Dict, Optional

import numpy as np
from qtpy import QtCore, QtGui, QtWidgets
from qtpy.QtCore import Signal
from qtpy.QtWidgets import (
    QComboBox,
    QCheckBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QWidget,
)

from annolid.agents.kokoro_tts import (
    DEFAULT_LANG,
    DEFAULT_SPEED,
    DEFAULT_VOICE,
    get_available_voices,
    get_suggested_languages,
)
from annolid.utils.tts_settings import load_tts_settings, save_tts_settings


_DISABLE_VALUES = {"1", "true", "yes", "on"}


def _audio_disabled_by_env() -> bool:
    return os.getenv("ANNOLID_DISABLE_AUDIO", "").strip().lower() in _DISABLE_VALUES


class _VoicePromptRecordWorker(QtCore.QObject):
    finished = QtCore.Signal(str, str)  # path, error

    def __init__(self, *, duration_s: float, sample_rate: int, out_path: str) -> None:
        super().__init__()
        self._duration_s = float(duration_s)
        self._sample_rate = int(sample_rate)
        self._out_path = str(out_path)

    @QtCore.Slot()
    def run(self) -> None:
        try:
            if _audio_disabled_by_env():
                raise RuntimeError(
                    "Audio is disabled via ANNOLID_DISABLE_AUDIO.")
            try:
                import sounddevice as sd  # type: ignore
            except Exception as exc:
                raise RuntimeError(
                    f"sounddevice is not available ({exc})") from exc

            try:
                devices = sd.query_devices()
            except Exception as exc:
                raise RuntimeError(
                    f"Could not enumerate audio devices ({exc})") from exc
            if not any(d.get("max_input_channels", 0) > 0 for d in devices):
                raise RuntimeError("No usable audio input device detected.")

            frames = max(1, int(round(self._duration_s * self._sample_rate)))
            recording = sd.rec(
                frames,
                samplerate=self._sample_rate,
                channels=1,
                dtype="float32",
            )
            sd.wait()
            samples = np.asarray(recording).reshape(-1)
            samples = np.clip(samples, -1.0, 1.0)
            pcm16 = (samples * 32767.0).astype(np.int16)

            out_path = Path(self._out_path)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            import wave

            with wave.open(str(out_path), "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(self._sample_rate)
                wf.writeframes(pcm16.tobytes())

            self.finished.emit(str(out_path), "")
        except Exception as exc:
            self.finished.emit("", str(exc))


class _VoicePromptRecorderDialog(QtWidgets.QDialog):
    def __init__(self, parent: Optional[QtWidgets.QWidget] = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Record Voice Prompt")
        self.setModal(True)
        self._recorded_path: str = ""

        layout = QtWidgets.QVBoxLayout(self)

        self.status_label = QtWidgets.QLabel(
            "Record a short voice prompt (recommended: 3–10 seconds)."
        )
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        form = QtWidgets.QFormLayout()
        layout.addLayout(form)

        self.duration_spin = QtWidgets.QDoubleSpinBox(self)
        self.duration_spin.setRange(1.0, 30.0)
        self.duration_spin.setSingleStep(0.5)
        self.duration_spin.setDecimals(1)
        self.duration_spin.setValue(6.0)
        form.addRow("Duration (s):", self.duration_spin)

        self.sample_rate_spin = QtWidgets.QSpinBox(self)
        self.sample_rate_spin.setRange(8000, 48000)
        self.sample_rate_spin.setSingleStep(1000)
        self.sample_rate_spin.setValue(24000)
        form.addRow("Sample rate:", self.sample_rate_spin)

        button_row = QtWidgets.QHBoxLayout()
        layout.addLayout(button_row)

        self.record_button = QtWidgets.QPushButton("Record", self)
        icon = QtGui.QIcon.fromTheme("audio-input-microphone")
        if not icon.isNull():
            self.record_button.setIcon(icon)
        self.record_button.clicked.connect(self._start_recording)
        button_row.addWidget(self.record_button)

        self.cancel_button = QtWidgets.QPushButton("Cancel", self)
        self.cancel_button.clicked.connect(self.reject)
        button_row.addWidget(self.cancel_button)
        button_row.addStretch(1)

        self._thread: Optional[QtCore.QThread] = None
        self._worker: Optional[_VoicePromptRecordWorker] = None

    def recorded_path(self) -> str:
        return self._recorded_path

    def _start_recording(self) -> None:
        self.record_button.setEnabled(False)
        self.cancel_button.setEnabled(False)
        self.status_label.setText("Recording…")

        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = Path.home() / ".annolid" / "tts_prompts"
        out_path = out_dir / f"voice_prompt_{ts}.wav"

        duration_s = float(self.duration_spin.value())
        sample_rate = int(self.sample_rate_spin.value())

        thread = QtCore.QThread(self)
        worker = _VoicePromptRecordWorker(
            duration_s=duration_s, sample_rate=sample_rate, out_path=str(
                out_path)
        )
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.finished.connect(self._on_finished)
        worker.finished.connect(thread.quit)
        worker.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)

        self._thread = thread
        self._worker = worker
        thread.start()

    @QtCore.Slot(str, str)
    def _on_finished(self, path: str, error: str) -> None:
        self.record_button.setEnabled(True)
        self.cancel_button.setEnabled(True)
        if error:
            QtWidgets.QMessageBox.warning(self, "Recording failed", error)
            self.status_label.setText("Recording failed.")
            return
        self._recorded_path = path
        self.status_label.setText(f"Recorded: {path}")
        self.accept()


class TtsControlsWidget(QWidget):
    """Compact controls for selecting TTS voice, language, and speed."""

    settingsChanged = Signal(dict)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._settings = load_tts_settings()
        self._prewarm_key: tuple[str, str] | None = None
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QFormLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)

        self.engine_combo = QComboBox(self)
        self.engine_combo.addItem("Kokoro", "kokoro")
        self.engine_combo.addItem("Chatterbox", "chatterbox")
        self.engine_combo.addItem("gTTS", "gtts")
        current_engine = str(self._settings.get(
            "engine", "kokoro")).strip().lower()
        idx = self.engine_combo.findData(current_engine)
        if idx >= 0:
            self.engine_combo.setCurrentIndex(idx)
        self.engine_combo.setToolTip("Select the text-to-speech engine.")
        layout.addRow("Engine:", self.engine_combo)

        self.voice_combo = QComboBox(self)
        self.voice_combo.setEditable(True)
        self.voice_combo.setInsertPolicy(QComboBox.NoInsert)
        voices = get_available_voices()
        if voices:
            self.voice_combo.addItems(voices)
        self.voice_combo.setCurrentText(
            str(self._settings.get("voice", DEFAULT_VOICE))
        )
        self.voice_combo.setToolTip(
            "Select a voice or type a custom voice ID."
        )
        layout.addRow("Kokoro voice:", self.voice_combo)

        self.language_combo = QComboBox(self)
        self.language_combo.setEditable(True)
        self.language_combo.setInsertPolicy(QComboBox.NoInsert)
        languages = get_suggested_languages()
        if languages:
            self.language_combo.addItems(languages)
        self.language_combo.setCurrentText(
            str(self._settings.get("lang", DEFAULT_LANG))
        )
        self.language_combo.setToolTip(
            "Language code passed to Kokoro (e.g., en-us)."
        )
        layout.addRow("Language:", self.language_combo)

        self.speed_spin = QDoubleSpinBox(self)
        self.speed_spin.setRange(0.5, 2.0)
        self.speed_spin.setSingleStep(0.05)
        self.speed_spin.setDecimals(2)
        self.speed_spin.setValue(
            float(self._settings.get("speed", DEFAULT_SPEED))
        )
        self.speed_spin.setToolTip("Speech rate multiplier.")
        layout.addRow("Speed:", self.speed_spin)

        self._chatterbox_voice_path = str(
            self._settings.get("chatterbox_voice_path", "")
        ).strip()
        self._chatterbox_dtype = str(
            self._settings.get("chatterbox_dtype", "fp32")
        ).strip() or "fp32"
        self._chatterbox_max_new_tokens = int(
            self._settings.get("chatterbox_max_new_tokens", 1024)
        )
        self._chatterbox_repetition_penalty = float(
            self._settings.get("chatterbox_repetition_penalty", 1.2)
        )
        self._chatterbox_apply_watermark = bool(
            self._settings.get("chatterbox_apply_watermark", False)
        )

        self.prompt_path_edit = QLineEdit(self)
        self.prompt_path_edit.setPlaceholderText(
            "Voice prompt WAV (for Chatterbox)")
        self.prompt_path_edit.setText(self._chatterbox_voice_path)
        self.prompt_path_edit.setToolTip(
            "Path to a short voice prompt WAV used for voice cloning."
        )
        layout.addRow("Chatterbox prompt:", self.prompt_path_edit)

        self.upload_prompt_button = QPushButton("Upload…", self)
        self.upload_prompt_button.clicked.connect(self._select_voice_prompt)
        layout.addRow("Upload:", self.upload_prompt_button)

        self.record_prompt_button = QPushButton("Record…", self)
        mic_icon = QtGui.QIcon.fromTheme("audio-input-microphone")
        if not mic_icon.isNull():
            self.record_prompt_button.setIcon(mic_icon)
        self.record_prompt_button.clicked.connect(self._record_voice_prompt)
        layout.addRow("Record:", self.record_prompt_button)

        self.clear_prompt_button = QPushButton("Clear", self)
        self.clear_prompt_button.clicked.connect(self._clear_voice_prompt)
        layout.addRow("Clear:", self.clear_prompt_button)

        self.chatterbox_dtype_combo = QComboBox(self)
        self.chatterbox_dtype_combo.addItems(
            ["fp32", "fp16", "q8", "q4", "q4f16"])
        self.chatterbox_dtype_combo.setCurrentText(self._chatterbox_dtype)
        layout.addRow("Dtype:", self.chatterbox_dtype_combo)

        self.chatterbox_max_tokens_spin = QSpinBox(self)
        self.chatterbox_max_tokens_spin.setRange(128, 4096)
        self.chatterbox_max_tokens_spin.setSingleStep(128)
        self.chatterbox_max_tokens_spin.setValue(
            self._chatterbox_max_new_tokens)
        layout.addRow("Max tokens:", self.chatterbox_max_tokens_spin)

        self.chatterbox_rep_penalty_spin = QDoubleSpinBox(self)
        self.chatterbox_rep_penalty_spin.setRange(1.0, 3.0)
        self.chatterbox_rep_penalty_spin.setSingleStep(0.05)
        self.chatterbox_rep_penalty_spin.setDecimals(2)
        self.chatterbox_rep_penalty_spin.setValue(
            self._chatterbox_repetition_penalty)
        layout.addRow("Repetition:", self.chatterbox_rep_penalty_spin)

        self.chatterbox_watermark_check = QCheckBox(self)
        self.chatterbox_watermark_check.setChecked(
            self._chatterbox_apply_watermark)
        layout.addRow("Watermark:", self.chatterbox_watermark_check)

        self._layout = layout

        self.voice_combo.currentTextChanged.connect(self._persist_settings)
        self.language_combo.currentTextChanged.connect(self._persist_settings)
        self.speed_spin.valueChanged.connect(self._persist_settings)
        self.engine_combo.currentTextChanged.connect(self._persist_settings)
        self.prompt_path_edit.editingFinished.connect(self._prompt_path_edited)
        self.chatterbox_dtype_combo.currentTextChanged.connect(
            self._persist_settings)
        self.chatterbox_max_tokens_spin.valueChanged.connect(
            self._persist_settings)
        self.chatterbox_rep_penalty_spin.valueChanged.connect(
            self._persist_settings)
        self.chatterbox_watermark_check.toggled.connect(self._persist_settings)
        self._sync_visible_rows()

    def current_settings(self) -> Dict[str, object]:
        engine = self.engine_combo.currentData() or "kokoro"
        voice = self.voice_combo.currentText().strip() or DEFAULT_VOICE
        lang = self.language_combo.currentText().strip() or DEFAULT_LANG
        speed = float(self.speed_spin.value())
        return {
            "engine": engine,
            "voice": voice,
            "lang": lang,
            "speed": speed,
            "chatterbox_voice_path": self._chatterbox_voice_path,
            "chatterbox_dtype": self.chatterbox_dtype_combo.currentText().strip()
            or "fp32",
            "chatterbox_max_new_tokens": int(self.chatterbox_max_tokens_spin.value()),
            "chatterbox_repetition_penalty": float(self.chatterbox_rep_penalty_spin.value()),
            "chatterbox_apply_watermark": bool(self.chatterbox_watermark_check.isChecked()),
        }

    def set_language_and_voice(
        self,
        lang: str,
        voice: str,
        *,
        persist: bool = True,
    ) -> None:
        """Programmatically set language + voice and optionally persist once."""
        lang = (lang or "").strip() or DEFAULT_LANG
        voice = (voice or "").strip() or DEFAULT_VOICE
        try:
            self.voice_combo.blockSignals(True)
            self.language_combo.blockSignals(True)
            self.voice_combo.setCurrentText(voice)
            self.language_combo.setCurrentText(lang)
        finally:
            self.voice_combo.blockSignals(False)
            self.language_combo.blockSignals(False)
        if persist:
            self._persist_settings()

    def _persist_settings(self) -> None:
        settings = self.current_settings()
        save_tts_settings(settings)
        self.settingsChanged.emit(settings)
        self._sync_visible_rows()

    def _set_row_visible(self, field: QtWidgets.QWidget, visible: bool) -> None:
        label = self._layout.labelForField(field)
        if label is not None:
            label.setVisible(visible)
        field.setVisible(visible)

    def _sync_visible_rows(self) -> None:
        engine = str(self.engine_combo.currentData()
                     or "kokoro").strip().lower()

        show_kokoro = engine == "kokoro"
        show_chatterbox = engine == "chatterbox"

        self._set_row_visible(self.voice_combo, show_kokoro)
        self._set_row_visible(self.language_combo, show_kokoro)
        self._set_row_visible(self.speed_spin, show_kokoro)

        self._set_row_visible(self.prompt_path_edit, show_chatterbox)
        self._set_row_visible(self.upload_prompt_button, show_chatterbox)
        self._set_row_visible(self.record_prompt_button, show_chatterbox)
        self._set_row_visible(
            self.clear_prompt_button, show_chatterbox and bool(
                self._chatterbox_voice_path)
        )
        self._set_row_visible(self.chatterbox_dtype_combo, show_chatterbox)
        self._set_row_visible(self.chatterbox_max_tokens_spin, show_chatterbox)
        self._set_row_visible(
            self.chatterbox_rep_penalty_spin, show_chatterbox)
        self._set_row_visible(self.chatterbox_watermark_check, show_chatterbox)

        if show_chatterbox:
            self._prewarm_chatterbox_async()

    def _select_voice_prompt(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Voice Prompt Audio",
            str(Path.home()),
            "Audio Files (*.wav *.flac *.mp3);;All Files (*)",
        )
        if not path:
            return
        self._chatterbox_voice_path = path
        self.prompt_path_edit.setText(path)
        self._persist_settings()

    def _clear_voice_prompt(self) -> None:
        self._chatterbox_voice_path = ""
        self.prompt_path_edit.setText("")
        self._persist_settings()

    def _prompt_path_edited(self) -> None:
        self._chatterbox_voice_path = self.prompt_path_edit.text().strip()
        self._persist_settings()

    def _record_voice_prompt(self) -> None:
        dialog = _VoicePromptRecorderDialog(self)
        if dialog.exec_() != QtWidgets.QDialog.Accepted:
            return
        path = dialog.recorded_path()
        if not path:
            return
        self._chatterbox_voice_path = path
        self.prompt_path_edit.setText(path)
        self._persist_settings()

    def _prewarm_chatterbox_async(self) -> None:
        dtype = self.chatterbox_dtype_combo.currentText().strip() or "fp32"
        voice_path = (self._chatterbox_voice_path or "").strip()
        key = (dtype, voice_path)
        if self._prewarm_key == key:
            return
        self._prewarm_key = key

        class _Runnable(QtCore.QRunnable):
            def __init__(self, dtype: str, voice_path: str) -> None:
                super().__init__()
                self._dtype = dtype
                self._voice_path = voice_path

            def run(self) -> None:  # pragma: no cover - background warmup
                try:
                    from annolid.agents.chatterbox_tts import prewarm

                    prewarm(voice_wav_path=self._voice_path, dtype=self._dtype)
                except Exception:
                    return

        QtCore.QThreadPool.globalInstance().start(_Runnable(dtype, voice_path))
