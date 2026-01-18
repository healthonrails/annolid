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
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QStackedWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from annolid.agents.kokoro_tts import (
    DEFAULT_LANG,
    DEFAULT_SPEED,
    DEFAULT_VOICE,
    get_available_voices,
    get_suggested_languages,
)
from annolid.agents.pocket_tts import (
    DEFAULT_VOICE as POCKET_DEFAULT_VOICE,
    get_available_voices as get_pocket_voices,
)
from annolid.utils.tts_settings import load_tts_settings, save_tts_settings


_DISABLE_VALUES = {"1", "true", "yes", "on"}


def _audio_disabled_by_env() -> bool:
    return os.getenv("ANNOLID_DISABLE_AUDIO", "").strip().lower() in _DISABLE_VALUES


def _theme_icon(name: str) -> QtGui.QIcon:
    icon = QtGui.QIcon.fromTheme(name)
    return icon if not icon.isNull() else QtGui.QIcon()


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
        self._thread: Optional[QtCore.QThread] = None
        self._worker: Optional[_VoicePromptRecordWorker] = None
        self._recording = False

        root = QVBoxLayout(self)

        self.status_label = QLabel(
            "Record a short voice prompt (recommended: 3–10 seconds)."
        )
        self.status_label.setWordWrap(True)
        root.addWidget(self.status_label)

        form = QFormLayout()
        root.addLayout(form)

        self.duration_spin = QDoubleSpinBox(self)
        self.duration_spin.setRange(1.0, 30.0)
        self.duration_spin.setSingleStep(0.5)
        self.duration_spin.setDecimals(1)
        self.duration_spin.setValue(6.0)
        form.addRow("Duration (s):", self.duration_spin)

        self.sample_rate_spin = QSpinBox(self)
        self.sample_rate_spin.setRange(8000, 48000)
        self.sample_rate_spin.setSingleStep(1000)
        self.sample_rate_spin.setValue(24000)
        form.addRow("Sample rate:", self.sample_rate_spin)

        row = QHBoxLayout()
        root.addLayout(row)

        self.record_button = QPushButton("Record", self)
        mic_icon = _theme_icon("audio-input-microphone")
        if not mic_icon.isNull():
            self.record_button.setIcon(mic_icon)
        self.record_button.clicked.connect(self._start_recording)
        row.addWidget(self.record_button)

        self.cancel_button = QPushButton("Cancel", self)
        self.cancel_button.clicked.connect(self.reject)
        row.addWidget(self.cancel_button)

        row.addStretch(1)

    def recorded_path(self) -> str:
        return self._recorded_path

    def _start_recording(self) -> None:
        self.record_button.setEnabled(False)
        self.cancel_button.setEnabled(False)
        self.status_label.setText("Recording…")
        self._recording = True

        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = Path.home() / ".annolid" / "tts_prompts"
        out_path = out_dir / f"voice_prompt_{ts}.wav"

        duration_s = float(self.duration_spin.value())
        sample_rate = int(self.sample_rate_spin.value())

        thread = QtCore.QThread(self)
        worker = _VoicePromptRecordWorker(
            duration_s=duration_s,
            sample_rate=sample_rate,
            out_path=str(out_path),
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
        self._recording = False

        if error:
            QtWidgets.QMessageBox.warning(self, "Recording failed", error)
            self.status_label.setText("Recording failed.")
            return

        self._recorded_path = path
        self.status_label.setText(f"Recorded: {path}")
        thread = self._thread
        if thread is not None:
            thread.wait(1000)
        self.accept()

    def closeEvent(self, event: QtGui.QCloseEvent) -> None:
        if self._recording:
            event.ignore()
            return
        super().closeEvent(event)


class _PromptPathRow(QWidget):
    """
    Prompt selector block:

      [ line edit ......................... ]
      [ Record… ]
      [ Browse… ]
      [ Clear  ]
    """

    pathChanged = Signal(str)
    recordRequested = Signal()

    def __init__(
        self,
        parent: Optional[QWidget],
        *,
        placeholder: str,
        dialog_title: str,
    ) -> None:
        super().__init__(parent)
        self._dialog_title = dialog_title
        self._path = ""

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        self.line_edit = QLineEdit(self)
        self.line_edit.setPlaceholderText(placeholder)
        self.line_edit.editingFinished.connect(self._on_edit_finished)
        layout.addWidget(self.line_edit)

        # Row 1: Browse…
        self.btn_browse = QPushButton("Browse…", self)
        open_icon = _theme_icon("document-open")
        if not open_icon.isNull():
            self.btn_browse.setIcon(open_icon)
        self.btn_browse.clicked.connect(self._on_browse)
        layout.addWidget(self.btn_browse)

        # Row 2: Record…
        self.btn_record = QPushButton("Record…", self)
        mic = _theme_icon("audio-input-microphone")
        if not mic.isNull():
            self.btn_record.setIcon(mic)
        self.btn_record.clicked.connect(lambda: self.recordRequested.emit())
        layout.addWidget(self.btn_record)

        # Row 3: Clear
        self.btn_clear = QPushButton("Clear", self)
        clear_icon = _theme_icon("edit-clear")
        if not clear_icon.isNull():
            self.btn_clear.setIcon(clear_icon)
        self.btn_clear.clicked.connect(lambda: self.set_path(""))
        layout.addWidget(self.btn_clear)

    def _on_edit_finished(self) -> None:
        self.set_path(self.line_edit.text().strip())

    def _on_browse(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            self._dialog_title,
            str(Path.home()),
            "Audio Files (*.wav *.flac *.mp3);;All Files (*)",
        )
        if path:
            self.set_path(path)

    def set_path(self, path: str) -> None:
        path = path or ""
        if path == self._path:
            return
        self._path = path
        try:
            self.line_edit.blockSignals(True)
            self.line_edit.setText(path)
        finally:
            self.line_edit.blockSignals(False)
        self.pathChanged.emit(path)

    def current_path(self) -> str:
        return self._path


class TtsControlsWidget(QWidget):
    """Modernized controls for selecting TTS engine, voice, language, and speed."""

    settingsChanged = Signal(dict)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._settings = load_tts_settings()
        self._prewarm_key: tuple[str, str] | None = None
        self._prewarm_pocket_key: tuple[str, str] | None = None

        # cached settings
        self._pocket_prompt_path = str(
            self._settings.get("pocket_prompt_path", "")).strip()
        self._chatterbox_voice_path = str(
            self._settings.get("chatterbox_voice_path", "")).strip()
        self._chatterbox_dtype = str(self._settings.get(
            "chatterbox_dtype", "fp32")).strip() or "fp32"
        self._chatterbox_max_new_tokens = int(
            self._settings.get("chatterbox_max_new_tokens", 1024))
        self._chatterbox_repetition_penalty = float(
            self._settings.get("chatterbox_repetition_penalty", 1.2))
        self._chatterbox_apply_watermark = bool(
            self._settings.get("chatterbox_apply_watermark", False))

        self._build_ui()
        self._sync_engine_page()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 0, 0, 0)

        # --- Top: Engine chooser
        top = QHBoxLayout()
        root.addLayout(top)

        lbl = QLabel("Engine:", self)
        top.addWidget(lbl)

        self.engine_combo = QComboBox(self)
        self.engine_combo.addItem("Kokoro", "kokoro")
        self.engine_combo.addItem("Pocket (Kyutai)", "pocket")
        self.engine_combo.addItem("Chatterbox", "chatterbox")
        self.engine_combo.addItem("gTTS", "gtts")
        self.engine_combo.setToolTip("Select the text-to-speech engine.")

        current_engine = str(self._settings.get(
            "engine", "kokoro")).strip().lower()
        idx = self.engine_combo.findData(current_engine)
        if idx >= 0:
            self.engine_combo.setCurrentIndex(idx)

        top.addWidget(self.engine_combo, 1)

        self.btn_reset = QPushButton("Reset", self)
        self.btn_reset.setToolTip(
            "Reset to recommended defaults for the current engine")
        self.btn_reset.clicked.connect(self._reset_defaults_for_engine)
        top.addWidget(self.btn_reset)

        # --- Stacked engine pages
        self.pages = QStackedWidget(self)
        root.addWidget(self.pages, 1)

        self.page_kokoro = QWidget(self)
        self.page_pocket = QWidget(self)
        self.page_chatterbox = QWidget(self)
        self.page_gtts = QWidget(self)

        self.pages.addWidget(self.page_kokoro)
        self.pages.addWidget(self.page_pocket)
        self.pages.addWidget(self.page_chatterbox)
        self.pages.addWidget(self.page_gtts)

        self._build_kokoro_page()
        self._build_pocket_page()
        self._build_chatterbox_page()
        self._build_gtts_page()

        # signals
        self.engine_combo.currentIndexChanged.connect(self._on_engine_changed)

    # -------------------- Pages

    def _build_kokoro_page(self) -> None:
        layout = QVBoxLayout(self.page_kokoro)

        box = QGroupBox("Kokoro Settings", self.page_kokoro)
        layout.addWidget(box)

        form = QFormLayout(box)
        form.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)

        self.voice_combo = QComboBox(self.page_kokoro)
        self.voice_combo.setEditable(True)
        self.voice_combo.setInsertPolicy(QComboBox.NoInsert)
        voices = get_available_voices()
        if voices:
            self.voice_combo.addItems(voices)
        self.voice_combo.setCurrentText(
            str(self._settings.get("voice", DEFAULT_VOICE)))
        self.voice_combo.setToolTip(
            "Select a voice or type a custom voice ID.")
        form.addRow("Voice:", self.voice_combo)

        self.language_combo = QComboBox(self.page_kokoro)
        self.language_combo.setEditable(True)
        self.language_combo.setInsertPolicy(QComboBox.NoInsert)
        languages = get_suggested_languages()
        if languages:
            self.language_combo.addItems(languages)
        self.language_combo.setCurrentText(
            str(self._settings.get("lang", DEFAULT_LANG)))
        self.language_combo.setToolTip(
            "Language code passed to Kokoro (e.g., en-us).")
        form.addRow("Language:", self.language_combo)

        self.speed_spin = QDoubleSpinBox(self.page_kokoro)
        self.speed_spin.setRange(0.5, 2.0)
        self.speed_spin.setSingleStep(0.05)
        self.speed_spin.setDecimals(2)
        self.speed_spin.setValue(
            float(self._settings.get("speed", DEFAULT_SPEED)))
        self.speed_spin.setToolTip("Speech rate multiplier.")
        form.addRow("Speed:", self.speed_spin)

        layout.addStretch(1)

        self.voice_combo.currentTextChanged.connect(self._persist_settings)
        self.language_combo.currentTextChanged.connect(self._persist_settings)
        self.speed_spin.valueChanged.connect(self._persist_settings)

    def _build_pocket_page(self) -> None:
        layout = QVBoxLayout(self.page_pocket)

        box = QGroupBox("Pocket (Kyutai) Settings", self.page_pocket)
        layout.addWidget(box)

        form = QFormLayout(box)
        form.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)

        self.pocket_voice_combo = QComboBox(self.page_pocket)
        self.pocket_voice_combo.setEditable(True)
        self.pocket_voice_combo.setInsertPolicy(QComboBox.NoInsert)
        pocket_voices = get_pocket_voices()
        if pocket_voices:
            self.pocket_voice_combo.addItems(pocket_voices)
        self.pocket_voice_combo.setCurrentText(
            str(self._settings.get("pocket_voice", POCKET_DEFAULT_VOICE))
        )
        self.pocket_voice_combo.setToolTip(
            "Select a Pocket voice, or type one.")
        form.addRow("Voice:", self.pocket_voice_combo)

        self.pocket_speed_spin = QDoubleSpinBox(self.page_pocket)
        self.pocket_speed_spin.setRange(0.5, 2.0)
        self.pocket_speed_spin.setSingleStep(0.05)
        self.pocket_speed_spin.setDecimals(2)
        self.pocket_speed_spin.setValue(
            float(self._settings.get("pocket_speed", 1.0)))
        self.pocket_speed_spin.setToolTip(
            "Playback speed multiplier for Pocket TTS.")
        form.addRow("Speed:", self.pocket_speed_spin)

        self.pocket_prompt_row = _PromptPathRow(
            self.page_pocket,
            placeholder="Voice prompt WAV (optional)",
            dialog_title="Select Pocket TTS Voice Prompt",
        )
        self.pocket_prompt_row.set_path(self._pocket_prompt_path)
        self.pocket_prompt_row.pathChanged.connect(
            self._on_pocket_prompt_changed)
        self.pocket_prompt_row.recordRequested.connect(
            self._record_pocket_prompt)
        form.addRow("Prompt:", self.pocket_prompt_row)

        hint = QLabel(
            "Tip: you can either pick a built-in voice, or provide a prompt file path.",
            self.page_pocket,
        )
        hint.setWordWrap(True)
        hint.setEnabled(False)
        layout.addWidget(hint)

        layout.addStretch(1)

        self.pocket_voice_combo.currentTextChanged.connect(
            self._persist_settings)
        self.pocket_speed_spin.valueChanged.connect(self._persist_settings)

    def _build_chatterbox_page(self) -> None:
        layout = QVBoxLayout(self.page_chatterbox)

        box = QGroupBox("Chatterbox Settings", self.page_chatterbox)
        layout.addWidget(box)

        form = QFormLayout(box)
        form.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)

        self.chatterbox_prompt_row = _PromptPathRow(
            self.page_chatterbox,
            placeholder="Voice prompt WAV (recommended)",
            dialog_title="Select Voice Prompt Audio",
        )
        self.chatterbox_prompt_row.set_path(self._chatterbox_voice_path)
        self.chatterbox_prompt_row.pathChanged.connect(
            self._on_chatterbox_prompt_changed)
        self.chatterbox_prompt_row.recordRequested.connect(
            self._record_voice_prompt)
        form.addRow("Prompt:", self.chatterbox_prompt_row)

        self.chatterbox_dtype_combo = QComboBox(self.page_chatterbox)
        self.chatterbox_dtype_combo.addItems(
            ["fp32", "fp16", "q8", "q4", "q4f16"])
        self.chatterbox_dtype_combo.setCurrentText(self._chatterbox_dtype)
        form.addRow("Dtype:", self.chatterbox_dtype_combo)

        self.chatterbox_max_tokens_spin = QSpinBox(self.page_chatterbox)
        self.chatterbox_max_tokens_spin.setRange(128, 4096)
        self.chatterbox_max_tokens_spin.setSingleStep(128)
        self.chatterbox_max_tokens_spin.setValue(
            self._chatterbox_max_new_tokens)
        form.addRow("Max tokens:", self.chatterbox_max_tokens_spin)

        self.chatterbox_rep_penalty_spin = QDoubleSpinBox(self.page_chatterbox)
        self.chatterbox_rep_penalty_spin.setRange(1.0, 3.0)
        self.chatterbox_rep_penalty_spin.setSingleStep(0.05)
        self.chatterbox_rep_penalty_spin.setDecimals(2)
        self.chatterbox_rep_penalty_spin.setValue(
            self._chatterbox_repetition_penalty)
        form.addRow("Repetition:", self.chatterbox_rep_penalty_spin)

        self.chatterbox_watermark_check = QCheckBox(self.page_chatterbox)
        self.chatterbox_watermark_check.setChecked(
            self._chatterbox_apply_watermark)
        form.addRow("Watermark:", self.chatterbox_watermark_check)

        layout.addStretch(1)

        self.chatterbox_dtype_combo.currentTextChanged.connect(
            self._persist_settings)
        self.chatterbox_max_tokens_spin.valueChanged.connect(
            self._persist_settings)
        self.chatterbox_rep_penalty_spin.valueChanged.connect(
            self._persist_settings)
        self.chatterbox_watermark_check.toggled.connect(self._persist_settings)

    def _build_gtts_page(self) -> None:
        layout = QVBoxLayout(self.page_gtts)
        msg = QLabel(
            "gTTS uses Google Text-to-Speech.\n"
            "This engine typically only needs the app’s global network settings.\n"
            "Select Kokoro/Pocket/Chatterbox for local voices and prompts.",
            self.page_gtts,
        )
        msg.setWordWrap(True)
        layout.addWidget(msg)
        layout.addStretch(1)

    # -------------------- Public API

    def current_settings(self) -> Dict[str, object]:
        engine = self.engine_combo.currentData() or "kokoro"

        voice = self.voice_combo.currentText().strip() or DEFAULT_VOICE
        lang = self.language_combo.currentText().strip() or DEFAULT_LANG
        speed = float(self.speed_spin.value())

        pocket_voice = self.pocket_voice_combo.currentText().strip() or POCKET_DEFAULT_VOICE
        pocket_speed = float(self.pocket_speed_spin.value())
        pocket_prompt_path = (self._pocket_prompt_path or "").strip()

        return {
            "engine": engine,
            "voice": voice,
            "lang": lang,
            "speed": speed,
            "pocket_voice": pocket_voice,
            "pocket_speed": pocket_speed,
            "pocket_prompt_path": pocket_prompt_path,
            "chatterbox_voice_path": (self._chatterbox_voice_path or "").strip(),
            "chatterbox_dtype": self.chatterbox_dtype_combo.currentText().strip() or "fp32",
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

    # -------------------- Internal

    def _on_engine_changed(self) -> None:
        self._sync_engine_page()
        self._persist_settings()

    def _sync_engine_page(self) -> None:
        engine = str(self.engine_combo.currentData()
                     or "kokoro").strip().lower()
        page_map = {
            "kokoro": self.page_kokoro,
            "pocket": self.page_pocket,
            "chatterbox": self.page_chatterbox,
            "gtts": self.page_gtts,
        }
        self.pages.setCurrentWidget(page_map.get(engine, self.page_kokoro))

        if engine == "chatterbox":
            self._prewarm_chatterbox_async()
        if engine == "pocket":
            self._prewarm_pocket_async()

    def _persist_settings(self) -> None:
        settings = self.current_settings()
        save_tts_settings(settings)
        self.settingsChanged.emit(settings)

    def _reset_defaults_for_engine(self) -> None:
        engine = str(self.engine_combo.currentData()
                     or "kokoro").strip().lower()
        if engine == "kokoro":
            self.voice_combo.setCurrentText(DEFAULT_VOICE)
            self.language_combo.setCurrentText(DEFAULT_LANG)
            self.speed_spin.setValue(float(DEFAULT_SPEED))
        elif engine == "pocket":
            self.pocket_voice_combo.setCurrentText(POCKET_DEFAULT_VOICE)
            self.pocket_speed_spin.setValue(1.0)
            self.pocket_prompt_row.set_path("")
        elif engine == "chatterbox":
            self.chatterbox_prompt_row.set_path("")
            self.chatterbox_dtype_combo.setCurrentText("fp32")
            self.chatterbox_max_tokens_spin.setValue(1024)
            self.chatterbox_rep_penalty_spin.setValue(1.2)
            self.chatterbox_watermark_check.setChecked(False)
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

            def run(self) -> None:  # pragma: no cover
                try:
                    from annolid.agents.chatterbox_tts import prewarm

                    prewarm(voice_wav_path=self._voice_path, dtype=self._dtype)
                except Exception:
                    return

        QtCore.QThreadPool.globalInstance().start(_Runnable(dtype, voice_path))

    def _prewarm_pocket_async(self) -> None:
        voice = self.pocket_voice_combo.currentText().strip() or POCKET_DEFAULT_VOICE
        prompt_path = (self._pocket_prompt_path or "").strip()
        key = (voice, prompt_path)
        if self._prewarm_pocket_key == key:
            return
        self._prewarm_pocket_key = key

        voice_spec = prompt_path or voice

        class _Runnable(QtCore.QRunnable):
            def __init__(self, voice_spec: str) -> None:
                super().__init__()
                self._voice_spec = voice_spec

            def run(self) -> None:  # pragma: no cover
                try:
                    from annolid.agents.pocket_tts import prewarm

                    prewarm(self._voice_spec)
                except Exception:
                    return

        QtCore.QThreadPool.globalInstance().start(_Runnable(voice_spec))

    def _record_voice_prompt(self) -> None:
        dialog = _VoicePromptRecorderDialog(self)
        if dialog.exec_() != QtWidgets.QDialog.Accepted:
            return
        path = dialog.recorded_path()
        if path:
            self.chatterbox_prompt_row.set_path(path)

    def _record_pocket_prompt(self) -> None:
        dialog = _VoicePromptRecorderDialog(self)
        if dialog.exec_() != QtWidgets.QDialog.Accepted:
            return
        path = dialog.recorded_path()
        if path:
            self.pocket_prompt_row.set_path(path)

    def _on_chatterbox_prompt_changed(self, path: str) -> None:
        self._chatterbox_voice_path = path
        self._persist_settings()

    def _on_pocket_prompt_changed(self, path: str) -> None:
        self._pocket_prompt_path = path
        self._persist_settings()
