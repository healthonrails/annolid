from __future__ import annotations

from typing import Dict

from qtpy.QtCore import Signal
from qtpy.QtWidgets import QComboBox, QDoubleSpinBox, QHBoxLayout, QLabel, QWidget

from annolid.agents.kokoro_tts import (
    DEFAULT_LANG,
    DEFAULT_SPEED,
    DEFAULT_VOICE,
    get_available_voices,
    get_suggested_languages,
)
from annolid.utils.tts_settings import load_tts_settings, save_tts_settings


class TtsControlsWidget(QWidget):
    """Compact controls for selecting TTS voice, language, and speed."""

    settingsChanged = Signal(dict)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._settings = load_tts_settings()
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        layout.addWidget(QLabel("Voice"))
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
        layout.addWidget(self.voice_combo)

        layout.addWidget(QLabel("Language"))
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
        layout.addWidget(self.language_combo)

        layout.addWidget(QLabel("Speed"))
        self.speed_spin = QDoubleSpinBox(self)
        self.speed_spin.setRange(0.5, 2.0)
        self.speed_spin.setSingleStep(0.05)
        self.speed_spin.setDecimals(2)
        self.speed_spin.setValue(
            float(self._settings.get("speed", DEFAULT_SPEED))
        )
        self.speed_spin.setToolTip("Speech rate multiplier.")
        layout.addWidget(self.speed_spin)

        self.voice_combo.currentTextChanged.connect(self._persist_settings)
        self.language_combo.currentTextChanged.connect(self._persist_settings)
        self.speed_spin.valueChanged.connect(self._persist_settings)

    def current_settings(self) -> Dict[str, object]:
        voice = self.voice_combo.currentText().strip() or DEFAULT_VOICE
        lang = self.language_combo.currentText().strip() or DEFAULT_LANG
        speed = float(self.speed_spin.value())
        return {"voice": voice, "lang": lang, "speed": speed}

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
