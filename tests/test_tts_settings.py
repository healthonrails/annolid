from __future__ import annotations

from pathlib import Path

from annolid.agents.pocket_tts import DEFAULT_VOICE as POCKET_DEFAULT_VOICE


def test_tts_settings_defaults_include_engine(tmp_path: Path, monkeypatch) -> None:
    from annolid.utils import tts_settings as mod

    monkeypatch.setattr(mod, "_SETTINGS_DIR", tmp_path)
    monkeypatch.setattr(mod, "_SETTINGS_FILE", tmp_path / "tts_settings.json")

    settings = mod.load_tts_settings()
    assert settings["engine"] == "kokoro"
    assert "chatterbox_voice_path" in settings
    assert settings["pocket_voice"] == POCKET_DEFAULT_VOICE
    assert settings["pocket_prompt_path"] == ""
    assert settings["pocket_speed"] == 1.0


def test_save_tts_settings_merges_partial_updates(tmp_path: Path, monkeypatch) -> None:
    from annolid.utils import tts_settings as mod

    monkeypatch.setattr(mod, "_SETTINGS_DIR", tmp_path)
    monkeypatch.setattr(mod, "_SETTINGS_FILE", tmp_path / "tts_settings.json")

    mod.save_tts_settings(
        {"engine": "chatterbox", "chatterbox_voice_path": "/tmp/voice.wav"}
    )
    mod.save_tts_settings({"voice": "af_sarah"})

    persisted = mod.load_tts_settings()
    assert persisted["engine"] == "chatterbox"
    assert persisted["chatterbox_voice_path"] == "/tmp/voice.wav"
    assert persisted["voice"] == "af_sarah"
