from __future__ import annotations

from annolid.gui.widgets.llm_settings_dialog import _extract_ollama_model_names
from qtpy import QtWidgets

from annolid.gui.widgets.llm_settings_dialog import LLMSettingsDialog


def test_extract_ollama_model_names_from_dict_response() -> None:
    payload = {
        "models": [
            {"name": "qwen3:8b"},
            {"name": "llama3.2-vision:latest"},
            {"name": "qwen3:8b"},
        ]
    }
    assert _extract_ollama_model_names(payload) == [
        "qwen3:8b",
        "llama3.2-vision:latest",
    ]


def test_extract_ollama_model_names_from_object_response() -> None:
    class _Model:
        def __init__(self, model: str) -> None:
            self.model = model

    class _Response:
        def __init__(self) -> None:
            self.models = [_Model("qwen3:14b"), _Model("qwen3:14b"), _Model("gemma3")]

    assert _extract_ollama_model_names(_Response()) == ["qwen3:14b", "gemma3"]


def test_llm_settings_dialog_constructs_without_runtime_tab_name_error() -> None:
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    _ = app
    dialog = LLMSettingsDialog(
        None,
        settings={
            "provider": "ollama",
            "ollama": {
                "host": "http://localhost:11434",
                "preferred_models": ["qwen3.5:0.8b"],
            },
            "agent": {},
        },
    )
    assert dialog._tabs.count() >= 1
    dialog.close()
