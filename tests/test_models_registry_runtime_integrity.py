from __future__ import annotations

from pathlib import Path

from annolid.gui.models_registry import (
    get_model_unavailable_reason,
    get_runtime_model_registry,
    validate_model_registry_entries,
)


def test_runtime_registry_uses_configured_model_path_defaults(tmp_path: Path) -> None:
    weight_path = tmp_path / "weights" / "best.pt"
    weight_path.parent.mkdir(parents=True, exist_ok=True)
    weight_path.write_bytes(b"")

    registry = get_runtime_model_registry(
        config={
            "ai": {
                "model_path_defaults": {
                    "dino_kpseg": str(weight_path),
                    "dino_kpseg_tracker": str(weight_path),
                    "videomt": str(weight_path),
                }
            }
        }
    )

    by_id = {cfg.identifier: cfg for cfg in registry}
    assert by_id["dino_kpseg"].weight_file == str(weight_path)
    assert by_id["dino_kpseg_tracker"].weight_file == str(weight_path)
    assert by_id["videomt"].weight_file == str(weight_path)


def test_validate_model_registry_entries_warns_for_missing_local_weights() -> None:
    registry = get_runtime_model_registry(
        config={
            "ai": {
                "model_path_defaults": {
                    "dino_kpseg": "/tmp/__missing_dino_kpseg__.pt",
                    "dino_kpseg_tracker": "/tmp/__missing_dino_kpseg_tracker__.pt",
                }
            }
        }
    )

    ok, errors, warnings = validate_model_registry_entries(registry)
    assert ok is True
    assert errors == []
    assert any("dino_kpseg" in msg.lower() for msg in warnings)


def test_get_model_unavailable_reason_only_for_local_required_entries() -> None:
    registry = get_runtime_model_registry(
        config={
            "ai": {
                "model_path_defaults": {
                    "dino_kpseg": "/tmp/__missing_dino_kpseg__.pt",
                    "dino_kpseg_tracker": "/tmp/__missing_dino_kpseg_tracker__.pt",
                }
            }
        }
    )
    by_id = {cfg.identifier: cfg for cfg in registry}

    assert get_model_unavailable_reason(by_id["dino_kpseg"]) is not None
    assert get_model_unavailable_reason(by_id["yolo11n"]) is None


def test_settings_override_takes_precedence_over_config_override(
    tmp_path: Path,
) -> None:
    class _Settings:
        def __init__(self, mapping):
            self._mapping = dict(mapping)

        def value(self, key, default=""):
            return self._mapping.get(key, default)

    settings_weight = tmp_path / "settings_best.pt"
    settings_weight.write_bytes(b"")
    config_weight = tmp_path / "config_best.pt"
    config_weight.write_bytes(b"")

    registry = get_runtime_model_registry(
        config={"ai": {"model_path_defaults": {"dino_kpseg": str(config_weight)}}},
        settings=_Settings({"ai/model_paths/dino_kpseg": str(settings_weight)}),
    )
    by_id = {cfg.identifier: cfg for cfg in registry}
    assert by_id["dino_kpseg"].weight_file == str(settings_weight)


def test_videomt_resolves_workspace_downloads_when_relative_path_exists(
    tmp_path: Path, monkeypatch
) -> None:
    fake_home = tmp_path / "home"
    expected = (
        fake_home
        / ".annolid"
        / "workspace"
        / "downloads"
        / "videomt_yt_2019_vit_small_52.8.onnx"
    )
    expected.parent.mkdir(parents=True, exist_ok=True)
    expected.write_bytes(b"onnx")

    monkeypatch.setattr("annolid.utils.model_assets.Path.home", lambda: fake_home)

    registry = get_runtime_model_registry(
        config={
            "ai": {
                "model_path_defaults": {
                    "videomt": "downloads/videomt_yt_2019_vit_small_52.8.onnx"
                }
            }
        }
    )
    by_id = {cfg.identifier: cfg for cfg in registry}
    assert by_id["videomt"].weight_file == str(expected)
    assert get_model_unavailable_reason(by_id["videomt"]) is None
