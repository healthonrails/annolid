from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

from annolid.engine import cli as engine_cli
from annolid.features import dino_models
from annolid.tracking.configuration import CutieDinoTrackerConfig


def test_resolve_dino_model_id_accepts_alias_and_display_name() -> None:
    assert (
        dino_models.resolve_dino_model_id("dinov3_vitl16")
        == "facebook/dinov3-vitl16-pretrain-lvd1689m"
    )
    assert (
        dino_models.resolve_dino_model_id("DINOv3 ViT-S/16 (gated, recommended)")
        == dino_models.DEFAULT_DINOV3_MODEL_ID
    )
    assert dino_models.resolve_dino_model_id("custom/repo") == "custom/repo"


def test_cutie_dino_tracker_config_normalizes_model_aliases() -> None:
    cfg = CutieDinoTrackerConfig(patch_model_name="dinov3_vits16plus")

    assert cfg.patch_model_name == "facebook/dinov3-vits16plus-pretrain-lvd1689m"
    assert cfg.dinov3_model_name == cfg.patch_model_name


def test_dino_runtime_helpers_resolve_and_sync_aliases() -> None:
    runtime = {"dinov3_model_name": "dinov3_vitb16"}

    assert (
        dino_models.resolve_dino_model_from_runtime(runtime)
        == "facebook/dinov3-vitb16-pretrain-lvd1426"
    )

    selected = dino_models.set_dino_model_on_runtime(runtime, "dinov3_vitl16")

    assert selected == "facebook/dinov3-vitl16-pretrain-lvd1689m"
    assert runtime["patch_model_name"] == selected
    assert runtime["dinov3_model_name"] == selected


def test_dino_runtime_resolver_prefers_non_default_alias() -> None:
    runtime = {
        "patch_model_name": dino_models.DEFAULT_DINO_FEATURE_MODEL_ID,
        "dinov3_model_name": "dinov3_vitl16",
    }

    assert (
        dino_models.resolve_dino_model_from_runtime(runtime)
        == "facebook/dinov3-vitl16-pretrain-lvd1689m"
    )


def test_cutie_dino_config_keeps_dinov3_model_when_patch_default() -> None:
    selected = "facebook/dinov3-vitl16-pretrain-lvd1689m"
    cfg = CutieDinoTrackerConfig(dinov3_model_name=selected)

    assert cfg.patch_model_name == selected
    assert cfg.dinov3_model_name == selected


def test_download_dino_model_uses_cache_dir_and_resolved_id(
    monkeypatch, tmp_path: Path
) -> None:
    captured: dict[str, object] = {}

    def fake_snapshot_download(**kwargs):
        captured.update(kwargs)
        return str(tmp_path / "snapshot")

    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub",
        SimpleNamespace(snapshot_download=fake_snapshot_download),
    )

    path = dino_models.download_dino_model(
        "dinov3_vits16",
        cache_dir=tmp_path / "hf-cache",
        local_files_only=True,
        token="token",
    )

    assert path == tmp_path / "snapshot"
    assert captured["repo_id"] == dino_models.DEFAULT_DINOV3_MODEL_ID
    assert captured["cache_dir"] == str(tmp_path / "hf-cache")
    assert captured["local_files_only"] is True
    assert captured["token"] == "token"


def test_engine_cli_dinov3_models_list_outputs_catalog(capsys) -> None:
    assert engine_cli.main(["dinov3-models", "--list", "--dinov3-only"]) == 0
    text = capsys.readouterr().out

    assert dino_models.DEFAULT_DINOV3_MODEL_ID in text
    assert "facebook/dinov2-base" not in text


def test_engine_cli_dinov3_models_download_uses_helper(
    monkeypatch, capsys, tmp_path: Path
) -> None:
    captured: dict[str, object] = {}

    def fake_download(model_name, **kwargs):
        captured["model_name"] = model_name
        captured.update(kwargs)
        return tmp_path / "snapshot"

    monkeypatch.setattr(dino_models, "download_dino_model", fake_download)

    assert (
        engine_cli.main(
            [
                "dinov3-models",
                "--model",
                "dinov3_vitb16",
                "--cache-dir",
                str(tmp_path / "cache"),
                "--local-files-only",
            ]
        )
        == 0
    )
    text = capsys.readouterr().out

    assert "facebook/dinov3-vitb16-pretrain-lvd1426" in text
    assert captured["model_name"] == "facebook/dinov3-vitb16-pretrain-lvd1426"
    assert captured["cache_dir"] == str(tmp_path / "cache")
    assert captured["local_files_only"] is True
