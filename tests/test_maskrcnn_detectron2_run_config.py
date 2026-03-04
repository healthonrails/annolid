from __future__ import annotations

import argparse

from annolid.engine.models.maskrcnn_detectron2 import _resolve_train_settings


def _build_args(**overrides):
    args = argparse.Namespace(
        dataset_dir=None,
        output_dir=None,
        run_config=None,
        max_iterations=None,
        batch_size=None,
        weights=None,
        model_config=None,
        score_threshold=None,
        overlap_threshold=None,
        base_lr=None,
        num_workers=None,
        checkpoint_period=None,
        roi_batch_size_per_image=None,
        sampler_train=None,
        repeat_threshold=None,
    )
    for key, value in overrides.items():
        setattr(args, key, value)
    return args


def test_resolve_train_settings_uses_defaults_with_cli_dataset(tmp_path):
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()

    settings = _resolve_train_settings(_build_args(dataset_dir=str(dataset_dir)))

    assert settings.dataset_dir == str(dataset_dir.resolve())
    assert settings.max_iterations == 3000
    assert settings.batch_size == 8
    assert (
        settings.model_config == "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
    )
    assert settings.base_lr == 0.0025


def test_resolve_train_settings_loads_yaml_and_cli_overrides(tmp_path):
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    output_dir = tmp_path / "outputs"
    run_config = tmp_path / "run.yaml"
    run_config.write_text(
        "\n".join(
            [
                "dataset:",
                f"  path: {dataset_dir}",
                "output:",
                f"  path: {output_dir}",
                "solver:",
                "  max_iterations: 5000",
                "  batch_size: 4",
                "  base_lr: 0.001",
                "dataloader:",
                "  num_workers: 6",
                "detectron2:",
                "  score_threshold: 0.2",
            ]
        ),
        encoding="utf-8",
    )

    settings = _resolve_train_settings(
        _build_args(
            run_config=str(run_config),
            batch_size=16,
        )
    )

    assert settings.dataset_dir == str(dataset_dir.resolve())
    assert settings.output_dir == str(output_dir.resolve())
    assert settings.max_iterations == 5000
    assert settings.batch_size == 16
    assert settings.base_lr == 0.001
    assert settings.num_workers == 6
    assert settings.score_threshold == 0.2


def test_resolve_train_settings_requires_dataset_dir():
    try:
        _resolve_train_settings(_build_args())
    except ValueError as exc:
        assert "dataset_dir is required" in str(exc)
    else:
        raise AssertionError("Expected ValueError when dataset_dir is missing")


def test_resolve_train_settings_default_model_arch_and_no_export(tmp_path):
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()

    settings = _resolve_train_settings(_build_args(dataset_dir=str(dataset_dir)))

    assert settings.model_arch == "mask_rcnn_R_50_FPN_3x"
    assert settings.export_torchscript is False


def test_resolve_train_settings_arch_maps_to_correct_config(tmp_path):
    from annolid.engine.models.maskrcnn_detectron2 import _ARCH_TO_CONFIG

    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()

    settings = _resolve_train_settings(
        _build_args(dataset_dir=str(dataset_dir), model_arch="mask_rcnn_R_101_FPN_3x")
    )

    expected_config = _ARCH_TO_CONFIG["mask_rcnn_R_101_FPN_3x"]
    assert settings.model_config == expected_config
    assert settings.model_arch == "mask_rcnn_R_101_FPN_3x"


def test_resolve_train_settings_explicit_model_config_overrides_arch(tmp_path):
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    custom_config = "MyOrg/custom_model.yaml"

    settings = _resolve_train_settings(
        _build_args(
            dataset_dir=str(dataset_dir),
            model_arch="vitdet_b",
            model_config=custom_config,
        )
    )

    # explicit --model-config should take precedence over --model-arch
    assert settings.model_config == custom_config
