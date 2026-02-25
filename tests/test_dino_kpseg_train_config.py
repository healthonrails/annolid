from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from annolid.segmentation.dino_kpseg.dataset_tools import generate_train_config
from annolid.segmentation.dino_kpseg.train import _load_train_config_defaults


def test_load_train_config_defaults_normalizes_hyphen_keys(tmp_path: Path) -> None:
    cfg = tmp_path / "train_cfg.yaml"
    cfg.write_text(
        "\n".join(
            [
                "data: /tmp/data.yaml",
                "data-format: coco",
                "schedule-profile: aggressive_s",
                "",
            ]
        ),
        encoding="utf-8",
    )
    payload = _load_train_config_defaults(cfg)
    assert payload["data"] == "/tmp/data.yaml"
    assert payload["data_format"] == "coco"
    assert payload["schedule_profile"] == "aggressive_s"


def test_generate_train_config_writes_yaml_and_launch_cmd(tmp_path: Path) -> None:
    data_yaml = tmp_path / "data_split.yaml"
    data_yaml.write_text("train: images/train\nval: images/val\n", encoding="utf-8")
    out_dir = tmp_path / "cfg"
    summary = generate_train_config(
        data_yaml=data_yaml,
        output=out_dir,
        schedule_profile="aggressive_s",
        data_format="auto",
        augment=True,
    )
    config_path = Path(summary["config_yaml"])
    assert config_path.exists()
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    assert payload["data"] == str(data_yaml.resolve())
    assert payload["schedule_profile"] == "aggressive_s"
    assert payload["augment"] is True
    assert payload["feature_merge"] == "concat"
    assert int(payload["feature_align_dim"]) == 0
    assert "--config" in str(summary["launch_cmd"])


def test_generate_train_config_rejects_unknown_schedule(tmp_path: Path) -> None:
    data_yaml = tmp_path / "data_split.yaml"
    data_yaml.write_text("train: images/train\nval: images/val\n", encoding="utf-8")
    out_dir = tmp_path / "cfg"
    with pytest.raises(ValueError):
        _ = generate_train_config(
            data_yaml=data_yaml,
            output=out_dir,
            schedule_profile="legacy_s",
        )
