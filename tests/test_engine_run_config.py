from __future__ import annotations

import argparse

from annolid.engine.models.dino_kpseg import DinoKPSEGPlugin
from annolid.engine.run_config import (
    build_cli_args_from_config,
    expand_argv_with_run_config,
    find_run_config_path,
    select_model_mode_config,
)


def test_find_run_config_path_variants():
    assert find_run_config_path(["--run-config", "a.yaml"]) == "a.yaml"
    assert find_run_config_path(["--run-config=b.yaml"]) == "b.yaml"
    assert find_run_config_path(["--data", "x"]) is None


def test_select_model_mode_config_prefers_model_scoped_train():
    payload = {
        "models": {
            "dino_kpseg": {
                "train": {
                    "epochs": 9,
                    "batch": 2,
                }
            }
        },
        "train": {"epochs": 1},
    }
    selected = select_model_mode_config(payload, model_name="dino_kpseg", mode="train")
    assert selected["epochs"] == 9
    assert selected["batch"] == 2


def test_build_cli_args_from_config_boolean_and_scalar():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--augment", dest="augment", action="store_true")
    group.add_argument("--no-augment", dest="augment", action="store_false")
    parser.add_argument("--epochs", type=int, default=1)
    args, unknown = build_cli_args_from_config(
        parser, {"augment": False, "epochs": 5, "unused_key": 7}
    )
    assert "--no-augment" in args
    assert "--epochs" in args
    assert "unused_key" in unknown


def test_expand_argv_with_run_config_for_dino(tmp_path):
    run_config = tmp_path / "run.yaml"
    run_config.write_text(
        "\n".join(
            [
                "models:",
                "  dino_kpseg:",
                "    train:",
                "      data: /tmp/data.yaml",
                "      epochs: 12",
                "      batch: 4",
                "      augment: false",
            ]
        ),
        encoding="utf-8",
    )

    parser = argparse.ArgumentParser()
    DinoKPSEGPlugin().add_train_args(parser)
    argv = ["--run-config", str(run_config), "--epochs", "20"]
    resolved = expand_argv_with_run_config(
        parser=parser,
        argv=argv,
        model_name="dino_kpseg",
        mode="train",
    )
    parsed = parser.parse_args(list(resolved))
    assert str(parsed.data) == "/tmp/data.yaml"
    assert int(parsed.epochs) == 20
    assert int(parsed.batch) == 4
    assert bool(parsed.augment) is False
