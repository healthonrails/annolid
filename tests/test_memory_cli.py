from annolid.engine import cli as engine_cli
from annolid.engine.registry import get_model, list_models
from annolid.interfaces.memory import cli as memory_cli


def test_engine_cli_dispatches_memory_subcommand(monkeypatch) -> None:
    captured = {}

    def fake_main(argv=None):
        captured["argv"] = argv
        return 0

    monkeypatch.setattr(memory_cli, "main", fake_main)

    assert engine_cli.main(["memory", "stats", "--scope", "global"]) == 0
    assert captured["argv"] == ["stats", "--scope", "global"]


def test_engine_cli_help_alias_normalization() -> None:
    assert engine_cli._normalize_help_argv(["help"]) == ["--help"]  # noqa: SLF001
    assert engine_cli._normalize_help_argv(["help", "train"]) == [  # noqa: SLF001
        "train",
        "--help",
    ]
    assert engine_cli._normalize_help_argv(["help", "annolid-run"]) == [  # noqa: SLF001
        "--help"
    ]
    assert engine_cli._normalize_help_argv(  # noqa: SLF001
        ["help", "annolid-run", "predict"]
    ) == ["predict", "--help"]
    assert engine_cli._normalize_help_argv(  # noqa: SLF001
        ["help", "train", "dino_kpseg"]
    ) == ["train", "dino_kpseg", "--help-model"]
    assert engine_cli._normalize_help_argv(  # noqa: SLF001
        ["help", "annolid-run", "predict", "dino_kpseg"]
    ) == ["predict", "dino_kpseg", "--help-model"]


def test_engine_cli_grouped_root_help_contains_sections() -> None:
    parser = engine_cli._build_root_parser()  # noqa: SLF001
    text = engine_cli._format_root_help(parser)  # noqa: SLF001
    assert "Models:" in text
    assert "Agent:" in text
    assert "Data:" in text
    assert "Utilities:" in text
    assert "annolid-run help predict" in text


def test_engine_cli_plugin_help_includes_quick_reference(capsys) -> None:
    assert engine_cli.main(["help", "predict", "yolo"]) == 0
    text = capsys.readouterr().out
    assert "Quick reference:" in text
    assert "Required inputs:" in text
    assert "Outputs and run location:" in text
    assert "--source" in text
    assert "--weights" in text


def test_engine_cli_train_plugin_help_includes_plugin_description(capsys) -> None:
    assert engine_cli.main(["help", "train", "dino_kpseg"]) == 0
    text = capsys.readouterr().out
    assert "DINOv3 feature + small conv head for keypoint mask segmentation." in text
    assert "Training controls:" in text
    assert "Data and augmentation:" in text
    assert "Model and features:" in text
    assert "Losses and reporting:" in text


def test_engine_cli_predict_plugin_help_prefers_plugin_sections(capsys) -> None:
    assert engine_cli.main(["help", "predict", "yolo"]) == 0
    text = capsys.readouterr().out
    assert "Inference controls:" in text
    assert "Saving and reporting:" in text
    assert "Training controls:" not in text


def test_builtin_model_plugins_define_explicit_help_sections() -> None:
    for info in list_models():
        plugin = get_model(info.name)
        if info.supports_train:
            assert plugin.get_help_sections("train"), info.name
        if info.supports_predict:
            assert plugin.get_help_sections("predict"), info.name
