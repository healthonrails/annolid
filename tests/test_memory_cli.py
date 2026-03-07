from annolid.engine import cli as engine_cli
from annolid.interfaces.memory import cli as memory_cli


def test_engine_cli_dispatches_memory_subcommand(monkeypatch) -> None:
    captured = {}

    def fake_main(argv=None):
        captured["argv"] = argv
        return 0

    monkeypatch.setattr(memory_cli, "main", fake_main)

    assert engine_cli.main(["memory", "stats", "--scope", "global"]) == 0
    assert captured["argv"] == ["stats", "--scope", "global"]
