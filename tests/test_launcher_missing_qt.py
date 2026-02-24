import pytest

from annolid.gui import launcher


def test_launcher_returns_helpful_error_when_qt_binding_missing(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(launcher, "parse_cli", lambda argv: ({}, object(), False))
    monkeypatch.setattr(launcher, "configure_logging", lambda: None)
    monkeypatch.setattr(launcher, "sanitize_qt_plugin_env", lambda env: None)

    qt_bindings_not_found_error = type("QtBindingsNotFoundError", (Exception,), {})

    def _raise_qt_binding_error(name: str):
        assert name == "annolid.gui.app"
        raise qt_bindings_not_found_error("No Qt bindings could be found")

    monkeypatch.setattr(launcher.importlib, "import_module", _raise_qt_binding_error)

    exit_code = launcher.main([])
    captured = capsys.readouterr()

    assert exit_code == 1
    assert "No Qt binding detected." in captured.err
    assert 'pip install -e ".[gui]"' in captured.err


def test_launcher_reraises_non_qt_import_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(launcher, "parse_cli", lambda argv: ({}, object(), False))
    monkeypatch.setattr(launcher, "configure_logging", lambda: None)
    monkeypatch.setattr(launcher, "sanitize_qt_plugin_env", lambda env: None)

    def _raise_import_error(name: str):
        assert name == "annolid.gui.app"
        raise ImportError("boom")

    monkeypatch.setattr(launcher.importlib, "import_module", _raise_import_error)

    with pytest.raises(ImportError, match="boom"):
        launcher.main([])
