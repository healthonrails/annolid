from pathlib import Path

from annolid.utils.log_paths import (
    APP_LOGS_DIRNAME,
    APP_REALTIME_LOGS_SUBDIR,
    resolve_annolid_logs_root,
    resolve_annolid_realtime_logs_root,
)


def test_resolve_annolid_logs_root_from_dataset(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    expected = (dataset_root / APP_LOGS_DIRNAME).resolve()
    assert resolve_annolid_logs_root(dataset_root) == expected


def test_resolve_realtime_logs_root_from_env(monkeypatch, tmp_path: Path) -> None:
    custom = tmp_path / "rt_logs"
    monkeypatch.setenv("ANNOLID_REALTIME_LOG_DIR", str(custom))
    assert resolve_annolid_realtime_logs_root() == custom.resolve()


def test_resolve_realtime_logs_root_defaults_under_logs(monkeypatch) -> None:
    monkeypatch.delenv("ANNOLID_REALTIME_LOG_DIR", raising=False)
    root = resolve_annolid_realtime_logs_root()
    assert root == (resolve_annolid_logs_root() / APP_REALTIME_LOGS_SUBDIR).resolve()
