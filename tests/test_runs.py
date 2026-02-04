from pathlib import Path

from annolid.utils.runs import allocate_run_dir, new_run_dir, shared_runs_root


def test_shared_runs_root_defaults_to_home(tmp_path, monkeypatch):
    monkeypatch.delenv("ANNOLID_RUNS_ROOT", raising=False)
    monkeypatch.delenv("ANNOLID_LOG_ROOT", raising=False)
    monkeypatch.delenv("ANNOLID_LOG_DIR", raising=False)
    root = shared_runs_root()
    assert root == (Path.home() / "annolid_logs" / "runs").resolve()


def test_shared_runs_root_env_override(tmp_path, monkeypatch):
    monkeypatch.setenv("ANNOLID_RUNS_ROOT", str(tmp_path / "runs"))
    root = shared_runs_root()
    assert root == (tmp_path / "runs").resolve()


def test_new_run_dir_is_under_root(tmp_path):
    run_dir = new_run_dir(
        task="yolo",
        model="yolov8n",
        runs_root=tmp_path,
        run_name="exp1",
        timestamp="20250101_000000",
    )
    assert run_dir == (tmp_path / "yolo" / "yolov8n" / "exp1").resolve()


def test_allocate_run_dir_is_unique_and_created(tmp_path):
    run_dir_1 = allocate_run_dir(
        task="yolo",
        model="yolov8n",
        runs_root=tmp_path,
        run_name="exp1",
    )
    run_dir_2 = allocate_run_dir(
        task="yolo",
        model="yolov8n",
        runs_root=tmp_path,
        run_name="exp1",
    )
    assert run_dir_1.exists()
    assert run_dir_2.exists()
    assert run_dir_1 != run_dir_2
