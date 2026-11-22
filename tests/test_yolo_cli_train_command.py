from pathlib import Path

from annolid.yolo import ultralytics_cli
from annolid.yolo.ultralytics_cli import (
    YOLOCLICompleted,
    build_yolo_train_command,
    build_yolo_val_command,
    format_yolo_cli_error,
    locate_yolo_cli,
    run_yolo_cli,
)


def test_build_yolo_train_command_matches_ultralytics_syntax():
    cmd = build_yolo_train_command(
        model="yolo11n-pose.pt",
        data="/content/YOLO_dataset/data.yaml",
        epochs=300,
        imgsz=640,
        yolo_cmd=["yolo"],
    )

    assert cmd[0:2] == ["yolo", "train"]
    assert "model=yolo11n-pose.pt" in cmd
    assert "data=/content/YOLO_dataset/data.yaml" in cmd
    assert "epochs=300" in cmd
    assert "imgsz=640" in cmd


def test_build_yolo_train_command_formats_common_options():
    cmd = build_yolo_train_command(
        model="/tmp/weights.pt",
        data="/tmp/data.yaml",
        epochs=1,
        imgsz=320,
        batch=16,
        device="mps",
        plots=True,
        workers=0,
        overrides={"lr0": 0.01, "optimizer": "AdamW"},
        yolo_cmd=["yolo"],
    )

    assert "batch=16" in cmd
    assert "device=mps" in cmd
    assert "plots=True" in cmd
    assert "workers=0" in cmd
    assert "lr0=0.01" in cmd
    assert "optimizer=AdamW" in cmd


def test_build_yolo_val_command_formats_common_options():
    cmd = build_yolo_val_command(
        model="/tmp/weights.pt",
        data="/tmp/data.yaml",
        imgsz=640,
        batch=4,
        device="mps",
        split="test",
        project="/tmp/runs",
        name="eval_pose",
        plots=True,
        save_json=True,
        workers=0,
        overrides={"conf": 0.25},
        yolo_cmd=["yolo"],
    )

    assert cmd[0:2] == ["yolo", "val"]
    assert "model=/tmp/weights.pt" in cmd
    assert "data=/tmp/data.yaml" in cmd
    assert "imgsz=640" in cmd
    assert "batch=4" in cmd
    assert "device=mps" in cmd
    assert "split=test" in cmd
    assert "project=/tmp/runs" in cmd
    assert "name=eval_pose" in cmd
    assert "plots=True" in cmd
    assert "save_json=True" in cmd
    assert "workers=0" in cmd
    assert "conf=0.25" in cmd


def test_command_builders_preserve_numeric_gpu_zero():
    train = build_yolo_train_command(
        model="weights.pt",
        data="data.yaml",
        epochs=1,
        imgsz=320,
        device=0,
        yolo_cmd=["yolo"],
    )
    val = build_yolo_val_command(
        model="weights.pt",
        data="data.yaml",
        device=0,
        yolo_cmd=["yolo"],
    )
    assert "device=0" in train
    assert "device=0" in val


def test_build_yolo_train_command_does_not_allow_overrides_to_clobber_explicit_flags():
    cmd = build_yolo_train_command(
        model="/runs/my_run/weights/best.pt",
        data="/datasets/real/data.yaml",
        epochs=50,
        imgsz=640,
        batch=8,
        device="mps",
        plots=False,
        resume=True,
        freeze=10,
        patience=40,
        cache="ram",
        overrides={
            "model": "yolo11n-seg.pt",
            "data": "/wrong/data.yaml",
            "device": "cpu",
            "plots": True,
            "lr0": 0.002,
            "cos_lr": True,
        },
        yolo_cmd=["yolo"],
    )

    assert [arg for arg in cmd if arg.startswith("model=")] == [
        "model=/runs/my_run/weights/best.pt"
    ]
    assert [arg for arg in cmd if arg.startswith("data=")] == [
        "data=/datasets/real/data.yaml"
    ]
    assert [arg for arg in cmd if arg.startswith("device=")] == ["device=mps"]
    assert [arg for arg in cmd if arg.startswith("plots=")] == ["plots=False"]
    assert "resume=True" in cmd
    assert "freeze=10" in cmd
    assert "patience=40" in cmd
    assert "cache=ram" in cmd
    assert "lr0=0.002" in cmd
    assert "cos_lr=True" in cmd


def test_locate_yolo_cli_prefers_sibling_venv_executable(tmp_path: Path, monkeypatch):
    fake_bin = tmp_path / "venv" / "bin"
    fake_bin.mkdir(parents=True)
    fake_python = fake_bin / "python"
    fake_yolo = fake_bin / "yolo"
    fake_python.write_text("", encoding="utf-8")
    fake_yolo.write_text("#!/bin/sh\n", encoding="utf-8")
    fake_yolo.chmod(0o755)

    monkeypatch.setattr(ultralytics_cli.sys, "executable", str(fake_python))
    monkeypatch.setattr(ultralytics_cli.shutil, "which", lambda _name: None)

    assert locate_yolo_cli() == [str(fake_yolo)]


def test_locate_yolo_cli_falls_back_to_python_entrypoint(tmp_path: Path, monkeypatch):
    fake_bin = tmp_path / "venv" / "bin"
    fake_bin.mkdir(parents=True)
    fake_python = fake_bin / "python"
    fake_python.write_text("", encoding="utf-8")

    monkeypatch.setattr(ultralytics_cli.sys, "executable", str(fake_python))
    monkeypatch.setattr(ultralytics_cli.shutil, "which", lambda _name: None)
    monkeypatch.setattr(
        ultralytics_cli.importlib.util,
        "find_spec",
        lambda name: object() if name == "ultralytics" else None,
    )

    cmd = locate_yolo_cli()
    assert cmd[0] == str(fake_python)
    assert cmd[1] == "-c"
    assert "entrypoint()" in cmd[2]


def test_run_yolo_cli_sets_mps_fallback_on_darwin(monkeypatch):
    import sys

    monkeypatch.setattr(ultralytics_cli.sys, "platform", "darwin")
    monkeypatch.delenv("PYTORCH_ENABLE_MPS_FALLBACK", raising=False)

    completed = run_yolo_cli(
        [
            sys.executable,
            "-c",
            "import os; print(os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK', ''), os.environ.get('MPLBACKEND', ''))",
        ]
    )
    assert completed.returncode == 0
    assert completed.output_tail[-1].strip() == "1 Agg"

    # Explicit user env override is preserved
    completed_override = run_yolo_cli(
        [
            sys.executable,
            "-c",
            "import os; print(os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK', ''))",
        ],
        env={"PYTORCH_ENABLE_MPS_FALLBACK": "0"},
    )
    assert completed_override.returncode == 0
    assert completed_override.output_tail[-1].strip() == "0"


def test_format_yolo_cli_error_includes_mps_and_oom_hints():
    mps_fail = YOLOCLICompleted(
        command=["yolo", "train", "device=mps"],
        returncode=1,
        output_tail=[
            "NotImplementedError: The operator 'aten::foo' is not currently implemented for the MPS device."
        ],
    )
    msg = format_yolo_cli_error(mps_fail, model="yolo11n-seg.pt", device="mps")
    assert "Apple MPS backend error" in msg
    assert "--device cpu" in msg

    oom_fail = YOLOCLICompleted(
        command=["yolo", "train", "device=0"],
        returncode=1,
        output_tail=["torch.OutOfMemoryError: CUDA out of memory."],
    )
    oom_msg = format_yolo_cli_error(oom_fail, model="yolo11x-seg.pt", device="0")
    assert "out-of-memory" in oom_msg.lower()
