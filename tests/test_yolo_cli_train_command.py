from annolid.yolo.ultralytics_cli import (
    build_yolo_train_command,
    build_yolo_val_command,
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
