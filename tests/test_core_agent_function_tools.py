from __future__ import annotations

import asyncio
import json
import shlex
import shutil
import subprocess
import sys
import types
from datetime import datetime, timedelta, timezone
from pathlib import Path

import cv2
import numpy as np
import pytest
import re
from PIL import Image
import yaml

from annolid.core.agent.config import BoxToolConfig, CalendarToolConfig
from annolid.core.agent.tools.function_base import FunctionTool
from annolid.core.agent.tools.function_builtin import (
    AnnolidDatasetInspectTool,
    AnnolidDatasetPrepareTool,
    AnnolidEvalReportTool,
    AnnolidEvalStartTool,
    AnnolidNoveltyCheckTool,
    AnnolidPaperRunReportTool,
    AnnolidRunTool,
    AnnolidTrainHelpTool,
    AnnolidTrainModelsTool,
    AnnolidTrainStartTool,
    BibtexListEntriesTool,
    BibtexRemoveEntryTool,
    BibtexUpsertEntryTool,
    CodeExplainTool,
    CodeSearchTool,
    CronTool,
    DownloadPdfTool,
    DownloadUrlTool,
    EditFileTool,
    ExecProcessTool,
    ExecStartTool,
    SandboxedExecTool,
    ExtractPdfImagesTool,
    ExtractPdfTextTool,
    GitCliTool,
    GitDiffTool,
    GitHubCliTool,
    GitHubPrChecksTool,
    GitHubPrStatusTool,
    GitLogTool,
    GitStatusTool,
    ListDirTool,
    MemoryGetTool,
    MemorySetTool,
    MemorySearchTool,
    OpenPdfTool,
    ReadFileTool,
    RenameFileTool,
    WebSearchTool,
    WriteFileTool,
    register_nanobot_style_tools,
)
from annolid.core.agent.tools.function_video import (
    VideoInfoTool,
    VideoListInferenceModelsTool,
    VideoProcessSegmentsTool,
    VideoRunModelInferenceTool,
    VideoSampleFramesTool,
    VideoSegmentTool,
)
from annolid.core.agent.tools.function_sam3 import Sam3AgentVideoTrackTool
from annolid.core.agent.tools.function_gui import register_annolid_gui_tools
from annolid.core.agent.tools.function_registry import FunctionToolRegistry
from annolid.core.agent.tools.mcp import MCPToolWrapper


class _EchoTool(FunctionTool):
    @property
    def name(self) -> str:
        return "echo"

    @property
    def description(self) -> str:
        return "Echo text."

    @property
    def parameters(self) -> dict[str, object]:
        return {
            "type": "object",
            "properties": {"text": {"type": "string"}},
            "required": ["text"],
        }

    async def execute(self, **kwargs) -> str:
        return str(kwargs.get("text", ""))


def _write_test_video(path: Path, *, fps: float = 10.0, frames: int = 8) -> None:
    width, height = 64, 48
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")
    writer = cv2.VideoWriter(str(path), fourcc, float(fps), (width, height))
    if not writer.isOpened():
        pytest.skip("OpenCV VideoWriter is not available in this environment.")
    try:
        for idx in range(int(frames)):
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            frame[..., 0] = (idx * 20) % 255
            writer.write(frame)
    finally:
        writer.release()


def _write_labelme_example(
    root: Path,
    *,
    stem: str,
    split: str = "",
    include_point: bool = True,
    include_polygon: bool = True,
) -> tuple[Path, Path]:
    base = root / split if split else root
    base.mkdir(parents=True, exist_ok=True)
    image_path = base / f"{stem}.png"
    Image.new("RGB", (64, 64), color=(120, 120, 120)).save(image_path)
    shapes: list[dict[str, object]] = []
    if include_polygon:
        shapes.append(
            {
                "label": "mouse",
                "shape_type": "polygon",
                "points": [[10, 10], [30, 10], [30, 30], [10, 30]],
                "group_id": 0,
            }
        )
    if include_point:
        shapes.append(
            {
                "label": "nose",
                "shape_type": "point",
                "points": [[20, 20]],
                "group_id": 0,
            }
        )
    json_path = base / f"{stem}.json"
    json_path.write_text(
        json.dumps(
            {
                "version": "5.0.1",
                "imagePath": image_path.name,
                "imageHeight": 64,
                "imageWidth": 64,
                "shapes": shapes,
            }
        ),
        encoding="utf-8",
    )
    return json_path, image_path


def _write_dlc_collected_data_csv(
    csv_path: Path, *, folder: str, image_name: str
) -> None:
    rows = [
        "scorer,,,hyn,hyn",
        "bodyparts,,,nose,nose",
        "coords,,,x,y",
        f"labeled-data,{folder},{image_name},10.0,20.0",
    ]
    csv_path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def _write_coco_detection_dataset(root: Path) -> None:
    images_dir = root / "images"
    annotations_dir = root / "annotations"
    images_dir.mkdir(parents=True, exist_ok=True)
    annotations_dir.mkdir(parents=True, exist_ok=True)

    Image.new("RGB", (64, 48), color=(80, 80, 80)).save(images_dir / "train.png")
    Image.new("RGB", (64, 48), color=(100, 100, 100)).save(images_dir / "val.png")

    categories = [{"id": 1, "name": "mouse"}]
    train_payload = {
        "images": [{"id": 1, "file_name": "train.png", "width": 64, "height": 48}],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "bbox": [8, 8, 16, 12],
                "area": 192,
                "iscrowd": 0,
            }
        ],
        "categories": categories,
    }
    val_payload = {
        "images": [{"id": 2, "file_name": "val.png", "width": 64, "height": 48}],
        "annotations": [
            {
                "id": 2,
                "image_id": 2,
                "category_id": 1,
                "bbox": [10, 10, 20, 14],
                "area": 280,
                "iscrowd": 0,
            }
        ],
        "categories": categories,
    }
    (annotations_dir / "instances_train.json").write_text(
        json.dumps(train_payload),
        encoding="utf-8",
    )
    (annotations_dir / "instances_val.json").write_text(
        json.dumps(val_payload),
        encoding="utf-8",
    )


def _write_coco_pose_dataset(root: Path) -> None:
    images_dir = root / "images"
    annotations_dir = root / "annotations"
    images_dir.mkdir(parents=True, exist_ok=True)
    annotations_dir.mkdir(parents=True, exist_ok=True)

    Image.new("RGB", (80, 60), color=(20, 20, 20)).save(images_dir / "train.png")
    Image.new("RGB", (80, 60), color=(40, 40, 40)).save(images_dir / "val.png")

    categories = [
        {
            "id": 1,
            "name": "mouse",
            "keypoints": ["nose", "tail"],
            "skeleton": [[1, 2]],
        }
    ]
    train_payload = {
        "images": [{"id": 1, "file_name": "train.png", "width": 80, "height": 60}],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "bbox": [10, 10, 30, 20],
                "area": 600,
                "iscrowd": 0,
                "num_keypoints": 2,
                "keypoints": [20, 20, 2, 30, 24, 2],
            }
        ],
        "categories": categories,
    }
    val_payload = {
        "images": [{"id": 2, "file_name": "val.png", "width": 80, "height": 60}],
        "annotations": [
            {
                "id": 2,
                "image_id": 2,
                "category_id": 1,
                "bbox": [12, 12, 28, 18],
                "area": 504,
                "iscrowd": 0,
                "num_keypoints": 2,
                "keypoints": [22, 20, 2, 32, 26, 1],
            }
        ],
        "categories": categories,
    }
    (annotations_dir / "person_keypoints_train.json").write_text(
        json.dumps(train_payload),
        encoding="utf-8",
    )
    (annotations_dir / "person_keypoints_val.json").write_text(
        json.dumps(val_payload),
        encoding="utf-8",
    )


def test_function_registry_validate_and_execute() -> None:
    registry = FunctionToolRegistry()
    registry.register(_EchoTool())
    bad = asyncio.run(registry.execute("echo", {"text": 123}))
    assert "Invalid parameters" in bad
    ok = asyncio.run(registry.execute("echo", {"text": "hi"}))
    assert ok == "hi"


def test_annolid_run_tool_executes_in_process_cli(monkeypatch, tmp_path: Path) -> None:
    observed: dict[str, object] = {}

    def _fake_main(argv: list[str]) -> int:
        observed["argv"] = list(argv)
        observed["cwd"] = str(Path.cwd())
        print("annolid-run ok")
        return 0

    monkeypatch.setattr(
        "annolid.core.agent.tools.annolid_run.annolid_run_main", _fake_main
    )

    tool = AnnolidRunTool(allowed_dir=tmp_path)
    result = asyncio.run(
        tool.execute(
            command="annolid-run agent-status",
            working_dir=str(tmp_path),
        )
    )
    payload = json.loads(result)
    assert payload["ok"] is True
    assert payload["exit_code"] == 0
    assert payload["argv"] == ["agent-status"]
    assert payload["working_dir"] == str(tmp_path.resolve())
    assert "annolid-run ok" in payload["stdout"]
    assert observed["argv"] == ["agent-status"]
    assert observed["cwd"] == str(tmp_path.resolve())


def test_annolid_run_tool_blocks_mutation_without_opt_in(tmp_path: Path) -> None:
    tool = AnnolidRunTool(allowed_dir=tmp_path)
    result = asyncio.run(tool.execute(command="annolid-run update run"))
    payload = json.loads(result)
    assert payload["ok"] is False
    assert payload["argv"] == ["update", "run"]
    assert "allow_mutation=true" in payload["error"]


def test_annolid_run_tool_normalizes_help_alias(tmp_path: Path) -> None:
    tool = AnnolidRunTool(allowed_dir=tmp_path)
    assert tool._normalize_argv("annolid-run help", None) == ["--help"]  # noqa: SLF001
    assert tool._normalize_argv("annolid-run help train", None) == [  # noqa: SLF001
        "train",
        "--help",
    ]
    assert tool._normalize_argv(  # noqa: SLF001
        "annolid-run help train dino_kpseg", None
    ) == ["train", "dino_kpseg", "--help-model"]
    assert tool._normalize_argv(  # noqa: SLF001
        "help annolid-run predict dino_kpseg", None
    ) == ["predict", "dino_kpseg", "--help-model"]


def test_register_nanobot_style_tools_includes_annolid_run(tmp_path: Path) -> None:
    registry = FunctionToolRegistry()
    asyncio.run(register_nanobot_style_tools(registry, allowed_dir=tmp_path))
    assert registry.has("annolid_run")
    assert registry.has("annolid_dataset_inspect")
    assert registry.has("annolid_dataset_prepare")
    assert registry.has("annolid_eval_report")
    assert registry.has("annolid_eval_start")
    assert registry.has("annolid_novelty_check")
    assert registry.has("annolid_paper_run_report")
    assert registry.has("annolid_train_models")
    assert registry.has("annolid_train_help")
    assert registry.has("annolid_train_start")


def test_annolid_train_models_lists_dino_and_yolo() -> None:
    tool = AnnolidTrainModelsTool()
    payload = json.loads(asyncio.run(tool.execute()))
    assert payload["ok"] is True
    names = {row["name"]: row for row in payload["models"]}
    assert "dino_kpseg" in names
    assert "yolo" in names
    assert "pose" in names["yolo"]["tasks"]
    assert "dino keypoint segmentation" in names["dino_kpseg"]["aliases"]


def test_annolid_dataset_inspect_reports_raw_labelme_pose_folder(
    tmp_path: Path,
) -> None:
    _write_labelme_example(tmp_path, stem="frame_0001", split="train")
    _write_labelme_example(tmp_path, stem="frame_0002", split="val")

    tool = AnnolidDatasetInspectTool(allowed_dir=tmp_path)
    payload = json.loads(asyncio.run(tool.execute(dataset_folder=str(tmp_path))))
    assert payload["ok"] is True
    dataset = payload["dataset"]
    assert dataset["ready_for_training"] is False
    assert "labelme" in dataset["dataset_kinds"]
    assert dataset["labelme"]["json_files"] == 2
    assert dataset["labelme"]["shape_type_counts"]["point"] >= 1
    assert any(row["model"] == "dino_kpseg" for row in dataset["recommended_models"])
    assert any("annolid_dataset_prepare" in step for step in dataset["next_actions"])


def test_annolid_dataset_prepare_generates_labelme_spec(tmp_path: Path) -> None:
    _write_labelme_example(tmp_path, stem="frame_0001", split="train")
    _write_labelme_example(tmp_path, stem="frame_0002", split="val")

    tool = AnnolidDatasetPrepareTool(allowed_dir=tmp_path)
    payload = json.loads(
        asyncio.run(
            tool.execute(
                dataset_folder=str(tmp_path),
                mode="labelme_spec",
                allow_mutation=True,
                val_size=0.5,
                test_size=0.0,
            )
        )
    )
    assert payload["ok"] is True
    result = payload["result"]
    assert Path(result["spec_path"]).exists()
    assert result["split_counts"]["train"] >= 1
    spec_payload = yaml.safe_load(Path(result["spec_path"]).read_text(encoding="utf-8"))
    assert spec_payload["format"] == "labelme"
    assert spec_payload["kpt_shape"][0] >= 1


def test_annolid_dataset_prepare_exports_yolo_dataset(tmp_path: Path) -> None:
    _write_labelme_example(
        tmp_path,
        stem="frame_0001",
        split="train",
        include_point=False,
        include_polygon=True,
    )
    _write_labelme_example(
        tmp_path,
        stem="frame_0002",
        split="val",
        include_point=False,
        include_polygon=True,
    )

    tool = AnnolidDatasetPrepareTool(allowed_dir=tmp_path)
    payload = json.loads(
        asyncio.run(
            tool.execute(
                dataset_folder=str(tmp_path),
                mode="yolo_from_labelme",
                allow_mutation=True,
                task="segmentation",
                dataset_name="seg_dataset",
            )
        )
    )
    assert payload["ok"] is True
    result = payload["result"]
    assert result["status"] == "ok"
    dataset_dir = Path(result["dataset_dir"])
    assert (dataset_dir / "data.yaml").exists()
    assert result["task"] == "segmentation"


def test_annolid_dataset_inspect_detects_coco_and_yolo_datasets(tmp_path: Path) -> None:
    coco_root = tmp_path / "coco_dataset"
    _write_coco_pose_dataset(coco_root)

    inspect_tool = AnnolidDatasetInspectTool(allowed_dir=tmp_path)
    coco_payload = json.loads(
        asyncio.run(inspect_tool.execute(dataset_folder=str(coco_root)))
    )
    assert coco_payload["ok"] is True
    coco_info = coco_payload["dataset"]
    assert coco_info["ready_for_training"] is False
    assert "coco" in coco_info["dataset_kinds"]
    assert coco_info["inferred_formats"] == ["coco"]
    assert coco_info["coco_spec"]["format"] == "coco"
    assert any(row["model"] == "dino_kpseg" for row in coco_info["recommended_models"])
    assert any(row["task"] == "pose" for row in coco_info["recommended_models"])
    assert any(
        "auto-stage an inferred COCO spec" in step for step in coco_info["next_actions"]
    )

    yolo_root = tmp_path / "yolo_dataset"
    yolo_root.mkdir()
    (yolo_root / "data.yaml").write_text(
        yaml.safe_dump(
            {
                "path": str(yolo_root),
                "train": "images/train",
                "val": "images/val",
                "nc": 1,
                "names": ["mouse"],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    yolo_payload = json.loads(
        asyncio.run(inspect_tool.execute(dataset_folder=str(yolo_root)))
    )
    assert yolo_payload["ok"] is True
    yolo_info = yolo_payload["dataset"]
    assert yolo_info["ready_for_training"] is True
    assert "yolo" in yolo_info["dataset_kinds"]
    assert yolo_info["yolo"]["path"] == str((yolo_root / "data.yaml").resolve())


def test_annolid_dataset_inspect_detects_deeplabcut_folder(tmp_path: Path) -> None:
    dataset = tmp_path / "dlc_dataset"
    labeled = dataset / "labeled-data" / "seg-1"
    labeled.mkdir(parents=True)
    image = labeled / "img00001.png"
    Image.new("RGB", (100, 80), color=(10, 20, 30)).save(image)
    _write_dlc_collected_data_csv(
        labeled / "CollectedData_test.csv",
        folder="seg-1",
        image_name="img00001.png",
    )

    tool = AnnolidDatasetInspectTool(allowed_dir=tmp_path)
    payload = json.loads(asyncio.run(tool.execute(dataset_folder=str(dataset))))
    assert payload["ok"] is True
    info = payload["dataset"]
    assert "deeplabcut" in info["dataset_kinds"]
    assert "deeplabcut" in info["external_formats"]
    assert any(row["task"] == "pose" for row in info["recommended_models"])


def test_annolid_dataset_prepare_imports_deeplabcut_training_data(
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "dlc_dataset"
    labeled = dataset / "labeled-data" / "seg-1"
    labeled.mkdir(parents=True)
    image = labeled / "img00001.png"
    Image.new("RGB", (100, 80), color=(10, 20, 30)).save(image)
    _write_dlc_collected_data_csv(
        labeled / "CollectedData_test.csv",
        folder="seg-1",
        image_name="img00001.png",
    )

    tool = AnnolidDatasetPrepareTool(allowed_dir=tmp_path)
    payload = json.loads(
        asyncio.run(
            tool.execute(
                dataset_folder=str(dataset),
                mode="deeplabcut_import",
                allow_mutation=True,
                write_pose_schema=True,
                write_index=True,
            )
        )
    )
    assert payload["ok"] is True
    result = payload["result"]
    assert result["json_written"] == 1
    assert Path(result["index_file"]).exists()
    assert result["index_summary"]["appended"] == 1
    assert (labeled / "img00001.json").exists()


def test_annolid_dataset_prepare_writes_coco_spec_and_materializes_yolo(
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "coco_dataset"
    _write_coco_detection_dataset(dataset)

    tool = AnnolidDatasetPrepareTool(allowed_dir=tmp_path)
    spec_payload = json.loads(
        asyncio.run(
            tool.execute(
                dataset_folder=str(dataset),
                mode="coco_spec",
                output_dir=str(dataset / "derived"),
                allow_mutation=True,
            )
        )
    )
    assert spec_payload["ok"] is True
    spec_path = Path(spec_payload["result"]["spec_path"])
    assert spec_path.exists()
    assert spec_path.name == "coco_spec.yaml"
    assert spec_payload["result"]["task"] == "detect"

    yolo_payload = json.loads(
        asyncio.run(
            tool.execute(
                dataset_folder=str(dataset),
                mode="coco_to_yolo",
                output_dir=str(dataset / "yolo_export"),
                allow_mutation=True,
            )
        )
    )
    assert yolo_payload["ok"] is True
    data_yaml = Path(yolo_payload["result"]["data_yaml"])
    assert data_yaml.exists()
    yolo_cfg = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    assert yolo_cfg["names"] == ["mouse"]
    assert (data_yaml.parent / "images" / "train").exists()
    assert (data_yaml.parent / "images" / "val").exists()


def test_annolid_train_help_delegates_to_annolid_run(
    monkeypatch, tmp_path: Path
) -> None:
    observed: dict[str, object] = {}

    async def _fake_execute(self, **kwargs):
        observed["kwargs"] = kwargs
        return json.dumps({"ok": True, "stdout": "train help"})

    monkeypatch.setattr(AnnolidRunTool, "execute", _fake_execute)
    tool = AnnolidTrainHelpTool(allowed_dir=tmp_path)
    payload = json.loads(asyncio.run(tool.execute(model="dino_kpseg")))
    assert payload["ok"] is True
    assert observed["kwargs"] == {"argv": ["train", "dino_kpseg", "--help-model"]}


def test_annolid_train_start_blocks_without_mutation(tmp_path: Path) -> None:
    tool = AnnolidTrainStartTool(allowed_dir=tmp_path)
    payload = json.loads(
        asyncio.run(
            tool.execute(
                model="yolo",
                data=str(tmp_path / "data.yaml"),
            )
        )
    )
    assert payload["ok"] is False
    assert "allow_mutation=true" in payload["error"]


def test_annolid_train_start_launches_yolo_pose_session(
    monkeypatch, tmp_path: Path
) -> None:
    data_yaml = tmp_path / "data.yaml"
    data_yaml.write_text("train: images/train\nval: images/val\n", encoding="utf-8")
    project_dir = tmp_path / "runs"

    class _FakeManager:
        def __init__(self) -> None:
            self.command = ""
            self.cwd = ""
            self.timeout_s = 0.0

        async def start(self, *, command: str, cwd: str, timeout_s: float = 0.0):
            self.command = command
            self.cwd = cwd
            self.timeout_s = timeout_s
            return types.SimpleNamespace(session_id="sh_train123")

        async def poll(self, session_id: str, *, wait_ms: int = 0):
            return {
                "ok": True,
                "running": False,
                "status": "completed",
                "return_code": 0,
                "session_id": session_id,
                "wait_ms": wait_ms,
            }

        async def log(self, session_id: str, *, tail_lines: int = 200):
            return {
                "ok": True,
                "session_id": session_id,
                "text": "done",
                "tail_lines": tail_lines,
            }

    manager = _FakeManager()
    monkeypatch.setattr(
        "annolid.core.agent.tools.training.get_shell_session_manager",
        lambda: manager,
    )
    monkeypatch.setattr(
        "annolid.core.agent.tools.training._resolve_training_python",
        lambda working_dir: str(working_dir / ".venv" / "bin" / "python"),
    )

    tool = AnnolidTrainStartTool(allowed_dir=tmp_path)
    payload = json.loads(
        asyncio.run(
            tool.execute(
                model="yolo",
                task="pose",
                data=str(data_yaml),
                output_dir=str(project_dir),
                epochs=5,
                batch=2,
                imgsz=640,
                allow_mutation=True,
            )
        )
    )
    assert payload["ok"] is True
    assert payload["task"] == "pose"
    assert payload["session_id"] == "sh_train123"
    assert "--weights" in payload["argv"]
    assert "yolo11n-pose.pt" in payload["argv"]
    assert manager.cwd == str(tmp_path.resolve())
    assert "train yolo" in manager.command
    assert "yolo11n-pose.pt" in manager.command


def test_annolid_train_start_maps_dino_short_side(monkeypatch, tmp_path: Path) -> None:
    data_yaml = tmp_path / "data.yaml"
    data_yaml.write_text("train: images/train\nval: images/val\n", encoding="utf-8")

    class _FakeManager:
        async def start(self, *, command: str, cwd: str, timeout_s: float = 0.0):
            return types.SimpleNamespace(session_id="sh_dino456")

        async def poll(self, session_id: str, *, wait_ms: int = 0):
            return {
                "ok": True,
                "running": False,
                "status": "completed",
                "return_code": 0,
            }

        async def log(self, session_id: str, *, tail_lines: int = 200):
            return {"ok": True, "text": ""}

    monkeypatch.setattr(
        "annolid.core.agent.tools.training.get_shell_session_manager",
        lambda: _FakeManager(),
    )

    tool = AnnolidTrainStartTool(allowed_dir=tmp_path)
    payload = json.loads(
        asyncio.run(
            tool.execute(
                model="dino_kpseg",
                data=str(data_yaml),
                short_side=896,
                extra_args=["--augment", "--schedule-profile", "aggressive_s"],
                allow_mutation=True,
            )
        )
    )
    assert payload["ok"] is True
    assert payload["argv"][0:2] == ["train", "dino_kpseg"]
    assert "--short-side" in payload["argv"]
    assert "896" in payload["argv"]


def test_annolid_train_start_resolves_dataset_folder(
    monkeypatch, tmp_path: Path
) -> None:
    _write_labelme_example(tmp_path, stem="frame_0001", split="train")
    _write_labelme_example(tmp_path, stem="frame_0002", split="val")
    prepare = AnnolidDatasetPrepareTool(allowed_dir=tmp_path)
    prepared = json.loads(
        asyncio.run(
            prepare.execute(
                dataset_folder=str(tmp_path),
                mode="labelme_spec",
                allow_mutation=True,
            )
        )
    )
    assert prepared["ok"] is True

    class _FakeManager:
        async def start(self, *, command: str, cwd: str, timeout_s: float = 0.0):
            return types.SimpleNamespace(session_id="sh_train_dataset")

        async def poll(self, session_id: str, *, wait_ms: int = 0):
            return {
                "ok": True,
                "running": False,
                "status": "completed",
                "return_code": 0,
            }

        async def log(self, session_id: str, *, tail_lines: int = 200):
            return {"ok": True, "text": ""}

    monkeypatch.setattr(
        "annolid.core.agent.tools.training.get_shell_session_manager",
        lambda: _FakeManager(),
    )

    tool = AnnolidTrainStartTool(allowed_dir=tmp_path)
    payload = json.loads(
        asyncio.run(
            tool.execute(
                model="dino_kpseg",
                dataset_folder=str(tmp_path),
                short_side=640,
                allow_mutation=True,
            )
        )
    )
    assert payload["ok"] is True
    assert "--data" in payload["argv"]
    assert str((tmp_path / "labelme_spec.yaml").resolve()) in payload["argv"]


def test_annolid_train_start_resolves_coco_dataset_folder_for_dino(
    monkeypatch, tmp_path: Path
) -> None:
    dataset = tmp_path / "coco_pose"
    _write_coco_pose_dataset(dataset)

    class _FakeManager:
        async def start(self, *, command: str, cwd: str, timeout_s: float = 0.0):
            return types.SimpleNamespace(session_id="sh_train_coco")

        async def poll(self, session_id: str, *, wait_ms: int = 0):
            return {
                "ok": True,
                "running": False,
                "status": "completed",
                "return_code": 0,
            }

        async def log(self, session_id: str, *, tail_lines: int = 200):
            return {"ok": True, "text": ""}

    monkeypatch.setattr(
        "annolid.core.agent.tools.training.get_shell_session_manager",
        lambda: _FakeManager(),
    )

    tool = AnnolidTrainStartTool(allowed_dir=tmp_path)
    payload = json.loads(
        asyncio.run(
            tool.execute(
                model="dino_kpseg",
                dataset_folder=str(dataset),
                allow_mutation=True,
            )
        )
    )
    assert payload["ok"] is True
    staged_spec = tmp_path / ".annolid" / "agent_cache" / "datasets"
    assert "--data" in payload["argv"]
    resolved_data = Path(payload["argv"][payload["argv"].index("--data") + 1])
    assert resolved_data.exists()
    assert staged_spec in resolved_data.parents
    assert "--data-format" in payload["argv"]
    assert "coco" in payload["argv"]


def test_annolid_train_start_rejects_coco_dataset_folder_for_yolo(
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "coco_detect"
    _write_coco_detection_dataset(dataset)
    (dataset / "coco_spec.yaml").write_text(
        yaml.safe_dump(
            {
                "format": "coco",
                "path": str(dataset.resolve()),
                "image_root": "images",
                "train": "annotations/instances_train.json",
                "val": "annotations/instances_val.json",
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    tool = AnnolidTrainStartTool(allowed_dir=tmp_path)
    payload = json.loads(
        asyncio.run(
            tool.execute(
                model="yolo",
                dataset_folder=str(dataset),
                allow_mutation=True,
            )
        )
    )
    assert payload["ok"] is False
    assert "mode=coco_to_yolo" in payload["error"]


def test_annolid_train_start_maps_behavior_classifier_args(
    monkeypatch, tmp_path: Path
) -> None:
    video_dir = tmp_path / "videos"
    video_dir.mkdir()
    ckpt_dir = tmp_path / "checkpoints"
    tb_dir = tmp_path / "tensorboard"

    class _FakeManager:
        async def start(self, *, command: str, cwd: str, timeout_s: float = 0.0):
            return types.SimpleNamespace(session_id="sh_behavior789")

        async def poll(self, session_id: str, *, wait_ms: int = 0):
            return {
                "ok": True,
                "running": False,
                "status": "completed",
                "return_code": 0,
            }

        async def log(self, session_id: str, *, tail_lines: int = 200):
            return {"ok": True, "text": ""}

    monkeypatch.setattr(
        "annolid.core.agent.tools.training.get_shell_session_manager",
        lambda: _FakeManager(),
    )

    tool = AnnolidTrainStartTool(allowed_dir=tmp_path)
    payload = json.loads(
        asyncio.run(
            tool.execute(
                model="behavior_classifier",
                video_folder=str(video_dir),
                checkpoint_dir=str(ckpt_dir),
                tensorboard_log_dir=str(tb_dir),
                batch_size=4,
                epochs=12,
                learning_rate=0.0005,
                validation_split=0.3,
                feature_backbone="dinov3",
                dinov3_model_name="facebook/dinov3-vits16-pretrain-lvd1689m",
                unfreeze_dinov3=True,
                allow_mutation=True,
            )
        )
    )
    assert payload["ok"] is True
    assert "--video-folder" in payload["argv"]
    assert str(video_dir.resolve()) in payload["argv"]
    assert "--checkpoint-dir" in payload["argv"]
    assert str(ckpt_dir.resolve()) in payload["argv"]
    assert "--tensorboard-log-dir" in payload["argv"]
    assert "--batch-size" in payload["argv"]
    assert "--learning-rate" in payload["argv"]
    assert "--validation-split" in payload["argv"]
    assert "--unfreeze-dinov3" in payload["argv"]


def test_annolid_train_start_maps_detectron_dataset_args(
    monkeypatch, tmp_path: Path
) -> None:
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    output_dir = tmp_path / "outputs"

    class _FakeManager:
        async def start(self, *, command: str, cwd: str, timeout_s: float = 0.0):
            return types.SimpleNamespace(session_id="sh_detectron012")

        async def poll(self, session_id: str, *, wait_ms: int = 0):
            return {
                "ok": True,
                "running": False,
                "status": "completed",
                "return_code": 0,
            }

        async def log(self, session_id: str, *, tail_lines: int = 200):
            return {"ok": True, "text": ""}

    monkeypatch.setattr(
        "annolid.core.agent.tools.training.get_shell_session_manager",
        lambda: _FakeManager(),
    )

    tool = AnnolidTrainStartTool(allowed_dir=tmp_path)
    payload = json.loads(
        asyncio.run(
            tool.execute(
                model="maskrcnn_detectron2",
                dataset_dir=str(dataset_dir),
                output_dir=str(output_dir),
                max_iterations=100,
                batch_size=2,
                base_lr=0.001,
                num_workers=0,
                checkpoint_period=25,
                score_threshold=0.5,
                overlap_threshold=0.8,
                model_arch="maskrcnn_resnet50_fpn_v2",
                allow_mutation=True,
            )
        )
    )
    assert payload["ok"] is True
    assert "--dataset-dir" in payload["argv"]
    assert str(dataset_dir.resolve()) in payload["argv"]
    assert "--output-dir" in payload["argv"]
    assert str(output_dir.resolve()) in payload["argv"]
    assert "--max-iterations" in payload["argv"]
    assert "--batch-size" in payload["argv"]
    assert "--base-lr" in payload["argv"]
    assert "--num-workers" in payload["argv"]
    assert "--checkpoint-period" in payload["argv"]
    assert "--score-threshold" in payload["argv"]
    assert "--overlap-threshold" in payload["argv"]
    assert "--model-arch" in payload["argv"]


def test_annolid_eval_report_builds_dino_report_from_summary_json(
    tmp_path: Path,
) -> None:
    metrics_path = tmp_path / "dino_eval.json"
    metrics_path.write_text(
        json.dumps(
            {
                "images_total": 10,
                "images_used": 10,
                "instances_total": 10,
                "keypoints_visible_total": 100,
                "mean_error_px": 2.25,
                "pck": {"4.0": 0.7, "8.0": 0.9},
                "pck_counts": {"4.0": 70, "8.0": 90},
            }
        ),
        encoding="utf-8",
    )
    tool = AnnolidEvalReportTool(allowed_dir=tmp_path)
    payload = json.loads(
        asyncio.run(
            tool.execute(
                path=str(metrics_path),
                model_family="dino_kpseg",
                dataset_name="mouse_pose",
                model_name="dino_best",
                split="test",
            )
        )
    )
    assert payload["ok"] is True
    report = payload["report"]
    assert report["metadata"]["model_family"] == "dino_kpseg"
    assert "PCK@4px" in report["paper_table"]["markdown"]
    assert "95% CI" in report["paper_table"]["markdown"]
    assert report["quality_status"] == "pass"
    assert any(
        item["id"] == "pck_ci_coverage" and item["status"] == "pass"
        for item in report["quality_checks"]
    )


def test_annolid_eval_report_builds_yolo_report_and_writes_files(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "yolo_run"
    run_dir.mkdir()
    (run_dir / "results.csv").write_text(
        "\n".join(
            [
                "epoch,metrics/precision(P),metrics/recall(P),metrics/mAP50(P),metrics/mAP50-95(P)",
                "0,0.2,0.3,0.4,0.25",
                "1,0.5,0.6,0.7,0.55",
            ]
        ),
        encoding="utf-8",
    )
    (run_dir / "results.png").write_bytes(b"fake")
    out_dir = tmp_path / "reports"
    tool = AnnolidEvalReportTool(allowed_dir=tmp_path)
    blocked = json.loads(
        asyncio.run(
            tool.execute(
                path=str(run_dir),
                model_family="yolo",
                dataset_name="mouse_pose",
                model_name="yolo_pose_best",
                split="test",
                report_dir=str(out_dir),
            )
        )
    )
    assert blocked["ok"] is False
    assert "allow_mutation=true" in blocked["error"]

    payload = json.loads(
        asyncio.run(
            tool.execute(
                path=str(run_dir),
                model_family="yolo",
                dataset_name="mouse_pose",
                model_name="yolo_pose_best",
                split="test",
                report_dir=str(out_dir),
                report_basename="paper_ready",
                allow_mutation=True,
            )
        )
    )
    assert payload["ok"] is True
    report = payload["report"]
    assert report["metadata"]["model_family"] == "yolo"
    assert "mAP@50-95" in report["paper_table"]["markdown"]
    assert report["quality_status"] == "warn"
    assert any(
        item["id"] == "prediction_json_present" and item["status"] == "warn"
        for item in report["quality_checks"]
    )
    assert any(
        item["id"] == "ci_coverage" and item["status"] == "warn"
        for item in report["quality_checks"]
    )
    written = report["written_files"]
    assert Path(written["json"]).exists()
    assert Path(written["markdown"]).exists()
    assert Path(written["csv"]).exists()
    assert Path(written["latex"]).exists()


def test_annolid_eval_report_builds_yolo_report_with_bootstrap_ci(
    tmp_path: Path,
) -> None:
    dataset_root = tmp_path / "dataset"
    annotations_dir = dataset_root / "annotations"
    annotations_dir.mkdir(parents=True)
    run_dir = tmp_path / "yolo_run"
    run_dir.mkdir()
    data_yaml = dataset_root / "data.yaml"
    data_yaml.write_text(
        "path: .\ntrain: images/train\nval: images/val\ntest: images/test\n",
        encoding="utf-8",
    )
    (annotations_dir / "instances_test.json").write_text(
        json.dumps(
            {
                "images": [
                    {"id": 1, "width": 64, "height": 64, "file_name": "img1.jpg"}
                ],
                "annotations": [
                    {
                        "id": 1,
                        "image_id": 1,
                        "category_id": 1,
                        "bbox": [10, 10, 20, 20],
                        "area": 400,
                        "iscrowd": 0,
                    }
                ],
                "categories": [{"id": 1, "name": "mouse"}],
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "results.csv").write_text(
        "\n".join(
            [
                "epoch,metrics/precision(B),metrics/recall(B),metrics/mAP50(B),metrics/mAP50-95(B)",
                "0,1.0,1.0,1.0,1.0",
            ]
        ),
        encoding="utf-8",
    )
    (run_dir / "predictions.json").write_text(
        json.dumps(
            [
                {
                    "image_id": 1,
                    "category_id": 1,
                    "bbox": [10, 10, 20, 20],
                    "score": 0.99,
                }
            ]
        ),
        encoding="utf-8",
    )
    (run_dir / "args.yaml").write_text(
        f"data: {data_yaml}\nsplit: test\n",
        encoding="utf-8",
    )
    tool = AnnolidEvalReportTool(allowed_dir=tmp_path)
    payload = json.loads(
        asyncio.run(
            tool.execute(
                path=str(run_dir),
                model_family="yolo",
                dataset_name="mouse_pose",
                model_name="yolo_detect_best",
                split="test",
                bootstrap_samples=8,
                bootstrap_seed=7,
            )
        )
    )
    assert payload["ok"] is True
    report = payload["report"]
    assert report["metadata"]["annotation_json"].endswith("instances_test.json")
    assert report["metadata"]["prediction_json"].endswith("predictions.json")
    assert report["quality_status"] == "pass"
    assert any(
        item["id"] == "ci_coverage" and item["status"] == "pass"
        for item in report["quality_checks"]
    )
    rows = {row["metric"]: row for row in report["paper_table"]["rows"]}
    assert rows["mAP@50"]["ci95"] != "NA"
    assert rows["mAP@50-95"]["ci95"] != "NA"
    assert "map50_ci95_low" in report["paper_table"]["csv"]
    assert any("predictions.json" in artifact for artifact in report["artifacts"])


def test_annolid_eval_report_builds_behavior_report(tmp_path: Path) -> None:
    run_dir = tmp_path / "behavior_run"
    run_dir.mkdir()
    (run_dir / "metrics.json").write_text(
        json.dumps(
            {
                "test_metrics": {
                    "loss": 0.4,
                    "accuracy": 0.8,
                    "macro_f1": 0.75,
                    "macro_map": 0.77,
                    "per_class": {
                        "grooming": {
                            "precision": 0.8,
                            "recall": 0.7,
                            "f1-score": 0.75,
                            "support": 12,
                        }
                    },
                    "per_class_ap": {"grooming": 0.82},
                },
                "predictions": [
                    {
                        "video_name": "video_0.mpg",
                        "target_index": 0,
                        "predicted_index": 0,
                        "class_probabilities": [0.9, 0.1],
                    },
                    {
                        "video_name": "video_1.mpg",
                        "target_index": 0,
                        "predicted_index": 1,
                        "class_probabilities": [0.2, 0.8],
                    },
                    {
                        "video_name": "video_2.mpg",
                        "target_index": 1,
                        "predicted_index": 1,
                        "class_probabilities": [0.1, 0.9],
                    },
                    {
                        "video_name": "video_3.mpg",
                        "target_index": 1,
                        "predicted_index": 1,
                        "class_probabilities": [0.3, 0.7],
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    (run_dir / "confusion_matrix.png").write_bytes(b"fake")
    tool = AnnolidEvalReportTool(allowed_dir=tmp_path)
    payload = json.loads(
        asyncio.run(
            tool.execute(
                path=str(run_dir),
                model_family="behavior_classifier",
                dataset_name="behavior_benchmark",
                model_name="polygon_frame_classifier",
                split="test",
            )
        )
    )
    assert payload["ok"] is True
    report = payload["report"]
    assert report["metadata"]["model_family"] == "behavior_classifier"
    assert "Macro F1" in report["paper_table"]["markdown"]
    assert report["paper_table"]["rows"][1]["ci95"] != "NA"
    assert "accuracy_ci95_low" in report["paper_table"]["csv"]
    assert report["per_class"][0]["precision_ci95"] != "NA"
    assert report["per_class"][0]["ap_ci95"] != "NA"
    assert report["quality_status"] == "warn"
    assert any(
        item["id"] == "prediction_sample_size" and item["status"] == "warn"
        for item in report["quality_checks"]
    )
    assert "grooming" in report["report_markdown"]
    assert "(" in report["report_markdown"]
    assert any("confusion_matrix.png" in artifact for artifact in report["artifacts"])
    assert report["per_class"][0]["label"] == "grooming"


def test_annolid_eval_report_citation_gate_fails_on_hallucinated_entries(
    tmp_path: Path,
) -> None:
    metrics_path = tmp_path / "dino_eval.json"
    metrics_path.write_text(
        json.dumps(
            {
                "images_total": 10,
                "images_used": 10,
                "instances_total": 10,
                "keypoints_visible_total": 100,
                "mean_error_px": 2.25,
                "pck": {"4.0": 0.7, "8.0": 0.9},
                "pck_counts": {"4.0": 70, "8.0": 90},
            }
        ),
        encoding="utf-8",
    )
    citation_report = tmp_path / "refs_batch.json"
    citation_report.write_text(
        json.dumps(
            {
                "summary": {
                    "total": 3,
                    "counts": {
                        "verified": 1,
                        "suspicious": 1,
                        "hallucinated": 1,
                        "skipped": 0,
                    },
                    "integrity_score": 0.42,
                }
            }
        ),
        encoding="utf-8",
    )
    tool = AnnolidEvalReportTool(allowed_dir=tmp_path)
    payload = json.loads(
        asyncio.run(
            tool.execute(
                path=str(metrics_path),
                model_family="dino_kpseg",
                dataset_name="mouse_pose",
                model_name="dino_best",
                split="test",
                citation_gate=True,
                citation_report_path=str(citation_report),
                citation_hallucinated_max=0,
                citation_suspicious_rate_warn=0.2,
                citation_integrity_min_warn=0.6,
            )
        )
    )
    assert payload["ok"] is True
    report = payload["report"]
    assert report["quality_status"] == "fail"
    assert any(
        item["id"] == "citation_hallucination_gate" and item["status"] == "fail"
        for item in report["quality_checks"]
    )
    assert "## Citation Quality Gate" in report["report_markdown"]
    assert "hallucinated=1" in report["report_markdown"]


def test_annolid_eval_report_citation_gate_required_fails_when_missing_report(
    tmp_path: Path,
) -> None:
    metrics_path = tmp_path / "dino_eval.json"
    metrics_path.write_text(
        json.dumps(
            {
                "images_total": 10,
                "images_used": 10,
                "instances_total": 10,
                "keypoints_visible_total": 100,
                "mean_error_px": 2.25,
                "pck": {"4.0": 0.7, "8.0": 0.9},
                "pck_counts": {"4.0": 70, "8.0": 90},
            }
        ),
        encoding="utf-8",
    )
    tool = AnnolidEvalReportTool(allowed_dir=tmp_path)
    payload = json.loads(
        asyncio.run(
            tool.execute(
                path=str(metrics_path),
                model_family="dino_kpseg",
                dataset_name="mouse_pose",
                model_name="dino_best",
                split="test",
                citation_gate=True,
                citation_gate_required=True,
            )
        )
    )
    assert payload["ok"] is True
    report = payload["report"]
    assert report["quality_status"] == "fail"
    assert any(
        item["id"] == "citation_report_presence" and item["status"] == "fail"
        for item in report["quality_checks"]
    )
    assert "## Citation Quality Gate" in report["report_markdown"]
    assert "report missing" in report["report_markdown"].lower()


def test_annolid_novelty_check_tool_recommends_differentiate(tmp_path: Path) -> None:
    tool = AnnolidNoveltyCheckTool(allowed_dir=tmp_path)
    payload = json.loads(
        asyncio.run(
            tool.execute(
                idea_title="Behavioral segmentation with active priors",
                idea_summary=(
                    "We combine segmentation and behavior classification using "
                    "active-learning priors over mouse interaction videos."
                ),
                related_work=[
                    {
                        "title": "Active learning for behavior classification",
                        "abstract": (
                            "We use active learning for behavior classification in "
                            "mouse social interaction videos."
                        ),
                    },
                    {
                        "title": "Segmentation priors in video models",
                        "abstract": (
                            "Priors improve segmentation for interaction videos with "
                            "self-training and weak supervision."
                        ),
                    },
                ],
                differentiate_overlap_threshold=0.2,
                abort_overlap_threshold=0.85,
            )
        )
    )
    assert payload["ok"] is True
    assert payload["recommendation"] in {"differentiate", "abort"}
    assert payload["coverage_quality"] in {"low", "medium", "high"}


def test_annolid_novelty_check_tool_supports_related_work_json_path(
    tmp_path: Path,
) -> None:
    json_path = tmp_path / "related.json"
    json_path.write_text(
        json.dumps(
            {
                "related_work": [
                    {
                        "title": "Non-overlapping approach",
                        "abstract": "Completely unrelated domain and setup.",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    tool = AnnolidNoveltyCheckTool(allowed_dir=tmp_path)
    payload = json.loads(
        asyncio.run(
            tool.execute(
                idea_summary=(
                    "A closed-loop neuroscience annotation framework with "
                    "human-in-the-loop model adaptation."
                ),
                related_work_json_path=str(json_path),
            )
        )
    )
    assert payload["ok"] is True
    assert payload["scores"]["related_work_count"] == 1


def test_annolid_paper_run_report_merges_eval_citation_and_novelty(
    tmp_path: Path,
) -> None:
    eval_report_path = tmp_path / "eval_report.json"
    eval_report_path.write_text(
        json.dumps(
            {
                "report": {
                    "metadata": {
                        "model": "dino_best",
                        "dataset": "mouse_pose",
                        "split": "test",
                        "model_family": "dino_kpseg",
                        "source_path": str(tmp_path / "dino_eval.json"),
                    },
                    "summary": {"images_total": 12, "images_used": 12},
                    "paper_table": {
                        "markdown": (
                            "| Metric | Value | 95% CI |\\n"
                            "|---|---:|---:|\\n"
                            "| PCK@4px | 0.8120 | [0.73, 0.88] |"
                        ),
                        "rows": [
                            {
                                "metric": "PCK@4px",
                                "value": "0.8120",
                                "ci95": "[0.73, 0.88]",
                            }
                        ],
                    },
                    "quality_status": "pass",
                    "quality_checks": [],
                }
            }
        ),
        encoding="utf-8",
    )
    citation_path = tmp_path / "citation_batch.json"
    citation_path.write_text(
        json.dumps(
            {
                "summary": {
                    "total": 10,
                    "counts": {
                        "verified": 8,
                        "suspicious": 2,
                        "hallucinated": 0,
                        "skipped": 0,
                    },
                    "integrity_score": 0.78,
                }
            }
        ),
        encoding="utf-8",
    )
    novelty_path = tmp_path / "novelty.json"
    novelty_path.write_text(
        json.dumps(
            {
                "ok": True,
                "recommendation": "differentiate",
                "reason": "Overlap indicates adjacent prior art.",
                "coverage_quality": "medium",
                "scores": {
                    "max_overlap": 0.51,
                    "mean_top3_overlap": 0.42,
                    "idea_token_coverage": 0.38,
                    "related_work_count": 6,
                },
            }
        ),
        encoding="utf-8",
    )

    tool = AnnolidPaperRunReportTool(allowed_dir=tmp_path)
    payload = json.loads(
        asyncio.run(
            tool.execute(
                eval_report_json_path=str(eval_report_path),
                citation_report_path=str(citation_path),
                novelty_report_path=str(novelty_path),
            )
        )
    )
    assert payload["ok"] is True
    report = payload["report"]
    assert report["evaluation"]["paper_table"]["rows"][0]["metric"] == "PCK@4px"
    assert report["citation"]["counts"]["suspicious"] == 2
    assert report["novelty"]["recommendation"] == "differentiate"
    assert report["quality_status"] == "warn"
    assert "## Reproducibility Checklist" in report["report_markdown"]


def test_annolid_paper_run_report_requires_eval_report(tmp_path: Path) -> None:
    tool = AnnolidPaperRunReportTool(allowed_dir=tmp_path)
    payload = json.loads(asyncio.run(tool.execute()))
    assert payload["ok"] is False
    assert "eval_report" in payload["error"]


def test_annolid_paper_run_report_blocks_export_when_gate_fails(
    tmp_path: Path,
) -> None:
    eval_report_path = tmp_path / "eval_report.json"
    eval_report_path.write_text(
        json.dumps(
            {
                "report": {
                    "metadata": {
                        "model": "dino_best",
                        "dataset": "mouse_pose",
                        "split": "test",
                        "model_family": "dino_kpseg",
                        "source_path": str(tmp_path / "dino_eval.json"),
                    },
                    "summary": {"images_total": 12, "images_used": 12},
                    "paper_table": {"markdown": "| Metric | Value | 95% CI |"},
                    "quality_status": "pass",
                    "quality_checks": [],
                }
            }
        ),
        encoding="utf-8",
    )
    citation_path = tmp_path / "citation_batch.json"
    citation_path.write_text(
        json.dumps(
            {
                "summary": {
                    "total": 4,
                    "counts": {
                        "verified": 2,
                        "suspicious": 2,
                        "hallucinated": 0,
                        "skipped": 0,
                    },
                    "integrity_score": 0.45,
                }
            }
        ),
        encoding="utf-8",
    )
    novelty_path = tmp_path / "novelty.json"
    novelty_path.write_text(
        json.dumps(
            {
                "ok": True,
                "recommendation": "differentiate",
                "coverage_quality": "low",
                "scores": {
                    "max_overlap": 0.41,
                    "mean_top3_overlap": 0.33,
                    "idea_token_coverage": 0.11,
                    "related_work_count": 2,
                },
            }
        ),
        encoding="utf-8",
    )
    report_dir = tmp_path / "report_out"
    tool = AnnolidPaperRunReportTool(allowed_dir=tmp_path)
    payload = json.loads(
        asyncio.run(
            tool.execute(
                eval_report_json_path=str(eval_report_path),
                citation_report_path=str(citation_path),
                novelty_report_path=str(novelty_path),
                paper_ready_gate=True,
                citation_integrity_floor=0.7,
                novelty_coverage_floor=0.3,
                report_dir=str(report_dir),
                allow_mutation=True,
            )
        )
    )
    assert payload["ok"] is False
    assert "Paper-ready gate failed" in payload["error"]
    assert payload["report"]["paper_ready"] is False
    assert not (report_dir / "paper_run_report.json").exists()


def test_annolid_paper_run_report_writes_export_when_gate_passes(
    tmp_path: Path,
) -> None:
    eval_report_path = tmp_path / "eval_report.json"
    eval_report_path.write_text(
        json.dumps(
            {
                "report": {
                    "metadata": {
                        "model": "dino_best",
                        "dataset": "mouse_pose",
                        "split": "test",
                        "model_family": "dino_kpseg",
                        "source_path": str(tmp_path / "dino_eval.json"),
                    },
                    "summary": {"images_total": 12, "images_used": 12},
                    "paper_table": {"markdown": "| Metric | Value | 95% CI |"},
                    "quality_status": "pass",
                    "quality_checks": [],
                }
            }
        ),
        encoding="utf-8",
    )
    citation_path = tmp_path / "citation_batch.json"
    citation_path.write_text(
        json.dumps(
            {
                "summary": {
                    "total": 4,
                    "counts": {
                        "verified": 4,
                        "suspicious": 0,
                        "hallucinated": 0,
                        "skipped": 0,
                    },
                    "integrity_score": 0.92,
                }
            }
        ),
        encoding="utf-8",
    )
    novelty_path = tmp_path / "novelty.json"
    novelty_path.write_text(
        json.dumps(
            {
                "ok": True,
                "recommendation": "proceed",
                "coverage_quality": "high",
                "scores": {
                    "max_overlap": 0.12,
                    "mean_top3_overlap": 0.1,
                    "idea_token_coverage": 0.58,
                    "related_work_count": 8,
                },
            }
        ),
        encoding="utf-8",
    )
    report_dir = tmp_path / "report_out"
    tool = AnnolidPaperRunReportTool(allowed_dir=tmp_path)
    payload = json.loads(
        asyncio.run(
            tool.execute(
                eval_report_json_path=str(eval_report_path),
                citation_report_path=str(citation_path),
                novelty_report_path=str(novelty_path),
                paper_ready_gate=True,
                citation_integrity_floor=0.7,
                novelty_coverage_floor=0.3,
                report_dir=str(report_dir),
                report_basename="paper_ready",
                allow_mutation=True,
            )
        )
    )
    assert payload["ok"] is True
    report = payload["report"]
    assert report["paper_ready"] is True
    assert report["paper_ready_gate"]["status"] == "pass"
    assert Path(report["written_files"]["json"]).exists()
    assert Path(report["written_files"]["markdown"]).exists()


def test_annolid_paper_run_report_rejects_invalid_gate_thresholds(
    tmp_path: Path,
) -> None:
    eval_report_path = tmp_path / "eval_report.json"
    eval_report_path.write_text(
        json.dumps(
            {
                "report": {
                    "metadata": {"source_path": str(tmp_path / "metrics.json")},
                    "summary": {},
                    "paper_table": {"markdown": "| Metric | Value | 95% CI |"},
                    "quality_status": "pass",
                    "quality_checks": [],
                }
            }
        ),
        encoding="utf-8",
    )
    tool = AnnolidPaperRunReportTool(allowed_dir=tmp_path)
    payload = json.loads(
        asyncio.run(
            tool.execute(
                eval_report_json_path=str(eval_report_path),
                paper_ready_gate=True,
                citation_integrity_floor=1.2,
            )
        )
    )
    assert payload["ok"] is False
    assert "citation_integrity_floor" in payload["error"]


def test_annolid_eval_start_blocks_without_mutation(tmp_path: Path) -> None:
    data_yaml = tmp_path / "data.yaml"
    weights = tmp_path / "best.pt"
    data_yaml.write_text("train: images/train\nval: images/val\n", encoding="utf-8")
    weights.write_text("fake", encoding="utf-8")
    tool = AnnolidEvalStartTool(allowed_dir=tmp_path)
    payload = json.loads(
        asyncio.run(
            tool.execute(
                model_family="dino_kpseg",
                data=str(data_yaml),
                weights=str(weights),
            )
        )
    )
    assert payload["ok"] is False
    assert "allow_mutation=true" in payload["error"]


def test_annolid_eval_start_builds_dino_command(monkeypatch, tmp_path: Path) -> None:
    data_yaml = tmp_path / "data.yaml"
    weights = tmp_path / "best.pt"
    out_json = tmp_path / "eval.json"
    report_dir = tmp_path / "paper"
    data_yaml.write_text("train: images/train\nval: images/val\n", encoding="utf-8")
    weights.write_text("fake", encoding="utf-8")

    class _FakeManager:
        def __init__(self) -> None:
            self.command = ""
            self.cwd = ""

        async def start(self, *, command: str, cwd: str, timeout_s: float = 0.0):
            self.command = command
            self.cwd = cwd
            return types.SimpleNamespace(session_id="sh_eval_dino")

        async def poll(self, session_id: str, *, wait_ms: int = 0):
            return {
                "ok": True,
                "running": False,
                "status": "completed",
                "return_code": 0,
            }

        async def log(self, session_id: str, *, tail_lines: int = 200):
            return {"ok": True, "text": ""}

    manager = _FakeManager()
    monkeypatch.setattr(
        "annolid.core.agent.tools.eval_start.get_shell_session_manager",
        lambda: manager,
    )
    monkeypatch.setattr(
        "annolid.core.agent.tools.eval_start._resolve_training_python",
        lambda cwd: str(cwd / ".venv" / "bin" / "python"),
    )

    tool = AnnolidEvalStartTool(allowed_dir=tmp_path)
    payload = json.loads(
        asyncio.run(
            tool.execute(
                model_family="dino_kpseg",
                data=str(data_yaml),
                weights=str(weights),
                split="test",
                data_format="coco",
                thresholds="4,8",
                per_keypoint=True,
                paper_report=True,
                dataset_name="mouse_pose",
                model_name="dino_best",
                out=str(out_json),
                report_dir=str(report_dir),
                report_basename="paper_metrics",
                allow_mutation=True,
            )
        )
    )
    assert payload["ok"] is True
    assert payload["session_id"] == "sh_eval_dino"
    assert payload["expected_output_path"] == str(out_json.resolve())
    assert "annolid.segmentation.dino_kpseg.eval" in manager.command
    assert "--paper-report" in manager.command
    assert "--per-keypoint" in manager.command
    assert "--report-dir" in manager.command


def test_annolid_eval_start_builds_yolo_command(monkeypatch, tmp_path: Path) -> None:
    data_yaml = tmp_path / "data.yaml"
    data_yaml.write_text("train: images/train\nval: images/val\n", encoding="utf-8")

    class _FakeManager:
        def __init__(self) -> None:
            self.command = ""

        async def start(self, *, command: str, cwd: str, timeout_s: float = 0.0):
            self.command = command
            return types.SimpleNamespace(session_id="sh_eval_yolo")

        async def poll(self, session_id: str, *, wait_ms: int = 0):
            return {
                "ok": True,
                "running": False,
                "status": "completed",
                "return_code": 0,
            }

        async def log(self, session_id: str, *, tail_lines: int = 200):
            return {"ok": True, "text": ""}

    monkeypatch.setattr(
        "annolid.core.agent.tools.eval_start.get_shell_session_manager",
        lambda: _FakeManager(),
    )
    monkeypatch.setattr(
        "annolid.core.agent.tools.eval_start.resolve_weight_path",
        lambda value: Path(value),
    )
    captured: dict[str, object] = {}

    def _fake_build_yolo_val_command(**kwargs):
        captured.update(kwargs)
        return [
            "yolo",
            "val",
            f"model={kwargs['model']}",
            f"data={kwargs['data']}",
            f"split={kwargs['split']}",
            f"project={kwargs['project']}",
            f"name={kwargs['name']}",
            f"save_json={kwargs['save_json']}",
            f"workers={kwargs['workers']}",
        ]

    monkeypatch.setattr(
        "annolid.core.agent.tools.eval_start.build_yolo_val_command",
        _fake_build_yolo_val_command,
    )

    tool = AnnolidEvalStartTool(allowed_dir=tmp_path)
    payload = json.loads(
        asyncio.run(
            tool.execute(
                model_family="yolo",
                data=str(data_yaml),
                weights="yolo11n-pose.pt",
                split="test",
                project=str(tmp_path / "runs"),
                run_name="eval_pose",
                imgsz=640,
                batch=4,
                save_json=True,
                workers=0,
                allow_mutation=True,
            )
        )
    )
    assert payload["ok"] is True
    assert payload["session_id"] == "sh_eval_yolo"
    assert payload["model_family"] == "yolo"
    assert "yolo val" in payload["command"]
    assert "split=test" in payload["command"]
    assert "save_json=True" in payload["command"]
    assert "workers=0" in payload["command"]
    assert captured["save_json"] is True
    assert captured["workers"] == 0


def test_annolid_eval_start_builds_behavior_command(
    monkeypatch, tmp_path: Path
) -> None:
    video_dir = tmp_path / "videos"
    video_dir.mkdir()
    checkpoint = tmp_path / "best_model.pth"
    checkpoint.write_text("fake", encoding="utf-8")
    out_json = tmp_path / "behavior_metrics.json"
    plot_dir = tmp_path / "behavior_plots"

    class _FakeManager:
        def __init__(self) -> None:
            self.command = ""

        async def start(self, *, command: str, cwd: str, timeout_s: float = 0.0):
            self.command = command
            return types.SimpleNamespace(session_id="sh_eval_behavior")

        async def poll(self, session_id: str, *, wait_ms: int = 0):
            return {
                "ok": True,
                "running": False,
                "status": "completed",
                "return_code": 0,
            }

        async def log(self, session_id: str, *, tail_lines: int = 200):
            return {"ok": True, "text": ""}

    manager = _FakeManager()
    monkeypatch.setattr(
        "annolid.core.agent.tools.eval_start.get_shell_session_manager",
        lambda: manager,
    )
    monkeypatch.setattr(
        "annolid.core.agent.tools.eval_start._resolve_training_python",
        lambda cwd: str(cwd / ".venv" / "bin" / "python"),
    )

    tool = AnnolidEvalStartTool(allowed_dir=tmp_path)
    payload = json.loads(
        asyncio.run(
            tool.execute(
                model_family="behavior_classifier",
                data=str(video_dir),
                weights=str(checkpoint),
                split="all",
                batch=2,
                feature_backbone="dinov3",
                dinov3_model_name="facebook/dinov3-vits16-pretrain-lvd1689m",
                feature_dim=768,
                transformer_dim=512,
                val_ratio=0.25,
                random_seed=7,
                out=str(out_json),
                plot_dir=str(plot_dir),
                allow_mutation=True,
            )
        )
    )
    assert payload["ok"] is True
    assert payload["session_id"] == "sh_eval_behavior"
    assert payload["expected_output_path"] == str(out_json.resolve())
    assert "annolid.behavior.eval" in manager.command
    assert "--video-folder" in manager.command
    assert "--checkpoint-path" in manager.command
    assert "--transformer-dim" in manager.command
    assert "--plot-dir" in manager.command


def test_filesystem_tools_round_trip(tmp_path: Path) -> None:
    write = WriteFileTool(allowed_dir=tmp_path)
    read = ReadFileTool(allowed_dir=tmp_path)
    edit = EditFileTool(allowed_dir=tmp_path)
    list_dir = ListDirTool(allowed_dir=tmp_path)
    file_path = tmp_path / "note.txt"

    wrote = asyncio.run(write.execute(path=str(file_path), content="hello"))
    assert "Successfully wrote" in wrote
    text = asyncio.run(read.execute(path=str(file_path)))
    assert text == "hello"
    edited = asyncio.run(
        edit.execute(path=str(file_path), old_text="hello", new_text="world")
    )
    assert "Successfully edited" in edited
    listed = asyncio.run(list_dir.execute(path=str(tmp_path)))
    assert "note.txt" in listed


def test_rename_file_tool_rename_and_overwrite(tmp_path: Path) -> None:
    writer = WriteFileTool(allowed_dir=tmp_path)
    renamer = RenameFileTool(allowed_dir=tmp_path)
    src = tmp_path / "old.pdf"
    dst = tmp_path / "new.pdf"
    conflict = tmp_path / "conflict.pdf"

    asyncio.run(writer.execute(path=str(src), content="v1"))
    asyncio.run(writer.execute(path=str(conflict), content="v2"))

    denied = asyncio.run(
        renamer.execute(path=str(src), new_path=str(conflict), overwrite=False)
    )
    assert "Target already exists" in denied

    renamed = asyncio.run(
        renamer.execute(path=str(src), new_path=str(dst), overwrite=False)
    )
    assert "Successfully renamed" in renamed
    assert not src.exists()
    assert dst.exists()

    replaced = asyncio.run(
        renamer.execute(path=str(conflict), new_path=str(dst), overwrite=True)
    )
    assert "Successfully renamed" in replaced
    assert not conflict.exists()
    assert dst.read_text(encoding="utf-8") == "v2"


def test_rename_file_tool_rejects_invalid_new_name(tmp_path: Path) -> None:
    writer = WriteFileTool(allowed_dir=tmp_path)
    renamer = RenameFileTool(allowed_dir=tmp_path)
    src = tmp_path / "paper.pdf"
    asyncio.run(writer.execute(path=str(src), content="pdf"))
    result = asyncio.run(
        renamer.execute(path=str(src), new_name="nested/path/illegal.pdf")
    )
    assert "new_name must be a base name" in result


def test_read_file_rejects_pdf_with_actionable_message(tmp_path: Path) -> None:
    tool = ReadFileTool(allowed_dir=tmp_path)
    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 fake")
    result = asyncio.run(tool.execute(path=str(pdf_path)))
    assert "extract_pdf_text" in result


def test_bibtex_tools_list_upsert_remove(tmp_path: Path) -> None:
    upsert = BibtexUpsertEntryTool(allowed_dir=tmp_path)
    listing = BibtexListEntriesTool(allowed_dir=tmp_path)
    remove = BibtexRemoveEntryTool(allowed_dir=tmp_path)
    bib_path = tmp_path / "refs.bib"

    created = asyncio.run(
        upsert.execute(
            path=str(bib_path),
            key="annolid2024",
            entry_type="article",
            fields={
                "title": "Annolid Toolkit",
                "author": "Liu, Jun",
                "year": "2024",
            },
        )
    )
    created_payload = json.loads(created)
    assert created_payload["created"] is True

    rows = asyncio.run(
        listing.execute(path=str(bib_path), query="toolkit", field="title")
    )
    rows_payload = json.loads(rows)
    assert rows_payload["returned"] == 1
    assert rows_payload["entries"][0]["key"] == "annolid2024"

    removed = asyncio.run(remove.execute(path=str(bib_path), key="annolid2024"))
    removed_payload = json.loads(removed)
    assert removed_payload["removed"] is True


def test_extract_pdf_text_tool_uses_fitz_backend(tmp_path: Path, monkeypatch) -> None:
    class _FakePage:
        def __init__(self, text: str) -> None:
            self._text = text

        def get_text(self, mode: str) -> str:
            assert mode == "text"
            return self._text

    class _FakeDoc:
        def __init__(self, pages: list[_FakePage]) -> None:
            self._pages = pages

        def __len__(self) -> int:
            return len(self._pages)

        def __getitem__(self, idx: int) -> _FakePage:
            return self._pages[idx]

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            del exc_type, exc, tb
            return None

    class _FakeFitz:
        @staticmethod
        def open(path: str) -> _FakeDoc:
            del path
            return _FakeDoc([_FakePage("Intro"), _FakePage("Results")])

    monkeypatch.setitem(sys.modules, "fitz", _FakeFitz)
    tool = ExtractPdfTextTool(allowed_dir=tmp_path)
    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 fake")
    result = asyncio.run(
        tool.execute(path=str(pdf_path), start_page=1, max_pages=2, max_chars=1000)
    )
    payload = json.loads(result)
    assert payload["backend"] == "pymupdf"
    assert payload["pages_read"] == 2
    assert "Intro" in payload["text"]
    assert "Results" in payload["text"]


def test_open_pdf_tool_uses_extract_pdf_text_backend(
    tmp_path: Path, monkeypatch
) -> None:
    class _FakePage:
        def __init__(self, text: str) -> None:
            self._text = text

        def get_text(self, mode: str) -> str:
            assert mode == "text"
            return self._text

    class _FakeDoc:
        def __init__(self, pages: list[_FakePage]) -> None:
            self._pages = pages

        def __len__(self) -> int:
            return len(self._pages)

        def __getitem__(self, idx: int) -> _FakePage:
            return self._pages[idx]

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            del exc_type, exc, tb
            return None

    class _FakeFitz:
        @staticmethod
        def open(path: str) -> _FakeDoc:
            del path
            return _FakeDoc([_FakePage("Page One"), _FakePage("Page Two")])

    monkeypatch.setitem(sys.modules, "fitz", _FakeFitz)
    tool = OpenPdfTool(allowed_dir=tmp_path)
    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 fake")
    result = asyncio.run(tool.execute(path=str(pdf_path), start_page=1, max_pages=1))
    payload = json.loads(result)
    assert payload["backend"] == "pymupdf"
    assert payload["pages_read"] == 1
    assert payload["text"] == "Page One"


def test_extract_pdf_images_tool_renders_pages(tmp_path: Path, monkeypatch) -> None:
    class _FakePixmap:
        def __init__(self, content: bytes) -> None:
            self._content = content

        def save(self, path: str) -> None:
            Path(path).write_bytes(self._content)

    class _FakePage:
        def __init__(self, number: int) -> None:
            self._number = number

        def get_pixmap(self, matrix=None, alpha=False):
            del matrix, alpha
            return _FakePixmap(f"page-{self._number}".encode("utf-8"))

    class _FakeDoc:
        def __init__(self, pages: list[_FakePage]) -> None:
            self._pages = pages

        def __len__(self) -> int:
            return len(self._pages)

        def __getitem__(self, idx: int) -> _FakePage:
            return self._pages[idx]

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            del exc_type, exc, tb
            return None

    class _FakeFitz:
        @staticmethod
        def Matrix(x: float, y: float):
            return (x, y)

        @staticmethod
        def open(path: str) -> _FakeDoc:
            del path
            return _FakeDoc([_FakePage(1), _FakePage(2)])

    monkeypatch.setitem(sys.modules, "fitz", _FakeFitz)
    tool = ExtractPdfImagesTool(allowed_dir=tmp_path)
    pdf_path = tmp_path / "paper.pdf"
    pdf_path.write_bytes(b"%PDF-1.4 fake")
    result = asyncio.run(
        tool.execute(path=str(pdf_path), start_page=1, max_pages=2, dpi=144)
    )
    payload = json.loads(result)
    assert payload["pages_rendered"] == 2
    for image_path in payload["images"]:
        image_file = Path(image_path)
        assert image_file.exists()
        assert image_file.suffix == ".png"


def test_video_info_tool_reads_metadata(tmp_path: Path) -> None:
    video_path = tmp_path / "tiny.avi"
    _write_test_video(video_path, fps=8.0, frames=6)
    tool = VideoInfoTool(allowed_dir=tmp_path)
    result = asyncio.run(tool.execute(path=str(video_path)))
    payload = json.loads(result)
    assert payload["total_frames"] == 6
    assert payload["fps"] > 0
    assert payload["width"] == 64
    assert payload["height"] == 48


def test_video_sample_frames_tool_stream_mode(tmp_path: Path) -> None:
    video_path = tmp_path / "tiny.avi"
    _write_test_video(video_path, fps=10.0, frames=10)
    tool = VideoSampleFramesTool(allowed_dir=tmp_path)
    result = asyncio.run(
        tool.execute(
            path=str(video_path),
            mode="stream",
            start_frame=2,
            step=2,
            max_frames=3,
        )
    )
    payload = json.loads(result)
    assert payload["count"] == 3
    frame_indices = [item["frame_index"] for item in payload["frames"]]
    assert frame_indices == [2, 4, 6]
    for item in payload["frames"]:
        assert Path(item["image_path"]).exists()


def test_video_segment_tool_exports_frame_range(tmp_path: Path) -> None:
    video_path = tmp_path / "tiny.avi"
    _write_test_video(video_path, fps=10.0, frames=12)
    out_path = tmp_path / "tiny_seg.avi"
    tool = VideoSegmentTool(allowed_dir=tmp_path)
    result = asyncio.run(
        tool.execute(
            path=str(video_path),
            output_path=str(out_path),
            start_frame=3,
            end_frame=6,
            overwrite=True,
        )
    )
    payload = json.loads(result)
    assert payload["frames_written"] == 4
    assert out_path.exists()


def test_video_process_segments_tool_exports_multiple_ranges(tmp_path: Path) -> None:
    video_path = tmp_path / "tiny.avi"
    _write_test_video(video_path, fps=10.0, frames=12)
    tool = VideoProcessSegmentsTool(allowed_dir=tmp_path)
    result = asyncio.run(
        tool.execute(
            path=str(video_path),
            segments=[
                {"start_frame": 0, "end_frame": 2},
                {"start_sec": 0.3, "end_sec": 0.5},
            ],
            overwrite=True,
        )
    )
    payload = json.loads(result)
    assert payload["segments_processed"] == 2
    assert len(payload["results"]) == 2
    for item in payload["results"]:
        assert item["frames_written"] > 0
        assert Path(item["output_path"]).exists()


def test_video_tools_allow_external_read_root_but_write_to_workspace(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    external = tmp_path / "external"
    external.mkdir(parents=True, exist_ok=True)
    video_path = external / "mouse.avi"
    _write_test_video(video_path, fps=10.0, frames=8)

    info_tool = VideoInfoTool(allowed_dir=workspace, allowed_read_roots=[str(external)])
    info_result = asyncio.run(info_tool.execute(path=str(video_path)))
    info_payload = json.loads(info_result)
    assert info_payload["total_frames"] == 8

    sample_tool = VideoSampleFramesTool(
        allowed_dir=workspace, allowed_read_roots=[str(external)]
    )
    sample_result = asyncio.run(
        sample_tool.execute(path=str(video_path), mode="stream", step=2, max_frames=2)
    )
    sample_payload = json.loads(sample_result)
    assert sample_payload["count"] == 2
    for frame in sample_payload["frames"]:
        image_path = Path(frame["image_path"])
        assert image_path.exists()
        assert str(image_path).startswith(str(workspace))


def test_video_list_inference_models_tool_reports_video_compatible_models(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from annolid.engine.registry import ModelInfo

    class _PredictPlugin:
        @classmethod
        def supports_predict(cls) -> bool:
            return True

        def add_predict_args(self, parser):  # noqa: ANN001
            parser.add_argument("--source", required=True)
            parser.add_argument("--output-dir", default=None)

    class _NoVideoPlugin:
        @classmethod
        def supports_predict(cls) -> bool:
            return True

        def add_predict_args(self, parser):  # noqa: ANN001
            parser.add_argument("--weights", default="model.pt")

    monkeypatch.setattr(
        "annolid.engine.registry.list_models",
        lambda load_builtins=True: [  # noqa: ARG005
            ModelInfo(
                name="video_model",
                description="video",
                supports_train=False,
                supports_predict=True,
            ),
            ModelInfo(
                name="non_video_model",
                description="non-video",
                supports_train=False,
                supports_predict=True,
            ),
        ],
    )
    monkeypatch.setattr(
        "annolid.engine.registry.get_model",
        lambda name: _PredictPlugin() if name == "video_model" else _NoVideoPlugin(),
    )

    tool = VideoListInferenceModelsTool()
    result = asyncio.run(tool.execute(video_only=True))
    payload = json.loads(result)
    assert payload["count"] == 1
    assert payload["models"][0]["name"] == "video_model"
    assert payload["models"][0]["video_compatible"] is True

    all_result = asyncio.run(tool.execute(video_only=False))
    all_payload = json.loads(all_result)
    assert all_payload["count"] == 2
    assert {m["name"] for m in all_payload["models"]} == {
        "video_model",
        "non_video_model",
    }


def test_video_run_model_inference_tool_infers_flags_and_executes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    video_path = tmp_path / "tiny.avi"
    _write_test_video(video_path, fps=8.0, frames=6)
    output_dir = tmp_path / "predictions"

    class _PredictPlugin:
        @classmethod
        def supports_predict(cls) -> bool:
            return True

        def add_predict_args(self, parser):  # noqa: ANN001
            parser.add_argument("--source", required=True)
            parser.add_argument("--output-dir", default=None)
            parser.add_argument("--weights", default="model.pt")

    captured: dict[str, object] = {}

    class _FakeProc:
        returncode = 0

        async def communicate(self):
            return (b"prediction completed", b"")

    async def _fake_create_subprocess_exec(*cmd, **kwargs):  # noqa: ANN001
        captured["cmd"] = list(cmd)
        captured["cwd"] = kwargs.get("cwd")
        return _FakeProc()

    monkeypatch.setattr(
        "annolid.engine.registry.get_model", lambda name: _PredictPlugin()
    )
    monkeypatch.setattr(
        asyncio,
        "create_subprocess_exec",
        _fake_create_subprocess_exec,
    )

    tool = VideoRunModelInferenceTool(allowed_dir=tmp_path)
    result = asyncio.run(
        tool.execute(
            model="fake_video_model",
            video_path=str(video_path),
            output_dir=str(output_dir),
            extra_args=["--weights", "custom.pt"],
        )
    )
    payload = json.loads(result)
    assert payload["ok"] is True
    assert payload["exit_code"] == 0
    assert payload["input_flag"] == "--source"
    assert payload["output_flag"] == "--output-dir"
    assert "prediction completed" in payload["stdout"]

    cmd = list(captured["cmd"])
    assert cmd[:5] == [
        sys.executable,
        "-m",
        "annolid.engine.cli",
        "predict",
        "fake_video_model",
    ]
    assert "--source" in cmd
    assert str(video_path.resolve()) in cmd
    assert "--output-dir" in cmd
    assert str(output_dir.resolve()) in cmd


def test_video_run_model_inference_tool_reports_when_input_flag_is_unknown(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    video_path = tmp_path / "tiny.avi"
    _write_test_video(video_path, fps=8.0, frames=6)

    class _PredictPlugin:
        @classmethod
        def supports_predict(cls) -> bool:
            return True

        def add_predict_args(self, parser):  # noqa: ANN001
            parser.add_argument("--weights", default="model.pt")

    monkeypatch.setattr(
        "annolid.engine.registry.get_model", lambda name: _PredictPlugin()
    )

    tool = VideoRunModelInferenceTool(allowed_dir=tmp_path)
    result = asyncio.run(
        tool.execute(model="fake_video_model", video_path=str(video_path))
    )
    payload = json.loads(result)
    assert "Cannot infer a video input argument" in payload["error"]
    assert "annolid-run predict fake_video_model" in payload["help"]


def test_sam3_agent_video_track_tool_executes_and_writes_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    video_path = tmp_path / "tiny.avi"
    _write_test_video(video_path, fps=8.0, frames=6)
    captured: dict[str, object] = {}

    def _fake_process_video_with_agent(**kwargs):  # noqa: ANN001
        captured.update(kwargs)
        return 11, 7

    monkeypatch.setattr(
        "annolid.segmentation.SAM.sam3.adapter.process_video_with_agent",
        _fake_process_video_with_agent,
    )

    tool = Sam3AgentVideoTrackTool(allowed_dir=tmp_path)
    result = asyncio.run(
        tool.execute(
            video_path=str(video_path),
            agent_prompt="mouse",
            window_size=6,
            stride=3,
            agent_det_thresh=0.4,
        )
    )
    payload = json.loads(result)
    expected_output_dir = tmp_path / "tiny_sam3_agent"
    expected_summary = expected_output_dir / "tiny_sam3_agent_tracking.json"

    assert payload["ok"] is True
    assert payload["frames_processed"] == 11
    assert payload["masks_written"] == 7
    assert payload["output_dir"] == str(expected_output_dir)
    assert payload["summary_path"] == str(expected_summary)
    assert captured["video_path"] == str(video_path.resolve())
    assert captured["agent_prompt"] == "mouse"
    assert captured["window_size"] == 6
    assert captured["stride"] == 3
    assert captured["agent_det_thresh"] == 0.4
    assert expected_summary.exists()
    assert json.loads(expected_summary.read_text(encoding="utf-8"))["ok"] is True


def test_sam3_agent_video_track_tool_dry_run_does_not_execute(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    video_path = tmp_path / "tiny.avi"
    _write_test_video(video_path, fps=8.0, frames=6)
    called = False

    def _fake_process_video_with_agent(**kwargs):  # noqa: ANN001
        nonlocal called
        called = True
        return 11, 7

    monkeypatch.setattr(
        "annolid.segmentation.SAM.sam3.adapter.process_video_with_agent",
        _fake_process_video_with_agent,
    )

    tool = Sam3AgentVideoTrackTool(allowed_dir=tmp_path)
    result = asyncio.run(
        tool.execute(
            video_path=str(video_path),
            agent_prompt="mouse",
            dry_run=True,
        )
    )
    payload = json.loads(result)
    expected_output_dir = tmp_path / "tiny_sam3_agent"

    assert payload["ok"] is True
    assert payload["dry_run"] is True
    assert called is False
    assert not expected_output_dir.exists()


def test_exec_tool_guard_blocks_dangerous() -> None:
    tool = SandboxedExecTool()
    result = asyncio.run(tool.execute(command="rm -rf /tmp/foo"))
    assert "blocked by safety guard" in result


def test_sandboxed_exec_tool_builds_hardened_docker_command(tmp_path: Path) -> None:
    tool = SandboxedExecTool(container_image="ubuntu:24.04")
    cmd = tool._build_docker_command(  # noqa: SLF001 - targeted unit test
        command="echo hi",
        cwd_path=tmp_path.resolve(),
    )
    joined = " ".join(cmd)
    assert "docker run --rm" in joined
    assert f"-v {tmp_path.resolve()}:{tmp_path.resolve()}:ro" in joined
    assert "--network none" in joined
    assert "--cap-drop ALL" in joined
    assert "--security-opt no-new-privileges" in joined
    assert "--pids-limit 256" in joined
    assert "--tmpfs /tmp:rw,noexec,nosuid,nodev,size=128m" in joined
    assert "ubuntu:24.04 bash -c echo hi" in joined


def test_sandboxed_exec_tool_can_enable_writable_host_mount(tmp_path: Path) -> None:
    tool = SandboxedExecTool(
        container_image="ubuntu:24.04",
        docker_host_mount_read_only=False,
    )
    cmd = tool._build_docker_command(  # noqa: SLF001 - targeted unit test
        command="echo hi",
        cwd_path=tmp_path.resolve(),
    )
    joined = " ".join(cmd)
    assert f"-v {tmp_path.resolve()}:{tmp_path.resolve()}" in joined
    assert f"-v {tmp_path.resolve()}:{tmp_path.resolve()}:ro" not in joined


def test_exec_start_foreground_returns_output(tmp_path: Path) -> None:
    tool = ExecStartTool()
    cmd = f"{shlex.quote(sys.executable)} -c \"print('annolid-shell-ok')\""
    result = asyncio.run(
        tool.execute(command=cmd, working_dir=str(tmp_path), background=False)
    )
    payload = json.loads(result)
    assert payload["ok"] is True
    assert payload["status"] == "completed"
    assert payload["return_code"] == 0
    assert "annolid-shell-ok" in payload["output"]


def test_exec_process_write_poll_log_and_kill(tmp_path: Path) -> None:
    async def _run() -> None:
        start = ExecStartTool()
        proc = ExecProcessTool()

        # Session that expects stdin and then exits.
        stdin_cmd = (
            f"{shlex.quote(sys.executable)} -u -c "
            "'import sys; print(sys.stdin.readline().strip())'"
        )
        started = json.loads(
            await start.execute(command=stdin_cmd, working_dir=str(tmp_path))
        )
        assert started["ok"] is True
        sid = started["session_id"]

        wrote = json.loads(
            await proc.execute(
                action="submit",
                session_id=sid,
                text="hello-shell",
            )
        )
        assert wrote["ok"] is True

        polled = {"ok": False, "status": "running"}
        for _ in range(12):
            polled = json.loads(
                await proc.execute(action="poll", session_id=sid, wait_ms=250)
            )
            if polled.get("status") != "running":
                break
        assert polled["ok"] is True
        assert polled["status"] in {"completed", "failed"}

        logs = json.loads(
            await proc.execute(action="log", session_id=sid, tail_lines=50)
        )
        assert logs["ok"] is True
        assert "hello-shell" in logs["text"]

        listed = json.loads(await proc.execute(action="list"))
        assert listed["ok"] is True
        assert any(item["session_id"] == sid for item in listed["sessions"])

        # Independent long-running session to verify kill path.
        sleep_cmd = f'{shlex.quote(sys.executable)} -c "import time; time.sleep(5)"'
        started_sleep = json.loads(
            await start.execute(command=sleep_cmd, working_dir=str(tmp_path))
        )
        sid_sleep = started_sleep["session_id"]
        killed = json.loads(await proc.execute(action="kill", session_id=sid_sleep))
        assert killed["ok"] is True

    asyncio.run(_run())


def test_web_search_tool_without_key_reports_config_error() -> None:
    tool = WebSearchTool(api_key="", backend="brave")
    result = asyncio.run(tool.execute(query="annolid"))
    assert "BRAVE_API_KEY not configured" in result


def test_web_search_tool_prefers_scrapling_backend(monkeypatch) -> None:
    tool = WebSearchTool(api_key="test-key")

    async def _fake_scrapling(*, query: str, count: int):
        del query, count
        return [
            {
                "title": "Scrapling Result",
                "url": "https://example.org/scrapling",
                "description": "from scrapling",
            }
        ]

    async def _fake_brave(*, query: str, count: int):
        del query, count
        return [
            {
                "title": "Brave Result",
                "url": "https://example.org/brave",
                "description": "from brave",
            }
        ]

    monkeypatch.setattr(tool, "_search_with_scrapling", _fake_scrapling)
    monkeypatch.setattr(tool, "_search_with_brave", _fake_brave)

    result = asyncio.run(tool.execute(query="annolid"))
    assert "Scrapling Result" in result
    assert "Brave Result" not in result


def test_web_search_tool_scrapling_fetchers_class_api(monkeypatch) -> None:
    called: dict[str, object] = {}

    class _Page:
        html = '<a class="result__a" href="https://example.org/docs">Example Docs</a>'

    class _AsyncFetcher:
        @classmethod
        async def fetch(cls, url: str, **kwargs):
            called["url"] = url
            called["kwargs"] = dict(kwargs)
            return _Page()

    scrapling_mod = types.ModuleType("scrapling")
    fetchers_mod = types.ModuleType("scrapling.fetchers")
    setattr(fetchers_mod, "AsyncFetcher", _AsyncFetcher)
    setattr(scrapling_mod, "fetchers", fetchers_mod)
    monkeypatch.setitem(sys.modules, "scrapling", scrapling_mod)
    monkeypatch.setitem(sys.modules, "scrapling.fetchers", fetchers_mod)

    tool = WebSearchTool(api_key="")
    result = asyncio.run(tool.execute(query="annolid", backend="scrapling"))
    assert "Results for: annolid" in result
    assert "Example Docs" in result
    assert str(called.get("url") or "").startswith("https://duckduckgo.com/html/?q=")


def test_web_search_tool_returns_no_results_when_scrapling_empty(monkeypatch) -> None:
    tool = WebSearchTool(api_key="")

    async def _fake_scrapling(*, query: str, count: int):
        del query, count
        return []

    async def _fake_brave(*, query: str, count: int):
        del query, count
        return None

    monkeypatch.setattr(tool, "_search_with_scrapling", _fake_scrapling)
    monkeypatch.setattr(tool, "_search_with_brave", _fake_brave)

    result = asyncio.run(tool.execute(query="annolid"))
    assert result == "No results for: annolid"


def test_web_search_tool_parses_duckduckgo_results() -> None:
    html = """
    <html><body>
      <a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fdoc">
        Example Title
      </a>
    </body></html>
    """
    rows = WebSearchTool._parse_duckduckgo_results(html, count=3)
    assert rows
    assert rows[0]["title"] == "Example Title"
    assert rows[0]["url"] == "https://example.com/doc"


def test_web_search_tool_parse_deduplicates_and_normalizes_duck_links() -> None:
    html = """
    <html><body>
      <a class='result__a' href='//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fdoc'>First</a>
      <a class='result__a' href='https://example.com/doc'>Duplicate</a>
      <a class='result__a' href='https://duckduckgo.com/?q=test'>Internal</a>
    </body></html>
    """
    rows = WebSearchTool._parse_duckduckgo_results(html, count=10)
    assert len(rows) == 1
    assert rows[0]["title"] == "First"
    assert rows[0]["url"] == "https://example.com/doc"


def test_cron_tool_add_list_remove(tmp_path: Path) -> None:
    tool = CronTool(store_path=tmp_path / "cron" / "jobs.json")
    tool.set_context("local", "user1")
    added = asyncio.run(
        tool.execute(action="add", message="ping", every_seconds=30, cron_expr=None)
    )
    assert "Created job" in added
    listed = asyncio.run(tool.execute(action="list"))
    assert "Scheduled jobs" in listed
    job_id = added.split("id: ")[-1].rstrip(")")
    checked = asyncio.run(tool.execute(action="check", job_id=job_id))
    assert f"id={job_id}" in checked
    assert "next_run_at_ms=" in checked
    checked_status = asyncio.run(tool.execute(action="check"))
    assert checked_status.startswith("Cron status:")
    canceled = asyncio.run(tool.execute(action="cancel", job_id=job_id))
    assert f"Removed job {job_id}" == canceled
    removed = asyncio.run(tool.execute(action="remove", job_id=job_id))
    assert f"Job {job_id} not found" == removed


def test_cron_tool_add_one_time_at_iso_datetime(tmp_path: Path) -> None:
    tool = CronTool(store_path=tmp_path / "cron" / "jobs.json")
    tool.set_context("local", "user1")
    at_value = (datetime.now(timezone.utc) + timedelta(minutes=10)).isoformat()
    added = asyncio.run(
        tool.execute(
            action="add",
            message="one-shot",
            every_seconds=None,
            cron_expr=None,
            at=at_value,
        )
    )
    assert "Created job" in added
    listed = asyncio.run(tool.execute(action="list"))
    assert "Scheduled jobs" in listed
    assert "at=" in listed


def test_cron_tool_accepts_timezone_for_cron_expr(tmp_path: Path) -> None:
    tool = CronTool(store_path=tmp_path / "cron" / "jobs.json")
    tool.set_context("local", "user1")
    added = asyncio.run(
        tool.execute(
            action="add",
            message="weekday check",
            cron_expr="0 9 * * 1-5",
            tz="America/New_York",
        )
    )
    assert "Created job" in added
    listed = asyncio.run(tool.execute(action="list"))
    assert "tz=America/New_York" in listed


def test_cron_tool_rejects_timezone_without_cron_expr(tmp_path: Path) -> None:
    tool = CronTool(store_path=tmp_path / "cron" / "jobs.json")
    tool.set_context("local", "user1")
    result = asyncio.run(
        tool.execute(
            action="add",
            message="bad tz usage",
            every_seconds=30,
            tz="America/New_York",
        )
    )
    assert "tz can only be used with cron_expr" in result


def test_cron_tool_rejects_invalid_timezone(tmp_path: Path) -> None:
    tool = CronTool(store_path=tmp_path / "cron" / "jobs.json")
    tool.set_context("local", "user1")
    result = asyncio.run(
        tool.execute(
            action="add",
            message="bad tz value",
            cron_expr="0 9 * * *",
            tz="Mars/Phobos",
        )
    )
    assert "unknown timezone" in result


def test_cron_tool_rejects_past_at_schedule(tmp_path: Path) -> None:
    tool = CronTool(store_path=tmp_path / "cron" / "jobs.json")
    tool.set_context("local", "user1")
    past_at = (datetime.now(timezone.utc) - timedelta(minutes=1)).isoformat()
    result = asyncio.run(
        tool.execute(
            action="add",
            message="past one-shot",
            at=past_at,
        )
    )
    assert "must be in the future" in result


def test_cron_tool_add_direct_scheduled_email(tmp_path: Path) -> None:
    tool = CronTool(store_path=tmp_path / "cron" / "jobs.json")
    tool.set_context("local", "user1")
    added = asyncio.run(
        tool.execute(
            action="add",
            message="weekly update email",
            every_seconds=3600,
            email_to="user@example.com",
            email_subject="Weekly Update",
            email_content="Status looks good.",
        )
    )
    assert "Created scheduled email job to user@example.com" in added
    job_id = added.split("id: ")[-1].rstrip(")")
    rows = tool._service.list_jobs(include_disabled=True)
    job = next(row for row in rows if row.id == job_id)
    assert job.payload.kind == "send_email"
    assert job.payload.email_to == "user@example.com"
    assert job.payload.email_subject == "Weekly Update"
    assert job.payload.email_content == "Status looks good."


def test_cron_tool_add_direct_scheduled_email_defaults_recipient_from_chat_id(
    tmp_path: Path,
) -> None:
    tool = CronTool(store_path=tmp_path / "cron" / "jobs.json")
    tool.set_context("email", "recipient@example.com")
    added = asyncio.run(
        tool.execute(
            action="add",
            message="follow up",
            every_seconds=300,
            email_content="Checking in.",
        )
    )
    assert "Created scheduled email job to recipient@example.com" in added


def test_cron_tool_add_direct_scheduled_email_with_schedule_time_alias(
    tmp_path: Path,
) -> None:
    tool = CronTool(store_path=tmp_path / "cron" / "jobs.json")
    tool.set_context("local", "user1")
    future_at = (datetime.now(timezone.utc) + timedelta(minutes=10)).isoformat()
    added = asyncio.run(
        tool.execute(
            action="add",
            message="Annolid joke email",
            schedule_time=future_at,
            email_to="cy384@cornell.edu",
            email_subject="Annolid Joke",
            email_content="Why did the segmentation model go to therapy?",
        )
    )
    assert "Created scheduled email job to cy384@cornell.edu" in added
    job_id = added.split("id: ")[-1].rstrip(")")
    rows = tool._service.list_jobs(include_disabled=True)
    job = next(row for row in rows if row.id == job_id)
    assert job.schedule.kind == "at"
    assert int(job.schedule.at_ms or 0) > 0


def test_cron_tool_default_store_path_does_not_use_cwd_annolid(monkeypatch) -> None:
    default_path = Path("/var/nonwritable/agent-workspace/cron/jobs.json")
    tmp_path = Path("/tmp/annolid/cron/jobs.json")

    monkeypatch.setattr(
        "annolid.core.agent.tools.cron.default_cron_store_path",
        lambda: default_path,
    )

    def _fake_writable(path: Path) -> bool:
        return path == tmp_path

    monkeypatch.setattr(
        CronTool,
        "_is_store_path_writable",
        staticmethod(_fake_writable),
    )

    resolved = CronTool._resolve_default_store_path()
    assert resolved == tmp_path
    assert ".annolid/cron/jobs.json" not in str(resolved).replace("\\", "/")


def test_memory_search_and_get_tools(tmp_path: Path) -> None:
    memory_dir = tmp_path / "memory"
    memory_dir.mkdir(parents=True, exist_ok=True)
    (memory_dir / "MEMORY.md").write_text(
        "# Long-term\n\nPreferred species: zebrafish\n",
        encoding="utf-8",
    )
    (memory_dir / "2026-02-11.md").write_text(
        "# 2026-02-11\n\nReviewed tracking thresholds.\n",
        encoding="utf-8",
    )

    search_tool = MemorySearchTool(workspace=tmp_path)
    result = asyncio.run(search_tool.execute(query="zebrafish preference", top_k=3))
    payload = json.loads(result)
    assert payload["count"] >= 1
    assert any(item["path"] == "memory/MEMORY.md" for item in payload["results"])

    get_tool = MemoryGetTool(workspace=tmp_path)
    got = asyncio.run(get_tool.execute(path="MEMORY.md", start_line=1, end_line=2))
    got_payload = json.loads(got)
    assert got_payload["path"] == "memory/MEMORY.md"
    assert "# Long-term" in got_payload["content"]

    blocked = asyncio.run(get_tool.execute(path="../secret.md"))
    blocked_payload = json.loads(blocked)
    assert "allowed" in blocked_payload["error"]


def test_memory_set_tool_writes_long_term_memory(tmp_path: Path) -> None:
    set_tool = MemorySetTool(workspace=tmp_path)
    first = asyncio.run(set_tool.execute(key="preferred_species", value="zebrafish"))
    first_payload = json.loads(first)
    assert first_payload["ok"] is True
    assert first_payload["path"] == "memory/MEMORY.md"

    second = asyncio.run(set_tool.execute(note="Use higher threshold for arena C"))
    second_payload = json.loads(second)
    assert second_payload["ok"] is True

    memory_text = (tmp_path / "memory" / "MEMORY.md").read_text(encoding="utf-8")
    assert "- preferred_species: zebrafish" in memory_text
    assert "- Use higher threshold for arena C" in memory_text


def test_memory_set_tool_upserts_existing_key(tmp_path: Path) -> None:
    set_tool = MemorySetTool(workspace=tmp_path)
    asyncio.run(set_tool.execute(key="preferred_species", value="zebrafish"))
    asyncio.run(set_tool.execute(key="preferred_species", value="medaka"))
    memory_text = (tmp_path / "memory" / "MEMORY.md").read_text(encoding="utf-8")
    assert "- preferred_species: medaka" in memory_text
    assert memory_text.count("- preferred_species:") == 1


def test_memory_set_tool_append_mode_keeps_duplicate_key_entries(
    tmp_path: Path,
) -> None:
    set_tool = MemorySetTool(workspace=tmp_path)
    asyncio.run(
        set_tool.execute(key="preferred_species", value="zebrafish", mode="append")
    )
    asyncio.run(
        set_tool.execute(key="preferred_species", value="medaka", mode="append")
    )
    memory_text = (tmp_path / "memory" / "MEMORY.md").read_text(encoding="utf-8")
    assert "- preferred_species: zebrafish" in memory_text
    assert "- preferred_species: medaka" in memory_text
    assert memory_text.count("- preferred_species:") == 2


def test_memory_set_tool_rejects_invalid_mode(tmp_path: Path) -> None:
    set_tool = MemorySetTool(workspace=tmp_path)
    result = asyncio.run(
        set_tool.execute(key="preferred_species", value="zebrafish", mode="merge")
    )
    payload = json.loads(result)
    assert "Invalid mode" in payload["error"]


def test_code_search_tool_finds_matches_with_context(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    target = workspace / "module.py"
    target.write_text(
        "def load_config(path):\n"
        "    return path\n"
        "\n"
        "def save_config(path, content):\n"
        "    return content\n",
        encoding="utf-8",
    )
    tool = CodeSearchTool(allowed_dir=workspace)
    result = asyncio.run(
        tool.execute(
            query="config",
            path=str(workspace),
            glob="*.py",
            context_lines=1,
            max_results=10,
        )
    )
    payload = json.loads(result)
    assert payload["count"] >= 2
    assert payload["truncated"] is False
    first = payload["results"][0]
    assert first["path"] == "module.py"
    assert "context" in first


def test_code_explain_tool_describes_module_and_symbol(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    target = workspace / "analyzer.py"
    target.write_text(
        '"""Behavior analysis helpers."""\n'
        "import json\n"
        "\n"
        "class Runner:\n"
        '    """Executes processing."""\n'
        "    def run(self, value):\n"
        "        return json.dumps(value)\n"
        "\n"
        "def normalize(data):\n"
        "    return str(data).strip()\n",
        encoding="utf-8",
    )
    tool = CodeExplainTool(allowed_dir=workspace)

    module_result = asyncio.run(tool.execute(path=str(target)))
    module_payload = json.loads(module_result)
    assert module_payload["module_docstring"] == "Behavior analysis helpers."
    assert any(item["name"] == "Runner" for item in module_payload["classes"])
    assert any(item["name"] == "normalize" for item in module_payload["functions"])

    symbol_result = asyncio.run(
        tool.execute(path=str(target), symbol="Runner.run", include_source=True)
    )
    symbol_payload = json.loads(symbol_result)
    assert symbol_payload["kind"] == "function"
    assert symbol_payload["name"] == "run"
    assert "json.dumps" in symbol_payload["calls"]
    assert "def run" in symbol_payload["source"]


def test_git_tools_status_diff_log(tmp_path: Path) -> None:
    if shutil.which("git") is None:
        pytest.skip("git is not available in this environment")
    repo = tmp_path / "repo"
    repo.mkdir(parents=True, exist_ok=True)

    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "annolid@example.com"],
        cwd=repo,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Annolid Bot"],
        cwd=repo,
        check=True,
        capture_output=True,
    )

    tracked = repo / "tracked.txt"
    tracked.write_text("line1\n", encoding="utf-8")
    subprocess.run(
        ["git", "add", "tracked.txt"], cwd=repo, check=True, capture_output=True
    )
    subprocess.run(
        ["git", "commit", "-m", "initial"],
        cwd=repo,
        check=True,
        capture_output=True,
    )
    tracked.write_text("line1\nline2\n", encoding="utf-8")

    status_tool = GitStatusTool(allowed_dir=repo)
    status_result = asyncio.run(status_tool.execute(repo_path=str(repo)))
    status_payload = json.loads(status_result)
    assert status_payload["exit_code"] == 0
    assert "tracked.txt" in status_payload["output"]

    diff_tool = GitDiffTool(allowed_dir=repo)
    diff_result = asyncio.run(diff_tool.execute(repo_path=str(repo)))
    diff_payload = json.loads(diff_result)
    assert diff_payload["exit_code"] == 0
    assert "+line2" in diff_payload["output"]

    log_tool = GitLogTool(allowed_dir=repo)
    log_result = asyncio.run(log_tool.execute(repo_path=str(repo), max_count=5))
    log_payload = json.loads(log_result)
    assert log_payload["exit_code"] == 0
    assert "initial" in log_payload["output"]


def test_github_tools_report_missing_gh_cli(tmp_path: Path, monkeypatch) -> None:
    async def _missing_command(*args, **kwargs):
        del args, kwargs
        raise FileNotFoundError("gh")

    monkeypatch.setattr(
        asyncio,
        "create_subprocess_exec",
        _missing_command,
    )
    status_tool = GitHubPrStatusTool(allowed_dir=tmp_path)
    checks_tool = GitHubPrChecksTool(allowed_dir=tmp_path)

    status_result = asyncio.run(status_tool.execute(repo_path=str(tmp_path)))
    checks_result = asyncio.run(checks_tool.execute(repo_path=str(tmp_path)))
    status_payload = json.loads(status_result)
    checks_payload = json.loads(checks_result)
    assert "Command not found: gh" in status_payload["error"]
    assert "Command not found: gh" in checks_payload["error"]


def test_git_cli_blocks_mutating_commands_without_explicit_allow(
    tmp_path: Path,
) -> None:
    tool = GitCliTool(allowed_dir=tmp_path)
    result = asyncio.run(
        tool.execute(repo_path=str(tmp_path), args=["commit", "-m", "x"])
    )
    payload = json.loads(result)
    assert "Blocked mutating git command" in payload["error"]


def test_git_cli_accepts_command_string_for_read_only_invocation(
    tmp_path: Path,
) -> None:
    if shutil.which("git") is None:
        pytest.skip("git is not available in this environment")
    repo = tmp_path / "repo"
    repo.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init"], cwd=repo, check=True, capture_output=True)

    tool = GitCliTool(allowed_dir=repo)
    result = asyncio.run(
        tool.execute(repo_path=str(repo), command="git status --short --branch")
    )
    payload = json.loads(result)
    assert int(payload.get("exit_code", 1)) == 0
    assert "## " in str(payload.get("output", ""))


def test_gh_cli_blocks_unknown_or_mutating_commands_without_explicit_allow(
    tmp_path: Path,
) -> None:
    tool = GitHubCliTool(allowed_dir=tmp_path)
    result = asyncio.run(
        tool.execute(repo_path=str(tmp_path), args=["pr", "comment", "123"])
    )
    payload = json.loads(result)
    assert "Blocked mutating gh command" in payload["error"]


def test_gh_cli_accepts_command_string_and_strips_prefix(tmp_path: Path) -> None:
    tool = GitHubCliTool(allowed_dir=tmp_path)
    result = asyncio.run(
        tool.execute(repo_path=str(tmp_path), command="gh pr comment 123")
    )
    payload = json.loads(result)
    assert "Blocked mutating gh command" in str(payload.get("error", ""))
    assert payload.get("command") == ["gh", "pr", "comment", "123"]


def test_register_nanobot_style_tools(tmp_path: Path) -> None:
    registry = FunctionToolRegistry()
    asyncio.run(register_nanobot_style_tools(registry, allowed_dir=tmp_path))
    assert registry.has("read_file")
    assert registry.has("rename_file")
    assert registry.has("code_search")
    assert registry.has("code_explain")
    assert registry.has("git_status")
    assert registry.has("git_cli")
    assert registry.has("git_diff")
    assert registry.has("git_log")
    assert registry.has("github_pr_status")
    assert registry.has("gh_cli")
    assert registry.has("github_pr_checks")
    assert registry.has("memory_search")
    assert registry.has("memory_get")
    assert registry.has("memory_set")
    assert registry.has("extract_pdf_text")
    assert registry.has("open_pdf")
    assert registry.has("extract_pdf_images")
    assert registry.has("video_info")
    assert registry.has("video_list_inference_models")
    assert registry.has("video_run_model_inference")
    assert registry.has("sam3_agent_video_track")
    assert registry.has("video_sample_frames")
    assert registry.has("video_segment")
    assert registry.has("video_process_segments")
    assert registry.has("camera_snapshot")
    assert registry.has("coding_session_start")
    assert registry.has("coding_session_send")
    assert registry.has("coding_session_poll")
    assert registry.has("coding_session_list")
    assert registry.has("coding_session_close")
    assert registry.has("automation_schedule")
    assert registry.has("exec")
    assert registry.has("exec_start")
    assert registry.has("exec_process")
    assert registry.has("annolid_dataset_inspect")
    assert registry.has("annolid_dataset_prepare")
    assert registry.has("annolid_train_start")
    assert registry.has("annolid_eval_start")
    assert registry.has("annolid_eval_report")
    assert registry.has("annolid_novelty_check")
    assert registry.has("annolid_paper_run_report")
    assert registry.has("cron")
    assert registry.has("download_url")
    assert registry.has("download_pdf")
    assert registry.has("bibtex_list_entries")
    assert registry.has("bibtex_upsert_entry")
    assert registry.has("bibtex_remove_entry")
    assert registry.has("clawhub_search_skills")
    assert registry.has("clawhub_install_skill")
    assert registry.has("box") is False


def test_register_nanobot_style_tools_accepts_custom_cron_store_path(
    tmp_path: Path,
) -> None:
    registry = FunctionToolRegistry()
    store_path = tmp_path / "workspace" / "cron" / "jobs.json"
    asyncio.run(
        register_nanobot_style_tools(
            registry,
            allowed_dir=tmp_path,
            cron_store_path=store_path,
        )
    )
    tool = registry.get("cron")
    assert isinstance(tool, CronTool)
    assert tool._service.store_path == store_path


def test_register_nanobot_style_tools_skips_calendar_when_deps_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry = FunctionToolRegistry()
    monkeypatch.setattr(
        "annolid.core.agent.tools.nanobot.GoogleCalendarTool.is_available",
        lambda: False,
    )
    asyncio.run(
        register_nanobot_style_tools(
            registry,
            allowed_dir=tmp_path,
            calendar_cfg=CalendarToolConfig(enabled=True, provider="google"),
        )
    )
    assert registry.has("google_calendar") is False


def test_register_nanobot_style_tools_skips_calendar_when_not_preflight_ready(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry = FunctionToolRegistry()
    monkeypatch.setattr(
        "annolid.core.agent.tools.nanobot.GoogleCalendarTool.is_available",
        lambda: True,
    )
    asyncio.run(
        register_nanobot_style_tools(
            registry,
            allowed_dir=tmp_path,
            calendar_cfg=CalendarToolConfig(
                enabled=True,
                provider="google",
                credentials_file=str(tmp_path / "missing_credentials.json"),
                token_file=str(tmp_path / "missing_token.json"),
                allow_interactive_auth=False,
            ),
        )
    )
    assert registry.has("google_calendar") is False


def test_register_nanobot_style_tools_skips_calendar_when_availability_check_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry = FunctionToolRegistry()

    def _raise_check() -> bool:
        raise ModuleNotFoundError("No module named 'google.auth'")

    monkeypatch.setattr(
        "annolid.core.agent.tools.nanobot.GoogleCalendarTool.is_available",
        _raise_check,
    )
    asyncio.run(
        register_nanobot_style_tools(
            registry,
            allowed_dir=tmp_path,
            calendar_cfg=CalendarToolConfig(enabled=True, provider="google"),
        )
    )
    assert registry.has("google_calendar") is False


def test_register_nanobot_style_tools_registers_calendar_when_interactive_auth_allowed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    registry = FunctionToolRegistry()
    credentials_path = tmp_path / "credentials.json"
    credentials_path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        "annolid.core.agent.tools.nanobot.GoogleCalendarTool.is_available",
        lambda: True,
    )
    asyncio.run(
        register_nanobot_style_tools(
            registry,
            allowed_dir=tmp_path,
            calendar_cfg=CalendarToolConfig(
                enabled=True,
                provider="google",
                credentials_file=str(credentials_path),
                token_file=str(tmp_path / "token.json"),
                allow_interactive_auth=True,
            ),
        )
    )
    assert registry.has("google_calendar") is True


def test_register_nanobot_style_tools_registers_box_when_enabled(
    tmp_path: Path,
) -> None:
    registry = FunctionToolRegistry()
    asyncio.run(
        register_nanobot_style_tools(
            registry,
            allowed_dir=tmp_path,
            box_cfg=BoxToolConfig(
                enabled=True,
                access_token="box-token",
            ),
        )
    )
    assert registry.has("box") is True


def test_mcp_tool_wrapper_sanitizes_name_and_schema() -> None:
    class _ToolDef:
        name = "search.web"
        description = "Search"
        inputSchema = {"properties": {"query": {"type": "string"}}}

    wrapper = MCPToolWrapper(
        session=object(),
        server_name="weather-server:v1",
        tool_def=_ToolDef(),
    )
    assert re.match(r"^[A-Za-z0-9_]+$", wrapper.name)
    assert len(wrapper.name) <= 64
    assert wrapper.parameters["type"] == "object"
    assert "query" in wrapper.parameters["properties"]


def test_mcp_tool_wrapper_execute_falls_back_to_structured_content(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _TextContent:
        def __init__(self, text: str) -> None:
            self.text = text

    fake_mcp = types.SimpleNamespace(
        types=types.SimpleNamespace(TextContent=_TextContent)
    )
    monkeypatch.setitem(sys.modules, "mcp", fake_mcp)

    class _ToolDef:
        name = "tool"
        description = "Tool"
        inputSchema = {"type": "object", "properties": {}}

    class _Session:
        async def call_tool(self, name: str, arguments: dict) -> object:
            assert name == "tool"
            assert arguments == {"x": 1}
            return types.SimpleNamespace(
                content=[],
                structuredContent={"ok": True, "value": 1},
                isError=False,
            )

    wrapper = MCPToolWrapper(
        session=_Session(),
        server_name="s",
        tool_def=_ToolDef(),
    )
    payload = asyncio.run(wrapper.execute(x=1))
    assert json.loads(payload) == {"ok": True, "value": 1}


def test_download_url_tool_saves_file_and_blocks_outside_dir(
    tmp_path: Path, monkeypatch
) -> None:
    class _FakeResponse:
        status_code = 200
        headers = {"content-type": "text/plain; charset=utf-8"}
        url = "https://example.org/note.txt"

        def raise_for_status(self) -> None:
            return None

        async def aiter_bytes(self):
            yield b"hello "
            yield b"agent"

    class _FakeStreamContext:
        async def __aenter__(self):
            return _FakeResponse()

        async def __aexit__(self, exc_type, exc, tb):
            del exc_type, exc, tb
            return None

    class _FakeClient:
        def __init__(self, *args, **kwargs):
            del args, kwargs

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            del exc_type, exc, tb
            return None

        def stream(self, method, url, headers=None):
            del method, url, headers
            return _FakeStreamContext()

    fake_httpx = types.SimpleNamespace(AsyncClient=_FakeClient)
    monkeypatch.setitem(sys.modules, "httpx", fake_httpx)

    tool = DownloadUrlTool(allowed_dir=tmp_path)
    out_path = tmp_path / "downloads" / "note.txt"
    result = asyncio.run(
        tool.execute(
            url="https://example.org/note.txt",
            output_path=str(out_path),
            content_type_prefixes=["text/plain"],
        )
    )
    payload = json.loads(result)
    assert payload["output_path"] == str(out_path)
    assert out_path.read_text(encoding="utf-8") == "hello agent"

    blocked = asyncio.run(
        tool.execute(
            url="https://example.org/note.txt",
            output_path=str(tmp_path.parent / "escape.txt"),
        )
    )
    blocked_payload = json.loads(blocked)
    assert "outside allowed directory" in blocked_payload["error"]


def test_download_pdf_tool_enforces_pdf_content_type(
    tmp_path: Path, monkeypatch
) -> None:
    class _FakePdfResponse:
        status_code = 200
        headers = {"content-type": "application/pdf"}
        url = "https://example.org/paper.pdf"

        def raise_for_status(self) -> None:
            return None

        async def aiter_bytes(self):
            yield b"%PDF-1.4 fake"

    class _FakeTextResponse:
        status_code = 200
        headers = {"content-type": "text/html; charset=utf-8"}
        url = "https://example.org/not-a-pdf"

        def raise_for_status(self) -> None:
            return None

        async def aiter_bytes(self):
            yield b"<html>hello</html>"

    class _FakeStreamContext:
        def __init__(self, response):
            self._response = response

        async def __aenter__(self):
            return self._response

        async def __aexit__(self, exc_type, exc, tb):
            del exc_type, exc, tb
            return None

    class _FakeClient:
        def __init__(self, *args, **kwargs):
            del args, kwargs

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            del exc_type, exc, tb
            return None

        def stream(self, method, url, headers=None):
            del method, headers
            if str(url).endswith(".pdf"):
                return _FakeStreamContext(_FakePdfResponse())
            return _FakeStreamContext(_FakeTextResponse())

    fake_httpx = types.SimpleNamespace(AsyncClient=_FakeClient)
    monkeypatch.setitem(sys.modules, "httpx", fake_httpx)

    tool = DownloadPdfTool(allowed_dir=tmp_path)
    out_path = tmp_path / "downloads" / "paper.pdf"
    ok = asyncio.run(
        tool.execute(url="https://example.org/paper.pdf", output_path=str(out_path))
    )
    ok_payload = json.loads(ok)
    assert ok_payload["is_pdf"] is True
    assert Path(ok_payload["output_path"]).exists()

    bad = asyncio.run(
        tool.execute(
            url="https://example.org/not-a-pdf",
            output_path=str(tmp_path / "downloads" / "bad.pdf"),
        )
    )
    bad_payload = json.loads(bad)
    assert "not allowed" in str(bad_payload.get("error", ""))


def test_download_pdf_tool_renames_generic_pdf_filename(
    tmp_path: Path, monkeypatch
) -> None:
    class _FakePdfResponse:
        status_code = 200
        headers = {"content-type": "application/pdf"}
        url = "https://example.org/pdf"

        def raise_for_status(self) -> None:
            return None

        async def aiter_bytes(self):
            yield b"%PDF-1.4 fake"

    class _FakeStreamContext:
        async def __aenter__(self):
            return _FakePdfResponse()

        async def __aexit__(self, exc_type, exc, tb):
            del exc_type, exc, tb
            return None

    class _FakeClient:
        def __init__(self, *args, **kwargs):
            del args, kwargs

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            del exc_type, exc, tb
            return None

        def stream(self, method, url, headers=None):
            del method, url, headers
            return _FakeStreamContext()

    fake_httpx = types.SimpleNamespace(AsyncClient=_FakeClient)
    monkeypatch.setitem(sys.modules, "httpx", fake_httpx)

    tool = DownloadPdfTool(allowed_dir=tmp_path)
    monkeypatch.setattr(
        tool,
        "_extract_pdf_title",
        lambda _path: "Neural Circuit Dynamics in Mouse Cortex",
    )
    result = asyncio.run(tool.execute(url="https://example.org/pdf"))
    payload = json.loads(result)

    output_path = Path(str(payload["output_path"]))
    assert payload["is_pdf"] is True
    assert payload["renamed"] is True
    assert output_path.name == "Neural_Circuit_Dynamics_in_Mouse_Cortex.pdf"
    assert output_path.exists()
    assert not (tmp_path / "downloads" / "pdf.pdf").exists()


def test_download_pdf_tool_renames_non_generic_when_title_differs(
    tmp_path: Path, monkeypatch
) -> None:
    class _FakePdfResponse:
        status_code = 200
        headers = {"content-type": "application/pdf"}
        url = "https://www.biorxiv.org/content/10.64898/2026.01.20.700446v2.full.pdf"

        def raise_for_status(self) -> None:
            return None

        async def aiter_bytes(self):
            yield b"%PDF-1.4 fake"

    class _FakeStreamContext:
        async def __aenter__(self):
            return _FakePdfResponse()

        async def __aexit__(self, exc_type, exc, tb):
            del exc_type, exc, tb
            return None

    class _FakeClient:
        def __init__(self, *args, **kwargs):
            del args, kwargs

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            del exc_type, exc, tb
            return None

        def stream(self, method, url, headers=None):
            del method, url, headers
            return _FakeStreamContext()

    fake_httpx = types.SimpleNamespace(AsyncClient=_FakeClient)
    monkeypatch.setitem(sys.modules, "httpx", fake_httpx)

    tool = DownloadPdfTool(allowed_dir=tmp_path)
    monkeypatch.setattr(
        tool,
        "_extract_pdf_title",
        lambda _path: "A Better Paper Title",
    )
    result = asyncio.run(
        tool.execute(
            url="https://www.biorxiv.org/content/10.64898/2026.01.20.700446v2.full.pdf"
        )
    )
    payload = json.loads(result)

    output_path = Path(str(payload["output_path"]))
    assert payload["is_pdf"] is True
    assert payload["renamed"] is True
    assert output_path.name == "A_Better_Paper_Title.pdf"
    assert output_path.exists()
    assert not (tmp_path / "downloads" / "2026.01.20.700446v2.full.pdf").exists()


def test_download_pdf_tool_retries_pmc_with_download_query(
    tmp_path: Path, monkeypatch
) -> None:
    class _ForbiddenResponse:
        status_code = 403
        headers = {"content-type": "text/html; charset=utf-8"}
        url = "https://pmc.ncbi.nlm.nih.gov/articles/PMC8219259/pdf/nihms-1556781.pdf"

        def raise_for_status(self) -> None:
            raise RuntimeError("403 Forbidden")

        async def aiter_bytes(self):
            yield b""

    class _PdfResponse:
        status_code = 200
        headers = {"content-type": "application/pdf"}
        url = "https://pmc.ncbi.nlm.nih.gov/articles/PMC8219259/pdf/nihms-1556781.pdf?download=1"

        def raise_for_status(self) -> None:
            return None

        async def aiter_bytes(self):
            yield b"%PDF-1.4 fake"

    class _FakeStreamContext:
        def __init__(self, response):
            self._response = response

        async def __aenter__(self):
            return self._response

        async def __aexit__(self, exc_type, exc, tb):
            del exc_type, exc, tb
            return None

    class _FakeClient:
        def __init__(self, *args, **kwargs):
            del args, kwargs

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            del exc_type, exc, tb
            return None

        def stream(self, method, url, headers=None):
            del method, headers
            if str(url).endswith("?download=1"):
                return _FakeStreamContext(_PdfResponse())
            return _FakeStreamContext(_ForbiddenResponse())

    fake_httpx = types.SimpleNamespace(AsyncClient=_FakeClient)
    monkeypatch.setitem(sys.modules, "httpx", fake_httpx)

    tool = DownloadPdfTool(allowed_dir=tmp_path)
    result = asyncio.run(
        tool.execute(
            url="https://pmc.ncbi.nlm.nih.gov/articles/PMC8219259/pdf/nihms-1556781.pdf"
        )
    )
    payload = json.loads(result)
    assert payload["is_pdf"] is True
    assert Path(str(payload["output_path"])).exists()


def test_download_pdf_tool_uses_pmc_oa_metadata_after_pmc_403(
    tmp_path: Path, monkeypatch
) -> None:
    class _ForbiddenResponse:
        status_code = 403
        headers = {"content-type": "text/html; charset=utf-8"}
        url = "https://pmc.ncbi.nlm.nih.gov/articles/PMC8219259/pdf/nihms-1556781.pdf"

        def raise_for_status(self) -> None:
            raise RuntimeError("403 Forbidden")

        async def aiter_bytes(self):
            yield b""

    class _OaPdfResponse:
        status_code = 200
        headers = {"content-type": "application/pdf"}
        url = (
            "https://ftp.ncbi.nlm.nih.gov/pub/pmc/articles/PMC8219259/nihms-1556781.pdf"
        )

        def raise_for_status(self) -> None:
            return None

        async def aiter_bytes(self):
            yield b"%PDF-1.4 fake"

    class _OaXmlResponse:
        status_code = 200
        text = (
            "<?xml version='1.0' encoding='UTF-8'?>"
            "<OA><records><record id='PMC8219259'>"
            "<link format='pdf' href='ftp://ftp.ncbi.nlm.nih.gov/pub/pmc/articles/PMC8219259/nihms-1556781.pdf'/>"
            "</record></records></OA>"
        )

        def raise_for_status(self) -> None:
            return None

    class _FakeStreamContext:
        def __init__(self, response):
            self._response = response

        async def __aenter__(self):
            return self._response

        async def __aexit__(self, exc_type, exc, tb):
            del exc_type, exc, tb
            return None

    class _FakeClient:
        def __init__(self, *args, **kwargs):
            del args, kwargs

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            del exc_type, exc, tb
            return None

        async def get(self, url, headers=None):
            del headers
            if "oa.fcgi?id=PMC8219259" in str(url):
                return _OaXmlResponse()
            raise RuntimeError("unexpected metadata URL")

        def stream(self, method, url, headers=None):
            del method, headers
            text = str(url)
            if text.startswith("https://ftp.ncbi.nlm.nih.gov/"):
                return _FakeStreamContext(_OaPdfResponse())
            return _FakeStreamContext(_ForbiddenResponse())

    fake_httpx = types.SimpleNamespace(AsyncClient=_FakeClient)
    monkeypatch.setitem(sys.modules, "httpx", fake_httpx)

    tool = DownloadPdfTool(allowed_dir=tmp_path)
    result = asyncio.run(
        tool.execute(
            url="https://pmc.ncbi.nlm.nih.gov/articles/PMC8219259/pdf/nihms-1556781.pdf"
        )
    )
    payload = json.loads(result)
    assert payload["is_pdf"] is True
    assert payload["finalUrl"].startswith("https://ftp.ncbi.nlm.nih.gov/")


def test_download_pdf_tool_prefers_pmc_oa_before_direct_pdf_urls(
    tmp_path: Path, monkeypatch
) -> None:
    attempts: list[str] = []

    class _OaPdfResponse:
        status_code = 200
        headers = {"content-type": "application/pdf"}
        url = (
            "https://ftp.ncbi.nlm.nih.gov/pub/pmc/articles/PMC8219259/nihms-1556781.pdf"
        )

        def raise_for_status(self) -> None:
            return None

        async def aiter_bytes(self):
            yield b"%PDF-1.4 fake"

    class _OaXmlResponse:
        status_code = 200
        text = (
            "<?xml version='1.0' encoding='UTF-8'?>"
            "<OA><records><record id='PMC8219259'>"
            "<link format='pdf' href='ftp://ftp.ncbi.nlm.nih.gov/pub/pmc/articles/PMC8219259/nihms-1556781.pdf'/>"
            "</record></records></OA>"
        )

        def raise_for_status(self) -> None:
            return None

    class _FakeStreamContext:
        def __init__(self, response):
            self._response = response

        async def __aenter__(self):
            return self._response

        async def __aexit__(self, exc_type, exc, tb):
            del exc_type, exc, tb
            return None

    class _FakeClient:
        def __init__(self, *args, **kwargs):
            del args, kwargs

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            del exc_type, exc, tb
            return None

        async def get(self, url, headers=None):
            del headers
            if "oa.fcgi?id=PMC8219259" in str(url):
                return _OaXmlResponse()
            raise RuntimeError("unexpected metadata URL")

        def stream(self, method, url, headers=None):
            del method, headers
            attempts.append(str(url))
            if str(url).startswith("https://ftp.ncbi.nlm.nih.gov/"):
                return _FakeStreamContext(_OaPdfResponse())
            raise RuntimeError("direct PMC URL should not be attempted first")

    fake_httpx = types.SimpleNamespace(AsyncClient=_FakeClient)
    monkeypatch.setitem(sys.modules, "httpx", fake_httpx)

    tool = DownloadPdfTool(allowed_dir=tmp_path)
    result = asyncio.run(
        tool.execute(
            url="https://pmc.ncbi.nlm.nih.gov/articles/PMC8219259/pdf/nihms-1556781.pdf"
        )
    )
    payload = json.loads(result)
    assert payload["is_pdf"] is True
    assert attempts and attempts[0].startswith("https://ftp.ncbi.nlm.nih.gov/")


def test_download_pdf_tool_solves_pmc_pow_challenge(
    tmp_path: Path, monkeypatch
) -> None:
    challenge = "abc123:token"

    class _TextHtmlResponse:
        status_code = 200
        headers = {"content-type": "text/html; charset=utf-8"}
        url = "https://pmc.ncbi.nlm.nih.gov/articles/PMC12139829/pdf/file.pdf"
        text = (
            '<script>const POW_CHALLENGE = "abc123:token";'
            'const POW_DIFFICULTY = "1";'
            'const POW_COOKIE_NAME = "cloudpmc-viewer-pow";'
            'const POW_COOKIE_PATH = "/";</script>'
        )

        def raise_for_status(self) -> None:
            return None

        async def aiter_bytes(self):
            yield b"<html>Preparing to download</html>"

    class _PdfResponse:
        status_code = 200
        headers = {"content-type": "application/pdf"}
        url = "https://pmc.ncbi.nlm.nih.gov/articles/PMC12139829/pdf/file.pdf"

        def raise_for_status(self) -> None:
            return None

        async def aiter_bytes(self):
            yield b"%PDF-1.4 fake"

    class _FakeCookies:
        def __init__(self) -> None:
            self.values: dict[str, str] = {}

        def set(self, name, value, domain=None, path=None):
            del domain, path
            self.values[str(name)] = str(value)

    class _FakeStreamContext:
        def __init__(self, response):
            self._response = response

        async def __aenter__(self):
            return self._response

        async def __aexit__(self, exc_type, exc, tb):
            del exc_type, exc, tb
            return None

    class _FakeClient:
        def __init__(self, *args, **kwargs):
            del args, kwargs
            self.cookies = _FakeCookies()

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, tb):
            del exc_type, exc, tb
            return None

        async def get(self, url, headers=None):
            del headers
            if "pmc.ncbi.nlm.nih.gov" in str(url):
                return _TextHtmlResponse()
            raise RuntimeError("unexpected url")

        def stream(self, method, url, headers=None):
            del method, headers
            if "pmc.ncbi.nlm.nih.gov" in str(url):
                cookie = self.cookies.values.get("cloudpmc-viewer-pow", "")
                if cookie.startswith(challenge + ","):
                    return _FakeStreamContext(_PdfResponse())
                return _FakeStreamContext(_TextHtmlResponse())
            raise RuntimeError("unexpected url")

    fake_httpx = types.SimpleNamespace(AsyncClient=_FakeClient)
    monkeypatch.setitem(sys.modules, "httpx", fake_httpx)

    tool = DownloadPdfTool(allowed_dir=tmp_path)
    result = asyncio.run(
        tool.execute(
            url="https://pmc.ncbi.nlm.nih.gov/articles/PMC12139829/pdf/file.pdf"
        )
    )
    payload = json.loads(result)
    assert payload["is_pdf"] is True
    assert payload.get("pmc_pow_solved") is True
    assert Path(str(payload["output_path"])).exists()


def test_register_annolid_gui_tools_and_context_payload() -> None:
    calls: list[tuple[str, object]] = []

    def _mark(name: str, value: object = None) -> dict[str, object]:
        calls.append((name, value))
        return {"ok": True}

    registry = FunctionToolRegistry()
    register_annolid_gui_tools(
        registry,
        context_callback=lambda: {"provider": "ollama", "frame_number": 12},
        image_path_callback=lambda: "/tmp/shared.png",
        open_video_callback=lambda path: _mark("open_video", path),
        open_url_callback=lambda url: _mark("open_url", url),
        open_in_browser_callback=lambda url: _mark("open_in_browser", url),
        open_threejs_callback=lambda path_or_url: _mark("open_threejs", path_or_url),
        open_threejs_example_callback=lambda example_id="": _mark(
            "open_threejs_example", example_id
        ),
        web_get_dom_text_callback=lambda max_chars=8000: _mark(
            "web_get_dom_text", max_chars
        ),
        web_capture_screenshot_callback=lambda max_width=1600: _mark(
            "web_capture_screenshot", max_width
        ),
        web_describe_view_callback=lambda max_width=1600: _mark(
            "web_describe_view", max_width
        ),
        web_extract_structured_callback=lambda **kwargs: _mark(
            "web_extract_structured", kwargs
        ),
        web_click_callback=lambda selector: _mark("web_click", selector),
        web_type_callback=lambda selector, text, submit=False: _mark(
            "web_type", {"selector": selector, "text": text, "submit": bool(submit)}
        ),
        web_scroll_callback=lambda delta_y=800: _mark("web_scroll", delta_y),
        web_find_forms_callback=lambda: _mark("web_find_forms"),
        web_save_current_callback=lambda: _mark("web_save_current"),
        web_run_steps_callback=lambda steps, stop_on_error=True, max_steps=12: _mark(
            "web_run_steps",
            {
                "steps": steps,
                "stop_on_error": bool(stop_on_error),
                "max_steps": int(max_steps),
            },
        ),
        open_pdf_callback=lambda path="": _mark("open_pdf", path or None),
        pdf_get_state_callback=lambda: _mark("pdf_get_state"),
        pdf_get_text_callback=lambda max_chars=8000,
        pages=2,
        start_page=0,
        path="": _mark(
            "pdf_get_text",
            {
                "max_chars": int(max_chars),
                "pages": int(pages),
                "start_page": int(start_page),
                "path": str(path or ""),
            },
        ),
        pdf_summarize_callback=lambda path="",
        max_pages=80,
        max_extract_chars=350000: _mark(
            "pdf_summarize",
            {
                "path": str(path or ""),
                "max_pages": int(max_pages),
                "max_extract_chars": int(max_extract_chars),
            },
        ),
        pdf_find_sections_callback=lambda max_sections=20, max_pages=12: _mark(
            "pdf_find_sections",
            {"max_sections": int(max_sections), "max_pages": int(max_pages)},
        ),
        set_frame_callback=lambda frame_index: _mark("set_frame", frame_index),
        set_prompt_callback=lambda text: _mark("set_prompt", text),
        send_prompt_callback=lambda: _mark("send_prompt"),
        set_chat_model_callback=lambda provider, model: _mark(
            "set_chat_model", f"{provider}:{model}"
        ),
        select_annotation_model_callback=lambda model_name: _mark(
            "select_model", model_name
        ),
        track_next_frames_callback=lambda to_frame: _mark("track", to_frame),
        set_ai_text_prompt_callback=lambda text, use_countgd=False: _mark(
            "set_ai_text_prompt", f"{text}|{bool(use_countgd)}"
        ),
        run_ai_text_segmentation_callback=lambda: _mark("run_ai_text_segmentation"),
        segment_track_video_callback=lambda **kwargs: _mark(
            "segment_track_video", kwargs
        ),
        label_behavior_segments_callback=lambda **kwargs: _mark(
            "label_behavior_segments", kwargs
        ),
        process_video_behaviors_callback=lambda **kwargs: _mark(
            "process_video_behaviors", kwargs
        ),
        analyze_tracking_stats_callback=lambda **kwargs: _mark(
            "analyze_tracking_stats", kwargs
        ),
        start_realtime_stream_callback=lambda **kwargs: _mark(
            "start_realtime_stream", kwargs
        ),
        stop_realtime_stream_callback=lambda: _mark("stop_realtime_stream"),
        get_realtime_status_callback=lambda: _mark("get_realtime_status"),
        list_realtime_models_callback=lambda: _mark("list_realtime_models"),
        list_realtime_logs_callback=lambda: _mark("list_realtime_logs"),
        list_logs_callback=lambda: _mark("list_logs"),
        open_log_folder_callback=lambda target: _mark("open_log_folder", target),
        remove_log_folder_callback=lambda target: _mark("remove_log_folder", target),
        list_log_files_callback=lambda target,
        pattern="*",
        limit=200,
        recursive=True,
        sort_by="name",
        descending=False: _mark(
            "list_log_files",
            {
                "target": target,
                "pattern": pattern,
                "limit": int(limit),
                "recursive": bool(recursive),
                "sort_by": str(sort_by),
                "descending": bool(descending),
            },
        ),
        read_log_file_callback=lambda path, max_chars=12000, tail_lines=200: _mark(
            "read_log_file",
            {"path": path, "max_chars": int(max_chars), "tail_lines": int(tail_lines)},
        ),
        search_logs_callback=lambda query,
        target="logs",
        pattern="*",
        case_sensitive=False,
        use_regex=False,
        max_matches=100,
        max_files=50: _mark(
            "search_logs",
            {
                "query": query,
                "target": target,
                "pattern": pattern,
                "case_sensitive": bool(case_sensitive),
                "use_regex": bool(use_regex),
                "max_matches": int(max_matches),
                "max_files": int(max_files),
            },
        ),
        save_citation_callback=lambda **kwargs: _mark("save_citation", kwargs),
        verify_citations_callback=lambda **kwargs: _mark("verify_citations", kwargs),
        generate_annolid_tutorial_callback=lambda **kwargs: _mark(
            "generate_annolid_tutorial", kwargs
        ),
        self_update_callback=lambda **kwargs: _mark("self_update", kwargs),
    )
    assert registry.has("gui_context")
    assert registry.has("gui_shared_image_path")
    assert registry.has("gui_open_video")
    assert registry.has("gui_open_url")
    assert registry.has("gui_open_in_browser")
    assert registry.has("gui_open_threejs")
    assert registry.has("gui_open_threejs_example")
    assert registry.has("gui_web_get_dom_text")
    assert registry.has("gui_web_capture_screenshot")
    assert registry.has("gui_web_describe_view")
    assert registry.has("gui_web_extract_structured")
    assert registry.has("gui_web_click")
    assert registry.has("gui_web_type")
    assert registry.has("gui_web_scroll")
    assert registry.has("gui_web_find_forms")
    assert registry.has("gui_web_save_current")
    assert registry.has("gui_web_run_steps")
    assert registry.has("gui_open_pdf")
    assert registry.has("gui_pdf_get_state")
    assert registry.has("gui_pdf_get_text")
    assert registry.has("gui_pdf_summarize")
    assert registry.has("gui_pdf_find_sections")
    assert registry.has("gui_set_frame")
    assert registry.has("gui_set_chat_prompt")
    assert registry.has("gui_send_chat_prompt")
    assert registry.has("gui_set_chat_model")
    assert registry.has("gui_select_annotation_model")
    assert registry.has("gui_track_next_frames")
    assert registry.has("gui_set_ai_text_prompt")
    assert registry.has("gui_run_ai_text_segmentation")
    assert registry.has("gui_segment_track_video")
    assert registry.has("gui_label_behavior_segments")
    assert registry.has("gui_process_video_behaviors")
    assert registry.has("gui_analyze_tracking_stats")
    assert registry.has("gui_start_realtime_stream")
    assert registry.has("gui_stop_realtime_stream")
    assert registry.has("gui_get_realtime_status")
    assert registry.has("gui_list_realtime_models")
    assert registry.has("gui_list_realtime_logs")
    assert registry.has("gui_list_logs")
    assert registry.has("gui_open_log_folder")
    assert registry.has("gui_remove_log_folder")
    assert registry.has("gui_list_log_files")
    assert registry.has("gui_read_log_file")
    assert registry.has("gui_search_logs")
    assert registry.has("gui_save_citation")
    assert registry.has("gui_verify_citations")
    assert registry.has("gui_generate_annolid_tutorial")
    assert registry.has("gui_self_update")
    ctx = asyncio.run(registry.execute("gui_context", {}))
    ctx_payload = json.loads(ctx)
    assert ctx_payload["provider"] == "ollama"
    image = asyncio.run(registry.execute("gui_shared_image_path", {}))
    image_payload = json.loads(image)
    assert image_payload["image_path"] == "/tmp/shared.png"
    result = asyncio.run(registry.execute("gui_open_video", {"path": "/tmp/a.mp4"}))
    assert json.loads(result)["ok"] is True
    open_url = asyncio.run(
        registry.execute("gui_open_url", {"url": "https://example.org"})
    )
    assert json.loads(open_url)["ok"] is True
    open_in_browser = asyncio.run(
        registry.execute("gui_open_in_browser", {"url": "https://example.org"})
    )
    assert json.loads(open_in_browser)["ok"] is True
    open_threejs = asyncio.run(
        registry.execute(
            "gui_open_threejs",
            {"path_or_url": "/tmp/annolid_threejs_examples/two_mice.html"},
        )
    )
    assert json.loads(open_threejs)["ok"] is True
    open_threejs_example = asyncio.run(
        registry.execute(
            "gui_open_threejs_example",
            {"example_id": "two_mice_html"},
        )
    )
    assert json.loads(open_threejs_example)["ok"] is True
    web_get_dom_text = asyncio.run(
        registry.execute("gui_web_get_dom_text", {"max_chars": 1200})
    )
    assert json.loads(web_get_dom_text)["ok"] is True
    web_capture_screenshot = asyncio.run(
        registry.execute("gui_web_capture_screenshot", {"max_width": 1280})
    )
    assert json.loads(web_capture_screenshot)["ok"] is True
    web_describe_view = asyncio.run(
        registry.execute("gui_web_describe_view", {"max_width": 1280})
    )
    assert json.loads(web_describe_view)["ok"] is True
    web_extract_structured = asyncio.run(
        registry.execute(
            "gui_web_extract_structured",
            {
                "fields": ["title", "summary"],
                "max_chars": 1200,
                "include_excerpt": True,
            },
        )
    )
    assert json.loads(web_extract_structured)["ok"] is True
    web_click = asyncio.run(
        registry.execute("gui_web_click", {"selector": "button.submit"})
    )
    assert json.loads(web_click)["ok"] is True
    web_type = asyncio.run(
        registry.execute(
            "gui_web_type",
            {"selector": "input[name='q']", "text": "annolid", "submit": True},
        )
    )
    assert json.loads(web_type)["ok"] is True
    web_scroll = asyncio.run(registry.execute("gui_web_scroll", {"delta_y": 600}))
    assert json.loads(web_scroll)["ok"] is True
    web_find_forms = asyncio.run(registry.execute("gui_web_find_forms", {}))
    assert json.loads(web_find_forms)["ok"] is True
    web_save_current = asyncio.run(registry.execute("gui_web_save_current", {}))
    assert json.loads(web_save_current)["ok"] is True
    web_run_steps = asyncio.run(
        registry.execute(
            "gui_web_run_steps",
            {
                "steps": [{"action": "open_url", "url": "https://example.org"}],
                "stop_on_error": True,
                "max_steps": 5,
            },
        )
    )
    assert json.loads(web_run_steps)["ok"] is True
    open_pdf = asyncio.run(registry.execute("gui_open_pdf", {}))
    assert json.loads(open_pdf)["ok"] is True
    open_pdf_with_path = asyncio.run(
        registry.execute("gui_open_pdf", {"path": "/tmp/paper.pdf"})
    )
    assert json.loads(open_pdf_with_path)["ok"] is True
    pdf_state = asyncio.run(registry.execute("gui_pdf_get_state", {}))
    assert json.loads(pdf_state)["ok"] is True
    pdf_text = asyncio.run(
        registry.execute("gui_pdf_get_text", {"max_chars": 1200, "pages": 2})
    )
    assert json.loads(pdf_text)["ok"] is True
    pdf_text_with_path = asyncio.run(
        registry.execute(
            "gui_pdf_get_text",
            {"max_chars": 1200, "pages": 2, "path": "/tmp/paper.pdf"},
        )
    )
    assert json.loads(pdf_text_with_path)["ok"] is True
    pdf_summary = asyncio.run(
        registry.execute(
            "gui_pdf_summarize",
            {
                "path": "/tmp/paper.pdf",
                "max_pages": 30,
                "max_extract_chars": 250000,
            },
        )
    )
    assert json.loads(pdf_summary)["ok"] is True
    pdf_sections = asyncio.run(
        registry.execute("gui_pdf_find_sections", {"max_sections": 10, "max_pages": 8})
    )
    assert json.loads(pdf_sections)["ok"] is True
    asyncio.run(registry.execute("gui_set_frame", {"frame_index": 3}))
    asyncio.run(registry.execute("gui_set_chat_prompt", {"text": "describe this"}))
    asyncio.run(registry.execute("gui_send_chat_prompt", {}))
    asyncio.run(
        registry.execute(
            "gui_set_chat_model", {"provider": "ollama", "model": "qwen3:8b"}
        )
    )
    asyncio.run(
        registry.execute(
            "gui_select_annotation_model", {"model_name": "Segment Anything 2"}
        )
    )
    asyncio.run(registry.execute("gui_track_next_frames", {"to_frame": 120}))
    asyncio.run(
        registry.execute(
            "gui_set_ai_text_prompt",
            {"text": "mouse", "use_countgd": True},
        )
    )
    asyncio.run(registry.execute("gui_run_ai_text_segmentation", {}))
    asyncio.run(
        registry.execute(
            "gui_segment_track_video",
            {
                "path": "/tmp/a.mp4",
                "text_prompt": "mouse",
                "mode": "track",
                "to_frame": 120,
            },
        )
    )
    asyncio.run(
        registry.execute(
            "gui_label_behavior_segments",
            {
                "path": "/tmp/a.mp4",
                "behavior_labels": ["walking", "eating"],
                "segment_mode": "uniform",
                "segment_frames": 30,
            },
        )
    )
    asyncio.run(
        registry.execute(
            "gui_process_video_behaviors",
            {
                "path": "/tmp/a.mp4",
                "text_prompt": "mouse",
                "run_tracking": True,
                "run_behavior_labeling": True,
            },
        )
    )
    asyncio.run(
        registry.execute(
            "gui_analyze_tracking_stats",
            {"root_dir": "/tmp/results", "video_id": "mouse", "top_k": 5},
        )
    )
    asyncio.run(
        registry.execute(
            "gui_start_realtime_stream",
            {
                "camera_source": "0",
                "model_name": "mediapipe_face",
                "classify_eye_blinks": True,
            },
        )
    )
    asyncio.run(registry.execute("gui_stop_realtime_stream", {}))
    asyncio.run(registry.execute("gui_get_realtime_status", {}))
    asyncio.run(registry.execute("gui_list_realtime_models", {}))
    asyncio.run(registry.execute("gui_list_realtime_logs", {}))
    asyncio.run(registry.execute("gui_list_logs", {}))
    asyncio.run(
        registry.execute(
            "gui_open_log_folder",
            {"target": "logs"},
        )
    )
    asyncio.run(
        registry.execute(
            "gui_remove_log_folder",
            {"target": "realtime"},
        )
    )
    asyncio.run(
        registry.execute(
            "gui_list_log_files",
            {
                "target": "logs",
                "pattern": "*.log",
                "limit": 20,
                "recursive": True,
                "sort_by": "mtime",
                "descending": True,
            },
        )
    )
    asyncio.run(
        registry.execute(
            "gui_read_log_file",
            {"path": "/tmp/logs/app/annolid.log", "max_chars": 2000, "tail_lines": 50},
        )
    )
    asyncio.run(
        registry.execute(
            "gui_search_logs",
            {
                "query": "error",
                "target": "logs",
                "pattern": "*.log",
                "case_sensitive": False,
                "use_regex": False,
                "max_matches": 10,
                "max_files": 5,
            },
        )
    )
    asyncio.run(
        registry.execute(
            "gui_save_citation",
            {"key": "annolid2024", "bib_file": "references.bib", "source": "pdf"},
        )
    )
    asyncio.run(
        registry.execute(
            "gui_verify_citations",
            {"bib_file": "references.bib", "limit": 20},
        )
    )
    asyncio.run(
        registry.execute(
            "gui_generate_annolid_tutorial",
            {
                "topic": "Realtime camera workflow",
                "level": "beginner",
                "save_to_file": True,
                "include_code_refs": True,
            },
        )
    )
    asyncio.run(
        registry.execute(
            "gui_self_update",
            {
                "channel": "stable",
                "execute": False,
                "run_post_check": True,
                "require_signature": False,
            },
        )
    )
    assert calls == [
        ("open_video", "/tmp/a.mp4"),
        ("open_url", "https://example.org"),
        ("open_in_browser", "https://example.org"),
        ("open_threejs", "/tmp/annolid_threejs_examples/two_mice.html"),
        ("open_threejs_example", "two_mice_html"),
        ("web_get_dom_text", 1200),
        ("web_capture_screenshot", 1280),
        ("web_describe_view", 1280),
        (
            "web_extract_structured",
            {
                "fields": ["title", "summary"],
                "max_chars": 1200,
                "include_excerpt": True,
            },
        ),
        ("web_click", "button.submit"),
        (
            "web_type",
            {"selector": "input[name='q']", "text": "annolid", "submit": True},
        ),
        ("web_scroll", 600),
        ("web_find_forms", None),
        ("web_save_current", None),
        (
            "web_run_steps",
            {
                "steps": [{"action": "open_url", "url": "https://example.org"}],
                "stop_on_error": True,
                "max_steps": 5,
            },
        ),
        ("open_pdf", None),
        ("open_pdf", "/tmp/paper.pdf"),
        ("pdf_get_state", None),
        (
            "pdf_get_text",
            {"max_chars": 1200, "pages": 2, "start_page": 0, "path": ""},
        ),
        (
            "pdf_get_text",
            {
                "max_chars": 1200,
                "pages": 2,
                "start_page": 0,
                "path": "/tmp/paper.pdf",
            },
        ),
        (
            "pdf_summarize",
            {
                "path": "/tmp/paper.pdf",
                "max_pages": 30,
                "max_extract_chars": 250000,
            },
        ),
        ("pdf_find_sections", {"max_sections": 10, "max_pages": 8}),
        ("set_frame", 3),
        ("set_prompt", "describe this"),
        ("send_prompt", None),
        ("set_chat_model", "ollama:qwen3:8b"),
        ("select_model", "Segment Anything 2"),
        ("track", 120),
        ("set_ai_text_prompt", "mouse|True"),
        ("run_ai_text_segmentation", None),
        (
            "segment_track_video",
            {
                "path": "/tmp/a.mp4",
                "text_prompt": "mouse",
                "mode": "track",
                "to_frame": 120,
            },
        ),
        (
            "label_behavior_segments",
            {
                "path": "/tmp/a.mp4",
                "behavior_labels": ["walking", "eating"],
                "segment_mode": "uniform",
                "segment_frames": 30,
            },
        ),
        (
            "process_video_behaviors",
            {
                "path": "/tmp/a.mp4",
                "text_prompt": "mouse",
                "run_tracking": True,
                "run_behavior_labeling": True,
            },
        ),
        (
            "analyze_tracking_stats",
            {"root_dir": "/tmp/results", "video_id": "mouse", "top_k": 5},
        ),
        (
            "start_realtime_stream",
            {
                "camera_source": "0",
                "model_name": "mediapipe_face",
                "classify_eye_blinks": True,
            },
        ),
        ("stop_realtime_stream", None),
        ("get_realtime_status", None),
        ("list_realtime_models", None),
        ("list_realtime_logs", None),
        ("list_logs", None),
        ("open_log_folder", "logs"),
        ("remove_log_folder", "realtime"),
        (
            "list_log_files",
            {
                "target": "logs",
                "pattern": "*.log",
                "limit": 20,
                "recursive": True,
                "sort_by": "mtime",
                "descending": True,
            },
        ),
        (
            "read_log_file",
            {"path": "/tmp/logs/app/annolid.log", "max_chars": 2000, "tail_lines": 50},
        ),
        (
            "search_logs",
            {
                "query": "error",
                "target": "logs",
                "pattern": "*.log",
                "case_sensitive": False,
                "use_regex": False,
                "max_matches": 10,
                "max_files": 5,
            },
        ),
        (
            "save_citation",
            {"key": "annolid2024", "bib_file": "references.bib", "source": "pdf"},
        ),
        (
            "verify_citations",
            {"bib_file": "references.bib", "limit": 20},
        ),
        (
            "generate_annolid_tutorial",
            {
                "topic": "Realtime camera workflow",
                "level": "beginner",
                "save_to_file": True,
                "include_code_refs": True,
            },
        ),
        (
            "self_update",
            {
                "channel": "stable",
                "execute": False,
                "run_post_check": True,
                "require_signature": False,
            },
        ),
    ]
