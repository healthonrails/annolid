from __future__ import annotations

import json
import os
from pathlib import Path

from PIL import Image
from qtpy import QtCore, QtWidgets

from annolid.gui.mixins.project_workflow_mixin import ProjectWorkflowMixin
from annolid.gui.model_manager import AIModelManager
from annolid.gui.widgets.inference_wizard import InferenceWizard, InferenceWorker
from annolid.gui.widgets.train_model_dialog import TrainModelDialog
from annolid.gui.widgets.training_wizard import TrainingWizard
from annolid.gui.yolo_training_manager import YOLOTrainingManager

os.environ.setdefault("QT_QPA_PLATFORM", "minimal")

_QAPP = None


def _ensure_qapp():
    global _QAPP
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
    _QAPP = app
    return _QAPP


def _make_coco_dir(root: Path) -> Path:
    (root / "images").mkdir(parents=True)
    (root / "annotations").mkdir(parents=True)
    Image.new("RGB", (32, 24)).save(root / "images" / "a.png")
    Image.new("RGB", (32, 24)).save(root / "images" / "b.png")
    train_payload = {
        "images": [{"id": 1, "file_name": "a.png", "width": 32, "height": 24}],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 1,
                "bbox": [2, 2, 8, 8],
                "area": 64,
                "iscrowd": 0,
            }
        ],
        "categories": [{"id": 1, "name": "mouse"}],
    }
    val_payload = {
        "images": [{"id": 2, "file_name": "b.png", "width": 32, "height": 24}],
        "annotations": [
            {
                "id": 2,
                "image_id": 2,
                "category_id": 1,
                "bbox": [2, 2, 8, 8],
                "area": 64,
                "iscrowd": 0,
            }
        ],
        "categories": [{"id": 1, "name": "mouse"}],
    }
    (root / "annotations" / "instances_train.json").write_text(
        json.dumps(train_payload), encoding="utf-8"
    )
    (root / "annotations" / "instances_val.json").write_text(
        json.dumps(val_payload), encoding="utf-8"
    )
    return root


def test_train_model_dialog_overrides_exclude_reserved_keys_and_apply_presets() -> None:
    _ensure_qapp()
    dlg = TrainModelDialog()
    try:
        dlg.radio_btn_yolo.setChecked(True)
        dlg.apply_yolo_preset("small_dataset")

        overrides = dlg.get_yolo_train_overrides()
        assert "model" not in overrides
        assert "device" not in overrides
        assert "plots" not in overrides
        assert overrides["freeze"] == 10
        assert overrides["optimizer"] == "AdamW"
        assert overrides["cos_lr"] is True
        assert abs(overrides["lr0"] - 0.002) < 1e-9
        assert dlg.yolo_epochs == 120

        # Non-registry Family/Task/Size combination updates yolo_model_file properly
        fam_idx = dlg.yolo_family_combo.findData("yolo26")
        assert fam_idx >= 0
        dlg.yolo_family_combo.setCurrentIndex(fam_idx)
        dlg.yolo_task_combo.setCurrentText("Pose Estimation")
        dlg.yolo_size_combo.setCurrentText("S")
        assert dlg.yolo_model_file == "yolo26s-pose.pt"

        # Resume checkbox enables only when checkpoint path is provided
        assert dlg.yolo_resume_checkbox.isEnabled() is False
        dlg.trainedModelLineEdit.setText("/tmp/last.pt")
        assert dlg.yolo_resume_checkbox.isEnabled() is True
        dlg.trainedModelLineEdit.setText("")
        assert dlg.yolo_resume_checkbox.isEnabled() is False
    finally:
        dlg.close()


def test_training_wizard_handles_coco_export_non_registry_weights_and_forwards_config(
    tmp_path: Path,
) -> None:
    _ensure_qapp()
    coco_root = _make_coco_dir(tmp_path / "coco_dataset")

    wizard = TrainingWizard()
    try:
        # Export completion for COCO switches to COCO tab and populates coco_picker
        wizard.select_dataset_page._on_dataset_export_complete(coco_root, "coco")
        assert wizard.select_dataset_page.get_dataset_type() == "coco"
        assert wizard.select_dataset_page.get_dataset_path() == str(coco_root)
        assert wizard.select_dataset_page.isComplete() is True

        # Entering Configure page auto-matches task & recommends small_dataset for 1 train img
        wizard.configure_params_page.initializePage()
        assert wizard.configure_params_page.yolo_task_combo.currentText() == "Detection"
        assert (
            wizard.configure_params_page.yolo_train_preset_combo.currentData()
            == "small_dataset"
        )

        # Switching to YOLO26 S Pose Estimation updates combo data instead of silently keeping old model
        fam_idx = wizard.configure_params_page.yolo_family_combo.findData("yolo26")
        wizard.configure_params_page.yolo_family_combo.setCurrentIndex(fam_idx)
        wizard.configure_params_page.yolo_task_combo.setCurrentText("Pose Estimation")
        wizard.configure_params_page.yolo_size_combo.setCurrentText("S")
        cfg = wizard.get_training_config()
        assert cfg["model"] == "yolo26s-pose.pt"
        assert cfg["freeze"] == 10
        assert cfg["optimizer"] == "AdamW"
        assert cfg["cos_lr"] is True
        assert cfg["patience"] == 40
    finally:
        wizard.close()

    # Verify ProjectWorkflowMixin._start_yolo_training_from_wizard forwards all overrides + local checkpoint
    local_ckpt = tmp_path / "my_last.pt"
    local_ckpt.write_bytes(b"pt")
    recorded = {}

    class _FakeManager:
        def start_training(self, **kwargs):
            recorded.update(kwargs)
            return True

    class _FakeStatusBar:
        def showMessage(self, *_args, **_kwargs):
            pass

    class _FakeWindow(ProjectWorkflowMixin):
        def ensure_yolo_training_manager(self):
            return _FakeManager()

        def statusBar(self):
            return _FakeStatusBar()

        def tr(self, text):
            return text

    win = _FakeWindow()
    win._start_yolo_training_from_wizard(
        {
            "model": str(local_ckpt),
            "dataset_path": str(coco_root),
            "epochs": 80,
            "imgsz": 640,
            "batch": 8,
            "device": "mps",
            "lr0": 0.002,
            "patience": 40,
            "freeze": 10,
            "optimizer": "AdamW",
            "cos_lr": True,
            "cache": True,
            "plots": False,
            "resume": True,
            "output_dir": str(tmp_path / "out"),
        }
    )

    assert recorded["model_path"] == str(local_ckpt.resolve())
    assert recorded["yolo_model_file"] == "my_last.pt"
    assert recorded["plots"] is False
    assert recorded["resume"] is True
    assert recorded["train_overrides"]["lr0"] == 0.002
    assert recorded["train_overrides"]["patience"] == 40
    assert recorded["train_overrides"]["freeze"] == 10
    assert recorded["train_overrides"]["optimizer"] == "AdamW"
    assert recorded["train_overrides"]["cos_lr"] is True


def test_ai_model_manager_discovers_managed_yolo_runs_and_selects_trained_model(
    tmp_path: Path, monkeypatch
) -> None:
    _ensure_qapp()
    runs_root = tmp_path / "runs"
    best_pt = (
        runs_root / "yolo" / "yolo11s-seg" / "20260501_100000" / "weights" / "best.pt"
    )
    best_pt.parent.mkdir(parents=True)
    best_pt.write_bytes(b"pt")
    monkeypatch.setenv("ANNOLID_RUNS_ROOT", str(runs_root))

    settings = QtCore.QSettings(
        str(tmp_path / "settings.ini"), QtCore.QSettings.IniFormat
    )
    parent = QtWidgets.QWidget()
    combo = QtWidgets.QComboBox(parent)
    mgr = AIModelManager(
        parent=parent,
        combo=combo,
        settings=settings,
        base_config={"ai": {}},
        canvas_getter=lambda: None,
    )
    mgr.initialize()

    discovered_weights = [
        str(Path(cfg.weight_file).resolve()) for cfg in mgr._auto_discovered_configs
    ]
    assert str(best_pt.resolve()) in discovered_weights

    # Selecting another custom checkpoint registers & activates it immediately
    custom_pt = tmp_path / "custom_run" / "weights" / "best.pt"
    custom_pt.parent.mkdir(parents=True)
    custom_pt.write_bytes(b"pt")
    selected_name = mgr.select_or_add_custom_model(custom_pt)
    assert selected_name is not None
    assert combo.currentText() == selected_name
    assert str(Path(mgr.get_current_weight()).resolve()) == str(custom_pt.resolve())


def test_inference_wizard_and_worker_forward_yolo_controls_and_count_detections(
    tmp_path: Path, monkeypatch
) -> None:
    _ensure_qapp()
    video_file = tmp_path / "clip.mp4"
    video_file.write_bytes(b"fake")
    ckpt = tmp_path / "best.pt"
    ckpt.write_bytes(b"pt")

    wizard = InferenceWizard(
        model_path="yolo11n-seg.pt",
        video_path=str(video_file),
        recent_models=[str(ckpt)],
    )
    try:
        # Built-in YOLO weight names are accepted cleanly without local file on disk
        assert wizard.select_model_page.isComplete() is True
        assert wizard.select_model_page.recent_list.count() == 1

        # Tracking toggles frame batch size availability
        assert wizard.configure_page.batch_size_spin.isEnabled() is False
        wizard.configure_page.enable_tracking_check.setChecked(False)
        assert wizard.configure_page.batch_size_spin.isEnabled() is True
        wizard.configure_page.batch_size_spin.setValue(4)
        wizard.configure_page.iou_threshold_spin.setValue(0.6)
        wizard.configure_page.imgsz_spin.setValue(960)
        cfg = wizard.get_inference_config()
        assert cfg["batch_size"] == 4
        assert abs(cfg["iou_threshold"] - 0.6) < 1e-6
        assert cfg["imgsz"] == 960
    finally:
        wizard.close()

    # Verify InferenceWorker forwards all YOLO controls and accumulates detection_count
    calls = {}

    class _FakeProc:
        def __init__(self, **kwargs):
            calls["init"] = kwargs
            self.detection_count = 0

        def run_inference(self, **kwargs):
            calls["run"] = kwargs
            self.detection_count = 17
            return "yolo#3"

    monkeypatch.setattr("annolid.segmentation.yolos.InferenceProcessor", _FakeProc)

    worker = InferenceWorker(
        {
            "model_type": "yolo",
            "model_path": str(ckpt),
            "videos": [str(video_file)],
            "score_threshold": 0.42,
            "iou_threshold": 0.55,
            "imgsz": 800,
            "top_k": 30,
            "device": "cpu",
            "batch_size": 4,
            "skip_existing": False,
            "enable_tracking": False,
            "save_labelme": False,
            "save_csv": False,
            "save_video": True,
            "output_dir": str(tmp_path / "out"),
        }
    )

    summaries = []
    worker.finished.connect(lambda s: summaries.append(s))
    worker.run()

    assert calls["init"]["persist_json"] is False
    assert calls["init"]["conf"] == 0.42
    assert calls["init"]["iou"] == 0.55
    assert calls["init"]["imgsz"] == 800
    assert calls["init"]["max_det"] == 30
    assert calls["init"]["device"] == "cpu"
    assert calls["run"]["batch_size"] == 4
    assert calls["run"]["skip_existing"] is False
    assert calls["run"]["save_video_path"] is not None
    assert summaries and summaries[0]["total_detections"] == 17
    assert summaries[0]["processed_frames"] == 3


def test_yolo_training_manager_stop_training_signals_active_worker() -> None:
    _ensure_qapp()
    window = QtWidgets.QMainWindow()
    manager = YOLOTrainingManager(window)
    try:
        stopped = []

        class _FakeWorker:
            def request_stop(self):
                stopped.append(True)

            def deleteLater(self):
                pass

        manager._active_jobs.append((None, _FakeWorker(), ""))  # type: ignore[arg-type]
        assert manager.stop_training() is True
        assert stopped == [True]
        manager._active_jobs.clear()
    finally:
        manager.cleanup()
        window.close()
