from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import time
from types import SimpleNamespace
import zipfile

import pytest
from qtpy import QtCore, QtGui, QtWidgets

from annolid.domain import default_behavior_spec, save_behavior_spec
from annolid.gui.controllers.menu import MenuController
from annolid.gui.controllers.project_controller import ProjectController
from annolid.gui.file_dock import FileDockMixin
from annolid.gui.managers.settings_manager import SettingsManager
from annolid.gui.mixins.file_browser_mixin import FileBrowserMixin
from annolid.gui.mixins.persistence_lifecycle_mixin import PersistenceLifecycleMixin
from annolid.gui.mixins.project_workflow_mixin import ProjectWorkflowMixin
from annolid.gui.services.project_service import ProjectService
from annolid.gui.widgets.dashboard import RecentItemWidget, RecentProject
from annolid.gui.widgets.dataset_wizard import DatasetExportWizard
from annolid.gui.widgets.file_audit_widget import FileAuditDialog, FileAuditWidget
from annolid.gui.widgets.label_collection_dialog import LabelCollectionDialog
from annolid.gui.widgets.project_hub_dialog import ProjectHubDialog
from annolid.gui.widgets.project_wizard import ProjectWizard
from annolid.infrastructure import AnnotationStore
from annolid.services.project_workspace import (
    PROJECT_CONFIG_FILENAME,
    audit_project_annotations,
    compute_project_workflow_status,
    export_project_bundle,
    normalize_project_config,
    prune_missing_recent_entries,
    register_project_videos,
    rename_annotation_file_pair,
    scan_annotation_audit,
    scan_project_statistics,
    upsert_recent_project_entries,
    validate_project_name,
)


def _wait_until(predicate, *, timeout_s: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        QtCore.QCoreApplication.processEvents()
        if predicate():
            return True
        time.sleep(0.01)
    QtCore.QCoreApplication.processEvents()
    return bool(predicate())


@pytest.fixture(scope="module")
def qapp():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
    return app


# ---------------------------------------------------------------------------
# Layer 2: Project Workspace & File-Pair Service
# ---------------------------------------------------------------------------


def test_validate_project_name_blocks_traversal_and_control_chars() -> None:
    assert validate_project_name("MouseExp_01") == (True, "")
    for bad in ("", "   ", ".", "..", "../escape", "a/b", "a\\b", "bad\nname"):
        ok, reason = validate_project_name(bad)
        assert ok is False
        assert reason


def test_normalize_and_dual_storage_project_statistics_and_workflow(
    tmp_path: Path,
) -> None:
    proj = tmp_path / "demo_project"
    (proj / "images" / "nested").mkdir(parents=True)
    (proj / "annotations" / "clip1").mkdir(parents=True)
    (proj / "models").mkdir(parents=True)
    (proj / "videos").mkdir(parents=True)
    (proj / "exports").mkdir(parents=True)

    # Start with a schema-only project.annolid.json (as produced by older wizard/save_schema).
    schema = default_behavior_spec()
    save_behavior_spec(schema, proj / PROJECT_CONFIG_FILENAME)
    raw = json.loads((proj / PROJECT_CONFIG_FILENAME).read_text(encoding="utf-8"))
    normalized, upgraded = normalize_project_config(proj, raw)
    assert upgraded is True
    assert normalized["name"] == "demo_project"
    assert normalized["directories"]["annotations"] == "annotations"
    assert normalized["behaviors"]  # behavior schema preserved

    # Add nested image, per-frame JSON (frame 0), AND dual-storage NDJSON store (frames 0, 1, 2).
    (proj / "images" / "nested" / "frame_000000000.png").write_bytes(b"\x89PNG")
    clip_dir = proj / "annotations" / "clip1"
    frame0_json = clip_dir / "clip1_000000000.json"
    frame0_json.write_text(json.dumps({"shapes": []}), encoding="utf-8")

    store = AnnotationStore.for_frame_path(frame0_json)
    store.append_frame({"frame": 0, "shapes": []})
    store.append_frame({"frame": 1, "shapes": []})
    store.append_frame({"frame": 2, "shapes": []})

    # Register an external video + a missing video.
    ext_video = tmp_path / "external_trial.mp4"
    ext_video.write_bytes(b"fake-mp4")
    missing_video = tmp_path / "deleted_trial.mp4"
    register_project_videos(proj, normalized, [ext_video, missing_video])

    stats = scan_project_statistics(proj, normalized)
    assert stats["total_images"] == 1
    # Frame 0 deduplicated between JSON & NDJSON store + frames 1 & 2 from store = 3 total!
    assert stats["total_annotations"] == 3
    assert stats["store_only_annotations"] == 2
    assert stats["total_videos"] == 1
    assert stats["missing_videos"] == 1
    assert stats["has_behavior_schema"] is True

    wf = compute_project_workflow_status(proj, normalized, stats)
    assert wf["steps"]["project"] == "completed"
    assert wf["steps"]["collect"] == "completed"
    assert wf["steps"]["annotate"] == "completed"
    assert wf["steps"]["train"] == "active"
    assert wf["next_step"] == "train"

    # Add model checkpoint, tracking CSV, and HTML report -> all 6 steps complete.
    (proj / "models" / "best.pt").write_bytes(b"weights")
    (proj / "exports" / "tracks.csv").write_text("frame,x,y\n0,1,2\n", encoding="utf-8")
    (proj / "exports" / "report.html").write_text("<html></html>", encoding="utf-8")
    stats2 = scan_project_statistics(proj, normalized)
    wf2 = compute_project_workflow_status(proj, normalized, stats2)
    assert wf2["completed_count"] == 6
    assert all(status == "completed" for status in wf2["steps"].values())


def test_export_project_bundle_guards_self_nesting_and_exports_zip(
    tmp_path: Path,
) -> None:
    svc = ProjectService()
    ok, _msg = svc.create_project(tmp_path, "bundle_proj")
    assert ok is True
    proj = tmp_path / "bundle_proj"
    (proj / "pose_schema.json").write_text('{"keypoints": ["nose"]}', encoding="utf-8")
    (proj / "annotations" / "clip_annotations.ndjson").write_text(
        '{"frame": 0}\n', encoding="utf-8"
    )
    (proj / "images" / "img1.png").write_bytes(b"img")

    # Exporting into project directory or descendant must be rejected.
    bad_ok, bad_msg = export_project_bundle(proj, proj / "exports" / "nested")
    assert bad_ok is False
    assert "inside" in bad_msg.lower()

    # Exporting as ZIP succeeds atomically and includes schema, ignore, NDJSON, and images.
    zip_out = tmp_path / "bundle_proj.zip"
    zip_ok, _zip_msg = export_project_bundle(
        proj, zip_out, include_data=True, include_models=False, as_zip=True
    )
    assert zip_ok is True
    assert zip_out.is_file()
    with zipfile.ZipFile(zip_out, "r") as zf:
        names = set(zf.namelist())
    assert PROJECT_CONFIG_FILENAME in names
    assert "pose_schema.json" in names
    assert ".annolidignore" in names
    assert "annotations/clip_annotations.ndjson" in names
    assert "images/img1.png" in names


def test_rename_annotation_file_pair_syncs_sidecar_json_and_image_path(
    tmp_path: Path,
) -> None:
    img = tmp_path / "frame_001.png"
    img.write_bytes(b"\x89PNG")
    label = tmp_path / "frame_001.json"
    label.write_text(
        json.dumps({"version": "5.0", "imagePath": "frame_001.png", "shapes": []}),
        encoding="utf-8",
    )

    result = rename_annotation_file_pair(img, "grooming_001.png")
    new_img = tmp_path / "grooming_001.png"
    new_label = tmp_path / "grooming_001.json"
    assert result["renamed_sidecar"] is True
    assert new_img.is_file() and not img.exists()
    assert new_label.is_file() and not label.exists()
    payload = json.loads(new_label.read_text(encoding="utf-8"))
    assert payload["imagePath"] == "grooming_001.png"

    # Rejects path traversal and existing targets.
    with pytest.raises(ValueError):
        rename_annotation_file_pair(new_img, "../escape.png")
    other = tmp_path / "taken.png"
    other.write_bytes(b"\x89PNG")
    with pytest.raises(FileExistsError):
        rename_annotation_file_pair(new_img, "taken.png")


def test_recent_project_helpers_deduplicate_and_prune(tmp_path: Path) -> None:
    p1 = tmp_path / "proj_a"
    p2 = tmp_path / "proj_b"
    p1.mkdir()
    p2.mkdir()
    entries = upsert_recent_project_entries(
        [], p1, name="Alpha", timestamp="2026-01-01T10:00:00"
    )
    entries = upsert_recent_project_entries(
        entries, p2, name="Beta", timestamp="2026-01-02T10:00:00"
    )
    entries = upsert_recent_project_entries(
        entries, p1, name="Alpha", timestamp="2026-01-03T10:00:00"
    )
    assert [e["name"] for e in entries] == ["Alpha", "Beta"]

    p2.rmdir()
    pruned = prune_missing_recent_entries(entries)
    assert [e["name"] for e in pruned] == ["Alpha"]


# ---------------------------------------------------------------------------
# ProjectService + ProjectController + SettingsManager
# ---------------------------------------------------------------------------


def test_project_service_and_controller_full_lifecycle(qapp, tmp_path: Path) -> None:
    del qapp
    controller = ProjectController()
    created_events: list[dict] = []
    loaded_events: list[dict] = []
    closed_events: list[bool] = []
    errors: list[str] = []
    controller.project_created.connect(created_events.append)
    controller.project_loaded.connect(loaded_events.append)
    controller.project_closed.connect(lambda: closed_events.append(True))
    controller.project_error.connect(errors.append)

    # Rejects invalid traversal name.
    assert controller.create_new_project(tmp_path, "../bad") is False
    assert errors

    # Creates valid project, emits BOTH project_created and project_loaded.
    assert controller.create_new_project(tmp_path, "lab_study") is True
    assert len(created_events) == 1
    assert len(loaded_events) == 1

    info = controller.get_project_info()
    assert info is not None
    assert info["directories"]["annotations"] == "annotations"
    assert info["settings"]["annotation_format"] == "labelme"
    assert "workflow_status" in info
    assert controller.get_project_directory("annotations") == (
        tmp_path / "lab_study" / "annotations"
    )
    valid, val_errors = controller.validate_project_structure(tmp_path / "lab_study")
    assert valid is True
    assert val_errors == []

    # Register & copy video into project.
    vid = tmp_path / "trial1.mp4"
    vid.write_bytes(b"video-bytes")
    ok_reg, reg_paths = controller.register_project_videos(
        [vid], copy_into_project=True
    )
    assert ok_reg is True
    assert len(reg_paths) == 1
    assert Path(reg_paths[0]).parent == (tmp_path / "lab_study" / "videos").resolve()

    # Load legacy schema-only project via direct path to project.annolid.json.
    legacy_dir = tmp_path / "legacy_schema_only"
    legacy_dir.mkdir()
    save_behavior_spec(default_behavior_spec(), legacy_dir / PROJECT_CONFIG_FILENAME)
    assert controller.load_project(legacy_dir / PROJECT_CONFIG_FILENAME) is True
    assert (legacy_dir / "images").is_dir()
    assert (legacy_dir / "annotations").is_dir()

    controller.close_project()
    assert controller.is_project_loaded() is False
    assert closed_events == [True]


def test_settings_manager_recent_projects_roundtrip(qapp, tmp_path: Path) -> None:
    del qapp
    sm = SettingsManager(organization="AnnolidTest", application="RecentProjectsTest")
    sm._qt_settings = QtCore.QSettings(
        str(tmp_path / "settings.ini"), QtCore.QSettings.IniFormat
    )
    proj = tmp_path / "proj_one"
    proj.mkdir()
    sm.add_recent_project(str(proj), name="Project One")
    loaded = sm.get_recent_projects(prune_missing=True)
    assert len(loaded) == 1
    assert loaded[0]["name"] == "Project One"
    sm.clear_recent_projects()
    assert sm.get_recent_projects() == []


# ---------------------------------------------------------------------------
# ProjectWizard, DashboardWidget, ProjectHubDialog, ProjectWorkflowMixin, Menu
# ---------------------------------------------------------------------------


def test_project_wizard_creates_loadable_workspace_and_respects_validation(
    qapp, tmp_path: Path
) -> None:
    del qapp
    wizard = ProjectWizard()
    try:
        wizard.project_info_page.location_edit.setText(str(tmp_path))
        wizard.project_info_page.name_edit.setText("../escape")
        assert wizard.project_info_page.isComplete() is False
        assert "Invalid name" in wizard.project_info_page.path_preview_label.text()

        wizard.project_info_page.name_edit.setText("WizardStudy")
        wizard.project_info_page.description_edit.setPlainText("Open-field mouse assay")
        assert wizard.project_info_page.isComplete() is True

        vid = tmp_path / "clip_a.mp4"
        vid.write_bytes(b"mp4-data")
        wizard.video_import_page.video_list.addItem(str(vid))
        wizard.video_import_page.copy_videos_check.setChecked(True)

        emitted: list[tuple] = []
        wizard.project_created.connect(lambda p, s, v: emitted.append((p, s, v)))
        wizard.accept()

        assert len(emitted) == 1
        proj_dir, _schema, videos = emitted[0]
        assert proj_dir == tmp_path / "WizardStudy"
        assert (proj_dir / "images").is_dir()
        assert (proj_dir / "annotations").is_dir()
        assert (proj_dir / ".annolidignore").is_file()
        assert len(videos) == 1
        assert Path(videos[0]).parent == (proj_dir / "videos").resolve()

        # Verify ProjectService can load the wizard-created project directly!
        svc = ProjectService()
        ok, msg = svc.load_project(proj_dir)
        assert ok is True, msg
        info = svc.get_project_info()
        assert info is not None
        assert info["description"] == "Open-field mouse assay"
        assert info["statistics"]["total_videos"] == 1
    finally:
        wizard.close()


class _ProjectHostWindow(ProjectWorkflowMixin, QtWidgets.QMainWindow):
    def __init__(self, settings_path: Path) -> None:
        super().__init__()
        self.settings = QtCore.QSettings(str(settings_path), QtCore.QSettings.IniFormat)
        self.project_controller = ProjectController(parent=self)
        self.project_schema = None
        self.project_schema_path = None
        self.lastOpenDir = None
        self.recentFiles: list[str] = []
        self.opened_videos: list[str] = []
        self.loaded_files: list[str] = []
        self.added_manager_videos: list[str] = []
        self.imported_dirs: list[str] = []
        self.behavior_controller = SimpleNamespace(
            configure_from_schema=lambda _s: None
        )
        self.video_manager_widget = SimpleNamespace(
            add_video=lambda p: self.added_manager_videos.append(str(p))
        )
        self.setStatusBar(QtWidgets.QStatusBar(self))

    def _populate_behavior_controls_from_schema(self, _schema) -> None:
        pass

    def _sync_behavior_flags_from_schema(self, _schema) -> None:
        pass

    def _update_modifier_controls_for_behavior(self, _ev) -> None:
        pass

    def openVideo(self, *, from_video_list=False, video_path=None) -> None:  # noqa: N802
        del from_video_list
        if video_path:
            self.opened_videos.append(str(video_path))

    def loadFile(self, file_path: str) -> bool:  # noqa: N802
        self.loaded_files.append(str(file_path))
        return True

    def importDirImages(self, dirpath: str, load: bool = True) -> None:  # noqa: N802
        del load
        self.imported_dirs.append(str(dirpath))


def test_project_workflow_open_hub_and_recent_menu(
    qapp, tmp_path: Path, monkeypatch
) -> None:
    del qapp
    host = _ProjectHostWindow(tmp_path / "host.ini")
    try:
        # Create project on disk with 2 videos.
        svc = ProjectService()
        assert svc.create_project(tmp_path, "SocialAssay")[0] is True
        proj_dir = tmp_path / "SocialAssay"
        save_behavior_spec(default_behavior_spec(), proj_dir / PROJECT_CONFIG_FILENAME)
        v1 = tmp_path / "v1.mp4"
        v2 = tmp_path / "v2.mp4"
        v1.write_bytes(b"v1")
        v2.write_bytes(b"v2")

        # _on_project_created with open_first_video=False should NOT open video,
        # but SHOULD register all videos and sync ProjectController.
        host._on_project_created(
            proj_dir,
            default_behavior_spec(),
            [str(v1), str(v2)],
            open_first_video=False,
        )
        assert host.project_controller.is_project_loaded() is True
        assert host.opened_videos == []
        assert host.added_manager_videos == [str(v1), str(v2)]
        assert len(host.get_recent_projects()) == 1

        # Open project explicitly opens registered videos cleanly.
        assert host.project_controller.register_project_videos([v1, v2])[0] is True
        assert host.open_project(proj_dir, open_first_video=True) is True
        assert host.opened_videos == [str(v1.resolve())]

        # ProjectHubDialog reflects active project, live workflow status, and recent projects.
        hub = ProjectHubDialog(
            project_controller=host.project_controller,
            recent_projects_getter=host.get_recent_projects,
            recent_files_getter=lambda: host.recentFiles,
            parent=host,
        )
        try:
            assert hub.project_name_label.text() == "SocialAssay"
            assert "Videos: 2" in hub.stats_label.text()
            assert hub.dashboard._step_widgets["project"].status == "completed"
            assert hub.dashboard._step_widgets["collect"].status == "completed"
            assert hub.dashboard._step_widgets["annotate"].status == "active"

            # Verify DashboardWidget dark/light theme switching and live status update.
            for hex_bg in ("#121418", "#ffffff"):
                pal = hub.dashboard.palette()
                pal.setColor(QtGui.QPalette.Window, QtGui.QColor(hex_bg))
                hub.dashboard.setPalette(pal)
                hub.dashboard._apply_theme_styles()
            hub.dashboard.update_workflow_status(
                {"annotate": "completed"}, summary="Done"
            )
            assert hub.dashboard._step_widgets["annotate"].status == "completed"
            assert hub.dashboard.workflow_summary_label.text() == "Done"
        finally:
            hub.close()

        # Export current project asynchronously as ZIP via mocked save dialog.
        zip_target = tmp_path / "exported_social.zip"
        monkeypatch.setattr(
            QtWidgets.QFileDialog,
            "getSaveFileName",
            lambda *_args, **_kwargs: (str(zip_target), "ZIP Archive (*.zip)"),
        )
        assert host.export_current_project() is True
        assert _wait_until(
            lambda: getattr(host, "_project_export_thread", None) is None
            and zip_target.is_file()
        )
        assert getattr(host, "_project_export_worker", None) is None
        assert getattr(host, "_project_export_progress", None) is None
        with zipfile.ZipFile(zip_target, "r") as zf:
            assert PROJECT_CONFIG_FILENAME in set(zf.namelist())

        # Close active project clears ProjectController and project_schema_path cleanly.
        host.close_active_project()
        assert host.project_controller.is_project_loaded() is False
        assert host.project_schema_path is None

        # MenuController._populate_recent_menu populates recent projects, files, and clear action.
        sample_file = tmp_path / "frame0.json"
        sample_file.write_text("{}", encoding="utf-8")
        host.recentFiles = [str(sample_file)]
        host.menus = SimpleNamespace(recent=QtWidgets.QMenu(host))
        menu_ctrl = MenuController(host)
        menu_ctrl._populate_recent_menu()
        action_texts = [
            a.text() for a in host.menus.recent.actions() if not a.isSeparator()
        ]
        assert any("SocialAssay" in t for t in action_texts)
        assert any("frame0.json" in t for t in action_texts)
        assert any("Clear Recent History" in t for t in action_texts)

        host.clear_recent_history()
        assert host.get_recent_projects() == []
        assert host.recentFiles == []
    finally:
        host.close()


# ---------------------------------------------------------------------------
# FileDock, FileBrowser, FileAudit, and Atomic Persistence
# ---------------------------------------------------------------------------


class _FileDockBrowserProbe(FileDockMixin, FileBrowserMixin, QtWidgets.QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.imageList: list[str] = []
        self.filename: str | None = None
        self.imagePath: str | None = None
        self.output_dir: str | None = None
        self.lastOpenDir: str | None = None
        self.recentFiles: list[str] = []
        self.errors: list[tuple[str, str]] = []
        self._init_file_dock_ui()
        self.setStatusBar(QtWidgets.QStatusBar(self))

    def _annotation_store_has_frame(self, _label_file: str) -> bool:
        return False

    def mayContinue(self) -> bool:  # noqa: N802
        return True

    def errorMessage(self, title: str, message: str) -> None:  # noqa: N802
        self.errors.append((str(title), str(message)))


def test_file_dock_status_filter_paired_rename_and_copy_path(
    qapp, tmp_path: Path, monkeypatch
) -> None:
    del qapp
    win = _FileDockBrowserProbe()
    try:
        img1 = tmp_path / "labeled_01.png"
        img1.write_bytes(b"\x89PNG")
        lbl1 = tmp_path / "labeled_01.json"
        lbl1.write_text(
            json.dumps({"imagePath": "labeled_01.png", "shapes": []}),
            encoding="utf-8",
        )
        img2 = tmp_path / "unlabeled_02.png"
        img2.write_bytes(b"\x89PNG")

        win._addItem(str(img1), str(lbl1))
        win._addItem(str(img2), str(tmp_path / "unlabeled_02.json"))
        win.imageList = [str(img1), str(img2)]
        win.filename = str(img1)
        win.imagePath = str(img1)
        win.recentFiles = [str(img1)]

        # Filter Labeled vs Unlabeled vs All.
        win.fileStatusFilterCombo.setCurrentIndex(
            win.fileStatusFilterCombo.findData("labeled")
        )
        assert win.fileListWidget.item(0).isHidden() is False
        assert win.fileListWidget.item(1).isHidden() is True

        win.fileStatusFilterCombo.setCurrentIndex(
            win.fileStatusFilterCombo.findData("unlabeled")
        )
        assert win.fileListWidget.item(0).isHidden() is True
        assert win.fileListWidget.item(1).isHidden() is False

        win.fileStatusFilterCombo.setCurrentIndex(
            win.fileStatusFilterCombo.findData("all")
        )
        assert win.fileListWidget.item(0).isHidden() is False
        assert win.fileListWidget.item(1).isHidden() is False

        # Copy path copies full path to clipboard.
        win.fileListWidget.setCurrentRow(0)
        win._copy_selected_file_path()
        assert QtWidgets.QApplication.clipboard().text() == str(img1)

        # Rename image renames paired .json and updates imagePath + window state + recentFiles.
        monkeypatch.setattr(
            QtWidgets.QInputDialog,
            "getText",
            lambda *_args, **_kwargs: ("renamed_01.png", True),
        )
        win._rename_selected_file_from_dock()
        new_img = tmp_path / "renamed_01.png"
        new_lbl = tmp_path / "renamed_01.json"
        assert new_img.is_file()
        assert new_lbl.is_file()
        assert json.loads(new_lbl.read_text(encoding="utf-8"))["imagePath"] == (
            "renamed_01.png"
        )
        assert win.filename == str(new_img)
        assert win.imagePath == str(new_img)
        assert win.recentFiles == [str(new_img)]

        # Async refresh selection restore via _finalize_import_dir_scan.
        win._dir_scan_token = 7
        win._dir_scan_running = True
        win._dir_scan_load = False
        win._dir_scan_pending_select = str(img2)
        win.fileListWidget.setCurrentRow(0)
        win._finalize_import_dir_scan(7)
        assert win.fileListWidget.currentItem().text() == str(img2)
    finally:
        win.close()


def test_file_audit_widget_search_summary_open_and_export(qapp, tmp_path: Path) -> None:
    del qapp
    w = FileAuditWidget()
    try:
        j1 = tmp_path / "clip_001.json"
        i1 = tmp_path / "clip_001.png"
        j2 = tmp_path / "clip_002.json"
        w.set_items(
            [
                {
                    "status": "valid",
                    "json_path": j1,
                    "image_path": i1,
                    "shape_count": 3,
                },
                {
                    "status": "missing_image",
                    "json_path": j2,
                    "image_path": None,
                    "shape_count": 0,
                },
            ]
        )
        assert "Total: 2 | Valid: 1 | Issues: 1" in w.summary_label.text()

        w.search_edit.setText("002")
        assert w.table.rowCount() == 1
        assert "Showing: 1/2" in w.summary_label.text()

        opened: list[str] = []
        w.fileOpenRequested.connect(opened.append)
        w.search_edit.clear()
        w._on_item_double_clicked(w.table.item(0, 1))
        assert opened == [str(i1)]

        out_csv = tmp_path / "audit.csv"
        saved = w.export_csv(out_csv)
        assert saved == out_csv
        csv_text = out_csv.read_text(encoding="utf-8")
        assert "clip_001.json" in csv_text
        assert "missing_image" in csv_text
    finally:
        w.close()


def test_persistence_atomic_writes_for_tracking_stats_and_labels(
    qapp, tmp_path: Path, monkeypatch
) -> None:
    del qapp

    class _PersistProbe(PersistenceLifecycleMixin, QtWidgets.QWidget):
        def __init__(self) -> None:
            super().__init__()
            self.here = tmp_path / "gui"
            self.uniqLabelList = QtWidgets.QListWidget()
            item = QtWidgets.QListWidgetItem("mouse")
            item.setData(QtCore.Qt.UserRole, "mouse")
            self.uniqLabelList.addItem(item)

    probe = _PersistProbe()
    try:
        pred_dir = tmp_path / "vid_results"
        pred_dir.mkdir()
        stats_path = pred_dir / "vid_results_tracking_stats.json"
        stats_path.write_text(
            json.dumps(
                {
                    "frame_stats": {
                        "0": {"sources": ["manual_seed"]},
                        "1": {"sources": ["predicted"]},
                    },
                    "bad_shape_events": [],
                    "prediction_segments": [],
                }
            ),
            encoding="utf-8",
        )
        probe._prune_tracking_stats_frames(
            pred_dir, start_frame=1, end_frame=None, protected_frames={0}
        )
        pruned = json.loads(stats_path.read_text(encoding="utf-8"))
        assert "0" in pruned["frame_stats"]
        assert "1" not in pruned["frame_stats"]

        labels_out = tmp_path / "labels.txt"
        monkeypatch.setattr(
            QtWidgets.QFileDialog,
            "getSaveFileName",
            lambda *_args, **_kwargs: (str(labels_out), "*.txt"),
        )
        probe.save_labels()
        lines = labels_out.read_text(encoding="utf-8").splitlines()
        assert lines == ["__ignore__", "_background_", "mouse"]
    finally:
        probe.close()


# ---------------------------------------------------------------------------
# Async Export Cancellation, Dual-Storage Audit Dialog & Wiring
# ---------------------------------------------------------------------------


def test_export_project_bundle_progress_cancellation_and_zip_conflict_guard(
    tmp_path: Path,
) -> None:
    svc = ProjectService()
    assert svc.create_project(tmp_path, "cancel_proj")[0] is True
    proj = tmp_path / "cancel_proj"
    (proj / "images" / "a.png").write_bytes(b"img-a" * 1024)
    (proj / "images" / "b.png").write_bytes(b"img-b" * 1024)
    (proj / "annotations" / "a.json").write_text('{"shapes": []}', encoding="utf-8")

    # Progress callback receives monotonic updates up to total_files.
    progress_events: list[tuple[int, int, str]] = []
    out_folder = tmp_path / "exported_folder"
    ok, _msg = export_project_bundle(
        proj,
        out_folder,
        include_data=True,
        as_zip=False,
        progress_callback=lambda done, total, rel: progress_events.append(
            (done, total, rel)
        ),
    )
    assert ok is True
    assert progress_events
    assert progress_events[0][0] == 0
    assert progress_events[-1][0] == progress_events[-1][1]
    assert (out_folder / "images" / "b.png").is_file()

    # Mid-export cancellation in folder mode cleans up newly created folder.
    cancelled_folder = tmp_path / "cancelled_folder"
    seen_calls = 0

    def _cancel_after_two() -> bool:
        nonlocal seen_calls
        seen_calls += 1
        return seen_calls > 2

    ok_c, msg_c = export_project_bundle(
        proj,
        cancelled_folder,
        include_data=True,
        as_zip=False,
        cancel_check=_cancel_after_two,
    )
    assert ok_c is False
    assert "cancelled" in msg_c.lower()
    assert not cancelled_folder.exists()

    # Mid-export cancellation in ZIP mode removes temp archive and never writes target.
    cancelled_zip = tmp_path / "cancelled_bundle.zip"
    zip_calls = 0

    def _cancel_zip() -> bool:
        nonlocal zip_calls
        zip_calls += 1
        return zip_calls > 2

    ok_z, msg_z = export_project_bundle(
        proj,
        cancelled_zip,
        include_data=True,
        as_zip=True,
        cancel_check=_cancel_zip,
    )
    assert ok_z is False
    assert "cancelled" in msg_z.lower()
    assert not cancelled_zip.exists()

    # Rejects ZIP destination colliding with an active project file.
    ok_conflict, msg_conflict = export_project_bundle(
        proj,
        proj / PROJECT_CONFIG_FILENAME,
        as_zip=True,
    )
    assert ok_conflict is False
    assert "conflicts" in msg_conflict.lower()


def test_audit_project_annotations_excludes_metadata_and_detects_ndjson(
    tmp_path: Path,
) -> None:
    svc = ProjectService()
    assert svc.create_project(tmp_path, "audit_proj")[0] is True
    proj = tmp_path / "audit_proj"
    (proj / "pose_schema.json").write_text('{"keypoints": ["nose"]}', encoding="utf-8")

    ann_dir = proj / "annotations" / "trial1"
    ann_dir.mkdir(parents=True)
    # Tracking stats + segments metadata should be excluded from LabelMe audit.
    (ann_dir / "trial1_tracking_stats.json").write_text("{}", encoding="utf-8")
    (ann_dir / "trial1.mp4.segments.json").write_text("[]", encoding="utf-8")

    # Valid image+JSON pair, missing-image JSON, empty JSON, corrupt JSON.
    (ann_dir / "trial1_000000000.png").write_bytes(b"\x89PNG")
    (ann_dir / "trial1_000000000.json").write_text(
        json.dumps(
            {
                "imagePath": "trial1_000000000.png",
                "shapes": [{"label": "mouse"}],
            }
        ),
        encoding="utf-8",
    )
    (ann_dir / "trial1_000000001.json").write_text(
        json.dumps(
            {"imagePath": "missing.png", "shapes": [{"label": "mouse"}]},
        ),
        encoding="utf-8",
    )
    (ann_dir / "trial1_000000002.png").write_bytes(b"\x89PNG")
    (ann_dir / "trial1_000000002.json").write_text(
        json.dumps({"imagePath": "trial1_000000002.png", "shapes": []}),
        encoding="utf-8",
    )
    (ann_dir / "trial1_000000003.json").write_text("{corrupt", encoding="utf-8")

    store = AnnotationStore.for_frame_path(ann_dir / "trial1_000000000.json")
    store.append_frame({"frame": 0, "shapes": []})
    store.append_frame({"frame": 5, "shapes": []})

    result = audit_project_annotations(proj)
    assert result["cancelled"] is False
    by_name = {Path(it["json_path"]).name: it["status"] for it in result["items"]}
    assert PROJECT_CONFIG_FILENAME not in by_name
    assert "pose_schema.json" not in by_name
    assert "trial1_tracking_stats.json" not in by_name
    assert "trial1.mp4.segments.json" not in by_name
    assert by_name == {
        "trial1_000000000.json": "valid",
        "trial1_000000001.json": "missing_image",
        "trial1_000000002.json": "empty",
        "trial1_000000003.json": "error",
    }
    assert result["ndjson_total_frames"] == 2
    assert len(result["ndjson_stores"]) == 1
    assert result["labels"] == ["mouse"]

    # Cancelled scan returns immediately.
    cancelled = scan_annotation_audit([proj], cancel_check=lambda: True)
    assert cancelled["cancelled"] is True


def test_file_audit_dialog_project_hub_wizard_and_dashboard_edge_cases(
    qapp, tmp_path: Path
) -> None:
    del qapp
    host = _ProjectHostWindow(tmp_path / "audit_host.ini")
    try:
        svc = ProjectService()
        assert svc.create_project(tmp_path, "AuditStudy")[0] is True
        proj = tmp_path / "AuditStudy"
        ann_dir = proj / "annotations"
        img_path = ann_dir / "clip_000000000.png"
        json_path = ann_dir / "clip_000000000.json"
        img_path.write_bytes(b"\x89PNG")
        json_path.write_text(
            json.dumps(
                {
                    "imagePath": "clip_000000000.png",
                    "shapes": [{"label": "rat"}],
                }
            ),
            encoding="utf-8",
        )
        store = AnnotationStore.for_frame_path(json_path)
        store.append_frame({"frame": 0, "shapes": []})
        store.append_frame({"frame": 1, "shapes": []})

        assert host.open_project(proj, open_first_video=False) is True

        # Open project-scoped FileAuditDialog asynchronously and verify NDJSON banner + open signal.
        audit_dlg = host.open_annotation_audit_dialog()
        try:
            assert isinstance(audit_dlg, FileAuditDialog)
            assert _wait_until(
                lambda: audit_dlg._scan_thread is None
                and audit_dlg.audit_widget.table.rowCount() == 1
            )
            assert audit_dlg.store_info_label.isHidden() is False
            assert "2 indexed frame(s)" in audit_dlg.store_info_label.text()

            # Double-clicking row emits fileOpenRequested -> routes to host.loadFile().
            audit_dlg.audit_widget._on_item_double_clicked(
                audit_dlg.audit_widget.table.item(0, 1)
            )
            assert host.loaded_files == [str(img_path.resolve())]
        finally:
            audit_dlg.close()

        # _open_audited_file routes videos to openVideo and ignores missing files safely.
        vid_file = tmp_path / "session.mp4"
        vid_file.write_bytes(b"mp4")
        assert host._open_audited_file(str(vid_file)) is True
        assert host.opened_videos == [str(vid_file)]
        assert host._open_audited_file(str(tmp_path / "does_not_exist.png")) is False

        # ProjectHubDialog exposes auditAnnotationsRequested and closeProjectRequested.
        hub = ProjectHubDialog(project_controller=host.project_controller, parent=host)
        try:
            audit_clicks: list[bool] = []
            close_clicks: list[bool] = []
            hub.auditAnnotationsRequested.connect(lambda: audit_clicks.append(True))
            hub.closeProjectRequested.connect(lambda: close_clicks.append(True))
            assert hub.close_project_btn.isEnabled() is True
            hub.audit_annotations_btn.click()
            hub.close_project_btn.click()
            assert audit_clicks == [True]
            assert close_clicks == [True]
        finally:
            hub.close()

        # DatasetExportWizard skips project.annolid.json and forwards file_open_requested.
        wizard = DatasetExportWizard(source_dir=str(proj), parent=host)
        try:
            wiz_opens: list[str] = []
            wizard.file_open_requested.connect(wiz_opens.append)
            scanned_names = [
                Path(it["json_path"]).name
                for it in wizard.select_annotations_page.audit_widget.items
            ]
            assert PROJECT_CONFIG_FILENAME not in scanned_names
            assert "clip_000000000.json" in scanned_names
            wizard.select_annotations_page.audit_widget.fileOpenRequested.emit(
                str(img_path)
            )
            assert wiz_opens == [str(img_path)]
        finally:
            wizard.close()

        # LabelCollectionDialog skips project.annolid.json and forwards file_open_requested.
        collect_dlg = LabelCollectionDialog(settings=host.settings, parent=host)
        try:
            collect_opens: list[str] = []
            collect_dlg.file_open_requested.connect(collect_opens.append)
            collect_dlg.sources_list.clear()
            collect_dlg.sources_list.addItem(str(proj))
            collect_dlg._scan_sources()
            scanned_names = [
                Path(it["json_path"]).name for it in collect_dlg.audit_widget.items
            ]
            assert PROJECT_CONFIG_FILENAME not in scanned_names
            assert "clip_000000000.json" in scanned_names
            collect_dlg.audit_widget.fileOpenRequested.emit(str(img_path))
            assert collect_opens == [str(img_path)]
        finally:
            collect_dlg.close()

        # RecentItemWidget._format_time_ago handles timezone-aware and future datetimes safely.
        utc_item = RecentProject(
            name="UTC Proj",
            path=str(proj),
            last_opened=datetime.now(timezone.utc) - timedelta(hours=2),
            kind="project",
        )
        future_item = RecentProject(
            name="Future Proj",
            path=str(proj),
            last_opened=datetime.now(timezone.utc) + timedelta(minutes=30),
            kind="project",
        )
        w_utc = RecentItemWidget(utc_item)
        w_future = RecentItemWidget(future_item)
        try:
            assert w_utc._time_label.text() in {"1h ago", "2h ago"}
            assert w_future._time_label.text() == "Just now"
        finally:
            w_utc.close()
            w_future.close()
    finally:
        host._cancel_active_project_export(wait_ms=2000)
        host.close()
