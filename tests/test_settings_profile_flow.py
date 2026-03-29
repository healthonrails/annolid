from __future__ import annotations

import os
from dataclasses import dataclass
from types import SimpleNamespace

from qtpy import QtCore, QtWidgets

from annolid.gui.widgets.advanced_parameters_dialog import AdvancedParametersDialog
from annolid.gui.widgets import settings_profile_flow as spf
from annolid.interfaces.memory.adapters.settings_model import SettingsProfile


os.environ.setdefault("QT_QPA_PLATFORM", "minimal")

_QAPP = None


def _ensure_qapp():
    global _QAPP
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
    _QAPP = app
    return _QAPP


class _DummyStatusBar:
    def __init__(self):
        self.last_message = ""

    def showMessage(self, message: str, timeout_ms: int = 0):
        _ = timeout_ms
        self.last_message = message


class _DummyOpticalFlowManager:
    def __init__(self):
        self.backend = None
        self.compute = None

    def set_backend(self, backend: str):
        self.backend = backend

    def set_compute_optical_flow(self, enabled: bool):
        self.compute = bool(enabled)


class _DummyWindow:
    def __init__(self, ini_path: str):
        self.settings = QtCore.QSettings(ini_path, QtCore.QSettings.IniFormat)
        self._status_bar = _DummyStatusBar()
        self._config = {}
        self.optical_flow_manager = _DummyOpticalFlowManager()
        self.tracker_runtime_config = SimpleNamespace()
        self.patch_similarity_model = "dinov2"
        self.patch_similarity_alpha = 0.55
        self.pca_map_model = "dinov2"
        self.pca_map_alpha = 0.65
        self.pca_map_clusters = 0

    def statusBar(self):
        return self._status_bar


def test_apply_optical_flow_profile(monkeypatch, tmp_path) -> None:
    _ensure_qapp()
    window = _DummyWindow(str(tmp_path / "settings.ini"))
    profile = SettingsProfile(
        name="OF profile",
        workflow="optical_flow",
        settings={
            "backend": "raft",
            "raft_model": "large",
            "visualization": "quiver",
            "opacity": 88,
        },
    )

    class _Adapter:
        def retrieve_settings_profiles(self, **kwargs):
            _ = kwargs
            return [profile]

    monkeypatch.setattr(spf, "_workspace_adapter", lambda _window: _Adapter())
    monkeypatch.setattr(spf, "_choose_profile_dialog", lambda *args, **kwargs: profile)

    ok = spf.apply_profile_for_workflow(window, "optical_flow", "Optical Flow")
    assert ok is True
    assert window.optical_flow_manager.backend == "raft"
    assert window.settings.value("optical_flow/raft_model") == "large"
    assert int(window.settings.value("optical_flow/opacity")) == 88


def test_apply_advanced_profile_updates_tracker_and_sam3(monkeypatch, tmp_path) -> None:
    _ensure_qapp()
    window = _DummyWindow(str(tmp_path / "advanced.ini"))
    profile = SettingsProfile(
        name="Advanced profile",
        workflow="advanced_parameters",
        settings={
            "epsilon_for_polygon": 4.0,
            "automatic_pause_enabled": True,
            "follow_prediction_progress": False,
            "tracker_runtime_config": {"track_buffer": 45},
            "sam3_runtime": {"compile": True},
            "optical_flow_enabled": True,
            "optical_flow_backend": "farneback",
        },
    )

    class _Adapter:
        def retrieve_settings_profiles(self, **kwargs):
            _ = kwargs
            return [profile]

    monkeypatch.setattr(spf, "_workspace_adapter", lambda _window: _Adapter())
    monkeypatch.setattr(spf, "_choose_profile_dialog", lambda *args, **kwargs: profile)

    ok = spf.apply_profile_for_workflow(
        window, "advanced_parameters", "Advanced Parameters"
    )
    assert ok is True
    assert float(window.epsilon_for_polygon) == 4.0
    assert bool(window.automatic_pause_enabled) is True
    assert bool(window._follow_prediction_progress) is False
    assert int(window.tracker_runtime_config.track_buffer) == 45
    assert bool(window._config["sam3"]["compile"]) is True
    assert window.optical_flow_manager.compute is True


def test_save_current_profile_for_patch_similarity(monkeypatch, tmp_path) -> None:
    _ensure_qapp()
    window = _DummyWindow(str(tmp_path / "save.ini"))
    window.patch_similarity_model = "dinov2_vits14"
    window.patch_similarity_alpha = 0.72

    saved = {}

    class _Adapter:
        def store_settings_profile(self, profile):
            saved["profile"] = profile
            return "mem-123"

    responses = iter(
        [("Patch Preset", True), ("vision,quick", True), ("for test", True)]
    )
    monkeypatch.setattr(
        QtWidgets.QInputDialog,
        "getText",
        lambda *args, **kwargs: next(responses),
    )
    monkeypatch.setattr(spf, "_workspace_adapter", lambda _window: _Adapter())

    ok = spf.save_current_profile_for_workflow(
        window, "patch_similarity", "Patch Similarity"
    )
    assert ok is True
    profile = saved["profile"]
    assert profile.name == "Patch Preset"
    assert profile.workflow == "patch_similarity"
    assert profile.settings["model"] == "dinov2_vits14"
    assert abs(float(profile.settings["alpha"]) - 0.72) < 1e-6


def test_choose_profile_dialog_handles_duplicate_names(monkeypatch) -> None:
    _ensure_qapp()
    profiles = [
        SettingsProfile(name="Preset", workflow="optical_flow", settings={}),
        SettingsProfile(name="Preset", workflow="optical_flow", settings={"x": 1}),
    ]

    monkeypatch.setattr(
        QtWidgets.QInputDialog,
        "getItem",
        lambda *args, **kwargs: ("2. Preset (optical_flow)", True),
    )

    selected = spf._choose_profile_dialog(None, profiles, "Optical Flow")
    assert selected is profiles[1]


def test_collect_advanced_parameters_serializes_dataclass_tracker() -> None:
    @dataclass
    class _TrackerCfg:
        tracker_preset: str | None = "rodent_30fps_occlusions"
        motion_search_gain: float = 1.25
        progress_hook: object = None

    window = SimpleNamespace(
        tracker_runtime_config=_TrackerCfg(),
        epsilon_for_polygon=2.0,
        t_max_value=5,
        automatic_pause_enabled=False,
        use_cpu_only=False,
        save_video_with_color_mask=False,
        auto_recovery_missing_instances=False,
        _follow_prediction_progress=True,
        videomt_mask_threshold=0.5,
        videomt_logit_threshold=-2.0,
        videomt_seed_iou_threshold=0.01,
        videomt_window=8,
        videomt_input_height=0,
        videomt_input_width=0,
        optical_flow_manager=SimpleNamespace(
            compute_optical_flow=True,
            optical_flow_backend="farneback",
        ),
        _config={"sam3": {}},
    )

    payload = spf._collect_advanced_parameters(window)
    assert (
        payload["tracker_runtime_config"]["tracker_preset"] == "rodent_30fps_occlusions"
    )
    assert float(payload["tracker_runtime_config"]["motion_search_gain"]) == 1.25
    assert "progress_hook" not in payload["tracker_runtime_config"]


def test_advanced_parameters_dialog_loads_window_state() -> None:
    _ensure_qapp()
    dialog = AdvancedParametersDialog()
    window = SimpleNamespace(
        epsilon_for_polygon=3.5,
        t_max_value=11,
        automatic_pause_enabled=True,
        use_cpu_only=True,
        save_video_with_color_mask=True,
        auto_recovery_missing_instances=True,
        _follow_prediction_progress=False,
        videomt_mask_threshold=0.6,
        videomt_logit_threshold=-1.5,
        videomt_seed_iou_threshold=0.2,
        videomt_window=12,
        videomt_input_height=720,
        videomt_input_width=1280,
    )
    optical_flow_manager = SimpleNamespace(
        compute_optical_flow=False,
        optical_flow_backend="raft",
    )

    dialog.load_window_state(window, optical_flow_manager=optical_flow_manager)

    assert dialog.automatic_pause_checkbox.isChecked() is True
    assert dialog.cpu_only_checkbox.isChecked() is True
    assert dialog.save_video_with_color_mask_checkbox.isChecked() is True
    assert dialog.auto_recovery_missing_instances_checkbox.isChecked() is True
    assert dialog.compute_optical_flow_checkbox.isChecked() is False
    assert dialog.follow_prediction_progress_checkbox.isChecked() is False
    assert dialog.optical_flow_backend_combo.currentText().startswith("raft")
    assert float(dialog.epsilon_spinbox.value()) == 3.5
    assert int(dialog.t_max_spinbox.value()) == 11


def test_advanced_parameters_dialog_tabs_are_scrollable() -> None:
    _ensure_qapp()
    dialog = AdvancedParametersDialog()
    tabs = dialog.findChild(QtWidgets.QTabWidget)
    assert tabs is not None
    assert tabs.count() == 3

    for index in range(tabs.count()):
        tab_page = tabs.widget(index)
        assert isinstance(tab_page, QtWidgets.QScrollArea)
        assert tab_page.widgetResizable() is True
