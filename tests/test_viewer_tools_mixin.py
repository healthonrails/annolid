from __future__ import annotations

import json
import os
import subprocess
import time
from pathlib import Path

from annolid.gui.mixins.viewer_tools_mixin import (
    ViewerToolsMixin,
    _is_recent_live_flybody_payload,
    _prepare_live_flybody_view_payload,
    _run_logged_subprocess,
    _supports_builtin_stack_viewer,
)
from qtpy import QtWidgets


def test_is_recent_live_flybody_payload_accepts_recent_live_payload(
    tmp_path: Path,
) -> None:
    payload_path = tmp_path / "flybody_live_rollout.json"
    payload_path.write_text(
        json.dumps(
            {
                "kind": "annolid-simulation-v1",
                "adapter": "flybody-live",
                "metadata": {
                    "run_metadata": {
                        "payload_version": 3,
                        "behavior": "walk_imitation",
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    assert (
        _is_recent_live_flybody_payload(
            payload_path,
            max_age_seconds=60.0,
            behavior="walk_imitation",
        )
        is True
    )


def test_is_recent_live_flybody_payload_rejects_stale_payload(tmp_path: Path) -> None:
    payload_path = tmp_path / "flybody_live_rollout.json"
    payload_path.write_text(
        json.dumps(
            {
                "kind": "annolid-simulation-v1",
                "adapter": "flybody-live",
                "metadata": {
                    "run_metadata": {
                        "payload_version": 3,
                        "behavior": "walk_imitation",
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    stale_time = time.time() - 120.0
    os.utime(payload_path, (stale_time, stale_time))

    assert (
        _is_recent_live_flybody_payload(
            payload_path,
            max_age_seconds=5.0,
            behavior="walk_imitation",
        )
        is False
    )


def test_is_recent_live_flybody_payload_rejects_non_live_payload(
    tmp_path: Path,
) -> None:
    payload_path = tmp_path / "flybody_live_rollout.json"
    payload_path.write_text(
        json.dumps(
            {
                "kind": "annolid-simulation-v1",
                "adapter": "flybody",
                "metadata": {
                    "run_metadata": {
                        "payload_version": 3,
                        "behavior": "walk_imitation",
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    assert (
        _is_recent_live_flybody_payload(
            payload_path,
            max_age_seconds=60.0,
            behavior="walk_imitation",
        )
        is False
    )


def test_is_recent_live_flybody_payload_rejects_wrong_behavior(tmp_path: Path) -> None:
    payload_path = tmp_path / "flybody_live_rollout.json"
    payload_path.write_text(
        json.dumps(
            {
                "kind": "annolid-simulation-v1",
                "adapter": "flybody-live",
                "metadata": {
                    "run_metadata": {
                        "payload_version": 3,
                        "behavior": "walk_imitation",
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    assert (
        _is_recent_live_flybody_payload(
            payload_path,
            max_age_seconds=60.0,
            behavior="flight_imitation",
        )
        is False
    )


class _DummyStatusBar:
    def __init__(self) -> None:
        self.messages: list[tuple[str, int]] = []

    def showMessage(self, text: str, timeout: int = 0) -> None:
        self.messages.append((text, timeout))


class _DummyManager:
    def __init__(self) -> None:
        self.paths: list[Path] = []

    def show_simulation_in_viewer(self, path) -> bool:
        self.paths.append(Path(path))
        return True


class _DummyViewerHost(ViewerToolsMixin):
    def __init__(self) -> None:
        self.threejs_manager = _DummyManager()
        self._status = _DummyStatusBar()

    def statusBar(self):
        return self._status

    def tr(self, text: str) -> str:
        return text


def test_open_threejs_example_flybody_stays_on_fast_example_path(
    tmp_path: Path, monkeypatch
) -> None:
    widget = _DummyViewerHost()
    example_path = tmp_path / "flybody.json"
    example_path.write_text(
        json.dumps(
            {"kind": "annolid-simulation-v1", "adapter": "flybody", "frames": []}
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "annolid.gui.mixins.viewer_tools_mixin.generate_threejs_example",
        lambda example_id, out_dir: example_path,
    )
    monkeypatch.setattr(
        widget,
        "_start_live_flybody_example",
        lambda manager: (_ for _ in ()).throw(
            AssertionError("live path should not run")
        ),
    )

    widget.open_threejs_example("flybody_simulation_json")

    assert widget.threejs_manager.paths == [example_path]
    assert widget.statusBar().messages[-1][0].startswith("Loaded FlyBody 3D example.")


def test_open_threejs_example_flybody_does_not_prompt_for_install(
    tmp_path: Path, monkeypatch
) -> None:
    widget = _DummyViewerHost()
    example_path = tmp_path / "flybody.json"
    example_path.write_text(
        json.dumps(
            {"kind": "annolid-simulation-v1", "adapter": "flybody", "frames": []}
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "annolid.gui.mixins.viewer_tools_mixin.generate_threejs_example",
        lambda example_id, out_dir: example_path,
    )
    monkeypatch.setattr(
        QtWidgets.QMessageBox,
        "question",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("question dialog should not run for FlyBody example")
        ),
    )

    widget.open_threejs_example("flybody_simulation_json")


def test_handle_flybody_viewer_command_routes_start_and_stop(monkeypatch) -> None:
    widget = _DummyViewerHost()
    calls = []

    monkeypatch.setattr(
        widget,
        "_start_live_flybody_behavior_example",
        lambda **kwargs: calls.append(("start", kwargs["behavior"])),
    )
    monkeypatch.setattr(
        widget,
        "_stop_live_flybody_example",
        lambda: calls.append(("stop", "")),
    )

    widget.handle_flybody_viewer_command("start", "flight_imitation")
    widget.handle_flybody_viewer_command("stop", "")

    assert calls == [("start", "flight_imitation"), ("stop", "")]


def test_start_live_flybody_example_shows_static_example_first_when_runtime_missing(
    tmp_path: Path, monkeypatch
) -> None:
    widget = _DummyViewerHost()
    example_path = tmp_path / "flybody.json"
    example_path.write_text(
        json.dumps(
            {"kind": "annolid-simulation-v1", "adapter": "flybody", "frames": []}
        ),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "annolid.gui.mixins.viewer_tools_mixin.generate_threejs_example",
        lambda example_id, out_dir: example_path,
    )
    monkeypatch.setattr(
        "annolid.gui.mixins.viewer_tools_mixin.pick_ready_flybody_runtime",
        lambda: (None, {}),
    )
    monkeypatch.setattr(
        "annolid.gui.mixins.viewer_tools_mixin._is_recent_live_flybody_payload",
        lambda *args, **kwargs: False,
    )
    shown: list[str] = []
    monkeypatch.setattr(
        QtWidgets.QMessageBox,
        "information",
        lambda *args, **kwargs: shown.append("info"),
    )

    widget.start_live_flybody_example()

    assert widget.threejs_manager.paths == [example_path]
    assert shown == ["info"]
    assert any(
        "Loaded FlyBody 3D example." in message
        for message, _timeout in widget.statusBar().messages
    )


def test_run_logged_subprocess_reports_timeout(monkeypatch, tmp_path: Path) -> None:
    def _fake_run(*args, **kwargs):
        raise subprocess.TimeoutExpired(cmd=["demo"], timeout=5)

    monkeypatch.setattr(subprocess, "run", _fake_run)

    try:
        _run_logged_subprocess(["demo"], cwd=tmp_path, timeout_seconds=5)
    except RuntimeError as exc:
        assert "Timed out after 5s." in str(exc)
    else:
        raise AssertionError("expected RuntimeError")


def test_run_logged_subprocess_reports_process_failure(
    monkeypatch, tmp_path: Path
) -> None:
    def _fake_run(*args, **kwargs):
        log_file = kwargs["stdout"]
        log_file.write("failure details\n")
        raise subprocess.CalledProcessError(returncode=7, cmd=["demo"])

    monkeypatch.setattr(subprocess, "run", _fake_run)

    try:
        _run_logged_subprocess(["demo"], cwd=tmp_path, timeout_seconds=5)
    except RuntimeError as exc:
        message = str(exc)
        assert "Command exited with status 7." in message
        assert "failure details" in message
    else:
        raise AssertionError("expected RuntimeError")


def test_prepare_live_flybody_view_payload_uses_combined_mesh_and_hides_overlays(
    monkeypatch, tmp_path: Path
) -> None:
    payload_path = tmp_path / "flybody_live_rollout.json"
    payload_path.write_text(
        json.dumps({"kind": "annolid-simulation-v1", "adapter": "flybody-live"}),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "annolid.gui.mixins.viewer_tools_mixin.attach_flybody_mesh",
        lambda payload, out_dir: {
            **payload,
            "mesh": {"type": "obj", "path": "fly.obj"},
        },
    )
    monkeypatch.setattr(
        "annolid.gui.mixins.viewer_tools_mixin.attach_flybody_floor",
        lambda payload: {
            **payload,
            "environment": {"floor": {"type": "plane", "position": [0.0, 0.0, -0.132]}},
        },
    )

    out_path = _prepare_live_flybody_view_payload(payload_path, base_dir=tmp_path)
    payload = json.loads(Path(out_path).read_text(encoding="utf-8"))

    assert payload["mesh"]["type"] == "obj"
    assert payload["environment"]["floor"]["type"] == "plane"
    assert payload["environment"]["floor"]["position"] == [0.0, -0.99, 0.0]
    assert payload["display"] == {
        "show_points": False,
        "show_labels": False,
        "show_edges": False,
        "show_trails": False,
    }


def test_supports_builtin_stack_viewer_only_for_tiff_like_sources() -> None:
    assert _supports_builtin_stack_viewer(Path("stack.tif")) is True
    assert _supports_builtin_stack_viewer(Path("stack.ome.tiff")) is True
    assert _supports_builtin_stack_viewer(Path("structural_brain.nii.gz")) is False


def test_open_3d_viewer_does_not_fallback_to_pil_for_nifti(
    monkeypatch, tmp_path: Path
) -> None:
    widget = _DummyViewerHost()
    nifti_path = tmp_path / "structural_brain.nii.gz"
    nifti_path.write_text("not-a-real-volume", encoding="utf-8")

    class _TiffStackVideo:
        pass

    class _VideoLoader(_TiffStackVideo):
        pass

    monkeypatch.setattr(
        "annolid.data.videos.TiffStackVideo",
        _TiffStackVideo,
    )
    widget.video_loader = _VideoLoader()
    widget.video_file = nifti_path
    widget.imagePath = ""
    widget.filename = ""

    monkeypatch.setattr(
        "annolid.gui.widgets.vtk_volume_viewer.VTKVolumeViewerDialog",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            RuntimeError("vtk viewer failed")
        ),
    )
    monkeypatch.setattr(
        "annolid.gui.widgets.volume_viewer.VolumeViewerDialog",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("built-in viewer must not run for NIfTI")
        ),
    )

    warnings: list[str] = []
    monkeypatch.setattr(
        QtWidgets.QMessageBox,
        "warning",
        lambda *args, **kwargs: warnings.append(args[2] if len(args) > 2 else ""),
    )

    widget.open_3d_viewer()

    assert warnings
    assert (
        "built-in fallback viewer only supports TIFF stacks".lower()
        in warnings[0].lower()
    )


def test_open_3d_viewer_retains_vtk_dialog_reference(
    monkeypatch, tmp_path: Path
) -> None:
    widget = _DummyViewerHost()
    nifti_path = tmp_path / "structural_brain.nii.gz"
    nifti_path.write_text("not-a-real-volume", encoding="utf-8")

    class _TiffStackVideo:
        pass

    class _VideoLoader(_TiffStackVideo):
        pass

    class _FakeDlg:
        def __init__(self, *args, **kwargs):
            self.destroyed = type(
                "_Signal",
                (),
                {"connect": lambda self, cb: None},
            )()

        def setModal(self, modal: bool):
            return None

        def show(self):
            return None

        def raise_(self):
            return None

        def activateWindow(self):
            return None

    monkeypatch.setattr("annolid.data.videos.TiffStackVideo", _TiffStackVideo)
    widget.video_loader = _VideoLoader()
    widget.video_file = nifti_path
    widget.imagePath = ""
    widget.filename = ""
    monkeypatch.setattr(
        "annolid.gui.widgets.vtk_volume_viewer.VTKVolumeViewerDialog",
        _FakeDlg,
    )
    monkeypatch.setattr(
        "annolid.gui.widgets.volume_viewer.VolumeViewerDialog",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("built-in viewer must not run")
        ),
    )
    monkeypatch.setattr(
        QtWidgets.QMessageBox,
        "warning",
        lambda *args, **kwargs: None,
    )

    widget.open_3d_viewer()

    assert hasattr(widget, "_vtk_volume_viewer_dialog")
    assert widget._vtk_volume_viewer_dialog is not None
