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
)
from annolid.gui.threejs_support import supports_threejs_canvas
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
        self.model_paths: list[Path] = []
        self.sim_paths: list[Path] = []

    def show_simulation_in_viewer(self, path) -> bool:
        p = Path(path)
        self.paths.append(p)
        self.sim_paths.append(p)
        return True

    def show_model_in_viewer(self, path) -> bool:
        p = Path(path)
        self.paths.append(p)
        self.model_paths.append(p)
        return supports_threejs_canvas(p)

    def is_supported(self, path) -> bool:
        return supports_threejs_canvas(path)


class _DummyViewerHost(ViewerToolsMixin):
    def __init__(self) -> None:
        self.threejs_manager = _DummyManager()
        self._status = _DummyStatusBar()

    def statusBar(self):
        return self._status

    def tr(self, text: str) -> str:
        return text


class _LazyThreeJsHost(ViewerToolsMixin):
    def __init__(self) -> None:
        self.threejs_manager = None
        self._status = _DummyStatusBar()
        self.ensure_calls = 0

    def ensure_threejs_manager(self):
        self.ensure_calls += 1
        if self.threejs_manager is None:
            self.threejs_manager = _DummyManager()
        return self.threejs_manager

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


def test_open_threejs_example_lazily_initializes_threejs_manager(
    tmp_path: Path, monkeypatch
) -> None:
    widget = _LazyThreeJsHost()
    example_path = tmp_path / "helix.csv"
    example_path.write_text("x,y,z\n0,0,0\n", encoding="utf-8")

    monkeypatch.setattr(
        "annolid.gui.mixins.viewer_tools_mixin.generate_threejs_example",
        lambda example_id, out_dir: example_path,
    )

    warnings = []
    monkeypatch.setattr(
        QtWidgets.QMessageBox,
        "warning",
        lambda *args, **kwargs: warnings.append((args, kwargs)),
    )

    widget.open_threejs_example("helix_points_csv")

    assert widget.ensure_calls >= 1
    assert widget.threejs_manager is not None
    assert widget.threejs_manager.model_paths == [example_path]
    assert not warnings


def test_normalize_3d_source_path_collapses_nested_zarr_entries(tmp_path: Path) -> None:
    widget = _DummyViewerHost()
    zarr_root = tmp_path / "atlas_interleaved_30um_image.zarr"
    nested = zarr_root / "185.0.0"
    nested.parent.mkdir(parents=True, exist_ok=True)
    nested.write_text("chunk", encoding="utf-8")

    normalized = widget._normalize_3d_source_path(str(nested))
    assert normalized == str(zarr_root)


def test_resolve_picked_3d_source_prefers_hinted_zarr_root(tmp_path: Path) -> None:
    widget = _DummyViewerHost()
    zarr_root = tmp_path / "atlas_interleaved_30um_image.zarr"
    nested = zarr_root / "185.0.0"
    nested.parent.mkdir(parents=True, exist_ok=True)
    nested.write_text("chunk", encoding="utf-8")

    resolved = widget._resolve_picked_3d_source(
        selected_paths=[],
        current_dir=str(zarr_root),
        hinted_path=str(nested),
    )

    assert resolved == str(zarr_root)


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


def test_open_3d_viewer_reports_unsupported_source_for_nifti(
    monkeypatch, tmp_path: Path
) -> None:
    widget = _DummyViewerHost()
    nifti_path = tmp_path / "structural_brain.nii.gz"
    nifti_path.write_text("not-a-real-volume", encoding="utf-8")
    widget.video_loader = None
    widget.video_file = None
    widget.imagePath = ""
    widget.filename = ""

    monkeypatch.setattr(widget, "_detect_existing_3d_source", lambda: str(nifti_path))

    infos: list[str] = []
    monkeypatch.setattr(
        QtWidgets.QMessageBox,
        "information",
        lambda *args, **kwargs: infos.append(args[2] if len(args) > 2 else ""),
    )
    monkeypatch.setattr(
        QtWidgets.QMessageBox,
        "warning",
        lambda *args, **kwargs: infos.append(args[2] if len(args) > 2 else ""),
    )

    widget.open_3d_viewer()

    assert infos
    message = infos[0].lower()
    assert "not supported by the three.js viewer" in message


def test_open_3d_viewer_routes_model_source_to_threejs(
    monkeypatch, tmp_path: Path
) -> None:
    widget = _DummyViewerHost()
    model_path = tmp_path / "mesh.obj"
    model_path.write_text("# fake obj", encoding="utf-8")
    widget.video_loader = None
    widget.video_file = None
    widget.imagePath = ""
    widget.filename = ""
    monkeypatch.setattr(widget, "_detect_existing_3d_source", lambda: str(model_path))

    widget.open_3d_viewer()

    assert widget.threejs_manager.model_paths == [model_path]


def test_open_3d_viewer_routes_json_to_simulation_view(
    monkeypatch, tmp_path: Path
) -> None:
    widget = _DummyViewerHost()
    payload = tmp_path / "sim.json"
    payload.write_text("{}", encoding="utf-8")
    widget.video_loader = None
    widget.video_file = None
    widget.imagePath = ""
    widget.filename = ""
    monkeypatch.setattr(widget, "_detect_existing_3d_source", lambda: str(payload))

    widget.open_3d_viewer()

    assert widget.threejs_manager.sim_paths == [payload]
