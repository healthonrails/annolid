from __future__ import annotations

import json
import tempfile
from pathlib import Path
import subprocess
import time

from qtpy import QtCore, QtWidgets

from annolid.gui.flybody_support import (
    FLYBODY_GITHUB_URL,
    build_clone_flybody_command,
    build_live_flybody_command,
    build_setup_flybody_command,
    default_flybody_install_dir,
    pick_ready_flybody_runtime,
    summarize_flybody_status,
)
from annolid.gui.models_registry import PATCH_SIMILARITY_MODELS
from annolid.gui.threejs_examples import (
    attach_flybody_floor,
    attach_flybody_mesh,
    generate_threejs_example,
)
from annolid.gui.widgets.vtk_volume_utils import path_matches_ext
from annolid.gui.workers import FlexibleWorker
from annolid.simulation import (
    SimulationRunRequest,
    build_default_output_path,
    run_simulation_workflow,
)
from annolid.utils.logger import logger

_LIVE_FLYBODY_EXAMPLE_STEPS = 96
_LIVE_FLYBODY_EXAMPLE_SEED = 7
_LIVE_FLYBODY_EXAMPLE_CACHE_TTL_SECONDS = 300.0
_LIVE_FLYBODY_SUBPROCESS_TIMEOUT_SECONDS = 45.0
_LIVE_FLYBODY_PAYLOAD_VERSION = 3
_BUILTIN_STACK_SUFFIXES = (
    ".tif",
    ".tiff",
    ".ome.tif",
    ".ome.tiff",
)
_POINT_CLOUD_SUFFIXES = {".ply", ".csv", ".xyz"}
_MESH_SUFFIXES = {".stl", ".obj"}
_THREEJS_SUFFIXES = _POINT_CLOUD_SUFFIXES | _MESH_SUFFIXES


def _supports_builtin_stack_viewer(path: str | Path | None) -> bool:
    try:
        if not path:
            return False
        return path_matches_ext(Path(path), _BUILTIN_STACK_SUFFIXES)
    except Exception:
        return False


def _normalize_volume_selection(raw: str) -> str:
    try:
        p = Path(raw)
        if p.is_file():
            if p.suffix.lower() in (".img", ".hdr"):
                stem_l = p.stem.lower()
                for child in p.parent.iterdir():
                    if (
                        child.is_file()
                        and child.stem.lower() == stem_l
                        and child.suffix.lower() == ".hdr"
                    ):
                        return str(child)
            if p.name.lower() == "zarr.json":
                return str(p.parent)
            if p.name.lower() == ".zgroup":
                return str(p.parent)
            if (p.parent / ".zarray").exists():
                return str(p.parent)
        cur = p
        for _ in range(3):
            if (
                cur.name.lower().endswith(".zarr")
                or (cur / ".zarray").exists()
                or (cur / "zarr.json").exists()
                or (cur / ".zgroup").exists()
            ):
                return str(cur)
            if (cur / "data" / ".zarray").exists() or (
                cur / "data" / "zarr.json"
            ).exists():
                return str(cur / "data")
            cur = cur.parent
    except Exception:
        pass
    return raw


def _retain_dialog(host: object, attr_name: str, dialog: object) -> None:
    try:
        setattr(host, attr_name, dialog)
    except Exception:
        return
    destroyed = getattr(dialog, "destroyed", None)
    if destroyed is not None:
        try:
            destroyed.connect(lambda *_args: setattr(host, attr_name, None))
        except Exception:
            pass


def _is_recent_live_flybody_payload(
    payload_path: str | Path,
    *,
    max_age_seconds: float = _LIVE_FLYBODY_EXAMPLE_CACHE_TTL_SECONDS,
    behavior: str | None = None,
) -> bool:
    path = Path(payload_path)
    if not path.exists():
        return False
    try:
        age_seconds = time.time() - path.stat().st_mtime
        if age_seconds < 0 or age_seconds > max_age_seconds:
            return False
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    if not isinstance(payload, dict):
        return False
    if payload.get("kind") != "annolid-simulation-v1":
        return False
    adapter = str(payload.get("adapter") or "")
    if adapter != "flybody-live":
        return False
    metadata = payload.get("metadata") or {}
    run_metadata = metadata.get("run_metadata") or {}
    if int(run_metadata.get("payload_version") or 0) != _LIVE_FLYBODY_PAYLOAD_VERSION:
        return False
    if behavior is not None and str(run_metadata.get("behavior") or "") != str(
        behavior
    ):
        return False
    return True


def _run_logged_subprocess(
    command: list[str],
    *,
    cwd: str | Path,
    timeout_seconds: float,
) -> None:
    with tempfile.TemporaryFile(mode="w+t", encoding="utf-8") as log_file:
        try:
            subprocess.run(
                command,
                cwd=str(Path(cwd)),
                check=True,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=timeout_seconds,
            )
            return
        except subprocess.TimeoutExpired as exc:
            log_file.seek(0)
            log_tail = log_file.read()[-4000:].strip()
            detail = (
                f"Timed out after {timeout_seconds:.0f}s."
                if not log_tail
                else f"Timed out after {timeout_seconds:.0f}s.\n\n{log_tail}"
            )
            raise RuntimeError(detail) from exc
        except subprocess.CalledProcessError as exc:
            log_file.seek(0)
            log_tail = log_file.read()[-4000:].strip()
            detail = (
                f"Command exited with status {exc.returncode}."
                if not log_tail
                else f"Command exited with status {exc.returncode}.\n\n{log_tail}"
            )
            raise RuntimeError(detail) from exc


def _read_log_tail(path: str | Path, *, max_chars: int = 4000) -> str:
    try:
        text = Path(path).read_text(encoding="utf-8")
    except Exception:
        return ""
    return text[-max_chars:].strip()


def _prepare_live_flybody_view_payload(
    payload_path: str | Path,
    *,
    base_dir: str | Path,
) -> str:
    payload_file = Path(payload_path)
    payload = json.loads(payload_file.read_text(encoding="utf-8"))
    payload = attach_flybody_mesh(payload, base_dir)
    payload = attach_flybody_floor(payload)
    floor = (payload.get("environment") or {}).get("floor") or {}
    model_scale = float(
        ((payload.get("frames") or [{}])[0].get("model_pose") or {}).get("scale") or 7.5
    )
    if isinstance(floor.get("position"), list) and len(floor["position"]) >= 3:
        floor_position = floor["position"]
        normalized_floor = dict(floor)
        normalized_floor["position"] = [
            0.0,
            float(floor_position[2]) * model_scale,
            0.0,
        ]
        payload["environment"] = dict(payload.get("environment") or {})
        payload["environment"]["floor"] = normalized_floor
    payload.setdefault("display", {})
    payload["display"].update(
        {
            "show_points": False,
            "show_labels": False,
            "show_edges": False,
            "show_trails": False,
        }
    )
    payload["playback"] = {
        "autoplay": True,
        "loop": True,
        "interval_ms": 90,
    }
    payload_file.write_text(
        json.dumps(payload, separators=(",", ":")),
        encoding="utf-8",
    )
    return str(payload_file)


class ViewerToolsMixin:
    """3D viewer and PCA map UI helpers."""

    def _show_flybody_static_example(self, manager) -> bool:
        try:
            base_dir = Path(tempfile.gettempdir()) / "annolid_threejs_examples"
            example_path = generate_threejs_example("flybody_simulation_json", base_dir)
        except Exception as exc:
            QtWidgets.QMessageBox.warning(
                self,
                self.tr("FlyBody 3D Example"),
                self.tr("Failed to generate FlyBody example:\n%1").replace(
                    "%1", str(exc)
                ),
            )
            return False
        if manager.show_simulation_in_viewer(example_path):
            self.statusBar().showMessage(
                self.tr(
                    "Loaded FlyBody 3D example. Start a live simulation when ready."
                ),
                5000,
            )
            return True
        return False

    def _detect_existing_3d_source(self) -> str | None:
        source_path = None
        try:
            from annolid.data import videos as _videos_mod

            if (
                isinstance(self.video_loader, _videos_mod.TiffStackVideo)
                and self.video_file
            ):
                source_path = str(self.video_file)
        except Exception:
            pass

        if source_path:
            return source_path

        current_image = Path(str(getattr(self, "imagePath", "") or "")).expanduser()
        large_backend = getattr(self, "large_image_backend", None)
        if current_image.exists() and current_image.suffix.lower() in {".tif", ".tiff"}:
            if (
                large_backend is not None
                and int(getattr(large_backend, "get_page_count", lambda: 1)() or 1) > 1
            ):
                return str(current_image)
        return None

    def _pick_3d_source_from_dialog(self) -> str | None:
        start_dir = (
            str(Path(self.filename).parent) if getattr(self, "filename", None) else "."
        )
        filters = self.tr(
            "3D sources (*.tif *.tiff *.ome.tif *.ome.tiff *.nii *.nii.gz *.hdr *.img *.dcm *.dicom *.ima *.IMA *.ply *.csv *.xyz *.stl *.STL *.obj *.OBJ *.zarr *.zarr.json *.zgroup);;All files (*.*)"
        )
        dialog = QtWidgets.QFileDialog(
            self, self.tr("Choose 3D Volume (TIFF/NIfTI/DICOM/Zarr)")
        )
        dialog.setDirectory(start_dir)
        dialog.setNameFilter(filters)
        dialog.setFileMode(QtWidgets.QFileDialog.ExistingFiles)
        dialog.setOption(QtWidgets.QFileDialog.DontUseNativeDialog, True)
        dialog.setOption(QtWidgets.QFileDialog.ReadOnly, True)
        paths: list[str] = []
        if dialog.exec_() == QtWidgets.QDialog.Accepted:
            paths = dialog.selectedFiles()
        if not paths:
            folder = QtWidgets.QFileDialog.getExistingDirectory(
                self,
                self.tr("Choose Volume Folder (DICOM/Zarr)"),
                start_dir,
                QtWidgets.QFileDialog.ShowDirsOnly | QtWidgets.QFileDialog.ReadOnly,
            )
            if not folder:
                return None
            paths = [folder]
        return _normalize_volume_selection(paths[0]) if paths else None

    def _open_vtk_volume_viewer(
        self, source_path: str
    ) -> tuple[bool, Exception | None, bool]:
        vtk_missing = False
        vtk_error = None
        try:
            from annolid.gui.widgets.vtk_volume_viewer import VTKVolumeViewerDialog  # type: ignore

            dlg = VTKVolumeViewerDialog(source_path, parent=self)
            _retain_dialog(self, "_vtk_volume_viewer_dialog", dlg)
            dlg.setModal(False)
            dlg.show()
            dlg.raise_()
            dlg.activateWindow()
            return True, None, False
        except ModuleNotFoundError as exc:
            vtk_error = exc
            vtk_missing = True
        except ImportError as exc:
            vtk_error = exc
            vtk_missing = True
        except Exception as exc:
            vtk_error = exc
        return False, vtk_error, vtk_missing

    def _probe_vtk_available(self) -> tuple[bool, str | None]:
        try:
            try:
                import vtkmodules  # noqa: F401
            except Exception:
                import vtk  # noqa: F401
            return True, None
        except Exception as exc:
            return False, str(exc)

    def _open_builtin_stack_viewer(self, source_path: str) -> bool:
        try:
            from annolid.gui.widgets.volume_viewer import VolumeViewerDialog

            dlg = VolumeViewerDialog(source_path, parent=self)
            _retain_dialog(self, "_builtin_volume_viewer_dialog", dlg)
            dlg.setModal(False)
            dlg.show()
            dlg.raise_()
            dlg.activateWindow()
            return True
        except Exception as exc:
            QtWidgets.QMessageBox.warning(
                self,
                self.tr("3D Viewer"),
                self.tr(f"Unable to open 3D viewer: {exc}"),
            )
            return False

    def open_3d_viewer(self):
        """Open Annolid's 3D viewer for volume, mesh, or point-cloud sources."""
        source_path = (
            self._detect_existing_3d_source() or self._pick_3d_source_from_dialog()
        )
        if not source_path:
            return

        opened, vtk_error, vtk_missing = self._open_vtk_volume_viewer(source_path)
        if opened:
            return

        try:
            suffix = Path(source_path).suffix.lower()
        except Exception:
            suffix = ""

        if suffix in _THREEJS_SUFFIXES:
            manager = getattr(self, "threejs_manager", None)
            if manager is not None:
                try:
                    if manager.show_model_in_viewer(source_path):
                        return
                except Exception:
                    pass

        requires_vtk = suffix in _POINT_CLOUD_SUFFIXES or suffix in _MESH_SUFFIXES
        vtk_ok, vtk_probe_error = self._probe_vtk_available()
        vtk_missing = not vtk_ok

        if requires_vtk:
            if vtk_missing:
                QtWidgets.QMessageBox.information(
                    self,
                    self.tr("Mesh/Point Cloud Viewer Requires VTK"),
                    self.tr(
                        "PLY/CSV/XYZ point clouds and STL/OBJ meshes require VTK with Qt support.\n\n"
                        f"Details: {vtk_probe_error or 'Unknown import error'}\n\n"
                        "Conda:  conda install -c conda-forge vtk\n"
                        "Pip:    pip install vtk"
                    ),
                )
            else:
                QtWidgets.QMessageBox.warning(
                    self,
                    self.tr("Mesh/Point Cloud Viewer"),
                    self.tr("Failed to open the VTK mesh/point cloud viewer.\n%s")
                    % (str(vtk_error) if vtk_error else ""),
                )
            return

        if not _supports_builtin_stack_viewer(source_path):
            details = f": {vtk_error}" if vtk_error else ""
            QtWidgets.QMessageBox.warning(
                self,
                self.tr("3D Viewer"),
                self.tr(
                    "Unable to open the VTK 3D viewer for this source.\n\n"
                    "The built-in fallback viewer only supports TIFF stacks.\n"
                    f"Source: {source_path}{details}"
                ),
            )
            return

        if not self._open_builtin_stack_viewer(source_path):
            return

        if vtk_missing:
            QtWidgets.QMessageBox.information(
                self,
                self.tr("True 3D Rendering (Optional)"),
                self.tr(
                    "For interactive 3D volume rendering, install VTK with Qt support.\n\n"
                    "Conda:  conda install -c conda-forge vtk\n"
                    "Pip:    pip install vtk\n\n"
                    "You are currently using the built-in slice/MIP viewer."
                ),
            )

    def open_threejs_example(self, example_id: str):
        manager = getattr(self, "threejs_manager", None)
        if manager is None:
            QtWidgets.QMessageBox.warning(
                self,
                self.tr("Three.js Examples"),
                self.tr("Three.js canvas is not available in this session."),
            )
            return
        try:
            base_dir = Path(tempfile.gettempdir()) / "annolid_threejs_examples"
            example_path = generate_threejs_example(example_id, base_dir)
        except Exception as exc:
            QtWidgets.QMessageBox.warning(
                self,
                self.tr("Three.js Examples"),
                self.tr("Failed to generate example:\n%1").replace("%1", str(exc)),
            )
            return
        if example_path.suffix.lower() == ".json":
            if manager.show_simulation_in_viewer(example_path):
                if example_id == "flybody_simulation_json":
                    self.statusBar().showMessage(
                        self.tr(
                            "Loaded FlyBody 3D example. Use the FlyBody controls in the 3D viewer for live motion."
                        ),
                        6000,
                    )
                return
        if not manager.show_model_in_viewer(example_path):
            # If it's an HTML file, we can try to open it directly if the manager supports it
            # or use the system browser.
            if example_path.suffix.lower() == ".html":
                from annolid.gui.widgets.threejs_viewer_server import (
                    _ensure_threejs_http_server,
                )

                base_url = _ensure_threejs_http_server()
                url = f"{base_url}/threejs/{example_path.name}"
                if hasattr(
                    manager, "show_url_in_viewer"
                ) and manager.show_url_in_viewer(url):
                    return
                import webbrowser

                webbrowser.open(url)
                return

            QtWidgets.QMessageBox.warning(
                self,
                self.tr("Three.js Examples"),
                self.tr("Unable to open generated example in Three.js viewer."),
            )

    def start_live_flybody_example(self) -> None:
        self._start_live_flybody_behavior_example(
            behavior="walk_imitation",
            label=self.tr("FlyBody Live Walk"),
        )

    def _start_live_flybody_behavior_example(
        self, *, behavior: str, label: str
    ) -> None:
        manager = getattr(self, "threejs_manager", None)
        if manager is None:
            QtWidgets.QMessageBox.warning(
                self,
                label,
                self.tr("Three.js canvas is not available in this session."),
            )
            return
        self._show_flybody_static_example(manager)
        if self._start_live_flybody_example(manager, behavior=behavior, label=label):
            return
        QtWidgets.QMessageBox.information(
            self,
            label,
            self.tr(
                "A ready FlyBody runtime was not found.\n\n"
                "Open 'FlyBody Status…' to inspect setup, or use 'Run Simulation to 3D Viewer…' with another backend."
            ),
        )

    def handle_flybody_viewer_command(self, action: str, behavior: str) -> None:
        if action == "stop":
            self._stop_live_flybody_example()
            return
        self._start_live_flybody_behavior_example(
            behavior=behavior or "walk_imitation",
            label=self.tr("FlyBody Live Simulation"),
        )

    def _flybody_runtime_summary_lines(self) -> tuple[list[str], bool]:
        summary = summarize_flybody_status()
        lines = [
            f"Repo: {summary['repo_root']}"
            if summary.get("repo_root")
            else "Repo: missing",
        ]
        for candidate in summary.get("candidates", []):
            python = str(candidate.get("python") or "").strip()
            exists = bool(candidate.get("exists"))
            if not python:
                continue
            if not exists:
                lines.append(f"Python: {python} [missing]")
                continue
            status = "ready" if candidate.get("ready") else "not ready"
            reason = str(
                candidate.get("error") or candidate.get("stderr") or ""
            ).strip()
            line = f"Python: {python} [{status}]"
            if reason:
                line += f" - {reason}"
            lines.append(line)
        if len(lines) == 1:
            lines.append("Python: no FlyBody runtime candidates found")
        return lines, bool(summary.get("ready"))

    def show_flybody_status_dialog(self) -> None:
        lines, ready = self._flybody_runtime_summary_lines()
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle(self.tr("FlyBody Setup"))
        layout = QtWidgets.QVBoxLayout(dialog)

        summary_label = QtWidgets.QLabel(
            self.tr("Optional FlyBody support for live 3D examples and simulation."),
            dialog,
        )
        summary_label.setWordWrap(True)
        layout.addWidget(summary_label)

        status_label = QtWidgets.QLabel(
            self.tr("Status: Ready") if ready else self.tr("Status: Setup Needed"),
            dialog,
        )
        layout.addWidget(status_label)

        details = QtWidgets.QPlainTextEdit(dialog)
        details.setReadOnly(True)
        details.setPlainText("\n".join(lines))
        details.setMinimumHeight(180)
        layout.addWidget(details, 1)

        buttons = QtWidgets.QDialogButtonBox(dialog)
        refresh_button = buttons.addButton(
            self.tr("Refresh"),
            QtWidgets.QDialogButtonBox.ActionRole,
        )
        open_button = buttons.addButton(
            self.tr("Open Example"),
            QtWidgets.QDialogButtonBox.ActionRole,
        )
        install_button = buttons.addButton(
            self.tr("Install / Update"),
            QtWidgets.QDialogButtonBox.ActionRole,
        )
        close_button = buttons.addButton(QtWidgets.QDialogButtonBox.Close)
        layout.addWidget(buttons)

        refresh_button.clicked.connect(dialog.accept)
        open_button.clicked.connect(lambda: dialog.done(2))
        install_button.clicked.connect(lambda: dialog.done(3))
        close_button.clicked.connect(dialog.reject)

        result = dialog.exec_()
        if result == QtWidgets.QDialog.Accepted:
            self.show_flybody_status_dialog()
            return
        if result == 2:
            self.open_threejs_example("flybody_simulation_json")
            return
        if result == 3:
            self.install_flybody_optional()

    def _start_live_flybody_example(
        self, manager, *, behavior: str, label: str
    ) -> bool:
        base_dir = Path(tempfile.gettempdir()) / "annolid_threejs_examples"
        base_dir.mkdir(parents=True, exist_ok=True)
        payload_path = base_dir / f"flybody_live_{behavior}.json"
        if _is_recent_live_flybody_payload(payload_path, behavior=behavior):
            logger.info("Reusing cached live FlyBody example payload: %s", payload_path)
            manager.update_simulation_in_viewer(
                str(payload_path), title=f"flybody_live_{behavior}"
            )
            self.statusBar().showMessage(
                self.tr("Reused recent FlyBody live example in 3D viewer."),
                4000,
            )
            return True

        runtime_python, _probe = pick_ready_flybody_runtime()
        if runtime_python is None:
            self.statusBar().showMessage(
                self.tr(
                    "FlyBody 3D example loaded. No ready FlyBody runtime found for live simulation."
                ),
                5000,
            )
            return False

        progress = QtWidgets.QProgressDialog(
            self.tr("Preparing live FlyBody simulation…"),
            "",
            0,
            0,
            self,
        )
        progress.setWindowTitle(label)
        progress.setCancelButton(None)
        progress.setMinimumDuration(0)
        progress.setWindowModality(QtCore.Qt.WindowModal)
        progress.setLabelText(
            self.tr(
                "FlyBody 3D example is open. Live motion will replace it when ready."
            )
        )

        progress.setCancelButtonText(self.tr("Cancel"))
        progress.setWindowModality(QtCore.Qt.NonModal)
        progress.setAutoClose(False)
        progress.setAutoReset(False)

        process = QtCore.QProcess(self)
        log_path = base_dir / "flybody_live_rollout.log"
        process.setWorkingDirectory(str(Path.cwd()))
        process.setProgram(str(runtime_python))
        process.setArguments(
            build_live_flybody_command(
                runtime_python,
                out_path=payload_path,
                steps=_LIVE_FLYBODY_EXAMPLE_STEPS,
                seed=_LIVE_FLYBODY_EXAMPLE_SEED,
                behavior=behavior,
            )[1:]
        )
        process.setProcessChannelMode(QtCore.QProcess.MergedChannels)
        process.setStandardOutputFile(str(log_path))

        started = time.perf_counter()
        timeout_timer = QtCore.QTimer(self)
        timeout_timer.setSingleShot(True)

        self._flybody_live_process = process
        self._flybody_live_progress = progress
        self._flybody_live_timeout_timer = timeout_timer

        def _cleanup_process() -> None:
            timeout_timer.stop()
            try:
                progress.close()
            except Exception:
                pass
            self._flybody_live_process = None
            self._flybody_live_progress = None
            self._flybody_live_timeout_timer = None
            process.deleteLater()
            timeout_timer.deleteLater()

        def _handle_failure(detail: str) -> None:
            _cleanup_process()
            QtWidgets.QMessageBox.warning(
                self,
                label,
                self.tr(
                    "Live FlyBody rollout failed. Falling back to the bundled example.\n%1"
                ).replace("%1", detail),
            )
            try:
                fallback_path = generate_threejs_example(
                    "flybody_simulation_json",
                    base_dir,
                )
                manager.show_simulation_in_viewer(fallback_path)
            except Exception:
                pass

        def _on_timeout() -> None:
            if process.state() != QtCore.QProcess.NotRunning:
                process.kill()
            detail = _read_log_tail(log_path) or (
                f"Timed out after {_LIVE_FLYBODY_SUBPROCESS_TIMEOUT_SECONDS:.0f}s."
            )
            _handle_failure(detail)

        def _on_canceled() -> None:
            if process.state() != QtCore.QProcess.NotRunning:
                process.kill()
            _cleanup_process()
            self.statusBar().showMessage(
                self.tr("Canceled live FlyBody simulation startup."),
                4000,
            )

        def _on_finished(exit_code: int, exit_status) -> None:
            if exit_status != QtCore.QProcess.NormalExit or exit_code != 0:
                detail = _read_log_tail(log_path) or (
                    f"Command exited with status {exit_code}."
                )
                _handle_failure(detail)
                return
            try:
                prepared_path = _prepare_live_flybody_view_payload(
                    payload_path,
                    base_dir=base_dir,
                )
            except Exception as exc:
                _handle_failure(str(exc))
                return
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            logger.info(
                "Prepared live FlyBody example in %.1fms using %s",
                elapsed_ms,
                prepared_path,
            )
            _cleanup_process()
            manager.update_simulation_in_viewer(
                prepared_path,
                title=f"flybody_live_{behavior}",
            )
            self.statusBar().showMessage(
                self.tr("FlyBody live rollout opened in 3D viewer."),
                4000,
            )

        timeout_timer.timeout.connect(_on_timeout)
        progress.canceled.connect(_on_canceled)
        process.finished.connect(_on_finished)
        progress.show()
        timeout_timer.start(int(_LIVE_FLYBODY_SUBPROCESS_TIMEOUT_SECONDS * 1000.0))
        process.start()
        return True

    def _stop_live_flybody_example(self) -> None:
        process = getattr(self, "_flybody_live_process", None)
        progress = getattr(self, "_flybody_live_progress", None)
        timeout_timer = getattr(self, "_flybody_live_timeout_timer", None)
        if process is not None and process.state() != QtCore.QProcess.NotRunning:
            process.kill()
        if timeout_timer is not None:
            timeout_timer.stop()
        if progress is not None:
            try:
                progress.close()
            except Exception:
                pass
        self._flybody_live_process = None
        self._flybody_live_progress = None
        self._flybody_live_timeout_timer = None
        self.statusBar().showMessage(
            self.tr("Stopped live FlyBody simulation."),
            4000,
        )

    def install_flybody_optional(self) -> None:
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle(self.tr("Install FlyBody (Optional)"))
        layout = QtWidgets.QFormLayout(dialog)

        repo_edit = QtWidgets.QLineEdit(FLYBODY_GITHUB_URL, dialog)
        dest_edit = QtWidgets.QLineEdit(str(default_flybody_install_dir()), dialog)
        venv_edit = QtWidgets.QLineEdit(str(Path.cwd() / ".venv311"), dialog)
        python_edit = QtWidgets.QLineEdit("3.12", dialog)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel,
            parent=dialog,
        )
        layout.addRow(self.tr("GitHub URL"), repo_edit)
        layout.addRow(self.tr("Clone Destination"), dest_edit)
        layout.addRow(self.tr("Virtual Env"), venv_edit)
        layout.addRow(self.tr("Python Version"), python_edit)
        layout.addRow(buttons)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)

        if dialog.exec_() != QtWidgets.QDialog.Accepted:
            return

        repo_url = repo_edit.text().strip() or FLYBODY_GITHUB_URL
        dest = Path(
            dest_edit.text().strip() or default_flybody_install_dir()
        ).expanduser()
        venv_dir = Path(
            venv_edit.text().strip() or (Path.cwd() / ".venv311")
        ).expanduser()
        python_version = python_edit.text().strip() or "3.12"

        progress = QtWidgets.QProgressDialog(
            self.tr("Installing optional FlyBody support…"),
            "",
            0,
            0,
            self,
        )
        progress.setWindowTitle(self.tr("Install FlyBody"))
        progress.setCancelButton(None)
        progress.setMinimumDuration(0)
        progress.setWindowModality(QtCore.Qt.WindowModal)

        def _task() -> str:
            if not dest.exists():
                clone_cmd = build_clone_flybody_command(repo_url, dest)
                _run_logged_subprocess(
                    clone_cmd,
                    cwd=str(Path.cwd()),
                    timeout_seconds=_LIVE_FLYBODY_SUBPROCESS_TIMEOUT_SECONDS,
                )
            setup_cmd = build_setup_flybody_command(
                repo_root=Path.cwd(),
                flybody_path=dest,
                venv_dir=venv_dir,
                python_version=python_version,
            )
            _run_logged_subprocess(
                setup_cmd,
                cwd=str(Path.cwd()),
                timeout_seconds=_LIVE_FLYBODY_SUBPROCESS_TIMEOUT_SECONDS,
            )
            return str(dest)

        thread = QtCore.QThread(self)
        worker = FlexibleWorker(_task)
        worker.moveToThread(thread)

        def _finish(result) -> None:
            try:
                progress.close()
            except Exception:
                pass
            thread.quit()
            thread.wait(2000)
            worker.deleteLater()
            thread.deleteLater()
            if isinstance(result, Exception):
                QtWidgets.QMessageBox.warning(
                    self,
                    self.tr("Install FlyBody"),
                    self.tr("FlyBody install failed:\n%1").replace("%1", str(result)),
                )
                return
            QtWidgets.QMessageBox.information(
                self,
                self.tr("Install FlyBody"),
                self.tr(
                    "FlyBody was prepared at:\n%1\n\n"
                    "You can now reopen the FlyBody Simulation Example or run the FlyBody 3D workflow."
                ).replace("%1", str(result)),
            )

        thread.started.connect(worker.run)
        worker.finished_signal.connect(_finish)
        progress.show()
        thread.start()

    def open_simulation_3d_viewer(self) -> None:
        manager = getattr(self, "threejs_manager", None)
        if manager is None:
            QtWidgets.QMessageBox.warning(
                self,
                self.tr("Simulation Viewer"),
                self.tr("Three.js canvas is not available in this session."),
            )
            return
        start_dir = getattr(self, "lastOpenDir", str(Path.home()))
        filters = self.tr(
            "Simulation Output (*.ndjson);;JSON Files (*.json);;All Files (*)"
        )
        filename, _ = QtWidgets.QFileDialog.getOpenFileName(
            self,
            self.tr("Open FlyBody / Simulation Output"),
            start_dir,
            filters,
        )
        if not filename:
            return
        self.lastOpenDir = str(Path(filename).parent)
        manager.show_simulation_in_viewer(filename)

    def run_simulation_3d_viewer(self) -> None:
        manager = getattr(self, "threejs_manager", None)
        if manager is None:
            QtWidgets.QMessageBox.warning(
                self,
                self.tr("Simulation Viewer"),
                self.tr("Three.js canvas is not available in this session."),
            )
            return

        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle(self.tr("Run Simulation to 3D Viewer"))
        layout = QtWidgets.QFormLayout(dialog)

        backend_combo = QtWidgets.QComboBox(dialog)
        backend_combo.addItem("FlyBody", "flybody")
        backend_combo.addItem("Identity", "identity")

        input_edit = QtWidgets.QLineEdit(dialog)
        mapping_edit = QtWidgets.QLineEdit(dialog)
        depth_edit = QtWidgets.QLineEdit(dialog)
        schema_edit = QtWidgets.QLineEdit(dialog)
        video_name_edit = QtWidgets.QLineEdit(dialog)
        default_z_spin = QtWidgets.QDoubleSpinBox(dialog)
        default_z_spin.setRange(-10000.0, 10000.0)
        default_z_spin.setDecimals(4)
        default_z_spin.setValue(0.0)
        dry_run_box = QtWidgets.QCheckBox(self.tr("Dry run"), dialog)
        dry_run_box.setChecked(True)
        smooth_combo = QtWidgets.QComboBox(dialog)
        for mode in ("none", "ema", "one_euro", "kalman"):
            smooth_combo.addItem(mode, mode)
        max_gap_spin = QtWidgets.QSpinBox(dialog)
        max_gap_spin.setRange(0, 1000)
        fps_spin = QtWidgets.QDoubleSpinBox(dialog)
        fps_spin.setRange(1.0, 1000.0)
        fps_spin.setValue(30.0)
        output_edit = QtWidgets.QLineEdit(dialog)

        def _pick_file(target: QtWidgets.QLineEdit, title: str, filters: str) -> None:
            start_dir = getattr(self, "lastOpenDir", str(Path.home()))
            filename, _ = QtWidgets.QFileDialog.getOpenFileName(
                dialog,
                title,
                start_dir,
                filters,
            )
            if filename:
                target.setText(filename)
                self.lastOpenDir = str(Path(filename).parent)

        def _pick_output(target: QtWidgets.QLineEdit) -> None:
            start_dir = getattr(self, "lastOpenDir", str(Path.home()))
            filename, _ = QtWidgets.QFileDialog.getSaveFileName(
                dialog,
                self.tr("Save Simulation Output"),
                start_dir,
                self.tr("Simulation Output (*.ndjson)"),
            )
            if filename:
                target.setText(filename)
                self.lastOpenDir = str(Path(filename).parent)

        input_browse = QtWidgets.QPushButton(self.tr("Browse…"), dialog)
        input_browse.clicked.connect(
            lambda: _pick_file(
                input_edit,
                self.tr("Open Pose Input"),
                self.tr("Pose Input (*.json *.ndjson);;All Files (*)"),
            )
        )
        mapping_browse = QtWidgets.QPushButton(self.tr("Browse…"), dialog)
        mapping_browse.clicked.connect(
            lambda: _pick_file(
                mapping_edit,
                self.tr("Open Simulation Mapping"),
                self.tr("Mapping Files (*.json *.yaml *.yml);;All Files (*)"),
            )
        )
        depth_browse = QtWidgets.QPushButton(self.tr("Browse…"), dialog)
        depth_browse.clicked.connect(
            lambda: _pick_file(
                depth_edit,
                self.tr("Open Depth NDJSON"),
                self.tr("Depth Output (*.ndjson);;All Files (*)"),
            )
        )
        schema_browse = QtWidgets.QPushButton(self.tr("Browse…"), dialog)
        schema_browse.clicked.connect(
            lambda: _pick_file(
                schema_edit,
                self.tr("Open Pose Schema"),
                self.tr("Schema Files (*.json *.yaml *.yml);;All Files (*)"),
            )
        )
        output_browse = QtWidgets.QPushButton(self.tr("Browse…"), dialog)
        output_browse.clicked.connect(lambda: _pick_output(output_edit))

        def _row_with_browse(widget: QtWidgets.QWidget, browse: QtWidgets.QWidget):
            row = QtWidgets.QWidget(dialog)
            row_layout = QtWidgets.QHBoxLayout(row)
            row_layout.setContentsMargins(0, 0, 0, 0)
            row_layout.addWidget(widget, 1)
            row_layout.addWidget(browse)
            return row

        layout.addRow(self.tr("Backend"), backend_combo)
        layout.addRow(
            self.tr("Pose Input"),
            _row_with_browse(input_edit, input_browse),
        )
        layout.addRow(
            self.tr("Mapping"),
            _row_with_browse(mapping_edit, mapping_browse),
        )
        layout.addRow(
            self.tr("Depth NDJSON"),
            _row_with_browse(depth_edit, depth_browse),
        )
        layout.addRow(
            self.tr("Pose Schema"),
            _row_with_browse(schema_edit, schema_browse),
        )
        layout.addRow(self.tr("Video Name"), video_name_edit)
        layout.addRow(self.tr("Default Z"), default_z_spin)
        layout.addRow(self.tr("Smoothing"), smooth_combo)
        layout.addRow(self.tr("FPS"), fps_spin)
        layout.addRow(self.tr("Max Gap Frames"), max_gap_spin)
        layout.addRow("", dry_run_box)
        layout.addRow(
            self.tr("Output NDJSON"),
            _row_with_browse(output_edit, output_browse),
        )

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel,
            parent=dialog,
        )
        layout.addRow(buttons)
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)

        if dialog.exec_() != QtWidgets.QDialog.Accepted:
            return

        input_path = input_edit.text().strip()
        mapping_path = mapping_edit.text().strip()
        if not input_path or not mapping_path:
            QtWidgets.QMessageBox.warning(
                self,
                self.tr("Simulation Viewer"),
                self.tr("Pose input and mapping are required."),
            )
            return

        backend = str(backend_combo.currentData() or "flybody")
        out_path = output_edit.text().strip() or str(
            build_default_output_path(input_path, backend=backend)
        )
        request = SimulationRunRequest(
            backend=backend,
            input_path=input_path,
            mapping_path=mapping_path,
            out_ndjson=out_path,
            pose_schema=schema_edit.text().strip() or None,
            depth_ndjson=depth_edit.text().strip() or None,
            video_name=video_name_edit.text().strip() or None,
            default_z=float(default_z_spin.value()),
            dry_run=bool(dry_run_box.isChecked()),
            smooth_mode=str(smooth_combo.currentData() or "none"),
            fps=float(fps_spin.value()),
            max_gap_frames=int(max_gap_spin.value()),
        )

        progress = QtWidgets.QProgressDialog(
            self.tr("Running simulation…"),
            "",
            0,
            0,
            self,
        )
        progress.setWindowTitle(self.tr("Simulation Viewer"))
        progress.setCancelButton(None)
        progress.setMinimumDuration(0)
        progress.setWindowModality(QtCore.Qt.WindowModal)

        def _task() -> str:
            return str(run_simulation_workflow(request))

        thread = QtCore.QThread(self)
        worker = FlexibleWorker(_task)
        worker.moveToThread(thread)

        def _finish(result) -> None:
            try:
                progress.close()
            except Exception:
                pass
            thread.quit()
            thread.wait(2000)
            worker.deleteLater()
            thread.deleteLater()
            if isinstance(result, Exception):
                QtWidgets.QMessageBox.warning(
                    self,
                    self.tr("Simulation Viewer"),
                    self.tr("Simulation run failed:\n%1").replace("%1", str(result)),
                )
            else:
                manager.show_simulation_in_viewer(str(result))
                self.statusBar().showMessage(
                    self.tr("Simulation ready in 3D viewer."),
                    3000,
                )

        thread.started.connect(worker.run)
        worker.finished_signal.connect(_finish)
        progress.show()
        thread.start()

    def _on_pca_map_started(self):
        self.statusBar().showMessage(self.tr("Computing PCA feature map…"))

    def _on_pca_map_finished(self, payload: dict) -> None:
        if not self.pca_map_action.isChecked():
            return
        overlay = payload.get("overlay_rgba")
        self.canvas.setPCAMapOverlay(overlay)
        cluster_labels = payload.get("cluster_labels") or []
        if cluster_labels:
            labels_text = ", ".join(cluster_labels)
            message = self.tr("PCA clustering ready (%s)") % labels_text
        else:
            message = self.tr("PCA feature map ready.")
        self.statusBar().showMessage(message, 4000)

    def _on_pca_map_error(self, message: str) -> None:
        self.canvas.setPCAMapOverlay(None)
        QtWidgets.QMessageBox.warning(
            self,
            self.tr("PCA Feature Map"),
            message,
        )
        self._deactivate_pca_map()

    def _open_pca_map_settings(self):
        dialog = QtWidgets.QDialog(self)
        dialog.setWindowTitle(self.tr("PCA Feature Map Settings"))
        layout = QtWidgets.QFormLayout(dialog)

        model_combo = QtWidgets.QComboBox(dialog)
        for cfg in PATCH_SIMILARITY_MODELS:
            model_combo.addItem(cfg.display_name, cfg.identifier)

        current_index = model_combo.findData(self.pca_map_model)
        if current_index >= 0:
            model_combo.setCurrentIndex(current_index)

        alpha_spin = QtWidgets.QDoubleSpinBox(dialog)
        alpha_spin.setRange(0.05, 1.0)
        alpha_spin.setSingleStep(0.05)
        alpha_spin.setValue(self.pca_map_alpha)

        cluster_spin = QtWidgets.QSpinBox(dialog)
        cluster_spin.setRange(0, 32)
        cluster_spin.setValue(max(0, int(getattr(self, "pca_map_clusters", 0))))

        layout.addRow(self.tr("Model"), model_combo)
        layout.addRow(self.tr("Overlay opacity"), alpha_spin)
        layout.addRow(self.tr("Cluster count"), cluster_spin)

        buttons = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel,
            parent=dialog,
        )
        layout.addRow(buttons)

        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)

        if dialog.exec_() == QtWidgets.QDialog.Accepted:
            self.pca_map_model = model_combo.currentData()
            self.pca_map_alpha = alpha_spin.value()
            self.pca_map_clusters = cluster_spin.value()
            self.settings.setValue("pca_map/model", self.pca_map_model)
            self.settings.setValue("pca_map/alpha", self.pca_map_alpha)
            self.settings.setValue("pca_map/clusters", self.pca_map_clusters)
            self.statusBar().showMessage(
                self.tr("PCA feature map preferences updated."),
                3000,
            )
            if self.pca_map_action.isChecked():
                self._request_pca_map()
