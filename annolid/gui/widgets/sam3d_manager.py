from __future__ import annotations

import functools
import json
import contextlib
import os
import threading
from pathlib import Path
from typing import Optional

from qtpy import QtCore, QtWidgets

from annolid.gui.workers import FlexibleWorker
from annolid.gui.widgets.sam3d_settings_dialog import Sam3DSettingsDialog
from annolid.three_d import sam3d_client
from annolid.three_d.sam3d_backend import Sam3DBackendError
from annolid.utils.logger import logger
from annolid.utils.qt2cv import convert_qt_image_to_rgb_cv_image
from annolid.gui.window_base import utils


class Sam3DManager(QtCore.QObject):
    """Encapsulates SAM 3D reconstruction wiring and settings."""

    def __init__(self, window) -> None:
        super().__init__(window)
        self.window = window
        self._sam3d_worker = None
        self._sam3d_worker_thread: Optional[QtCore.QThread] = None

    # ------------------------------------------------------------------ settings
    def _sam3d_client_config(self) -> sam3d_client.Sam3DClientConfig:
        cfg_block = {}
        try:
            cfg_block = dict((self.window._config or {}).get("sam3d", {}) or {})
        except Exception:
            cfg_block = {}
        try:
            persisted = (
                self.window.settings.value("sam3d")
                if hasattr(self.window, "settings")
                else None
            )
            if isinstance(persisted, str):
                try:
                    persisted = json.loads(persisted)
                except Exception:
                    persisted = None
            if isinstance(persisted, dict):
                cfg_block.update(persisted)
        except Exception:
            pass
        repo_path = cfg_block.get("repo_path") or os.environ.get(
            "SAM3D_HOME", "sam-3d-objects"
        )
        checkpoints_dir = cfg_block.get("checkpoints_dir")
        return sam3d_client.Sam3DClientConfig(
            repo_path=Path(repo_path),
            checkpoints_dir=Path(checkpoints_dir) if checkpoints_dir else None,
            checkpoint_tag=cfg_block.get("checkpoint_tag", "hf"),
            compile=bool(cfg_block.get("compile", False)),
            seed=cfg_block.get("seed"),
            python_executable=cfg_block.get("python_executable")
            or os.environ.get("SAM3D_PYTHON"),
            timeout_s=cfg_block.get("timeout_s"),
        )

    def configure_sam3d_settings(self) -> None:
        dialog = Sam3DSettingsDialog(
            parent=self.window,
            config=(self.window._config or {}).get("sam3d", {}),
        )
        if dialog.exec_() != QtWidgets.QDialog.Accepted:
            return
        values = dialog.values()
        self.window._config.setdefault("sam3d", {}).update(values)
        if isinstance(self.window.settings, QtCore.QSettings):
            self.window.settings.setValue("sam3d", values)
            try:
                self.window.settings.setValue("sam3d_json", json.dumps(values))
            except Exception:
                pass

    # ------------------------------------------------------------------ main action
    def run_sam3d_reconstruction(self) -> None:
        """Run SAM 3D on the current frame and selected instance mask."""
        w = self.window
        if not w.video_file:
            QtWidgets.QMessageBox.warning(
                w,
                w.tr("No video loaded"),
                w.tr("Open a video and select an instance before running SAM 3D."),
            )
            return
        selected = getattr(w.canvas, "selectedShapes", None)
        if not selected:
            QtWidgets.QMessageBox.information(
                w,
                w.tr("No shape selected"),
                w.tr("Select an instance to reconstruct in 3D."),
            )
            return
        frame_rgb = self._current_frame_rgb()
        if frame_rgb is None:
            QtWidgets.QMessageBox.warning(
                w,
                w.tr("Frame unavailable"),
                w.tr("Unable to read the current frame for SAM 3D."),
            )
            return
        shape = selected[0]
        pts = []
        for pt in shape.points:
            try:
                pts.append((pt.x(), pt.y()))
            except Exception:
                try:
                    pts.append((pt[0], pt[1]))
                except Exception:
                    pass
        if not pts:
            QtWidgets.QMessageBox.warning(
                w, w.tr("Invalid shape"), w.tr("No points found on shape.")
            )
            return
        mask = utils.shape_to_mask(frame_rgb.shape[:2], pts, shape.shape_type)
        try:
            cfg = self._sam3d_client_config()
            availability = sam3d_client.sam3d_available(cfg, mode="auto")
        except Exception as exc:
            availability = sam3d_client.Sam3DAvailability(
                ok=False, reason=str(exc), mode="auto"
            )
        if not availability.ok:
            QtWidgets.QMessageBox.information(
                w,
                w.tr("SAM 3D unavailable"),
                w.tr("SAM 3D Objects is not available.\n%s")
                % (availability.reason or w.tr("Install SAM 3D and set SAM3D_HOME.")),
            )
            return

        video_path = Path(w.video_file)
        out_dir_cfg = (
            (self.window._config or {}).get("sam3d", {}).get("output_dir", None)
        )
        output_dir = (
            Path(out_dir_cfg).expanduser()
            if out_dir_cfg
            else video_path.parent / f"{video_path.stem}_sam3d"
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        basename = (
            f"{video_path.stem}_f{w.frame_number:06d}_{shape.label or 'instance'}"
        )
        metadata = {
            "video": str(video_path),
            "frame_index": int(w.frame_number),
            "label": str(shape.label),
        }
        job = sam3d_client.Sam3DJobSpec(
            image=frame_rgb,
            mask=mask,
            output_dir=output_dir,
            basename=basename,
            metadata=metadata,
        )
        runner = (
            sam3d_client.run_subprocess
            if cfg.python_executable
            else sam3d_client.run_inprocess
        )
        worker = FlexibleWorker(runner, cfg, job)
        worker_thread = QtCore.QThread(w)
        worker.moveToThread(worker_thread)
        worker.finished_signal.connect(
            functools.partial(self._handle_sam3d_finished, worker_thread=worker_thread)
        )
        worker.finished_signal.connect(worker_thread.quit)
        worker_thread.started.connect(worker.run)
        worker_thread.start()
        self._sam3d_worker = worker
        self._sam3d_worker_thread = worker_thread
        w.statusBar().showMessage(w.tr("SAM 3D reconstruction started..."), 5000)

    # ------------------------------------------------------------------ internals
    def _current_frame_rgb(self):
        pixmap = getattr(self.window.canvas, "pixmap", None)
        if pixmap is None or pixmap.isNull():
            return None
        qimg = pixmap.toImage()
        try:
            return convert_qt_image_to_rgb_cv_image(qimg).copy()
        except Exception:
            return None

    def _handle_sam3d_finished(self, result, *, worker_thread: QtCore.QThread):
        with contextlib.suppress(Exception):
            worker_thread.quit()
            worker_thread.wait()
        self._sam3d_worker = None
        self._sam3d_worker_thread = None

        if isinstance(result, Sam3DBackendError):
            msg = str(result)
            QtWidgets.QMessageBox.critical(
                self.window,
                self.window.tr("SAM 3D failed"),
                self.window.tr("SAM 3D failed:\n%s") % msg,
            )
            self.window.statusBar().showMessage(self.window.tr("SAM 3D failed"), 5000)
            return
        if isinstance(result, Exception):
            msg = str(result)
            QtWidgets.QMessageBox.critical(
                self.window,
                self.window.tr("SAM 3D failed"),
                self.window.tr("SAM 3D failed:\n%s") % msg,
            )
            self.window.statusBar().showMessage(self.window.tr("SAM 3D failed"), 5000)
            return
        if isinstance(result, sam3d_client.Sam3DResult):
            QtWidgets.QMessageBox.information(
                self.window,
                self.window.tr("SAM 3D complete"),
                self.window.tr("PLY saved to:\n%s") % str(result.ply_path),
            )
            try:
                from annolid.gui.widgets.vtk_volume_viewer import (  # type: ignore
                    VTKVolumeViewerDialog,
                )

                dlg = VTKVolumeViewerDialog(str(result.ply_path), parent=self.window)
                dlg.setModal(False)
                dlg.show()
                dlg.raise_()
                dlg.activateWindow()
            except Exception as exc:
                logger.debug("Unable to open VTK viewer for SAM3D result: %s", exc)
            self.window.statusBar().showMessage(
                self.window.tr("SAM 3D complete: %s") % str(result.ply_path), 5000
            )
            self._maybe_offer_gradio_model3d(str(result.ply_path))
            return
        self.window.statusBar().showMessage(self.window.tr("SAM 3D finished"), 3000)

    def _maybe_offer_gradio_model3d(self, ply_path: str) -> None:
        """Offer to open the PLY in a Gradio Model3D viewer if available."""
        try:
            import gradio as gr  # type: ignore
        except Exception:
            return
        answer = QtWidgets.QMessageBox.question(
            self.window,
            self.window.tr("Open in Gradio?"),
            self.window.tr("Open the reconstructed PLY in a Gradio Model3D viewer?"),
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
        )
        if answer != QtWidgets.QMessageBox.Yes:
            return

        def _launch():
            try:
                with gr.Blocks() as demo:
                    gr.Markdown("# SAM 3D Reconstruction")
                    gr.Model3D(value=ply_path, label="PLY")
                demo.launch(
                    share=False,
                    inbrowser=True,
                    prevent_thread_lock=True,
                    show_error=True,
                )
            except Exception as exc:  # pragma: no cover - UI path
                logger.warning("Gradio viewer failed: %s", exc)

        threading.Thread(target=_launch, daemon=True).start()
