import atexit
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml
from qtpy import QtCore, QtWidgets

from annolid.gui.workers import FlexibleWorker
from annolid.utils.logger import logger


class YOLOTrainingManager(QtCore.QObject):
    """Encapsulates YOLO training orchestration for the Annolid GUI."""

    def __init__(self, window: QtWidgets.QMainWindow) -> None:
        super().__init__(parent=window)
        self._window = window
        self._temp_configs: List[Path] = []
        self._active_jobs: List[Tuple[QtCore.QThread,
                                      FlexibleWorker, str]] = []
        self._training_running = False
        self._start_dialog: Optional[QtWidgets.QMessageBox] = None
        atexit.register(self.cleanup)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def prepare_data_config(self, config_file: str) -> Optional[str]:
        """Resolve relative paths in a YOLO data.yaml and persist to a temp file."""
        config_path = Path(config_file).expanduser().resolve()
        if not config_path.exists():
            QtWidgets.QMessageBox.warning(
                self._window,
                "Invalid Dataset",
                f"Dataset configuration file not found:\n{config_path}",
            )
            return None

        try:
            with config_path.open("r", encoding="utf-8") as stream:
                data_cfg: Dict[str, Any] = yaml.safe_load(stream) or {}
        except Exception as exc:
            logger.exception(
                "Failed to load YOLO dataset config: %s", config_path)
            QtWidgets.QMessageBox.critical(
                self._window,
                "Dataset Error",
                f"Could not read dataset configuration:\n{config_path}\n\n{exc}",
            )
            return None

        base_dir = config_path.parent

        def is_remote(path_str: str) -> bool:
            return path_str.startswith(("http://", "https://", "rtsp://", "rtmp://"))

        def resolve_entry(entry: Any) -> Any:
            if isinstance(entry, str):
                if not entry.strip():
                    return entry
                if is_remote(entry):
                    return entry
                entry_path = Path(entry).expanduser()
                resolved = entry_path if entry_path.is_absolute() else base_dir / entry_path
                return str(resolved.resolve())
            if isinstance(entry, (list, tuple)):
                return [resolve_entry(item) for item in entry]
            return entry

        for split in ("train", "val", "test"):
            if split in data_cfg and data_cfg[split]:
                data_cfg[split] = resolve_entry(data_cfg[split])

        if "path" in data_cfg:
            if data_cfg["path"]:
                data_cfg["path"] = resolve_entry(data_cfg["path"])
        else:
            data_cfg["path"] = str(base_dir)

        missing_paths: List[str] = []
        for split in ("train", "val", "test"):
            value = data_cfg.get(split)
            if not value:
                continue
            entries = value if isinstance(value, list) else [value]
            for entry in entries:
                if isinstance(entry, str) and not is_remote(entry):
                    candidate = Path(entry)
                    if any(char in entry for char in ("*", "?", "[")):
                        continue
                    if not candidate.exists():
                        missing_paths.append(f"{split}: {entry}")

        if missing_paths:
            QtWidgets.QMessageBox.warning(
                self._window,
                "Dataset Missing Files",
                "The following dataset paths could not be found:\n"
                + "\n".join(missing_paths),
            )
            return None

        return self._write_temp_config(data_cfg, config_path)

    def start_training(
        self,
        *,
        yolo_model_file: str,
        model_path: Optional[str],
        data_config_path: str,
        epochs: int,
        image_size: int,
        out_dir: Optional[str],
    ) -> bool:
        """Launch YOLO training on a background thread."""
        if self._training_running:
            QtWidgets.QMessageBox.warning(
                self._window,
                "Training In Progress",
                "A YOLO training job is already running in the background.",
            )
            self._release_temp_config(data_config_path)
            return False

        from annolid.yolo import configure_ultralytics_cache, resolve_weight_path
        configure_ultralytics_cache()
        from ultralytics import YOLO
        self._training_running = True
        self._window.statusBar().showMessage(
            self._window.tr("YOLO training started in the background...")
        )
        self._show_start_notification()

        def train_task():
            weight_path = resolve_weight_path(yolo_model_file)
            model = YOLO(str(weight_path))
            if model_path:
                try:
                    model.load(model_path)
                except Exception as exc:
                    raise RuntimeError(
                        f"Failed to load trained model: {exc}"
                    ) from exc
            return model.train(
                data=data_config_path,
                epochs=epochs,
                imgsz=image_size,
                project=out_dir if out_dir else None,
                workers=0,  # Avoid multiprocessing issues when invoked from GUI threads
                plots=False,  # Matplotlib is not thread-safe on macOS; disable GUI plotting
            )

        worker = FlexibleWorker(train_task)
        thread = QtCore.QThread(self)
        worker.moveToThread(thread)
        self._active_jobs.append((thread, worker, data_config_path))

        def on_finished(outcome):
            self._handle_finished(outcome, thread, worker, data_config_path)

        worker.finished_signal.connect(on_finished, QtCore.Qt.QueuedConnection)
        thread.started.connect(worker.run, QtCore.Qt.QueuedConnection)
        thread.finished.connect(thread.deleteLater, QtCore.Qt.QueuedConnection)
        thread.start()
        return True

    def is_running(self) -> bool:
        return self._training_running

    def cleanup(self) -> None:
        """Stop running jobs and remove temporary config files."""
        while self._active_jobs:
            thread, worker, temp_config = self._active_jobs.pop()
            try:
                thread.quit()
                thread.wait()
            except RuntimeError:
                logger.info("YOLO training thread already stopped.")
            try:
                worker.deleteLater()
            except RuntimeError:
                logger.debug("YOLO training worker already deleted.")
            if temp_config:
                self._release_temp_config(temp_config)
        self._close_start_notification()
        self._cleanup_temp_configs()

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _handle_finished(
        self,
        outcome: Any,
        thread: QtCore.QThread,
        worker: FlexibleWorker,
        temp_config: str,
    ) -> None:
        self._training_running = False
        self._active_jobs = [
            job for job in self._active_jobs if job[0] is not thread
        ]

        try:
            thread.quit()
            thread.wait()
        except RuntimeError:
            logger.debug("YOLO training thread already stopped.")
        try:
            thread.deleteLater()
        except RuntimeError:
            pass
        try:
            worker.deleteLater()
        except RuntimeError:
            logger.debug("YOLO training worker already deleted.")

        self._release_temp_config(temp_config)
        self._close_start_notification()

        if isinstance(outcome, Exception):
            logger.error("YOLO training failed: %s", outcome)
            QtWidgets.QMessageBox.critical(
                self._window,
                "Training Error",
                f"An error occurred during YOLO training:\n{outcome}",
            )
            self._window.statusBar().showMessage(
                self._window.tr(
                    "YOLO training failed. Check logs for details.")
            )
            return

        save_dir = getattr(outcome, "save_dir", None)
        details = f"\nResults are saved to:\n{save_dir}" if save_dir else ""
        QtWidgets.QMessageBox.information(
            self._window,
            "Training Completed",
            "YOLO model training completed successfully!" + details,
        )
        self._window.statusBar().showMessage(
            self._window.tr("YOLO training completed.")
        )

    def _show_start_notification(self) -> None:
        self._close_start_notification()
        dialog = QtWidgets.QMessageBox(self._window)
        dialog.setIcon(QtWidgets.QMessageBox.Information)
        dialog.setWindowTitle("Training Started")
        dialog.setText(
            "YOLO training has started in the background.\n"
            "You can continue working; a notification will appear when training finishes."
        )
        dialog.setStandardButtons(QtWidgets.QMessageBox.Ok)
        dialog.setModal(False)
        dialog.finished.connect(lambda _=None: self._clear_start_dialog())
        dialog.open()
        self._start_dialog = dialog

    def _close_start_notification(self) -> None:
        if not self._start_dialog:
            return
        try:
            if self._start_dialog.isVisible():
                self._start_dialog.close()
            self._start_dialog.deleteLater()
        except RuntimeError:
            pass
        self._start_dialog = None

    def _clear_start_dialog(self) -> None:
        self._start_dialog = None

    def _write_temp_config(
        self, data_cfg: Dict[str, Any], config_path: Path
    ) -> str:
        fd, temp_path = tempfile.mkstemp(
            prefix=f"{config_path.stem}_resolved_",
            suffix=".yaml",
            dir=str(config_path.parent),
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                yaml.safe_dump(data_cfg, handle, sort_keys=False)
        except Exception:
            try:
                os.close(fd)
            except OSError:
                pass
            Path(temp_path).unlink(missing_ok=True)
            raise
        resolved_path = Path(temp_path)
        self._temp_configs.append(resolved_path)
        return str(resolved_path)

    def _release_temp_config(self, path_str: str) -> None:
        if not path_str:
            return
        path_obj = Path(path_str)
        try:
            self._temp_configs.remove(path_obj)
        except ValueError:
            pass
        try:
            path_obj.unlink()
        except FileNotFoundError:
            return
        except Exception:
            logger.debug(
                "Could not remove temporary YOLO config: %s", path_obj)

    def _cleanup_temp_configs(self) -> None:
        while self._temp_configs:
            temp_path = self._temp_configs.pop()
            try:
                temp_path.unlink()
            except FileNotFoundError:
                continue
            except Exception:
                logger.debug(
                    "Could not remove temporary YOLO config: %s", temp_path)
