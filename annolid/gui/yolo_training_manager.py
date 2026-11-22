import atexit
import os
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import yaml
from qtpy import QtCore, QtGui, QtWidgets

from annolid.gui.qt_threads import configure_worker_thread_stack
from annolid.gui.workers import FlexibleWorker
from annolid.utils.logger import logger
from annolid.utils.runs import allocate_run_dir, shared_runs_root
from annolid.yolo import infer_yolo_task
from annolid.yolo.dataset_prep import (
    YOLODatasetError,
    check_training_preflight,
    count_dataset_images,
    infer_flip_idx,
    prepare_yolo_dataset,
)

_RESERVED_OVERRIDE_KEYS = frozenset(
    {"model", "data", "device", "plots", "project", "name", "resume"}
)


@dataclass(frozen=True)
class _CLITrainingOutcome:
    save_dir: str
    best_weights: Optional[str] = None


def _locate_run_checkpoint(run_dir: Path) -> Optional[Path]:
    for filename in ("best.pt", "last.pt"):
        candidate = run_dir / "weights" / filename
        if candidate.is_file():
            return candidate
    return None


class YOLOTrainingManager(QtCore.QObject):
    """Encapsulates YOLO training orchestration for the Annolid GUI."""

    training_started = QtCore.Signal(object)
    training_finished = QtCore.Signal(object)

    def __init__(self, window: QtWidgets.QMainWindow) -> None:
        super().__init__(parent=window)
        self._window = window
        self._temp_configs: List[Path] = []
        self._temp_dataset_dirs: List[Path] = []
        self._active_jobs: List[Tuple[QtCore.QThread, FlexibleWorker, str]] = []
        self._training_running = False
        self._start_dialog: Optional[QtWidgets.QMessageBox] = None
        atexit.register(self.cleanup)

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #
    def prepare_data_config(
        self,
        config_file: str,
        *,
        expected_task: Optional[str] = None,
    ) -> Optional[str]:
        """Resolve training data config and persist a YOLO-compatible temp YAML."""
        try:
            prepared = prepare_yolo_dataset(
                config_file,
                expected_task=expected_task,
            )
        except YOLODatasetError as exc:
            if exc.critical:
                QtWidgets.QMessageBox.critical(self._window, exc.title, exc.message)
            else:
                QtWidgets.QMessageBox.warning(self._window, exc.title, exc.message)
            return None
        except Exception as exc:
            logger.exception("Failed to prepare YOLO dataset config: %s", config_file)
            QtWidgets.QMessageBox.critical(
                self._window,
                "Dataset Error",
                f"Could not prepare dataset configuration:\n{config_file}\n\n{exc}",
            )
            return None

        self._temp_configs.extend(prepared.temp_configs)
        self._temp_dataset_dirs.extend(prepared.temp_dataset_dirs)
        return str(prepared.data_yaml)

    def start_training(
        self,
        *,
        yolo_model_file: str,
        model_path: Optional[str],
        data_config_path: str,
        epochs: int,
        image_size: int,
        batch_size: int,
        device: Optional[str],
        plots: bool,
        train_overrides: Optional[Dict[str, Any]],
        out_dir: Optional[str],
        resume: bool = False,
    ) -> bool:
        """Launch YOLO training on a background thread."""
        expected_task = self._infer_expected_task(
            yolo_model_file,
            model_path=model_path,
        )
        config_path = Path(data_config_path).expanduser().resolve()
        if config_path not in self._temp_configs:
            prepared_cfg = self.prepare_data_config(
                str(config_path),
                expected_task=expected_task,
            )
            if not prepared_cfg:
                return False
            data_config_path = prepared_cfg
        else:
            data_config_path = str(config_path)

        if self._training_running:
            QtWidgets.QMessageBox.warning(
                self._window,
                "Training In Progress",
                "A YOLO training job is already running in the background.",
            )
            self._release_temp_config(data_config_path)
            return False

        from annolid.yolo import resolve_weight_path, resolve_yolo_device
        from annolid.yolo.ultralytics_cli import (
            build_yolo_train_command,
            ensure_parent_dir,
            format_yolo_cli_error,
            run_yolo_cli,
        )

        resolved_device = resolve_yolo_device(device)

        if not self._confirm_preflight(
            data_config_path=data_config_path,
            batch_size=batch_size,
            device=resolved_device,
            plots=plots,
            expected_task=expected_task,
        ):
            self._release_temp_config(data_config_path)
            return False

        if expected_task == "pose":
            try:
                payload = (
                    yaml.safe_load(Path(data_config_path).read_text(encoding="utf-8"))
                    or {}
                )
            except Exception:
                payload = {}
            if not payload.get("kpt_shape"):
                QtWidgets.QMessageBox.warning(
                    self._window,
                    "Pose Dataset Required",
                    "The selected YOLO pose model requires a dataset with 'kpt_shape'.\n\n"
                    "Choose a YOLO pose dataset or a COCO keypoints dataset.",
                )
                self._release_temp_config(data_config_path)
                return False

        self._training_running = True
        self._window.statusBar().showMessage(
            self._window.tr("YOLO training started in the background...")
        )
        self._show_start_notification()

        model_label = Path(model_path or yolo_model_file).stem or "yolo"
        runs_root = (
            Path(out_dir).expanduser().resolve() if out_dir else shared_runs_root()
        )
        run_dir = allocate_run_dir(
            task="yolo",
            model=model_label,
            runs_root=runs_root,
        )
        run_rel = run_dir.relative_to(runs_root)
        self.training_started.emit(
            {
                "task": "yolo",
                "model": model_label,
                "run_dir": str(run_dir),
            }
        )

        def train_task(*, stop_event=None):
            chosen_model = model_path or yolo_model_file
            if model_path:
                model_candidate = Path(model_path).expanduser()
                if not model_candidate.exists():
                    raise RuntimeError(
                        f"Selected model checkpoint does not exist:\n{model_candidate}"
                    )
                model_arg = str(model_candidate.resolve())
            else:
                weight_path = resolve_weight_path(yolo_model_file)
                model_arg = ensure_parent_dir(str(weight_path))

            run_dir.mkdir(parents=True, exist_ok=True)
            log_path = run_dir / "train.log"

            overrides: Dict[str, Any] = {}
            if train_overrides:
                for key, value in dict(train_overrides).items():
                    if value is None:
                        continue
                    key_str = str(key).strip()
                    if not key_str or key_str.lower() in _RESERVED_OVERRIDE_KEYS:
                        continue
                    overrides[key_str] = value

            cmd = build_yolo_train_command(
                model=str(model_arg),
                data=str(Path(data_config_path).expanduser().resolve()),
                epochs=int(epochs),
                imgsz=int(image_size),
                batch=int(batch_size),
                device=resolved_device,
                project=str(runs_root),
                name=str(run_rel),
                exist_ok=True,
                plots=bool(plots),
                workers=0,  # avoid multiprocessing when invoked from GUI threads
                resume=bool(resume),
                overrides=overrides,
            )

            try:
                (run_dir / "command.txt").write_text(
                    " ".join(cmd) + "\n", encoding="utf-8"
                )
            except Exception:
                pass

            from annolid.yolo.tensorboard_logging import YOLORunTensorBoardLogger

            tb_logger = YOLORunTensorBoardLogger(run_dir=run_dir)
            tb_logger.write_static_metadata(
                command=" ".join(cmd),
                hparams={
                    "model": str(model_arg),
                    "data": str(Path(data_config_path).expanduser().resolve()),
                    "epochs": int(epochs),
                    "imgsz": int(image_size),
                    "batch": int(batch_size),
                    "device": str(resolved_device or ""),
                    "plots": bool(plots),
                    "resume": bool(resume),
                    "run_dir": str(run_dir),
                    "started_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                },
            )

            poll_stop = threading.Event()

            def poll_loop() -> None:
                while not poll_stop.is_set():
                    try:
                        tb_logger.poll_and_log_metrics()
                        tb_logger.poll_and_log_images()
                    except Exception:
                        pass
                    poll_stop.wait(2.0)

            poll_thread = threading.Thread(
                target=poll_loop, name="annolid-yolo-tb-poll", daemon=True
            )
            poll_thread.start()

            log_fh = None
            try:
                log_path.parent.mkdir(parents=True, exist_ok=True)
                log_fh = log_path.open("a", encoding="utf-8")
            except Exception:
                log_fh = None

            def sink(line: str) -> None:
                line = line.rstrip("\n")
                if line:
                    logger.info("[yolo] %s", line)
                tb_logger.record_output_line(line)
                if log_fh is not None:
                    try:
                        log_fh.write(line + "\n")
                        log_fh.flush()
                    except Exception:
                        pass

            try:
                completed = run_yolo_cli(
                    cmd,
                    stop_event=stop_event,
                    output_sink=sink,
                )
                if stop_event is not None and stop_event.is_set():
                    tb_logger.poll_and_log_metrics()
                    tb_logger.finalize(ok=False, error="Training cancelled by user.")
                    raise RuntimeError("YOLO training cancelled by user.")
                if completed.returncode != 0:
                    raise RuntimeError(
                        format_yolo_cli_error(
                            completed,
                            model=str(chosen_model),
                            device=resolved_device,
                        )
                    )
                tb_logger.poll_and_log_metrics()
                tb_logger.finalize(ok=True)
                ckpt = _locate_run_checkpoint(run_dir)
                return _CLITrainingOutcome(
                    save_dir=str(run_dir),
                    best_weights=(str(ckpt) if ckpt is not None else None),
                )
            except Exception as exc:
                tb_logger.finalize(ok=False, error=str(exc))
                raise
            finally:
                poll_stop.set()
                try:
                    poll_thread.join(timeout=2.0)
                except Exception:
                    pass
                try:
                    if log_fh is not None:
                        log_fh.close()
                except Exception:
                    pass
                tb_logger.close()

        worker = FlexibleWorker(train_task)
        thread = configure_worker_thread_stack(QtCore.QThread(self))
        worker.moveToThread(thread)
        self._active_jobs.append((thread, worker, data_config_path))

        def on_finished(outcome):
            self._handle_finished(outcome, thread, worker, data_config_path)
            save_dir = getattr(outcome, "save_dir", None)
            best_weights = getattr(outcome, "best_weights", None)
            self.training_finished.emit(
                {
                    "task": "yolo",
                    "model": model_label,
                    "run_dir": str(save_dir or run_dir),
                    "best_weights": best_weights,
                    "ok": not isinstance(outcome, Exception),
                }
            )

        worker.finished_signal.connect(on_finished, QtCore.Qt.QueuedConnection)
        thread.started.connect(worker.run, QtCore.Qt.QueuedConnection)
        thread.finished.connect(thread.deleteLater, QtCore.Qt.QueuedConnection)
        QtCore.QTimer.singleShot(0, thread.start)
        return True

    def stop_training(self) -> bool:
        """Request graceful cancellation of the active YOLO training job."""
        if not self._active_jobs:
            return False
        stopped_any = False
        for _thread, worker, _temp_cfg in list(self._active_jobs):
            try:
                worker.request_stop()
                stopped_any = True
            except Exception:
                logger.debug("Could not signal YOLO worker stop.", exc_info=True)
        if stopped_any and self._window is not None:
            try:
                self._window.statusBar().showMessage(
                    self._window.tr("Stopping YOLO training...")
                )
            except Exception:
                pass
        return stopped_any

    def _infer_expected_task(
        self,
        yolo_model_file: str,
        *,
        model_path: Optional[str],
    ) -> Optional[str]:
        inferred = infer_yolo_task(model_path, fallback_weight=yolo_model_file)
        if inferred in {"pose", "seg", "obb", "cls"}:
            return inferred
        return None

    def _infer_flip_idx(self, data_cfg: Dict[str, Any]) -> Optional[List[int]]:
        return infer_flip_idx(data_cfg)

    def _confirm_preflight(
        self,
        *,
        data_config_path: str,
        batch_size: int,
        device: Optional[str],
        plots: bool,
        expected_task: Optional[str] = None,
    ) -> bool:
        warnings = check_training_preflight(
            data_config_path,
            batch_size=batch_size,
            device=device,
            plots=plots,
            expected_task=expected_task,
        )
        if not warnings:
            return True

        message = (
            "Potential training issues detected:\n\n- "
            + "\n- ".join(warnings)
            + "\n\nDo you want to start training anyway?"
        )
        reply = QtWidgets.QMessageBox.warning(
            self._window,
            "YOLO Training Preflight",
            message,
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.No,
        )
        return reply == QtWidgets.QMessageBox.Yes

    def _count_images(self, entry: Union[str, List[str], Tuple[str, ...], None]) -> int:
        return count_dataset_images(entry)

    def is_running(self) -> bool:
        return self._training_running

    def cleanup(self) -> None:
        """Stop running jobs and remove temporary config files."""
        while self._active_jobs:
            thread, worker, temp_config = self._active_jobs.pop()
            try:
                worker.request_stop()
            except Exception:
                pass
            try:
                self._stop_thread(thread, "YOLO training thread")
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
        self._cleanup_temp_dataset_dirs()

    def _stop_thread(self, thread: QtCore.QThread, label: str) -> None:
        """Attempt graceful thread shutdown; terminate as a last resort."""
        if thread is None:
            return
        try:
            if thread.isRunning():
                try:
                    thread.requestInterruption()
                except Exception:
                    pass
                thread.quit()
                if not thread.wait(2000):
                    logger.warning("%s did not stop in time; terminating.", label)
                    thread.terminate()
                    thread.wait(2000)
        except RuntimeError:
            logger.debug("%s already stopped.", label)

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _refresh_window_models(self) -> None:
        ai_mgr = getattr(self._window, "ai_model_manager", None)
        if ai_mgr is not None and hasattr(ai_mgr, "refresh"):
            try:
                ai_mgr.refresh()
            except Exception:
                logger.debug("Could not refresh AI model dropdown.", exc_info=True)

    def _activate_trained_model(self, weights_path: Path) -> None:
        ai_mgr = getattr(self._window, "ai_model_manager", None)
        if ai_mgr is None:
            return
        if hasattr(ai_mgr, "select_or_add_custom_model"):
            try:
                ai_mgr.select_or_add_custom_model(weights_path)
                return
            except Exception:
                logger.debug("Could not activate trained model.", exc_info=True)
        self._refresh_window_models()

    def _launch_inference_for_weights(self, weights_path: Path) -> None:
        if hasattr(self._window, "open_inference_wizard"):
            try:
                self._window.open_inference_wizard(model_path=str(weights_path))
                return
            except TypeError:
                try:
                    self._window.open_inference_wizard()
                    return
                except Exception:
                    pass
            except Exception:
                logger.debug("Could not launch InferenceWizard.", exc_info=True)

    def _handle_finished(
        self,
        outcome: Any,
        thread: QtCore.QThread,
        worker: FlexibleWorker,
        temp_config: str,
    ) -> None:
        self._training_running = False
        self._active_jobs = [job for job in self._active_jobs if job[0] is not thread]

        try:
            self._stop_thread(thread, "YOLO training thread")
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
                self._window.tr("YOLO training failed. Check logs for details.")
            )
            return

        save_dir = getattr(outcome, "save_dir", None)
        best_weights_str = getattr(outcome, "best_weights", None)
        best_weights = (
            Path(best_weights_str)
            if best_weights_str and Path(best_weights_str).is_file()
            else (_locate_run_checkpoint(Path(save_dir)) if save_dir else None)
        )

        # Immediately expose the newly trained checkpoint in the GUI model selector.
        self._refresh_window_models()

        self._window.statusBar().showMessage(
            self._window.tr("YOLO training completed.")
        )

        if best_weights is None:
            details = f"\nResults are saved to:\n{save_dir}" if save_dir else ""
            QtWidgets.QMessageBox.information(
                self._window,
                "Training Completed",
                "YOLO model training completed successfully!" + details,
            )
            return

        box = QtWidgets.QMessageBox(self._window)
        box.setIcon(QtWidgets.QMessageBox.Information)
        box.setWindowTitle("Training Completed")
        box.setText("YOLO model training completed successfully!")
        info_lines = []
        if best_weights is not None:
            info_lines.append(f"Best weights:\n{best_weights}")
        if save_dir:
            info_lines.append(f"Run folder:\n{save_dir}")
        box.setInformativeText("\n\n".join(info_lines))

        use_btn = box.addButton("Use Model Now", QtWidgets.QMessageBox.AcceptRole)
        infer_btn = box.addButton("Run Inference...", QtWidgets.QMessageBox.ActionRole)
        folder_btn = (
            box.addButton("Open Run Folder", QtWidgets.QMessageBox.ActionRole)
            if save_dir
            else None
        )
        box.addButton(QtWidgets.QMessageBox.Ok)
        box.exec_()

        clicked = box.clickedButton()
        if clicked == use_btn:
            self._activate_trained_model(best_weights)
        elif clicked == infer_btn:
            self._activate_trained_model(best_weights)
            self._launch_inference_for_weights(best_weights)
        elif folder_btn is not None and clicked == folder_btn and save_dir:
            QtGui.QDesktopServices.openUrl(
                QtCore.QUrl.fromLocalFile(str(Path(save_dir).resolve()))
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
        dialog = self._start_dialog
        if dialog is None:
            return
        self._start_dialog = None
        try:
            if dialog.isVisible():
                dialog.close()
            dialog.deleteLater()
        except RuntimeError:
            pass
        except Exception:
            logger.debug("Could not close start notification dialog.")

    def _clear_start_dialog(self) -> None:
        self._start_dialog = None

    def _write_temp_config(self, data_cfg: Dict[str, Any], config_path: Path) -> str:
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
        should_delete = False
        try:
            self._temp_configs.remove(path_obj)
            should_delete = True
        except ValueError:
            should_delete = False
        if not should_delete:
            return
        try:
            path_obj.unlink()
        except FileNotFoundError:
            return
        except Exception:
            logger.debug("Could not remove temporary YOLO config: %s", path_obj)

    def _cleanup_temp_configs(self) -> None:
        while self._temp_configs:
            temp_path = self._temp_configs.pop()
            try:
                temp_path.unlink()
            except FileNotFoundError:
                continue
            except Exception:
                logger.debug("Could not remove temporary YOLO config: %s", temp_path)

    def _cleanup_temp_dataset_dirs(self) -> None:
        while self._temp_dataset_dirs:
            dataset_dir = self._temp_dataset_dirs.pop()
            try:
                for child in sorted(dataset_dir.rglob("*"), reverse=True):
                    if child.is_file() or child.is_symlink():
                        child.unlink(missing_ok=True)
                    elif child.is_dir():
                        child.rmdir()
                dataset_dir.rmdir()
            except FileNotFoundError:
                continue
            except Exception:
                logger.debug(
                    "Could not remove temporary YOLO dataset dir: %s", dataset_dir
                )
