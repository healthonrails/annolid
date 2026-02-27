import atexit
import os
import platform
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import yaml
from qtpy import QtCore, QtWidgets

from annolid.gui.workers import FlexibleWorker
from annolid.segmentation.dino_kpseg.data import (
    load_coco_pose_spec,
    materialize_coco_pose_as_yolo,
)
from annolid.utils.logger import logger
from annolid.utils.runs import allocate_run_dir, shared_runs_root


@dataclass(frozen=True)
class _CLITrainingOutcome:
    save_dir: str


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
    def prepare_data_config(self, config_file: str) -> Optional[str]:
        """Resolve training data config and persist a YOLO-compatible temp YAML."""
        config_path = Path(config_file).expanduser().resolve()
        if not config_path.exists():
            QtWidgets.QMessageBox.warning(
                self._window,
                "Invalid Dataset",
                f"Dataset configuration file not found:\n{config_path}",
            )
            return None

        if config_path.is_dir():
            prepared = self._prepare_from_coco_annotations_dir(config_path)
            if prepared is None:
                return None
            config_path = prepared

        try:
            with config_path.open("r", encoding="utf-8") as stream:
                data_cfg: Dict[str, Any] = yaml.safe_load(stream) or {}
        except Exception as exc:
            logger.exception("Failed to load YOLO dataset config: %s", config_path)
            QtWidgets.QMessageBox.critical(
                self._window,
                "Dataset Error",
                f"Could not read dataset configuration:\n{config_path}\n\n{exc}",
            )
            return None

        if self._looks_like_coco_spec(data_cfg):
            prepared = self._materialize_coco_spec(config_path)
            if prepared is None:
                return None
            config_path = prepared
            try:
                with config_path.open("r", encoding="utf-8") as stream:
                    data_cfg = yaml.safe_load(stream) or {}
            except Exception as exc:
                logger.exception(
                    "Failed to load staged YOLO dataset config: %s", config_path
                )
                QtWidgets.QMessageBox.critical(
                    self._window,
                    "Dataset Error",
                    f"Could not read staged YOLO dataset config:\n{config_path}\n\n{exc}",
                )
                return None

        base_dir = config_path.parent

        def is_remote(path_str: str) -> bool:
            return path_str.startswith(("http://", "https://", "rtsp://", "rtmp://"))

        root_dir = base_dir
        if data_cfg.get("path"):
            root_candidate = Path(str(data_cfg["path"])).expanduser()
            if not root_candidate.is_absolute():
                root_candidate = (base_dir / root_candidate).resolve()
            root_dir = root_candidate
            data_cfg["path"] = str(root_dir)
        else:
            data_cfg["path"] = str(base_dir)

        def resolve_entry(entry: Any) -> Any:
            if isinstance(entry, str):
                if not entry.strip():
                    return entry
                if is_remote(entry):
                    return entry
                entry_path = Path(entry).expanduser()
                resolved = (
                    entry_path if entry_path.is_absolute() else root_dir / entry_path
                )
                return str(resolved.resolve())
            if isinstance(entry, (list, tuple)):
                return [resolve_entry(item) for item in entry]
            return entry

        for split in ("train", "val", "test"):
            if split in data_cfg and data_cfg[split]:
                data_cfg[split] = resolve_entry(data_cfg[split])

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

        # Pose datasets: auto-generate flip_idx if missing, so fliplr/flipud augmentations work.
        if data_cfg.get("kpt_shape") and not data_cfg.get("flip_idx"):
            flip_idx = self._infer_flip_idx(data_cfg)
            if flip_idx:
                data_cfg["flip_idx"] = flip_idx

        return self._write_temp_config(data_cfg, config_path)

    def _infer_flip_idx(self, data_cfg: Dict[str, Any]) -> Optional[List[int]]:
        kpt_shape = data_cfg.get("kpt_shape")
        if not isinstance(kpt_shape, (list, tuple)) or len(kpt_shape) < 1:
            return None
        try:
            nkpt = int(kpt_shape[0])
        except Exception:
            return None
        if nkpt <= 0:
            return None

        kpt_labels = data_cfg.get("kpt_labels")
        if not isinstance(kpt_labels, dict) or not kpt_labels:
            return None

        labels_by_idx: Dict[int, str] = {}
        for key, value in kpt_labels.items():
            try:
                idx = int(key)
            except Exception:
                continue
            if idx < 0 or idx >= nkpt:
                continue
            label = str(value or "").strip()
            if not label:
                continue
            labels_by_idx[idx] = label

        if len(labels_by_idx) != nkpt:
            return None

        idx_by_label = {lbl.lower(): i for i, lbl in labels_by_idx.items()}

        def counterpart(label: str) -> Optional[str]:
            s = label.lower()
            swaps = [
                ("left", "right"),
                ("right", "left"),
                ("l_", "r_"),
                ("r_", "l_"),
                ("l-", "r-"),
                ("r-", "l-"),
            ]
            for a, b in swaps:
                if a in s:
                    return s.replace(a, b)
            return None

        flip_idx: List[int] = []
        for i in range(nkpt):
            label = labels_by_idx[i]
            other = counterpart(label)
            if other and other in idx_by_label:
                flip_idx.append(idx_by_label[other])
            else:
                flip_idx.append(i)

        if len(flip_idx) != nkpt:
            return None
        return flip_idx

    def _looks_like_coco_spec(self, data_cfg: Dict[str, Any]) -> bool:
        fmt = str(data_cfg.get("format") or data_cfg.get("type") or "").strip().lower()
        if fmt in {"coco", "coco_pose", "coco_keypoints"}:
            return True

        # Heuristic: any split path pointing to a JSON annotation file.
        for split in ("train", "val", "test"):
            value = data_cfg.get(split)
            entries = value if isinstance(value, (list, tuple)) else [value]
            for entry in entries:
                if isinstance(entry, str) and entry.strip().lower().endswith(".json"):
                    return True
        return False

    def _prepare_from_coco_annotations_dir(
        self, annotations_dir: Path
    ) -> Optional[Path]:
        if not annotations_dir.is_dir():
            return None

        train_json = annotations_dir / "train.json"
        val_json = annotations_dir / "val.json"
        test_json = annotations_dir / "test.json"
        if not train_json.exists() and not val_json.exists():
            QtWidgets.QMessageBox.warning(
                self._window,
                "Invalid COCO Folder",
                "Expected train.json or val.json in the selected COCO folder.",
            )
            return None

        root_path = annotations_dir
        image_root = "."
        if (annotations_dir / "images").exists():
            image_root = "images"
        elif (annotations_dir.parent / "images").exists():
            root_path = annotations_dir.parent
            image_root = "images"

        payload: Dict[str, Any] = {
            "format": "coco",
            "path": str(root_path),
            "image_root": image_root,
        }
        if train_json.exists():
            payload["train"] = str(train_json.resolve().relative_to(root_path))
        if val_json.exists():
            payload["val"] = str(val_json.resolve().relative_to(root_path))
        if test_json.exists():
            payload["test"] = str(test_json.resolve().relative_to(root_path))

        cfg_stub = annotations_dir / "coco_pose_spec.yaml"
        temp_spec = Path(self._write_temp_config(payload, cfg_stub))
        return temp_spec

    def _materialize_coco_spec(self, config_path: Path) -> Optional[Path]:
        try:
            spec = load_coco_pose_spec(config_path)
        except Exception as exc:
            logger.exception("Failed to load COCO pose spec: %s", config_path)
            QtWidgets.QMessageBox.critical(
                self._window,
                "Dataset Error",
                f"Could not parse COCO pose spec:\n{config_path}\n\n{exc}",
            )
            return None

        try:
            stage_dir = Path(
                tempfile.mkdtemp(
                    prefix=f"{config_path.stem}_coco_yolo_",
                )
            ).resolve()
            self._temp_dataset_dirs.append(stage_dir)
            staged_yaml = materialize_coco_pose_as_yolo(
                spec=spec,
                output_dir=stage_dir,
                link_mode="hardlink",
            )
            return staged_yaml
        except Exception as exc:
            logger.exception("Failed to convert COCO spec to YOLO pose dataset.")
            QtWidgets.QMessageBox.critical(
                self._window,
                "Dataset Error",
                f"Failed to convert COCO annotations into a YOLO pose dataset.\n\n{exc}",
            )
            return None

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
    ) -> bool:
        """Launch YOLO training on a background thread."""
        # Ensure the dataset config is resolved to an absolute, temporary
        # YAML with paths fixed up for Ultralytics. If preparation fails,
        # abort early.
        prepared_cfg = self.prepare_data_config(data_config_path)
        if not prepared_cfg:
            return False
        # From this point forward use the prepared temp config path so
        # downstream cleanup/removal is safe.
        data_config_path = prepared_cfg

        if self._training_running:
            QtWidgets.QMessageBox.warning(
                self._window,
                "Training In Progress",
                "A YOLO training job is already running in the background.",
            )
            self._release_temp_config(data_config_path)
            return False

        from annolid.yolo import resolve_weight_path
        from annolid.yolo.ultralytics_cli import (
            build_yolo_train_command,
            ensure_parent_dir,
            run_yolo_cli,
        )

        if not self._confirm_preflight(
            data_config_path=data_config_path,
            batch_size=batch_size,
            device=device,
            plots=plots,
        ):
            self._release_temp_config(data_config_path)
            return False

        self._training_running = True
        self._window.statusBar().showMessage(
            self._window.tr("YOLO training started in the background...")
        )
        self._show_start_notification()

        runs_root = (
            Path(out_dir).expanduser().resolve() if out_dir else shared_runs_root()
        )
        run_dir = allocate_run_dir(
            task="yolo",
            model=Path(yolo_model_file).stem,
            runs_root=runs_root,
        )
        run_rel = run_dir.relative_to(runs_root)
        self.training_started.emit(
            {
                "task": "yolo",
                "model": Path(yolo_model_file).stem,
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
                    overrides[str(key)] = value

            cmd = build_yolo_train_command(
                model=str(model_arg),
                data=str(Path(data_config_path).expanduser().resolve()),
                epochs=int(epochs),
                imgsz=int(image_size),
                batch=int(batch_size),
                device=(str(device).strip() if device else None),
                project=str(runs_root),
                name=str(run_rel),
                exist_ok=True,
                plots=bool(plots),
                workers=0,  # avoid multiprocessing when invoked from GUI threads
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
                    "device": str(device or ""),
                    "plots": bool(plots),
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
                if completed.returncode != 0:
                    tail = "\n".join(completed.output_tail[-50:])
                    raise RuntimeError(
                        f"YOLO CLI failed (exit {completed.returncode}) while training {chosen_model!r}.\n\n"
                        f"Command: {' '.join(completed.command)}\n\n"
                        f"Last output:\n{tail}"
                    )
                tb_logger.poll_and_log_metrics()
                tb_logger.finalize(ok=True)
                return _CLITrainingOutcome(save_dir=str(run_dir))
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
        thread = QtCore.QThread(self)
        worker.moveToThread(thread)
        self._active_jobs.append((thread, worker, data_config_path))

        def on_finished(outcome):
            self._handle_finished(outcome, thread, worker, data_config_path)
            save_dir = getattr(outcome, "save_dir", None)
            self.training_finished.emit(
                {
                    "task": "yolo",
                    "model": Path(yolo_model_file).stem,
                    "run_dir": str(save_dir or run_dir),
                    "ok": not isinstance(outcome, Exception),
                }
            )

        worker.finished_signal.connect(on_finished, QtCore.Qt.QueuedConnection)
        thread.started.connect(worker.run, QtCore.Qt.QueuedConnection)
        thread.finished.connect(thread.deleteLater, QtCore.Qt.QueuedConnection)
        QtCore.QTimer.singleShot(0, thread.start)
        return True

    def _confirm_preflight(
        self,
        *,
        data_config_path: str,
        batch_size: int,
        device: Optional[str],
        plots: bool,
    ) -> bool:
        warnings: List[str] = []

        data_cfg: Dict[str, Any] = {}
        resolved_yaml = Path(data_config_path).expanduser()
        try:
            data_cfg = yaml.safe_load(resolved_yaml.read_text(encoding="utf-8")) or {}
        except Exception:
            data_cfg = {}

        train_count = self._count_images(data_cfg.get("train"))
        val_count = self._count_images(data_cfg.get("val"))

        if train_count and batch_size > train_count:
            warnings.append(
                f"Batch size ({batch_size}) is larger than the training set ({train_count} images). "
                "This can result in very few optimizer steps and poor convergence."
            )

        if train_count and train_count < 50:
            warnings.append(
                f"Training set is very small ({train_count} images). Consider labeling more frames for better accuracy."
            )

        if val_count and val_count < 10:
            warnings.append(
                f"Validation set is very small ({val_count} images). Metrics may be unstable/noisy."
            )

        kpt_shape = data_cfg.get("kpt_shape")
        if kpt_shape and not data_cfg.get("flip_idx"):
            warnings.append(
                "Pose dataset has 'kpt_shape' but no 'flip_idx' in data.yaml; Ultralytics disables flip augmentations."
            )

        device_str = str(device or "").strip().lower()
        if not device_str:
            warnings.append(
                "Device is set to Auto. If training is slow, explicitly select 'mps' (Apple Silicon) or CUDA GPU."
            )
        elif device_str == "cpu":
            warnings.append(
                "Training on CPU is usually much slower. If available, prefer 'mps' (Apple Silicon) or CUDA GPU."
            )

        if plots and platform.system().lower() == "darwin":
            warnings.append(
                "Ultralytics training plots may crash on macOS in GUI/threaded runs (Bus error). If this happens, disable plots."
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
        if not entry:
            return 0

        if isinstance(entry, (list, tuple)):
            return sum(self._count_images(item) for item in entry)

        path_str = str(entry).strip()
        if not path_str or path_str.startswith(("http://", "https://")):
            return 0
        if any(ch in path_str for ch in ("*", "?", "[")):
            return 0

        p = Path(path_str).expanduser()
        try:
            if p.is_dir():
                exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
                return sum(1 for f in p.rglob("*") if f.suffix.lower() in exts)
            if p.is_file() and p.suffix.lower() == ".txt":
                count = 0
                for line in p.read_text(encoding="utf-8").splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    candidate = Path(line)
                    if not candidate.is_absolute():
                        candidate = p.parent / candidate
                    if candidate.exists():
                        count += 1
                return count
        except OSError:
            return 0
        return 0

    def is_running(self) -> bool:
        return self._training_running

    def cleanup(self) -> None:
        """Stop running jobs and remove temporary config files."""
        while self._active_jobs:
            thread, worker, temp_config = self._active_jobs.pop()
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
        dialog = self._start_dialog
        if dialog is None:
            return
        # Clear the reference early to avoid races with the dialog's finished signal
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
        try:
            self._temp_configs.remove(path_obj)
        except ValueError:
            pass
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
