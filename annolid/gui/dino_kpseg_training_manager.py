from __future__ import annotations

import atexit
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml
from qtpy import QtCore, QtWidgets

from annolid.gui.workers import FlexibleWorker
from annolid.utils.logger import logger
from annolid.utils.runs import new_run_dir, shared_runs_root


class DinoKPSEGTrainingManager(QtCore.QObject):
    """Encapsulates DinoKPSEG training orchestration for the Annolid GUI."""

    training_started = QtCore.Signal(object)
    training_finished = QtCore.Signal(object)

    def __init__(self, window: QtWidgets.QMainWindow) -> None:
        super().__init__(parent=window)
        self._window = window
        self._temp_configs: List[Path] = []
        self._active_jobs: List[Tuple[QtCore.QThread, FlexibleWorker]] = []
        self._training_running = False
        self._start_dialog: Optional[QtWidgets.QMessageBox] = None
        atexit.register(self.cleanup)

    def is_running(self) -> bool:
        return self._training_running

    def cleanup(self) -> None:
        while self._active_jobs:
            thread, worker = self._active_jobs.pop()
            try:
                self._stop_thread(thread, "DinoKPSEG training thread")
            except RuntimeError:
                logger.debug("DinoKPSEG training thread already stopped.")
            try:
                worker.deleteLater()
            except RuntimeError:
                pass
        self._cleanup_temp_configs()

    def prepare_data_config(self, config_file: str) -> Optional[str]:
        """Resolve relative paths in a YOLO pose data.yaml and persist to a temp file."""
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
            logger.exception("Failed to load dataset config: %s", config_path)
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

        if not data_cfg.get("kpt_shape"):
            QtWidgets.QMessageBox.warning(
                self._window,
                "Pose Dataset Required",
                "DinoKPSEG training requires a YOLO pose dataset with 'kpt_shape' in data.yaml.",
            )
            return None

        return self._write_temp_config(data_cfg, config_path)

    def start_training(
        self,
        *,
        data_config_path: str,
        out_dir: Optional[str],
        model_name: str,
        short_side: int,
        layers: str,
        radius_px: float,
        hidden_dim: int,
        lr: float,
        epochs: int,
        threshold: float,
        device: Optional[str],
        cache_features: bool,
        early_stop_patience: int = 0,
        early_stop_min_delta: float = 0.0,
        early_stop_min_epochs: int = 0,
        augment: bool = False,
        hflip: float = 0.5,
        degrees: float = 0.0,
        translate: float = 0.0,
        scale: float = 0.0,
        brightness: float = 0.0,
        contrast: float = 0.0,
        saturation: float = 0.0,
        seed: Optional[int] = None,
        tb_add_graph: bool = False,
    ) -> bool:
        if self._training_running:
            QtWidgets.QMessageBox.warning(
                self._window,
                "Training In Progress",
                "A DinoKPSEG training job is already running in the background.",
            )
            return False

        if not self._preflight_ok(data_config_path=data_config_path):
            return False

        try:
            import transformers  # noqa: F401
        except Exception as exc:
            QtWidgets.QMessageBox.critical(
                self._window,
                "Missing Dependency",
                "DinoKPSEG training requires the optional dependency 'transformers'.\n\n"
                f"Import error: {exc}\n\n"
                "Install it with:\n  pip install 'transformers>=4.39'",
            )
            return False

        output_dir = self._resolve_output_dir(out_dir)
        self.training_started.emit(
            {
                "task": "dino_kpseg",
                "model": str(model_name),
                "run_dir": str(output_dir),
            }
        )

        def _tail_text(path: Path, *, max_lines: int = 80) -> str:
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except Exception:
                return ""
            lines = text.splitlines()
            tail = lines[-max(1, int(max_lines)):]
            return "\n".join(tail)

        def train_task():
            layer_tuple = self._parse_layers(layers)
            data_path = Path(data_config_path).expanduser().resolve()
            log_path = output_dir / "train.log"
            try:
                output_dir.mkdir(parents=True, exist_ok=True)
            except Exception:
                pass
            cmd: list[str] = [
                sys.executable,
                "-m",
                "annolid.segmentation.dino_kpseg.train",
                "--data",
                str(data_path),
                "--output",
                str(output_dir),
                "--model-name",
                str(model_name),
                "--short-side",
                str(int(short_side)),
                "--layers",
                ",".join(str(int(x)) for x in layer_tuple),
                "--radius-px",
                str(float(radius_px)),
                "--hidden-dim",
                str(int(hidden_dim)),
                "--lr",
                str(float(lr)),
                "--epochs",
                str(int(epochs)),
                "--threshold",
                str(float(threshold)),
                "--early-stop-patience",
                str(int(early_stop_patience)),
                "--early-stop-min-delta",
                str(float(early_stop_min_delta)),
                "--early-stop-min-epochs",
                str(int(early_stop_min_epochs)),
            ]
            if bool(tb_add_graph):
                cmd.append("--tb-add-graph")
            if device:
                cmd += ["--device", str(device).strip()]
            if not bool(cache_features):
                cmd.append("--no-cache")
            if bool(augment):
                cmd.append("--augment")
                cmd += ["--hflip", str(float(hflip))]
                cmd += ["--degrees", str(float(degrees))]
                cmd += ["--translate", str(float(translate))]
                cmd += ["--scale", str(float(scale))]
                cmd += ["--brightness", str(float(brightness))]
                cmd += ["--contrast", str(float(contrast))]
                cmd += ["--saturation", str(float(saturation))]
                if seed is not None:
                    cmd += ["--seed", str(int(seed))]

            env = dict(os.environ)
            env.setdefault("PYTHONUNBUFFERED", "1")
            env.setdefault("TOKENIZERS_PARALLELISM", "false")
            env.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

            with log_path.open("w", encoding="utf-8") as fh:
                fh.write("Command:\n")
                fh.write(" ".join(cmd) + "\n\n")
                fh.flush()
                completed = subprocess.run(
                    cmd,
                    env=env,
                    stdout=fh,
                    stderr=subprocess.STDOUT,
                    text=True,
                )

            if completed.returncode != 0:
                if completed.returncode < 0:
                    sig = -int(completed.returncode)
                    raise RuntimeError(
                        f"DinoKPSEG training crashed (signal {sig}).\n\n"
                        f"Log: {log_path}\n\n{_tail_text(log_path)}"
                    )
                raise RuntimeError(
                    f"DinoKPSEG training failed (exit code {completed.returncode}).\n\n"
                    f"Log: {log_path}\n\n{_tail_text(log_path)}"
                )

            best_path = output_dir / "weights" / "best.pt"
            if not best_path.exists():
                raise RuntimeError(
                    f"DinoKPSEG training completed but best checkpoint was not found:\n{best_path}\n\n"
                    f"Log: {log_path}\n\n{_tail_text(log_path)}"
                )

            return {
                "best": str(best_path),
                "output_dir": str(output_dir),
                "log_path": str(log_path),
            }

        self._training_running = True
        self._window.statusBar().showMessage(
            self._window.tr("DinoKPSEG training started in the background...")
        )
        self._show_start_notification()

        worker = FlexibleWorker(train_task)
        thread = QtCore.QThread(self)
        worker.moveToThread(thread)
        self._active_jobs.append((thread, worker))

        def on_finished(outcome: Any) -> None:
            self._handle_finished(outcome, thread, worker, data_config_path)
            resolved_out_dir = None
            if isinstance(outcome, dict):
                resolved_out_dir = outcome.get("output_dir")
            self.training_finished.emit(
                {
                    "task": "dino_kpseg",
                    "model": str(model_name),
                    "run_dir": str(resolved_out_dir or output_dir),
                    "ok": not isinstance(outcome, Exception),
                }
            )

        worker.finished_signal.connect(on_finished, QtCore.Qt.QueuedConnection)
        thread.started.connect(worker.run, QtCore.Qt.QueuedConnection)
        thread.finished.connect(thread.deleteLater, QtCore.Qt.QueuedConnection)
        QtCore.QTimer.singleShot(0, thread.start)
        return True

    def _resolve_output_dir(self, out_dir: Optional[str]) -> Path:
        base = Path(out_dir).expanduser().resolve(
        ) if out_dir else shared_runs_root()
        run_dir = new_run_dir(task="dino_kpseg", model="train", runs_root=base)
        try:
            run_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            # Defer failures to the training task (which will surface a clearer message).
            pass
        return run_dir

    @staticmethod
    def _parse_layers(value: str) -> Tuple[int, ...]:
        raw = str(value or "").strip()
        if not raw:
            return (-1,)
        items = []
        for token in raw.split(","):
            token = token.strip()
            if not token:
                continue
            items.append(int(token))
        return tuple(items) if items else (-1,)

    def _preflight_ok(self, *, data_config_path: str) -> bool:
        cfg_path = Path(data_config_path).expanduser()
        try:
            data_cfg = yaml.safe_load(
                cfg_path.read_text(encoding="utf-8")) or {}
        except Exception:
            data_cfg = {}

        if not isinstance(data_cfg, dict) or not data_cfg:
            QtWidgets.QMessageBox.warning(
                self._window,
                "Invalid Dataset",
                f"Could not read dataset YAML:\n{cfg_path}",
            )
            return False

        if not data_cfg.get("kpt_shape"):
            QtWidgets.QMessageBox.warning(
                self._window,
                "Pose Dataset Required",
                "DinoKPSEG training requires a YOLO pose dataset with 'kpt_shape' in data.yaml.",
            )
            return False

        return True

    def _stop_thread(self, thread: QtCore.QThread, label: str) -> None:
        try:
            if thread.isRunning():
                try:
                    thread.requestInterruption()
                except Exception:
                    pass
                thread.quit()
                if not thread.wait(2000):
                    logger.warning(
                        "%s did not stop in time; terminating.", label)
                    thread.terminate()
                    thread.wait(2000)
        except RuntimeError:
            logger.debug("%s already stopped.", label)

    def _handle_finished(
        self,
        outcome: Any,
        thread: QtCore.QThread,
        worker: FlexibleWorker,
        temp_config: str,
    ) -> None:
        self._training_running = False
        self._active_jobs = [
            job for job in self._active_jobs if job[0] is not thread]

        try:
            self._stop_thread(thread, "DinoKPSEG training thread")
        except RuntimeError:
            pass
        try:
            thread.deleteLater()
        except RuntimeError:
            pass
        try:
            worker.deleteLater()
        except RuntimeError:
            pass

        self._close_start_notification()
        self._release_temp_config(temp_config)

        if isinstance(outcome, Exception):
            logger.error("DinoKPSEG training failed: %s",
                         outcome, exc_info=True)
            QtWidgets.QMessageBox.critical(
                self._window,
                "Training Error",
                f"An error occurred during DinoKPSEG training:\n{outcome}",
            )
            self._window.statusBar().showMessage(
                self._window.tr(
                    "DinoKPSEG training failed. Check logs for details.")
            )
            return

        details = ""
        if isinstance(outcome, dict):
            best = outcome.get("best")
            out_dir = outcome.get("output_dir")
            log_path = outcome.get("log_path")
            if best:
                try:
                    best_path = Path(str(best)).expanduser().resolve()
                    if best_path.exists():
                        self._window.settings.setValue(
                            "ai/dino_kpseg_last_best", str(best_path)
                        )
                        self._window.settings.sync()
                except Exception:
                    pass
            if out_dir:
                details += f"\nResults are saved to:\n{out_dir}"
            if best:
                details += f"\n\nBest checkpoint:\n{best}"
            if log_path:
                details += f"\n\nTraining log:\n{log_path}"

        QtWidgets.QMessageBox.information(
            self._window,
            "Training Completed",
            "DinoKPSEG training completed successfully!" + details,
        )
        self._window.statusBar().showMessage(
            self._window.tr("DinoKPSEG training completed.")
        )

    def _show_start_notification(self) -> None:
        self._close_start_notification()
        dialog = QtWidgets.QMessageBox(self._window)
        dialog.setIcon(QtWidgets.QMessageBox.Information)
        dialog.setWindowTitle("Training Started")
        dialog.setText(
            "DinoKPSEG training has started in the background.\n"
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
        # Clear immediately to avoid races with the dialog's finished() handler.
        self._start_dialog = None
        try:
            if dialog.isVisible():
                dialog.close()
            dialog.deleteLater()
        except (RuntimeError, AttributeError):
            pass

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
            logger.debug("Could not remove temporary config: %s", path_obj)

    def _cleanup_temp_configs(self) -> None:
        while self._temp_configs:
            temp_path = self._temp_configs.pop()
            try:
                temp_path.unlink()
            except FileNotFoundError:
                continue
            except Exception:
                logger.debug(
                    "Could not remove temporary config: %s", temp_path)
