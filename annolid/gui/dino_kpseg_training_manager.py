from __future__ import annotations

import atexit
import os
import platform
import queue
import signal
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml
from qtpy import QtCore, QtWidgets

from annolid.gui.workers import FlexibleWorker
from annolid.utils.logger import logger
from annolid.utils.runs import new_run_dir, shared_runs_root


class _TrainingCancelled(RuntimeError):
    pass


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
        self._process_lock = threading.Lock()
        self._active_process: Optional[subprocess.Popen] = None
        atexit.register(self.cleanup)

    def is_running(self) -> bool:
        return self._training_running

    def cleanup(self) -> None:
        while self._active_jobs:
            thread, worker = self._active_jobs.pop()
            try:
                worker.request_stop()
            except Exception:
                pass
            try:
                self._stop_thread(thread, "DinoKPSEG training thread")
            except RuntimeError:
                logger.debug("DinoKPSEG training thread already stopped.")
            try:
                worker.deleteLater()
            except RuntimeError:
                pass
        self._terminate_active_process()
        self._close_start_notification()
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

        def has_glob(s: str) -> bool:
            return any(ch in s for ch in ("*", "?", "["))

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

        missing_paths: List[str] = []
        for split in ("train", "val", "test"):
            value = data_cfg.get(split)
            if not value:
                continue
            entries = value if isinstance(value, list) else [value]
            for entry in entries:
                if not isinstance(entry, str) or is_remote(entry):
                    continue
                if has_glob(entry):
                    continue
                candidate = Path(entry).expanduser()
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

    def start_training(
        self,
        *,
        data_config_path: str,
        out_dir: Optional[str],
        model_name: str,
        short_side: int,
        layers: str,
        radius_px: float,
        mask_type: str = "gaussian",
        heatmap_sigma: Optional[float] = None,
        instance_mode: str = "union",
        bbox_scale: float = 1.25,
        hidden_dim: int,
        lr: float,
        epochs: int,
        batch_size: int = 1,
        threshold: float,
        device: Optional[str],
        cache_features: bool,
        head_type: str = "conv",
        attn_heads: int = 4,
        attn_layers: int = 1,
        lr_pair_loss_weight: float = 0.0,
        lr_pair_margin_px: float = 0.0,
        lr_side_loss_weight: float = 0.0,
        lr_side_loss_margin: float = 0.0,
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
            self._release_temp_config(data_config_path)
            return False

        if not self._preflight_ok(
            data_config_path=data_config_path,
            augment=augment,
            device=device,
            epochs=epochs,
        ):
            self._release_temp_config(data_config_path)
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
            self._release_temp_config(data_config_path)
            return False

        if not str(model_name or "").strip():
            QtWidgets.QMessageBox.warning(
                self._window,
                "Missing Model Name",
                "Please select a DINO backbone model name before starting training.",
            )
            self._release_temp_config(data_config_path)
            return False

        output_dir = self._resolve_output_dir(
            out_dir=out_dir,
            model_name=model_name,
            data_config_path=data_config_path,
        )
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

        def train_task(*, stop_event=None):
            layer_tuple = self._parse_layers(layers)
            data_path = Path(data_config_path).expanduser().resolve()
            log_path = output_dir / "train.log"
            try:
                output_dir.mkdir(parents=True, exist_ok=True)
            except Exception:
                pass
            layers_arg = ",".join(str(int(x)) for x in layer_tuple)
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
                f"--layers={layers_arg}",
                "--radius-px",
                str(float(radius_px)),
                "--mask-type",
                str(mask_type or "gaussian"),
                "--instance-mode",
                str(instance_mode or "union"),
                "--bbox-scale",
                str(float(bbox_scale)),
                "--hidden-dim",
                str(int(hidden_dim)),
                "--lr",
                str(float(lr)),
                "--epochs",
                str(int(epochs)),
                "--batch",
                str(int(batch_size)),
                "--threshold",
                str(float(threshold)),
                "--head-type",
                str(head_type or "conv"),
                "--attn-heads",
                str(int(attn_heads)),
                "--attn-layers",
                str(int(attn_layers)),
                "--lr-pair-loss-weight",
                str(float(lr_pair_loss_weight)),
                "--lr-pair-margin-px",
                str(float(lr_pair_margin_px)),
                "--lr-side-loss-weight",
                str(float(lr_side_loss_weight)),
                "--lr-side-loss-margin",
                str(float(lr_side_loss_margin)),
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
            if heatmap_sigma is not None:
                cmd += ["--heatmap-sigma", str(float(heatmap_sigma))]
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

            try:
                (output_dir / "command.txt").write_text(
                    " ".join(cmd) + "\n", encoding="utf-8"
                )
            except Exception:
                pass

            with log_path.open("w", encoding="utf-8") as fh:
                fh.write("Command:\n")
                fh.write(" ".join(cmd) + "\n\n")
                fh.flush()

                creationflags = 0
                kwargs: Dict[str, Any] = {}
                if platform.system().lower() == "windows":
                    creationflags = getattr(
                        subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
                else:
                    kwargs["start_new_session"] = True

                proc = subprocess.Popen(
                    cmd,
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1,
                    creationflags=creationflags,
                    **kwargs,
                )
                self._set_active_process(proc)
                output_q: "queue.Queue[Optional[str]]" = queue.Queue()

                def reader() -> None:
                    try:
                        stream = proc.stdout
                        if stream is None:
                            output_q.put(None)
                            return
                        for line in stream:
                            output_q.put(line)
                    except Exception:
                        pass
                    finally:
                        output_q.put(None)

                reader_thread = threading.Thread(
                    target=reader, name="annolid-dino-kpseg-log", daemon=True
                )
                reader_thread.start()
                try:
                    while True:
                        if stop_event is not None and getattr(stop_event, "is_set", None) and stop_event.is_set():
                            self._terminate_process(proc)
                            raise _TrainingCancelled(
                                "DinoKPSEG training was cancelled.")
                        drained = 0
                        while drained < 200:
                            try:
                                line = output_q.get_nowait()
                            except queue.Empty:
                                break
                            if line is None:
                                drained = 10_000
                                break
                            drained += 1
                            if line:
                                fh.write(line)
                                fh.flush()
                                logger.info("[dino_kpseg] %s",
                                            line.rstrip("\n"))

                        if proc.poll() is not None:
                            # Drain remaining lines (up to a cap) after process exit.
                            for _ in range(10_000):
                                try:
                                    line = output_q.get_nowait()
                                except queue.Empty:
                                    break
                                if line is None:
                                    break
                                if line:
                                    fh.write(line)
                                    fh.flush()
                            break

                        if stop_event is not None:
                            stop_event.wait(0.2)
                        else:
                            time.sleep(0.2)
                finally:
                    try:
                        if proc.stdout is not None:
                            proc.stdout.close()
                    except Exception:
                        pass
                    try:
                        reader_thread.join(timeout=1.0)
                    except Exception:
                        pass
                    self._clear_active_process(proc)

            completed_rc = proc.returncode

            if completed_rc != 0:
                if completed_rc < 0:
                    sig = -int(completed_rc)
                    raise RuntimeError(
                        f"DinoKPSEG training crashed (signal {sig}).\n\n"
                        f"Log: {log_path}\n\n{_tail_text(log_path)}"
                    )
                raise RuntimeError(
                    f"DinoKPSEG training failed (exit code {completed_rc}).\n\n"
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

    def _resolve_output_dir(
        self,
        *,
        out_dir: Optional[str],
        model_name: str,
        data_config_path: str,
    ) -> Path:
        base = Path(out_dir).expanduser().resolve(
        ) if out_dir else shared_runs_root()
        run_name = self._infer_run_name(data_config_path)
        run_dir = new_run_dir(
            task="dino_kpseg",
            model=str(model_name or "train"),
            runs_root=base,
            run_name=run_name,
        )
        try:
            run_dir.mkdir(parents=True, exist_ok=True)
        except Exception:
            # Defer failures to the training task (which will surface a clearer message).
            pass
        return run_dir

    @staticmethod
    def _infer_run_name(data_config_path: str) -> Optional[str]:
        try:
            stem = Path(str(data_config_path)).expanduser().stem
        except Exception:
            return None
        if not stem:
            return None
        marker = "_resolved_"
        if marker in stem:
            stem = stem.split(marker, 1)[0]
        return stem or None

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

    def _preflight_ok(
        self,
        *,
        data_config_path: str,
        augment: bool,
        device: Optional[str],
        epochs: int,
    ) -> bool:
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

        warnings: List[str] = []
        train_count = self._count_images(data_cfg.get("train"))
        val_count = self._count_images(data_cfg.get("val"))

        if train_count and train_count < 50:
            warnings.append(
                f"Training set is very small ({train_count} images). Consider labeling more frames for better accuracy."
            )
        if val_count and val_count < 10:
            warnings.append(
                f"Validation set is very small ({val_count} images). Metrics may be unstable/noisy."
            )

        if bool(augment) and data_cfg.get("kpt_shape") and not data_cfg.get("flip_idx"):
            warnings.append(
                "Augmentations are enabled but the pose dataset has no 'flip_idx' in data.yaml; horizontal flip may be incorrect."
            )

        if int(epochs) >= 200 and train_count and train_count < 200:
            warnings.append(
                f"Epochs ({int(epochs)}) is high relative to a small dataset ({train_count} images); you may overfit."
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

        if not warnings:
            return True

        message = (
            "Potential training issues detected:\n\n- "
            + "\n- ".join(warnings)
            + "\n\nDo you want to start training anyway?"
        )
        reply = QtWidgets.QMessageBox.warning(
            self._window,
            "DinoKPSEG Training Preflight",
            message,
            QtWidgets.QMessageBox.Yes | QtWidgets.QMessageBox.No,
            QtWidgets.QMessageBox.No,
        )
        return reply == QtWidgets.QMessageBox.Yes

    def _count_images(self, entry: Any) -> int:
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
                exts = {".jpg", ".jpeg", ".png",
                        ".bmp", ".tif", ".tiff", ".webp"}
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

    def _set_active_process(self, proc: subprocess.Popen) -> None:
        with self._process_lock:
            self._active_process = proc

    def _clear_active_process(self, proc: subprocess.Popen) -> None:
        with self._process_lock:
            if self._active_process is proc:
                self._active_process = None

    def _terminate_active_process(self) -> None:
        with self._process_lock:
            proc = self._active_process
        if proc is None:
            return
        try:
            self._terminate_process(proc)
        except Exception:
            pass

    def _terminate_process(self, proc: subprocess.Popen) -> None:
        if proc.poll() is not None:
            return

        system = platform.system().lower()
        try:
            if system == "windows":
                try:
                    proc.send_signal(getattr(signal, "CTRL_BREAK_EVENT"))
                    proc.wait(timeout=2.0)
                    return
                except Exception:
                    pass
                try:
                    proc.terminate()
                    proc.wait(timeout=2.0)
                    return
                except Exception:
                    pass
                proc.kill()
                proc.wait(timeout=2.0)
                return

            # POSIX: terminate process group if possible.
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except Exception:
                try:
                    proc.terminate()
                except Exception:
                    pass
            try:
                proc.wait(timeout=3.0)
                return
            except Exception:
                pass
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
            try:
                proc.wait(timeout=2.0)
            except Exception:
                pass
        finally:
            self._clear_active_process(proc)

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
            if isinstance(outcome, _TrainingCancelled):
                logger.info("DinoKPSEG training cancelled: %s", outcome)
                QtWidgets.QMessageBox.information(
                    self._window,
                    "Training Cancelled",
                    str(outcome),
                )
                self._window.statusBar().showMessage(
                    self._window.tr("DinoKPSEG training cancelled.")
                )
                return
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
