from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from qtpy import QtCore, QtWidgets

from annolid.gui.workers import FlexibleWorker
from annolid.utils.logger import logger


class AgentAnalysisMixin:
    """Agent-run orchestration and event import helpers."""

    def open_agent_run_dialog(self) -> None:
        from annolid.gui.widgets import AgentRunDialog

        if not self.video_file:
            QtWidgets.QMessageBox.information(
                self,
                self.tr("Run Agent"),
                self.tr("Open a video before configuring an agent run."),
            )
            return

        if self.video_results_folder:
            results_dir = str(self.video_results_folder)
        else:
            try:
                results_dir = str(Path(self.video_file).with_suffix(""))
            except Exception:
                results_dir = ""

        config = self._load_agent_run_settings()
        dialog = AgentRunDialog(
            self,
            config=config,
            video_path=self.video_file,
            results_dir=results_dir,
        )
        if dialog.exec_() != QtWidgets.QDialog.Accepted:
            return

        values = dialog.values()
        self._agent_run_config = dict(values)
        self._save_agent_run_settings(values)
        self.statusBar().showMessage(self.tr("Agent run settings updated."), 4000)
        self.run_agent_analysis(values)

    def run_agent_analysis(self, config: Dict[str, Any]) -> None:
        if not self.video_file:
            QtWidgets.QMessageBox.information(
                self,
                self.tr("Run Agent"),
                self.tr("Open a video before running the agent."),
            )
            return

        if self._agent_thread is not None:
            try:
                if self._agent_thread.isRunning():
                    QtWidgets.QMessageBox.information(
                        self,
                        self.tr("Run Agent"),
                        self.tr("An agent run is already in progress."),
                    )
                    return
            except RuntimeError:
                self._agent_thread = None

        results_dir = None
        if self.video_results_folder:
            results_dir = str(self.video_results_folder)
        else:
            try:
                results_dir = str(Path(self.video_file).with_suffix(""))
            except Exception:
                results_dir = None

        state: Dict[str, Any] = {"status": "", "total": None}

        def _agent_task(
            *,
            pred_worker: Optional[FlexibleWorker] = None,
            stop_event: Optional[object] = None,
        ):
            from annolid.core.agent.runner import AgentRunConfig
            from annolid.core.agent.service import run_agent_to_results

            if bool(config.get("anchor_rerun")):
                try:
                    self._prepare_anchor_rerun()
                except Exception as exc:
                    logger.warning("Failed to apply anchor rerun: %s", exc)

            vision_model = None
            if str(config.get("vision_adapter") or "").strip().lower() == "maskrcnn":
                from annolid.core.models.adapters.maskrcnn_torchvision import (
                    TorchvisionMaskRCNNAdapter,
                )

                vision_model = TorchvisionMaskRCNNAdapter(
                    pretrained=bool(config.get("vision_pretrained", False)),
                    score_threshold=float(config.get("vision_score_threshold", 0.5)),
                    device=str(config.get("vision_device"))
                    if config.get("vision_device")
                    else None,
                )
            elif (
                str(config.get("vision_adapter") or "").strip().lower() == "dino_kpseg"
            ):
                from annolid.core.models.adapters.dino_kpseg_adapter import (
                    DinoKPSEGAdapter,
                )

                weight_path = self._resolve_dino_kpseg_weight(
                    config.get("vision_weights") or ""
                )
                if not weight_path:
                    raise ValueError(
                        "DinoKPSEG weights could not be resolved. "
                        "Select a checkpoint or train a DinoKPSEG model."
                    )
                vision_model = DinoKPSEGAdapter(
                    weight_path=weight_path,
                    device=str(config.get("vision_device"))
                    if config.get("vision_device")
                    else None,
                    score_threshold=float(config.get("vision_score_threshold", 0.5)),
                )

            llm_model = None
            if str(config.get("llm_adapter") or "").strip().lower() == "llm_chat":
                from annolid.core.models.adapters.llm_chat import LLMChatAdapter

                llm_model = LLMChatAdapter(
                    profile=config.get("llm_profile"),
                    provider=config.get("llm_provider"),
                    model=config.get("llm_model"),
                    persist=bool(config.get("llm_persist", False)),
                )

            run_config = AgentRunConfig(
                max_frames=config.get("max_frames"),
                stride=int(config.get("stride", 1)),
                include_llm_summary=bool(config.get("include_llm_summary", False)),
                llm_summary_prompt=str(
                    config.get("llm_summary_prompt")
                    or "Summarize the behaviors defined in this behavior spec."
                ),
            )

            def _progress(frame_idx: int, written: int, total: int | None) -> None:
                if total and total > 0:
                    total_frames = total
                    max_frames = run_config.max_frames
                    if max_frames is not None:
                        total_frames = min(total_frames, int(max_frames))
                    percent = int(round((written / max(total_frames, 1)) * 100))
                else:
                    percent = 0
                state["status"] = f"Frame {frame_idx} • Records {written}"
                state["total"] = total
                if pred_worker is not None:
                    pred_worker.report_progress(max(0, min(100, percent)))

            return run_agent_to_results(
                video_path=self.video_file,
                behavior_spec_path=config.get("schema_path"),
                results_dir=results_dir,
                vision_model=vision_model,
                llm_model=llm_model,
                config=run_config,
                progress_callback=_progress,
                stop_event=stop_event,
            )

        progress = QtWidgets.QProgressDialog(
            self.tr("Running agent analysis…"),
            self.tr("Cancel"),
            0,
            100,
            self,
        )
        progress.setWindowTitle(self.tr("Agent Analysis"))
        progress.setWindowModality(QtCore.Qt.WindowModal)
        progress.setAutoClose(False)
        progress.setAutoReset(False)
        progress.setMinimumDuration(500)
        self._agent_progress_dialog = progress

        self._agent_thread = QtCore.QThread(self)
        self._agent_worker = FlexibleWorker(task_function=_agent_task)
        self._agent_worker.moveToThread(self._agent_thread)
        self._agent_thread.started.connect(
            self._agent_worker.run, QtCore.Qt.QueuedConnection
        )

        def _update_progress(value: int) -> None:
            if self._agent_progress_dialog is None:
                return
            label = state.get("status") or self.tr("Running agent analysis…")
            self._agent_progress_dialog.setLabelText(label)
            self._agent_progress_dialog.setValue(int(value))

        def _finish_agent(result) -> None:
            if self._agent_progress_dialog is not None:
                self._agent_progress_dialog.close()
                self._agent_progress_dialog.deleteLater()
                self._agent_progress_dialog = None

            if isinstance(result, Exception):
                logger.exception("Agent run failed", exc_info=result)
                QtWidgets.QMessageBox.warning(
                    self,
                    self.tr("Agent Analysis Failed"),
                    self.tr("Agent run failed with error:\n%s") % result,
                )
                return

            self._apply_agent_results(result, config)

        self._agent_worker.progress_signal.connect(_update_progress)
        self._agent_worker.finished_signal.connect(_finish_agent)
        self._agent_worker.finished_signal.connect(self._agent_thread.quit)
        self._agent_worker.finished_signal.connect(self._agent_worker.deleteLater)
        self._agent_thread.finished.connect(self._agent_thread.deleteLater)

        progress.canceled.connect(self._agent_worker.request_stop)
        self._agent_thread.start()

    def _apply_agent_results(self, result, config: Dict[str, Any]) -> None:
        try:
            results_dir = Path(result.results_dir)
        except Exception:
            results_dir = None

        if results_dir is not None:
            self.video_results_folder = results_dir
            self.annotation_dir = results_dir
            self.behavior_controller.attach_annotation_store_for_video(results_dir)
            self._refresh_embedding_file_list()
            try:
                if hasattr(self, "embedding_search_widget"):
                    self.embedding_search_widget.set_annotation_dir(results_dir)
            except Exception:
                pass

        event_intervals = self._load_agent_event_intervals(result)
        if event_intervals:
            self._apply_agent_event_intervals(event_intervals)
        else:
            total_frames = result.total_frames or self.num_frames
            if total_frames is None or total_frames <= 0:
                return

            try:
                from annolid.core.behavior.spec import load_behavior_spec

                schema, _ = load_behavior_spec(
                    path=config.get("schema_path"),
                    video_path=self.video_file,
                )
                behavior_label = (
                    schema.behaviors[0].code if schema and schema.behaviors else "Agent"
                )
                subject = (
                    schema.subjects[0].name
                    if schema and schema.subjects
                    else self._current_subject_name()
                )
            except Exception:
                behavior_label = "Agent"
                subject = self._current_subject_name()

            mid = max(0, int(total_frames // 2))
            end = min(int(total_frames - 1), mid + max(1, int(self.fps or 30)))

            def _timestamp_provider(frame: int) -> Optional[float]:
                fps = self.fps if self.fps and self.fps > 0 else None
                if fps is None:
                    return None
                return float(frame) / float(fps)

            if behavior_label:
                self.behavior_controller.create_interval(
                    behavior=behavior_label,
                    start_frame=mid,
                    end_frame=end,
                    subject=subject,
                    timestamp_provider=_timestamp_provider,
                )

        fps_for_log = self.fps if self.fps and self.fps > 0 else 29.97
        if getattr(self, "behavior_log_widget", None) is not None:
            self.behavior_log_widget.set_events(
                list(self.behavior_controller.iter_events()),
                fps=fps_for_log,
            )
        if self.seekbar is not None:
            self.seekbar.setTickMarks()
        if self.frame_number is not None:
            self.loadPredictShapes(self.frame_number, self.filename)

    def _load_agent_event_intervals(self, result) -> List[Dict[str, Any]]:
        results_dir = None
        ndjson_path = None
        try:
            results_dir = Path(result.results_dir)
        except Exception:
            results_dir = None
        try:
            ndjson_path = Path(result.ndjson_path)
        except Exception:
            ndjson_path = None

        events: List[Dict[str, Any]] = []

        def _extend_from_file(path: Path) -> None:
            try:
                raw = path.read_text(encoding="utf-8").strip()
            except Exception:
                return
            if not raw:
                return
            try:
                payload = json.loads(raw)
            except Exception:
                return
            if isinstance(payload, list):
                for item in payload:
                    if isinstance(item, dict):
                        events.append(item)
            elif isinstance(payload, dict):
                if "events" in payload and isinstance(payload["events"], list):
                    for item in payload["events"]:
                        if isinstance(item, dict):
                            events.append(item)

        if results_dir is not None:
            json_path = results_dir / "agent_events.json"
            ndjson_events = results_dir / "agent_events.ndjson"
            if json_path.exists():
                _extend_from_file(json_path)
            if ndjson_events.exists():
                try:
                    for line in ndjson_events.read_text(encoding="utf-8").splitlines():
                        if not line.strip():
                            continue
                        try:
                            payload = json.loads(line)
                        except Exception:
                            continue
                        if isinstance(payload, dict):
                            events.append(payload)
                except Exception:
                    pass

        if ndjson_path is not None and ndjson_path.exists():
            try:
                for line in ndjson_path.read_text(encoding="utf-8").splitlines():
                    if not line.strip():
                        continue
                    try:
                        record = json.loads(line)
                    except Exception:
                        continue
                    if not isinstance(record, dict):
                        continue
                    other = record.get("otherData") or {}
                    if isinstance(other, dict):
                        for key in ("agent_events", "behavior_events"):
                            payload = other.get(key)
                            if isinstance(payload, list):
                                events.extend(
                                    [e for e in payload if isinstance(e, dict)]
                                )
                        behavior_block = other.get("annolid_behavior")
                        if isinstance(behavior_block, dict):
                            payload = behavior_block.get("events")
                            if isinstance(payload, list):
                                events.extend(
                                    [e for e in payload if isinstance(e, dict)]
                                )
            except Exception:
                pass

        return self._normalize_agent_events(events)

    def _normalize_agent_events(
        self, events: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        intervals: List[Dict[str, Any]] = []
        pending: Dict[tuple, int] = {}
        fps = self.fps if self.fps and self.fps > 0 else None

        def _frame_from_time(value: Any) -> Optional[int]:
            if value is None or fps is None:
                return None
            try:
                return int(round(float(value) * float(fps)))
            except (TypeError, ValueError):
                return None

        for entry in events:
            behavior = entry.get("behavior") or entry.get("label")
            if behavior is None:
                continue
            subject = entry.get("subject")
            start_frame = entry.get("start_frame")
            end_frame = entry.get("end_frame")

            if start_frame is None and "start_time" in entry:
                start_frame = _frame_from_time(entry.get("start_time"))
            if end_frame is None and "end_time" in entry:
                end_frame = _frame_from_time(entry.get("end_time"))

            if start_frame is not None and end_frame is not None:
                intervals.append(
                    {
                        "behavior": str(behavior),
                        "start_frame": int(start_frame),
                        "end_frame": int(end_frame),
                        "subject": subject,
                    }
                )
                continue

            event_label = entry.get("event")
            frame = entry.get("frame")
            if frame is None and "frame_index" in entry:
                frame = entry.get("frame_index")
            if frame is None and "timestamp_sec" in entry:
                frame = _frame_from_time(entry.get("timestamp_sec"))
            if event_label and frame is not None:
                key = (str(behavior), subject)
                if str(event_label).lower() == "start":
                    pending[key] = int(frame)
                elif str(event_label).lower() == "end":
                    start = pending.pop(key, int(frame))
                    intervals.append(
                        {
                            "behavior": str(behavior),
                            "start_frame": int(start),
                            "end_frame": int(frame),
                            "subject": subject,
                        }
                    )
                continue

            if "second" in entry and "behaviors" in entry:
                sec = entry.get("second")
                if sec is None:
                    continue
                try:
                    frame_idx = _frame_from_time(sec) if fps is not None else int(sec)
                except Exception:
                    continue
                behaviors = entry.get("behaviors") or []
                if not isinstance(behaviors, list):
                    continue
                for item in behaviors:
                    if not isinstance(item, dict):
                        continue
                    label = item.get("label") or item.get("behavior")
                    if not label:
                        continue
                    intervals.append(
                        {
                            "behavior": str(label),
                            "start_frame": int(frame_idx),
                            "end_frame": int(frame_idx),
                            "subject": subject,
                        }
                    )

        for (behavior, subject), start in pending.items():
            intervals.append(
                {
                    "behavior": str(behavior),
                    "start_frame": int(start),
                    "end_frame": int(start),
                    "subject": subject,
                }
            )
        return intervals

    def _apply_agent_event_intervals(self, intervals: List[Dict[str, Any]]) -> None:
        def _timestamp_provider(frame: int) -> Optional[float]:
            fps = self.fps if self.fps and self.fps > 0 else None
            if fps is None:
                return None
            return float(frame) / float(fps)

        for entry in intervals:
            try:
                behavior = str(entry.get("behavior") or "Agent")
                start = int(entry.get("start_frame"))
                end = int(entry.get("end_frame", start))
                subject = entry.get("subject")
                self.behavior_controller.create_interval(
                    behavior=behavior,
                    start_frame=start,
                    end_frame=end,
                    subject=subject,
                    timestamp_provider=_timestamp_provider,
                )
            except Exception:
                continue
