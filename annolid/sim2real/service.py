from __future__ import annotations

import subprocess
import sys
import threading
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Optional

import cv2

from annolid.colab.models import ColabPredictRequest, ColabTrainRequest
from annolid.colab.service import ColabWorkflowService

from .analyzer import analyze_example_videos, build_default_sim_config
from .evaluator import evaluate_iteration_outputs, suggest_next_iteration
from .generators import generate_sim_dataset
from .models import (
    Sim2RealIterationRecord,
    Sim2RealProjectState,
    Sim2RealProjectStore,
    SimDatasetArtifact,
    SimDatasetConfig,
    SimIterationEvaluation,
    VideoEnvironmentProfile,
    utc_now_iso,
)
from .reporting import generate_sim2real_reports

LogCallback = Optional[Callable[[str], None]]
ProgressCallback = Optional[Callable[[int, str], None]]


def _emit_log(cb: LogCallback, msg: str) -> None:
    if cb is not None:
        try:
            cb(msg)
        except Exception:
            pass


def _emit_progress(cb: ProgressCallback, pct: int, label: str) -> None:
    if cb is not None:
        try:
            cb(max(0, min(100, int(pct))), label)
        except Exception:
            pass


def _default_base_weights_for_task(task: str) -> str:
    t = str(task or "yolo_seg").strip().lower()
    if t == "yolo_pose":
        return "yolo11n-pose.pt"
    if t == "yolo_detect":
        return "yolo11n.pt"
    return "yolo11n-seg.pt"


def _find_best_weights(search_dir: Path) -> str:
    if not search_dir.exists():
        return ""
    for pattern in (
        "**/weights/best.pt",
        "**/best.pt",
        "**/weights/last.pt",
        "**/*.pt",
    ):
        hits = sorted(search_dir.glob(pattern), key=lambda p: p.stat().st_mtime)
        if hits:
            return str(hits[-1].resolve())
    return ""


def _prepare_eval_clip(
    source_path: Path,
    clip_output_dir: Path,
    *,
    max_frames: int = 24,
) -> Path:
    """Create a compact sampled video/image subset of an example video for fast evaluation.

    Avoids uploading multi-GB raw lab videos to Colab during iterative evaluation while
    preserving exact recording resolution and appearance.
    """
    clip_output_dir.mkdir(parents=True, exist_ok=True)
    if source_path.is_dir():
        return source_path

    if source_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}:
        return source_path

    cap = cv2.VideoCapture(str(source_path))
    if not cap.isOpened():
        return source_path
    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 15.0)
        if fps <= 0.1:
            fps = 15.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)

        if 0 < total <= max_frames:
            return source_path

        out_clip = clip_output_dir / f"{source_path.stem}_eval_clip.mp4"
        if out_clip.is_file() and out_clip.stat().st_size > 0:
            return out_clip

        frames: list[Any] = []
        if total > 0:
            import numpy as np

            indices = np.linspace(
                0, max(0, total - 1), num=max(1, max_frames), dtype=int
            )
            seen: set[int] = set()
            for idx in indices.tolist():
                if idx in seen:
                    continue
                seen.add(idx)
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ok, frame = cap.read()
                if ok and frame is not None:
                    frames.append(frame)
        if not frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            while len(frames) < max_frames:
                ok, frame = cap.read()
                if not ok or frame is None:
                    break
                frames.append(frame)
        if not frames:
            return source_path
        if width <= 0 or height <= 0:
            height, width = frames[0].shape[:2]

        writer = cv2.VideoWriter(
            str(out_clip),
            cv2.VideoWriter_fourcc(*"mp4v"),
            min(fps, 15.0),
            (width, height),
        )
        try:
            for fr in frames:
                if fr.shape[1] != width or fr.shape[0] != height:
                    fr = cv2.resize(fr, (width, height))
                writer.write(fr)
        finally:
            writer.release()
        return out_clip if out_clip.is_file() else source_path
    finally:
        cap.release()


class Sim2RealWorkflowService:
    """Orchestrates example-video profiling -> 3D Mouse/Fly simulation -> Colab train/eval -> iteration -> reporting."""

    def __init__(
        self,
        project_dir: str | Path,
        *,
        colab_service: Optional[ColabWorkflowService] = None,
    ) -> None:
        self.project_dir = Path(project_dir).expanduser().resolve()
        self.project_dir.mkdir(parents=True, exist_ok=True)
        self.store = Sim2RealProjectStore(self.project_dir)
        self._colab_service = colab_service

    @property
    def colab(self) -> ColabWorkflowService:
        if self._colab_service is None:
            self._colab_service = ColabWorkflowService()
        return self._colab_service

    def load_state(self) -> Sim2RealProjectState:
        return self.store.load()

    def save_state(self, state: Sim2RealProjectState) -> Sim2RealProjectState:
        return self.store.save(state)

    # ------------------------------------------------------------------ #
    # Step 1: Analyze real example videos & infer 3D sim configuration
    # ------------------------------------------------------------------ #

    def analyze_videos(
        self,
        video_paths: Iterable[str | Path],
        *,
        species_hint: str = "auto",
        task: str = "yolo_seg",
        export_level: str = "instances",
        frames_per_scenario: int = 24,
        max_samples_per_video: int = 24,
        log_callback: LogCallback = None,
    ) -> tuple[VideoEnvironmentProfile, SimDatasetConfig]:
        paths = [Path(p).expanduser().resolve() for p in video_paths]
        _emit_log(
            log_callback,
            f"Analyzing {len(paths)} example source(s) (species_hint={species_hint})...",
        )
        analysis_dir = self.project_dir / "environment_analysis"
        profile = analyze_example_videos(
            paths,
            species_hint=species_hint,
            output_dir=analysis_dir,
            max_samples_per_video=max_samples_per_video,
        )
        sim_cfg = build_default_sim_config(
            profile,
            task=task,
            export_level=export_level,
            frames_per_scenario=frames_per_scenario,
        )

        state = self.load_state()
        state.example_videos = list(profile.example_videos)
        state.profile = profile.to_dict()
        state.current_sim_config = sim_cfg.to_dict()
        self.save_state(state)

        _emit_log(
            log_callback,
            f"Matched environment -> species={profile.species}, arena={profile.recommended_arena}, "
            f"camera={profile.recommended_camera_view}, lighting={profile.recommended_lighting}, "
            f"coat={profile.recommended_coat_pair}, subjects~{profile.estimated_subject_count}.",
        )
        return profile, sim_cfg

    def update_sim_config(self, sim_config: SimDatasetConfig) -> Sim2RealProjectState:
        state = self.load_state()
        state.current_sim_config = sim_config.to_dict()
        return self.save_state(state)

    # ------------------------------------------------------------------ #
    # Step 2: Generate environment-matched 3D simulation dataset
    # ------------------------------------------------------------------ #

    def generate_dataset(
        self,
        sim_config: Optional[SimDatasetConfig] = None,
        *,
        output_dir: Optional[str | Path] = None,
        log_callback: LogCallback = None,
    ) -> SimDatasetArtifact:
        state = self.load_state()
        profile = (
            VideoEnvironmentProfile.from_dict(state.profile)
            if isinstance(state.profile, Mapping)
            else None
        )
        if sim_config is None:
            if isinstance(state.current_sim_config, Mapping):
                sim_config = SimDatasetConfig.from_dict(state.current_sim_config)
            elif profile is not None:
                sim_config = build_default_sim_config(profile)
            else:
                sim_config = SimDatasetConfig()
        else:
            state.current_sim_config = sim_config.to_dict()
            self.save_state(state)

        target_dir = (
            Path(output_dir).expanduser().resolve()
            if output_dir is not None
            else (
                self.project_dir
                / "datasets"
                / f"{sim_config.species}_seed{sim_config.seed}"
            )
        )
        return generate_sim_dataset(
            sim_config,
            target_dir,
            profile=profile,
            log_callback=log_callback,
        )

    # ------------------------------------------------------------------ #
    # Step 3 & 4: Train / fine-tune + evaluate on example videos
    # ------------------------------------------------------------------ #

    def _run_local_train(
        self,
        *,
        dataset_yaml: str,
        model: str,
        weights: str,
        epochs: int,
        imgsz: int,
        batch: Optional[int],
        output_dir: Path,
        log_callback: LogCallback,
    ) -> tuple[str, str]:
        output_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable,
            "-m",
            "annolid.engine.cli",
            "train",
            model,
            "--data",
            str(dataset_yaml),
            "--epochs",
            str(int(epochs)),
            "--imgsz",
            str(int(imgsz)),
            "--project",
            str(output_dir.parent),
            "--name",
            str(output_dir.name),
        ]
        if weights:
            cmd.extend(["--weights", str(weights)])
        if batch is not None:
            cmd.extend(["--batch", str(int(batch))])
        _emit_log(log_callback, f"Running local training: {' '.join(cmd)}")
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.stdout:
            for line in proc.stdout.splitlines()[-25:]:
                _emit_log(log_callback, line)
        if proc.returncode != 0:
            raise RuntimeError(
                f"Local training failed (exit {proc.returncode}): {(proc.stderr or proc.stdout or '')[-500:]}"
            )
        return str(output_dir), _find_best_weights(output_dir)

    def _run_local_predict(
        self,
        *,
        source_path: Path,
        weights_path: str,
        conf: float,
        output_dir: Path,
        log_callback: LogCallback,
    ) -> str:
        output_dir.mkdir(parents=True, exist_ok=True)
        cmd = [
            sys.executable,
            "-m",
            "annolid.engine.cli",
            "predict",
            "yolo_labelme",
            "--source",
            str(source_path),
            "--weights",
            str(weights_path),
            "--conf",
            str(float(conf)),
            "--output-dir",
            str(output_dir),
        ]
        _emit_log(log_callback, f"Running local evaluation inference: {' '.join(cmd)}")
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"Local inference failed (exit {proc.returncode}): {(proc.stderr or proc.stdout or '')[-500:]}"
            )
        return str(output_dir)

    def run_iteration(
        self,
        *,
        sim_config: Optional[SimDatasetConfig] = None,
        existing_dataset: Optional[SimDatasetArtifact] = None,
        train_backend: str = "colab",
        colab_backend: str = "auto",
        accelerator: str = "T4",
        high_mem: bool = False,
        session_name: Optional[str] = None,
        keep_runtime: bool = False,
        annolid_install_spec: str = "annolid",
        model: str = "yolo",
        weights: Optional[str] = None,
        epochs: int = 20,
        imgsz: int = 640,
        batch: Optional[int] = None,
        conf: float = 0.25,
        eval_max_frames: int = 24,
        ground_truth_labelme_dir: Optional[str | Path] = None,
        target_score: float = 0.88,
        log_callback: LogCallback = None,
        progress_callback: ProgressCallback = None,
        cancel_event: Optional[threading.Event] = None,
    ) -> Sim2RealIterationRecord:
        """Execute one complete Generate -> Train/Fine-Tune -> Test on Examples -> Evaluate -> Advise iteration."""
        state = self.load_state()
        profile = (
            VideoEnvironmentProfile.from_dict(state.profile)
            if isinstance(state.profile, Mapping)
            else None
        )
        prev_records = [
            Sim2RealIterationRecord.from_dict(it)
            for it in (state.iterations or [])
            if isinstance(it, Mapping)
        ]
        iter_num = (max((r.iteration for r in prev_records), default=0)) + 1
        iter_dir = self.project_dir / "iterations" / f"iter_{iter_num:02d}"
        iter_dir.mkdir(parents=True, exist_ok=True)

        if sim_config is None:
            if isinstance(state.current_sim_config, Mapping):
                sim_config = SimDatasetConfig.from_dict(state.current_sim_config)
            elif profile is not None:
                sim_config = build_default_sim_config(profile)
            else:
                sim_config = SimDatasetConfig()

        # Default weights: fine-tune from best checkpoint if available, else task base checkpoint.
        resolved_weights = str(weights or "").strip()
        if not resolved_weights:
            if state.best_weights_path and Path(state.best_weights_path).is_file():
                resolved_weights = state.best_weights_path
            else:
                resolved_weights = _default_base_weights_for_task(sim_config.task)

        record = Sim2RealIterationRecord(
            iteration=iter_num,
            status="running",
            sim_config=sim_config.to_dict(),
            train_backend=train_backend,
            train_model=model,
            train_weights_in=resolved_weights,
            train_epochs=int(epochs),
            train_imgsz=int(imgsz),
            train_batch=batch,
            accelerator=accelerator if train_backend == "colab" else "local",
        )
        state.iterations.append(record.to_dict())
        self.save_state(state)

        def _persist() -> None:
            record.updated_at = utc_now_iso()
            for idx, existing in enumerate(state.iterations):
                if int(existing.get("iteration", -1)) == iter_num:
                    state.iterations[idx] = record.to_dict()
                    break
            self.save_state(state)

        try:
            if cancel_event is not None and cancel_event.is_set():
                raise InterruptedError(
                    "Sim-to-Real iteration cancelled before dataset generation."
                )

            # 1. Dataset generation
            _emit_progress(
                progress_callback,
                10,
                f"Iter {iter_num}: Generating 3D simulation dataset...",
            )
            if existing_dataset is not None:
                ds_artifact = existing_dataset
                _emit_log(
                    log_callback,
                    f"[Iter {iter_num}] Reusing dataset: {ds_artifact.data_yaml_path}",
                )
            else:
                ds_artifact = generate_sim_dataset(
                    sim_config,
                    iter_dir / "dataset",
                    profile=profile,
                    log_callback=log_callback,
                )
            record.dataset = ds_artifact.to_dict()
            _persist()

            if cancel_event is not None and cancel_event.is_set():
                raise InterruptedError(
                    "Sim-to-Real iteration cancelled before training."
                )

            # 2. Train / Fine-tune
            _emit_progress(
                progress_callback,
                35,
                f"Iter {iter_num}: Training/fine-tuning on {train_backend} ({accelerator})...",
            )
            owned_session_name: Optional[str] = None
            train_out_dir = iter_dir / "train_output"
            if train_backend == "colab":
                train_req = ColabTrainRequest(
                    model=model,
                    dataset_path=ds_artifact.data_yaml_path,
                    weights=resolved_weights,
                    epochs=int(epochs),
                    batch=batch,
                    imgsz=int(imgsz),
                    plots=True,
                    accelerator=accelerator,
                    high_mem=high_mem,
                    backend=colab_backend,
                    session_name=session_name,
                    keep_runtime=keep_runtime or bool(state.example_videos),
                    output_dir=str(train_out_dir),
                    annolid_install_spec=annolid_install_spec,
                    job_name=f"sim2real_iter{iter_num:02d}",
                )
                job = self.colab.run_train(
                    train_req,
                    log_callback=log_callback,
                    progress_callback=lambda p, lbl: _emit_progress(
                        progress_callback,
                        35 + int(p * 0.35),
                        f"Iter {iter_num} Train: {lbl}",
                    ),
                    cancel_event=cancel_event,
                )
                record.colab_train_job_id = job.job_id
                record.train_output_dir = job.local_output_dir or str(train_out_dir)
                # Keep active session name for immediate evaluation reuse and
                # track ownership so we always tear it down on failure/cancel.
                if job.session_name and not session_name:
                    session_name = job.session_name
                    if train_req.keep_runtime and not keep_runtime:
                        owned_session_name = job.session_name
                if job.status == "notebook_ready":
                    record.status = "notebook_ready"
                    record.notes.append(
                        f"Colab notebook generated ({job.notebook_path}). Run it in Colab and pull results before evaluating."
                    )
                    _persist()
                    return record
                record.best_weights_path = _find_best_weights(
                    Path(record.train_output_dir)
                )
            else:
                out_dir_str, best_pt = self._run_local_train(
                    dataset_yaml=ds_artifact.data_yaml_path,
                    model=model,
                    weights=resolved_weights,
                    epochs=int(epochs),
                    imgsz=int(imgsz),
                    batch=batch,
                    output_dir=train_out_dir,
                    log_callback=log_callback,
                )
                record.train_output_dir = out_dir_str
                record.best_weights_path = best_pt

            _persist()

            if cancel_event is not None and cancel_event.is_set():
                raise InterruptedError(
                    "Sim-to-Real iteration cancelled before evaluation."
                )

            # 3. Test trained weights on real example video(s)
            pred_dirs: list[str] = []
            example_videos = list(state.example_videos or [])
            if example_videos and record.best_weights_path:
                _emit_progress(
                    progress_callback,
                    75,
                    f"Iter {iter_num}: Testing trained weights on {len(example_videos)} example video(s)...",
                )
                eval_clips_dir = self.project_dir / "eval_clips"
                for v_idx, raw_vid in enumerate(example_videos):
                    src_vid = Path(raw_vid)
                    clip_path = _prepare_eval_clip(
                        src_vid, eval_clips_dir, max_frames=int(eval_max_frames)
                    )
                    pred_out = (
                        iter_dir
                        / "predictions"
                        / f"video_{v_idx + 1:02d}_{src_vid.stem}"
                    )
                    if train_backend == "colab":
                        is_last_eval = v_idx == len(example_videos) - 1
                        pred_req = ColabPredictRequest(
                            model="yolo_labelme",
                            source_path=str(clip_path),
                            weights=record.best_weights_path,
                            conf=float(conf),
                            save_csv=True,
                            save_video=False,
                            accelerator=accelerator,
                            high_mem=high_mem,
                            backend=colab_backend,
                            session_name=session_name,
                            keep_runtime=keep_runtime if is_last_eval else True,
                            output_dir=str(pred_out),
                            annolid_install_spec=annolid_install_spec,
                            job_name=f"sim2real_eval_iter{iter_num:02d}_v{v_idx + 1}",
                        )
                        pjob = self.colab.run_predict(
                            pred_req,
                            log_callback=log_callback,
                            progress_callback=lambda p, lbl: _emit_progress(
                                progress_callback,
                                75 + int(p * 0.15),
                                f"Iter {iter_num} Eval: {lbl}",
                            ),
                            cancel_event=cancel_event,
                        )
                        record.colab_predict_job_ids.append(pjob.job_id)
                        pred_dirs.append(pjob.local_output_dir or str(pred_out))
                        if is_last_eval and not keep_runtime:
                            owned_session_name = None
                    else:
                        pred_dirs.append(
                            self._run_local_predict(
                                source_path=clip_path,
                                weights_path=record.best_weights_path,
                                conf=float(conf),
                                output_dir=pred_out,
                                log_callback=log_callback,
                            )
                        )
            elif not record.best_weights_path:
                record.notes.append(
                    "No trained checkpoint (`best.pt`) found in training output; skipped real-video inference."
                )

            record.prediction_dirs = pred_dirs

            # 4. Evaluate & propose next-iteration refinements
            _emit_progress(
                progress_callback,
                92,
                f"Iter {iter_num}: Evaluating metrics & advising next round...",
            )
            overlay_sheet = iter_dir / f"prediction_overlays_iter_{iter_num:02d}.png"
            evaluation = evaluate_iteration_outputs(
                example_videos=example_videos,
                prediction_dirs=pred_dirs,
                profile=profile,
                training_output_dir=record.train_output_dir,
                ground_truth_labelme_dir=ground_truth_labelme_dir,
                overlay_output_path=overlay_sheet,
            )
            record.evaluation = evaluation.to_dict()

            if prev_records:
                prev_eval = prev_records[-1].evaluation or {}
                if prev_eval.get("sim2real_score") is not None:
                    record.score_delta_vs_prev = round(
                        evaluation.sim2real_score - float(prev_eval["sim2real_score"]),
                        4,
                    )

            adjustment = suggest_next_iteration(
                current_config=sim_config,
                evaluation=evaluation,
                profile=profile,
                previous_iterations=prev_records,
                current_best_weights=record.best_weights_path,
                current_epochs=int(epochs),
                current_imgsz=int(imgsz),
                target_score=float(target_score),
            )
            record.adjustment = adjustment.to_dict()
            record.status = "completed"

            # Track best checkpoint across all iterations.
            if state.best_score is None or evaluation.sim2real_score >= float(
                state.best_score
            ):
                state.best_score = evaluation.sim2real_score
                state.best_iteration = iter_num
                if record.best_weights_path:
                    state.best_weights_path = record.best_weights_path

            _persist()

            # Refresh report after each completed iteration.
            state.report_paths = generate_sim2real_reports(state)
            self.save_state(state)

            _emit_progress(
                progress_callback,
                100,
                f"Iter {iter_num} complete (score={evaluation.sim2real_score:.3f})",
            )
            _emit_log(
                log_callback,
                f"[Iter {iter_num}] Completed: Sim2Real score={evaluation.sim2real_score:.3f}, "
                f"coverage={evaluation.detection_coverage:.1%}, mean_conf={evaluation.mean_confidence:.2f}, "
                f"val_map50={evaluation.val_mask_map50 or evaluation.val_box_map50}.",
            )
            return record
        except Exception as exc:
            record.status = (
                "cancelled" if isinstance(exc, InterruptedError) else "failed"
            )
            record.error = str(exc)
            _persist()
            raise
        finally:
            if (
                train_backend == "colab"
                and not keep_runtime
                and locals().get("owned_session_name")
                and record.status != "notebook_ready"
            ):
                try:
                    self.colab.stop_session(
                        str(locals()["owned_session_name"]),
                        backend=colab_backend,
                    )
                except Exception as stop_exc:
                    _emit_log(
                        log_callback,
                        f"[Iter {iter_num}] Warning: failed to stop Colab session "
                        f"'{locals()['owned_session_name']}': {stop_exc}",
                    )

    def apply_latest_recommendation(self) -> tuple[SimDatasetConfig, dict[str, Any]]:
        """Persist the latest iteration's recommended SimDatasetConfig as current config."""
        state = self.load_state()
        for raw_it in reversed(state.iterations or []):
            adj = raw_it.get("adjustment") or {}
            suggested = adj.get("suggested_sim_config")
            if isinstance(suggested, Mapping) and suggested:
                cfg = SimDatasetConfig.from_dict(dict(suggested))
                state.current_sim_config = cfg.to_dict()
                self.save_state(state)
                return cfg, dict(adj.get("suggested_train_overrides") or {})
        raise ValueError(
            "No completed iteration with recommendations found in project state."
        )

    def run_loop(
        self,
        *,
        iterations: int = 2,
        auto_apply_recommendations: bool = True,
        sim_config: Optional[SimDatasetConfig] = None,
        train_backend: str = "colab",
        colab_backend: str = "auto",
        accelerator: str = "T4",
        high_mem: bool = False,
        session_name: Optional[str] = None,
        keep_runtime: bool = False,
        annolid_install_spec: str = "annolid",
        model: str = "yolo",
        weights: Optional[str] = None,
        epochs: int = 20,
        imgsz: int = 640,
        batch: Optional[int] = None,
        conf: float = 0.25,
        eval_max_frames: int = 24,
        ground_truth_labelme_dir: Optional[str | Path] = None,
        target_score: float = 0.88,
        log_callback: LogCallback = None,
        progress_callback: ProgressCallback = None,
        cancel_event: Optional[threading.Event] = None,
    ) -> list[Sim2RealIterationRecord]:
        """Run multiple closed-loop Sim-to-Real iterations, refining dataset & fine-tuning automatically."""
        n_iters = max(1, int(iterations))
        state = self.load_state()
        cur_cfg = (
            sim_config
            if sim_config is not None
            else (
                SimDatasetConfig.from_dict(state.current_sim_config)
                if isinstance(state.current_sim_config, Mapping)
                else SimDatasetConfig()
            )
        )
        cur_weights = weights
        cur_epochs = int(epochs)
        cur_imgsz = int(imgsz)
        cur_conf = float(conf)

        # Share one warm Colab CLI session across loop iterations to save provisioning & install time.
        loop_session = session_name
        created_shared_session = False
        if (
            train_backend == "colab"
            and n_iters > 1
            and not loop_session
            and colab_backend in {"auto", "cli"}
        ):
            loop_session = f"annolid-s2r-{state.project_id[-6:]}"
            created_shared_session = True

        completed: list[Sim2RealIterationRecord] = []
        try:
            for round_idx in range(n_iters):
                if cancel_event is not None and cancel_event.is_set():
                    break
                is_last = round_idx == n_iters - 1
                _emit_log(
                    log_callback,
                    f"=== Starting Sim-to-Real Round {round_idx + 1}/{n_iters} ===",
                )
                rec = self.run_iteration(
                    sim_config=cur_cfg,
                    train_backend=train_backend,
                    colab_backend=colab_backend,
                    accelerator=accelerator,
                    high_mem=high_mem,
                    session_name=loop_session,
                    keep_runtime=True if not is_last else keep_runtime,
                    annolid_install_spec=annolid_install_spec,
                    model=model,
                    weights=cur_weights,
                    epochs=cur_epochs,
                    imgsz=cur_imgsz,
                    batch=batch,
                    conf=cur_conf,
                    eval_max_frames=eval_max_frames,
                    ground_truth_labelme_dir=ground_truth_labelme_dir,
                    target_score=target_score,
                    log_callback=log_callback,
                    progress_callback=lambda p, lbl, r=round_idx: _emit_progress(
                        progress_callback,
                        int(((r + p / 100.0) / float(n_iters)) * 100.0),
                        f"[Round {r + 1}/{n_iters}] {lbl}",
                    ),
                    cancel_event=cancel_event,
                )
                completed.append(rec)

                if rec.status != "completed":
                    break
                adj = rec.adjustment or {}
                if not adj.get("should_continue", True):
                    _emit_log(
                        log_callback,
                        f"Stopping loop early: {adj.get('stop_reason') or 'target reached'}.",
                    )
                    break
                if auto_apply_recommendations and not is_last:
                    suggested_cfg = adj.get("suggested_sim_config")
                    if isinstance(suggested_cfg, Mapping) and suggested_cfg:
                        cur_cfg = SimDatasetConfig.from_dict(dict(suggested_cfg))
                        self.update_sim_config(cur_cfg)
                    overrides = adj.get("suggested_train_overrides") or {}
                    if overrides.get("weights"):
                        cur_weights = str(overrides["weights"])
                    if overrides.get("epochs"):
                        cur_epochs = int(overrides["epochs"])
                    if overrides.get("imgsz"):
                        cur_imgsz = int(overrides["imgsz"])
                    if overrides.get("conf"):
                        cur_conf = float(overrides["conf"])
                    _emit_log(
                        log_callback,
                        f"Applied automated adjustments for next round: "
                        f"scenarios={cur_cfg.scenarios}, frames/scen={cur_cfg.frames_per_scenario}, "
                        f"photometry={cur_cfg.photometry_strength}, dr={cur_cfg.domain_randomization}, "
                        f"weights={cur_weights}, epochs={cur_epochs}, imgsz={cur_imgsz}.",
                    )
        finally:
            if created_shared_session and not keep_runtime and loop_session:
                try:
                    _emit_log(
                        log_callback,
                        f"Cleaning up shared Colab session '{loop_session}'...",
                    )
                    self.colab.stop_session(loop_session, backend=colab_backend)
                except Exception:
                    pass

        self.write_report()
        return completed

    def evaluate_existing_weights(
        self,
        *,
        weights_path: str | Path,
        train_backend: str = "colab",
        colab_backend: str = "auto",
        accelerator: str = "T4",
        session_name: Optional[str] = None,
        keep_runtime: bool = False,
        conf: float = 0.25,
        eval_max_frames: int = 24,
        ground_truth_labelme_dir: Optional[str | Path] = None,
        log_callback: LogCallback = None,
    ) -> SimIterationEvaluation:
        """Evaluate any checkpoint directly on the project's example video(s) without retraining."""
        state = self.load_state()
        profile = (
            VideoEnvironmentProfile.from_dict(state.profile)
            if isinstance(state.profile, Mapping)
            else None
        )
        example_videos = list(state.example_videos or [])
        if not example_videos:
            raise ValueError("Project has no example videos. Run `analyze` first.")

        w_path = str(Path(weights_path).expanduser().resolve())
        eval_dir = self.project_dir / "standalone_eval"
        eval_clips_dir = self.project_dir / "eval_clips"
        pred_dirs: list[str] = []
        created_session_name: Optional[str] = None
        try:
            for v_idx, raw_vid in enumerate(example_videos):
                src_vid = Path(raw_vid)
                clip_path = _prepare_eval_clip(
                    src_vid, eval_clips_dir, max_frames=int(eval_max_frames)
                )
                pred_out = eval_dir / f"video_{v_idx + 1:02d}_{src_vid.stem}"
                if train_backend == "colab":
                    pjob = self.colab.run_predict(
                        ColabPredictRequest(
                            model="yolo_labelme",
                            source_path=str(clip_path),
                            weights=w_path,
                            conf=float(conf),
                            save_csv=True,
                            save_video=False,
                            accelerator=accelerator,
                            backend=colab_backend,
                            session_name=session_name,
                            keep_runtime=keep_runtime
                            if v_idx == len(example_videos) - 1
                            else True,
                            output_dir=str(pred_out),
                        ),
                        log_callback=log_callback,
                    )
                    if pjob.session_name and not session_name:
                        session_name = pjob.session_name
                        created_session_name = pjob.session_name
                    pred_dirs.append(pjob.local_output_dir or str(pred_out))
                else:
                    pred_dirs.append(
                        self._run_local_predict(
                            source_path=clip_path,
                            weights_path=w_path,
                            conf=float(conf),
                            output_dir=pred_out,
                            log_callback=log_callback,
                        )
                    )
        finally:
            if train_backend == "colab" and not keep_runtime and created_session_name:
                try:
                    self.colab.stop_session(created_session_name, backend=colab_backend)
                except Exception:
                    pass

        return evaluate_iteration_outputs(
            example_videos=example_videos,
            prediction_dirs=pred_dirs,
            profile=profile,
            ground_truth_labelme_dir=ground_truth_labelme_dir,
            overlay_output_path=eval_dir / "prediction_overlays.png",
        )

    def write_report(self, output_dir: Optional[str | Path] = None) -> dict[str, str]:
        state = self.load_state()
        paths = generate_sim2real_reports(state, output_dir=output_dir)
        state.report_paths = paths
        self.save_state(state)
        return paths
