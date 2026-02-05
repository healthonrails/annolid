import gc
import torch
import logging
from pathlib import Path
from typing import List, Dict, Optional, TYPE_CHECKING

from qtpy.QtCore import QThread, Signal, Slot, QObject

from .tracking_jobs import VideoProcessingJob, JobType

from annolid.segmentation.cutie_vos.processor import SegmentedCutieExecutor
from annolid.segmentation.cutie_vos.engine import CutieEngine

# Utilities
from annolid.utils.files import (
    find_manual_labeled_json_files,
    get_frame_number_from_json,
)
from annolid.annotation.labelme2csv import convert_json_to_csv

from annolid.gui.label_file import LabelFile

if TYPE_CHECKING:
    from annolid.segmentation.SAM.edge_sam_bg import VideoProcessor


class TrackingWorker(QThread):
    """
    A unified worker thread for processing video tracking jobs.
    """

    progress = Signal(int, str)
    finished = Signal(str)
    error = Signal(str)
    video_job_started = Signal(str, str)  # video_path_str, output_folder_str
    video_job_finished = Signal(str)  # video_path_str

    def __init__(
        self,
        processing_jobs: List[VideoProcessingJob],
        global_config: Dict,
        parent: Optional[QObject] = None,
    ):
        super().__init__(parent)
        self.processing_jobs = self._validate_jobs(processing_jobs)
        self.global_config = global_config
        self._is_running = True
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self.current_video_path_being_processed_for_ui: Optional[str] = None
        self.device = self._select_device()  # Select device once for the worker

    def _validate_jobs(
        self, jobs: List[VideoProcessingJob]
    ) -> List[VideoProcessingJob]:
        valid_jobs = []
        if not jobs:
            self.logger.warning(
                "TrackingWorker received an empty list of processing jobs."
            )
            return []

        for job in jobs:
            if not isinstance(job, VideoProcessingJob):
                self.logger.error(
                    f"Invalid job type passed to TrackingWorker: {type(job)}. Skipping."
                )
                continue
            if not job.video_path.is_file():
                self.logger.error(
                    f"Video path in job does not exist or is not a file: {job.video_path}. Skipping job."
                )
                continue
            if job.job_type == JobType.VIDEO_SEGMENTS:
                if not job.segments_data:
                    self.logger.error(
                        f"Job for {job.video_path.name} is VIDEO_SEGMENTS but has no segment data. Skipping job."
                    )
                    continue
                if job.fps is None or job.fps <= 0:
                    self.logger.error(
                        f"Job for {job.video_path.name} is VIDEO_SEGMENTS but has invalid FPS ({job.fps}). Skipping job."
                    )
                    # Potentially try to get FPS from video here if absolutely necessary, but job should provide it
                    continue
            valid_jobs.append(job)
        return valid_jobs

    def _select_device(self) -> torch.device:
        if torch.cuda.is_available():
            return torch.device("cuda:0")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def _log_gpu_memory(self, task_name: str, stage: str):  # Helper
        video_name_for_log = (
            Path(self.current_video_path_being_processed_for_ui).stem
            if self.current_video_path_being_processed_for_ui
            else "UnknownVideo"
        )
        full_stage_name = f"{task_name} - {stage}"
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**2
            reserved = torch.cuda.memory_reserved() / 1024**2
            self.logger.info(
                f"GPU Memory (CUDA - {full_stage_name} - {video_name_for_log}): Allocated: {allocated:.2f} MB, Reserved: {reserved:.2f} MB"
            )
        elif (
            hasattr(torch.backends, "mps")
            and torch.backends.mps.is_available()
            and hasattr(torch.mps, "current_allocated_memory")
        ):
            allocated = torch.mps.current_allocated_memory() / 1024**2
            self.logger.info(
                f"GPU Memory (MPS - {full_stage_name} - {video_name_for_log}): Allocated: {allocated:.2f} MB"
            )
        else:
            self.logger.info(
                f"No GPU or memory tracking available ({full_stage_name} - {video_name_for_log})"
            )

    # Helper for VideoProcessor instances
    def _cleanup_general_processor(self, processor, device):
        if processor is not None:
            # If VideoProcessor has specific cleanup for Cutie, it should handle it.
            # This generic cleanup is for the VideoProcessor object itself.
            if (
                hasattr(processor, "cutie_processor")
                and processor.cutie_processor is not None
            ):
                processor.cutie_processor = None  # Let VP manage its Cutie instance
            del processor
        gc.collect()
        if device.type == "cuda":
            try:
                torch.cuda.empty_cache()
            except Exception as e:
                self.logger.info(f"CUDA cache clearing failed: {e}")
        elif device.type == "mps" and hasattr(torch.mps, "empty_cache"):
            try:
                torch.mps.empty_cache()
            except AttributeError:
                self.logger.info(
                    "MPS cache clearing not supported in this PyTorch version"
                )

    # --- Interface for pred_worker (if used by processors) ---

    def is_stopped(self) -> bool:
        return not self._is_running

    @Slot()
    def stop(self):
        self.logger.info("TrackingWorker stop requested.")
        self._is_running = False

    def run(self):
        self.logger.info(
            f"TrackingWorker started with {len(self.processing_jobs)} job(s) on device {self.device}."
        )
        total_jobs = len(self.processing_jobs)
        if total_jobs == 0:
            self.logger.warning("No valid jobs to process.")
            self.finished.emit("No valid jobs to process.")
            return

        jobs_completed_count = 0

        for job_idx, current_job in enumerate(self.processing_jobs):
            if not self._is_running:
                self.logger.info("TrackingWorker run loop interrupted by stop signal.")
                self.finished.emit("Tracking stopped by user.")
                return

            self.current_video_path_being_processed_for_ui = str(current_job.video_path)
            video_path_obj = current_job.video_path
            video_name = video_path_obj.stem
            output_folder = video_path_obj.with_suffix("")
            output_folder.mkdir(exist_ok=True, parents=True)

            self.logger.info(
                f"Starting Job {job_idx + 1}/{total_jobs}: Video {video_name} ({current_job.job_type.name})"
            )
            self.video_job_started.emit(str(video_path_obj), str(output_folder))
            self._log_gpu_memory(f"VideoJob {video_name}", "Before")

            # Effective config for this job (global overridden by job-specific)
            effective_job_config = {
                **self.global_config,
                **current_job.video_specific_config,
            }
            model_name_for_job = effective_job_config.get(
                "model_name", "Cutie"
            ).lower()  # Standardize

            job_successful_flag = False
            try:
                if current_job.job_type == JobType.VIDEO_SEGMENTS:
                    job_successful_flag = self._process_segmented_job(
                        current_job,
                        self.device,
                        effective_job_config,
                        model_name_for_job,
                        job_idx,
                        total_jobs,
                    )
                elif current_job.job_type == JobType.WHOLE_VIDEO:
                    job_successful_flag = self._process_whole_video_job(
                        current_job,
                        self.device,
                        effective_job_config,
                        model_name_for_job,
                        job_idx,
                        total_jobs,
                    )
                else:
                    self.logger.error(
                        f"Unknown job type: {current_job.job_type} for video {video_name}"
                    )
                    self.error.emit(f"Unknown job type for {video_name}")

            except Exception as e:  # Catch-all for safety during job execution
                self.logger.error(
                    f"Unhandled exception during job for {video_name}: {e}",
                    exc_info=True,
                )
                self.error.emit(f"Critical error processing {video_name}: {str(e)}")
            finally:
                # Signal GUI this video job is done
                self.video_job_finished.emit(str(video_path_obj))
                self.current_video_path_being_processed_for_ui = None
                # Log memory after job attempt
                self._log_gpu_memory(f"VideoJob {video_name}", "After")

            if job_successful_flag:
                jobs_completed_count += 1

            if not self._is_running:
                break  # Check after each job

        if self._is_running:
            self.finished.emit(
                f"All {jobs_completed_count}/{total_jobs} tracking jobs completed."
            )
        else:  # If stopped mid-way through all jobs
            if jobs_completed_count < total_jobs:
                self.finished.emit(
                    f"Tracking stopped by user. {jobs_completed_count}/{total_jobs} jobs attempted."
                )

        self._is_running = False  # Ensure state is final
        self.logger.info("TrackingWorker finished run method.")

    def _process_segmented_job(
        self,
        job: VideoProcessingJob,
        device: torch.device,
        job_config: Dict,
        model_name: str,
        job_idx: int,
        total_jobs: int,
    ) -> bool:
        video_name = job.video_path.stem
        output_folder = job.video_path.with_suffix("")  # For CSV conversion

        if not job.segments_data:  # Should be caught by _validate_jobs but double check
            self.logger.error(f"No segments defined for segmented job: {video_name}")
            return False

        # Converts dicts to TrackingSegment objects
        tracking_segments = job.get_tracking_segments()
        num_segments_in_job = len(tracking_segments)
        segments_processed_successfully_count = 0

        shared_cutie_engine: Optional[CutieEngine] = None
        if "cutie" in model_name:
            cutie_engine_config_overrides = {
                "mem_every": job_config.get("mem_every", 5),
                "max_mem_frames": job_config.get("t_max_value", 5),
            }
            try:
                shared_cutie_engine = CutieEngine(
                    cutie_config_overrides=cutie_engine_config_overrides,
                    device=device,
                )
            except Exception as exc:
                self.logger.error(
                    "Failed to initialize shared CutieEngine for %s: %s",
                    video_name,
                    exc,
                    exc_info=True,
                )
                self.error.emit(
                    f"Failed to initialize CutieEngine for {video_name}: {exc}"
                )
                return False

        for seg_idx, segment_obj in enumerate(tracking_segments):
            if not self._is_running:
                self.logger.info(
                    f"Segment processing for {video_name} interrupted by stop signal."
                )
                return False

            progress_overall = int(
                ((job_idx + (seg_idx / num_segments_in_job)) / total_jobs) * 100
            )
            self.progress.emit(
                progress_overall,
                f"Video {video_name}: Segment {seg_idx + 1}/{num_segments_in_job}...",
            )
            # Segment __str__ is informative
            self.logger.info(f"Starting: {segment_obj}")
            self._log_gpu_memory(f"Seg {video_name}-{seg_idx + 1}", "Before")

            segment_executor = None  # Scoped here
            try:
                if not segment_obj.is_annotation_valid():
                    msg = f"Skipping segment (ann:{segment_obj.annotated_frame}) for {video_name} due to missing annotation: {segment_obj.annotation_json_path}"
                    self.logger.warning(msg)
                    self.error.emit(msg)
                    continue

                # Handle "Cutie" or "cutie-base" etc.
                if "cutie" in model_name:
                    # Create executor with combined config for this segment
                    # Allow video config to override job_config for this segment
                    segment_config = {**job_config, **job.video_specific_config}

                    segment_executor = SegmentedCutieExecutor(
                        video_path_str=str(job.video_path),
                        segment_annotated_frame=segment_obj.annotated_frame,
                        segment_start_frame=segment_obj.segment_start_frame,
                        segment_end_frame=segment_obj.segment_end_frame,
                        processing_config=segment_config,  # Pass effective config
                        pred_worker=self,  # TrackingWorker itself
                        device=device,
                        cutie_engine=shared_cutie_engine,
                    )
                    status_message = segment_executor.process_segment()

                    if (
                        "Error" in status_message
                        or "not found" in status_message
                        or "Failed" in status_message
                    ):
                        self.error.emit(
                            f"Segment failed ({video_name}, ann:{segment_obj.annotated_frame}): {status_message}"
                        )
                    else:
                        segments_processed_successfully_count += 1
                        self.logger.info(
                            f"Segment ({video_name}, ann:{segment_obj.annotated_frame}) processed: {status_message}"
                        )
                else:
                    # Here you would dispatch to a different Segment Executor or use VideoProcessor for other models
                    # For now, this path is an error if not Cutie for segments.
                    self.logger.error(
                        f"Segmented tracking for model '{model_name}' not implemented. Only Cutie is supported via SegmentedCutieExecutor."
                    )
                    self.error.emit(
                        f"Unsupported model '{model_name}' for segmented tracking of {video_name}."
                    )
                    # To attempt with VideoProcessor (less efficient, needs VP to be segment-aware):
                    # success = self._run_segment_with_general_vp(job, segment_obj, device, segment_config, model_name)
                    # if success: segments_processed_successfully_count +=1

            except Exception as e:  # Catch errors from SegmentedCutieExecutor instantiation or process_segment
                self.error.emit(
                    f"Critical error in segment ({video_name}, ann:{segment_obj.annotated_frame}): {str(e)}"
                )
                self.logger.error(
                    f"Segment execution critical error for {segment_obj}", exc_info=True
                )
            finally:
                if segment_executor:
                    segment_executor.cleanup()
                self._log_gpu_memory(f"Seg {video_name}-{seg_idx + 1}", "After")

        if shared_cutie_engine is not None:
            try:
                shared_cutie_engine.cleanup()
            except Exception:
                # Best-effort cleanup; avoid failing the run due to teardown.
                pass

        if (
            segments_processed_successfully_count > 0
        ):  # If at least one segment was attempted/succeeded
            try:
                convert_json_to_csv(str(output_folder))
                self.logger.info(
                    f"CSV conversion run for {video_name} after processing {segments_processed_successfully_count} segments."
                )
            except Exception as e:
                self.error.emit(
                    f"CSV conversion for {video_name} (segmented) failed: {str(e)}"
                )

        # True if all segments in this job succeeded
        return segments_processed_successfully_count == num_segments_in_job

    def _process_whole_video_job(
        self,
        job: VideoProcessingJob,
        device: torch.device,
        job_config: Dict,
        model_name: str,
        job_idx: int,
        total_jobs: int,
    ) -> bool:
        video_name = job.video_path.stem
        output_folder = job.video_path.with_suffix("")
        self.progress.emit(
            int((job_idx / total_jobs) * 100), f"Video {video_name} (Whole)..."
        )
        self.logger.info(
            f"Processing WHOLE video job: {video_name} using model: {model_name}"
        )
        self._log_gpu_memory(f"WholeVideo {video_name}", "Before")

        vp_instance: Optional["VideoProcessor"] = None
        try:
            # Import lazily to avoid importing optional ONNX dependencies at GUI startup.
            from annolid.segmentation.SAM.edge_sam_bg import VideoProcessor

            annotated_frame_for_vp = job.initial_annotated_frame_for_whole_video
            if annotated_frame_for_vp is None:  # Try to find the "main" annotation
                json_files = find_manual_labeled_json_files(str(output_folder))
                if not json_files:
                    self.error.emit(
                        f"No initial JSON annotation found for WHOLE video {video_name}. Cannot track."
                    )
                    return False

                # Heuristic: use the latest modified, or one with most shapes, or a specific naming convention.
                # For now, using the one with the highest frame number among found JSONs.
                highest_frame_json = ""
                highest_frame_num = -1
                for json_file_str in json_files:
                    try:
                        num = get_frame_number_from_json(json_file_str)
                        if num > highest_frame_num:
                            highest_frame_num = num
                            highest_frame_json = json_file_str
                    except ValueError:
                        continue  # Invalid filename format

                if not highest_frame_json:
                    self.error.emit(
                        f"Could not determine a primary annotation JSON for {video_name}."
                    )
                    return False

                annotated_frame_for_vp = highest_frame_num
                # Validate this chosen JSON
                lf_check = LabelFile(str(highest_frame_json), is_video_frame=True)
                if not any(
                    s.get("shape_type") == "polygon" and len(s.get("points", [])) >= 3
                    for s in lf_check.shapes
                ):
                    self.error.emit(
                        f"Chosen JSON {Path(highest_frame_json).name} for {video_name} has no valid polygons."
                    )
                    return False
                self.logger.info(
                    f"Using annotation from frame {annotated_frame_for_vp} (JSON: {Path(highest_frame_json).name}) for whole video {video_name}."
                )

            # Create VideoProcessor instance
            vp_instance = VideoProcessor(
                video_path=str(job.video_path),
                # This tells VP which internal logic to use (Cutie, EdgeSAM, etc.)
                model_name=model_name,
                save_image_to_disk=False,
                device=device,
                **job_config,  # Pass all effective configs
            )
            vp_instance.set_pred_worker(self)
            total_video_frames = vp_instance.get_total_frames()

            # If model is Cutie, VideoProcessor's Cutie path needs reset.
            # The existing VideoProcessor.process_video_frames, when is_cutie=True,
            # calls process_video_with_cutite, which should ideally use a fresh Cutie processor
            # or one reset via is_new_segment.
            if "cutie" in model_name.lower() and hasattr(
                vp_instance, "reset_cutie_processor"
            ):
                vp_instance.reset_cutie_processor(
                    mem_every=job_config.get("mem_every", 5)
                )
                self.logger.info(
                    f"Reset Cutie processor within VideoProcessor for WHOLE video {video_name}"
                )

            # Call existing VideoProcessor.process_video_frames
            # Its `start_frame` must be the actual annotated frame.
            # Its `end_frame` is used for conceptual end; Cutie path inside VP might override.
            message = vp_instance.process_video_frames(
                start_frame=annotated_frame_for_vp,
                end_frame=total_video_frames - 1,
                is_cutie=("cutie" in model_name.lower()),
                is_new_segment=True,  # Treat as a new processing task for VP's internal state mgt
                **job_config,  # Pass other relevant parameters from job_config
            )

            if (
                "Error" in message
                or "not found" in message
                or "Failed" in message
                or "No valid polygon" in message
            ):
                self.error.emit(
                    f"WHOLE video processing for {video_name} failed: {message}"
                )
                return False

            convert_json_to_csv(str(output_folder))
            self.logger.info(
                f"WHOLE video processing for {video_name} completed successfully."
            )
            return True

        except Exception as e:
            self.error.emit(f"Error processing WHOLE video {video_name}: {str(e)}")
            self.logger.error(
                f"Error in _process_whole_video_job for {video_name}", exc_info=True
            )
            return False
        finally:
            if vp_instance:
                # Ensure VP instance is cleaned
                self._cleanup_general_processor(vp_instance, device)
