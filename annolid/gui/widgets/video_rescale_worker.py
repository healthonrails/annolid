from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from qtpy import QtCore

from annolid.utils.video_processing_reports import save_processing_summary
from annolid.utils.videos import compress_and_rescale_video


@dataclass(frozen=True)
class VideoRescaleJob:
    selected_videos: list[str]
    input_mode: str
    input_source: str
    input_folder: str
    output_folder: str
    scale_factor: float
    fps: float | None
    collect_only: bool
    rescale: bool
    apply_denoise: bool
    auto_contrast: bool
    auto_contrast_strength: float
    crop_params: tuple[int, int, int, int] | None
    per_video_overrides: dict[str, dict[str, object]] | None = None

    @property
    def is_single_input(self) -> bool:
        return self.input_mode == "single video"

    def effective_output_folder(self) -> str:
        if self.output_folder:
            return self.output_folder
        if self.input_folder:
            source_folder = Path(self.input_folder)
            return str(source_folder.with_name(f"{source_folder.name}_downsampled"))
        if self.selected_videos:
            source_video = Path(self.selected_videos[0])
            return str(source_video.with_name(f"{source_video.stem}_downsampled"))
        return ""


class VideoRescaleWorker(QtCore.QObject):
    progress = QtCore.Signal(int, str)
    finished = QtCore.Signal(dict)
    failed = QtCore.Signal(str)
    canceled = QtCore.Signal()

    def __init__(self, job: VideoRescaleJob, parent=None):
        super().__init__(parent)
        self.job = job
        self._cancel_requested = False

    @QtCore.Slot()
    def run(self) -> None:
        try:
            self._emit_progress(0, 100, "Starting")
            result = self._run_job()
        except RuntimeError as exc:
            if "cancelled" in str(exc).lower():
                self.canceled.emit()
                return
            self.failed.emit(str(exc))
            return
        except Exception as exc:  # pragma: no cover - exercised via signal path
            self.failed.emit(str(exc))
            return
        self.finished.emit(result)

    @QtCore.Slot()
    def cancel(self) -> None:
        self._cancel_requested = True

    def _is_cancelled(self) -> bool:
        return bool(self._cancel_requested)

    def _emit_progress(self, current: int, total: int, message: str) -> None:
        total_safe = max(int(total), 1)
        value = max(0, min(int((max(current, 0) / total_safe) * 100), 100))
        self.progress.emit(value, str(message))

    def _run_job(self) -> dict:
        job = self.job
        command_log: dict[str, str] = {}

        if job.collect_only:
            self._emit_progress(0, 1, "Collecting metadata")
            save_processing_summary(
                job.input_folder,
                video_paths=job.selected_videos if job.is_single_input else None,
                input_mode=job.input_mode,
                input_source=job.input_source,
                progress_callback=self._emit_progress,
                cancel_callback=self._is_cancelled,
            )
            self._emit_progress(100, 1, "Metadata complete")

        if job.rescale:
            output_folder = job.effective_output_folder()
            if not output_folder:
                raise RuntimeError("Please select a valid output folder.")

            self._emit_progress(0, len(job.selected_videos), "Starting downsample")
            command_log = compress_and_rescale_video(
                job.input_folder,
                output_folder,
                job.scale_factor,
                input_video_path=job.selected_videos[0]
                if job.is_single_input
                else None,
                fps=job.fps,
                apply_denoise=job.apply_denoise,
                auto_contrast=job.auto_contrast,
                auto_contrast_strength=job.auto_contrast_strength,
                crop_x=job.crop_params[0] if job.crop_params else None,
                crop_y=job.crop_params[1] if job.crop_params else None,
                crop_width=job.crop_params[2] if job.crop_params else None,
                crop_height=job.crop_params[3] if job.crop_params else None,
                per_video_overrides=job.per_video_overrides,
                progress_callback=self._emit_progress,
                cancel_callback=self._is_cancelled,
            )
            save_processing_summary(
                output_folder,
                video_paths=[
                    str(Path(output_folder) / output_name)
                    for output_name in sorted(command_log)
                ]
                or None,
                input_mode=job.input_mode,
                input_source=job.input_source,
                output_folder=output_folder,
                scale_factor=job.scale_factor,
                fps=job.fps,
                apply_denoise=job.apply_denoise,
                auto_contrast=job.auto_contrast,
                auto_contrast_strength=job.auto_contrast_strength,
                crop_params=job.crop_params,
                per_video_overrides=job.per_video_overrides,
                command_log=command_log,
                progress_callback=self._emit_progress,
                cancel_callback=self._is_cancelled,
            )

        if job.rescale:
            success_count = len(command_log)
            failed_count = max(0, len(job.selected_videos) - success_count)
        else:
            success_count = len(job.selected_videos)
            failed_count = 0
        effective_output_folder = job.effective_output_folder() if job.rescale else ""
        summary_lines = [
            "Video processing complete.",
            "",
            "Downsample Parameters:",
            f"- Input mode: {job.input_mode}",
            f"- Input source: {job.input_source}",
        ]
        if job.rescale:
            summary_lines.insert(
                3,
                f"- Processed videos: {len(job.selected_videos)}",
            )
        if effective_output_folder:
            summary_lines.append(f"- Output folder: {effective_output_folder}")
        summary_lines.extend(
            [
                f"- Scale factor: {job.scale_factor}",
                f"- FPS: {job.fps if job.fps is not None else 'original per-video FPS'}",
                f"- Apply denoise: {job.apply_denoise}",
                f"- Auto contrast: {job.auto_contrast}",
            ]
        )
        if job.auto_contrast:
            summary_lines.append(
                f"- Auto contrast strength: {job.auto_contrast_strength}"
            )
        if job.crop_params is not None:
            summary_lines.append(
                "Crop region: "
                f"x={job.crop_params[0]}, y={job.crop_params[1]}, "
                f"width={job.crop_params[2]}, height={job.crop_params[3]}"
            )
        override_count = len(job.per_video_overrides or {})
        if override_count > 0:
            summary_lines.append(f"- Per-video review selections: {override_count}")
        summary_lines.extend(
            [
                "",
                f"Successful: {success_count}",
                f"Failed: {failed_count}",
            ]
        )
        summary = "\n".join(summary_lines)
        self._emit_progress(100, 100, "Complete")
        return {
            "summary": summary,
            "command_log": command_log,
            "success_count": success_count,
            "failed_count": failed_count,
            "output_folder": effective_output_folder,
            "input_mode": job.input_mode,
            "input_source": job.input_source,
        }
