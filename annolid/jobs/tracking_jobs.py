from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import List, Optional, Dict
import os


class JobType(Enum):
    """Defines the type of tracking job."""
    WHOLE_VIDEO = auto()
    VIDEO_SEGMENTS = auto()


@dataclass
class TrackingSegment:
    """
    Represents a single segment to be tracked within a video.
    This is what the SegmentEditorDialog will primarily produce for a video.
    """
    video_path: Path             # Path to the video file this segment belongs to
    fps: float                   # Frames per second of the video
    # The frame number that contains the user's initial annotation (mask)
    annotated_frame: int
    # The actual video frame number where tracking for this segment should begin
    segment_start_frame: int
    # The actual video frame number where tracking for this segment should end
    segment_end_frame: int
    unique_id: str = field(default_factory=lambda: os.urandom(
        4).hex())  # For UI or logging

    @property
    def annotation_json_path(self) -> Path:
        """Constructs the expected path to the annotation JSON file for this segment."""
        video_folder = self.video_path.with_suffix('')
        return video_folder / f"{self.video_path.stem}_{self.annotated_frame:09d}.json"

    @property
    def duration_frames(self) -> int:
        """Calculates the duration of the segment in frames."""
        return self.segment_end_frame - self.segment_start_frame + 1

    @property
    def duration_sec(self) -> float:
        """Calculates the duration of the segment in seconds."""
        if self.fps > 0:
            return self.duration_frames / self.fps
        return 0.0

    def is_annotation_valid(self) -> bool:
        """Checks if the required annotation JSON file exists."""
        return self.annotation_json_path.exists()

    def __str__(self) -> str:
        return (f"Segment(Video: {self.video_path.name}, AnnFrame: {self.annotated_frame}, "
                f"TrackFrames: [{self.segment_start_frame}-{self.segment_end_frame}], "
                f"Duration: {self.duration_sec:.2f}s, ValidAnn: {self.is_annotation_valid()})")

    @staticmethod
    def _format_seconds(total_seconds: float) -> str:
        """Helper to format seconds into HH:MM:SS."""
        if total_seconds < 0:
            total_seconds = 0  # Handle potential negative from bad input
        hours = int(total_seconds // 3600)
        minutes = int((total_seconds % 3600) // 60)
        seconds = int(total_seconds % 60)
        return f"{hours:02}:{minutes:02}:{seconds:02}"

    def start_time_str(self) -> str:
        """Returns the start time of the segment as HH:MM:SS string."""
        if self.fps > 0:
            return self._format_seconds(self.segment_start_frame / self.fps)
        return "N/A"

    def end_time_str(self) -> str:
        """Returns the end time of the segment as HH:MM:SS string."""
        if self.fps > 0:
            # +1 because end_frame is inclusive
            return self._format_seconds((self.segment_end_frame + 1) / self.fps)
        return "N/A"

    def to_dict(self) -> Dict:
        """Serializes to a dictionary, useful for saving or passing around."""
        return {
            "video_path": str(self.video_path),  # Store path as string
            "fps": self.fps,
            "annotated_frame": self.annotated_frame,
            "segment_start_frame": self.segment_start_frame,
            "segment_end_frame": self.segment_end_frame,
            "unique_id": self.unique_id
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'TrackingSegment':
        """Deserializes from a dictionary."""
        return cls(
            video_path=Path(data["video_path"]),
            fps=data["fps"],
            annotated_frame=data["annotated_frame"],
            segment_start_frame=data["segment_start_frame"],
            segment_end_frame=data["segment_end_frame"],
            # Handle older data without unique_id
            unique_id=data.get("unique_id", os.urandom(4).hex())
        )


@dataclass
class VideoProcessingJob:
    """
    Represents a complete processing job for a single video, which can be
    either tracking the whole video or a list of specific segments.
    This is what the TrackingWorker will primarily consume.
    """
    video_path: Path
    job_type: JobType
    # Optional: Frame number of the primary annotation if tracking the whole video.
    # If None, the worker/VideoProcessor might try to find the most recent annotation.
    initial_annotated_frame_for_whole_video: Optional[int] = None
    # List of segments to process; only used if job_type is VIDEO_SEGMENTS.
    # These are dictionaries that can be converted to TrackingSegment objects or used directly.
    segments_data: Optional[List[Dict]] = field(default_factory=list)
    # Optional: Video-specific configurations that might override global ones.
    video_specific_config: Optional[Dict] = field(default_factory=dict)
    # Optional: To store computed FPS if not passed explicitly with segments
    fps: Optional[float] = None  # Can be pre-filled by the job creator

    def __post_init__(self):
        """Basic validation after initialization."""
        if self.job_type == JobType.VIDEO_SEGMENTS and not self.segments_data:
            raise ValueError(
                "JobType.VIDEO_SEGMENTS requires a non-empty list of segments_data.")
        if self.job_type == JobType.WHOLE_VIDEO and self.segments_data:
            # logger.warning("JobType.WHOLE_VIDEO should not have segments_data defined; ignoring.")
            # Or raise ValueError - for now, just a note.
            pass
        if not self.video_path.is_file():
            raise FileNotFoundError(
                f"Video path does not exist or is not a file: {self.video_path}")

    def get_tracking_segments(self) -> List[TrackingSegment]:
        """
        Converts segment data dictionaries to TrackingSegment objects.
        Requires self.fps to be set.
        """
        if self.job_type != JobType.VIDEO_SEGMENTS or not self.segments_data:
            return []
        if self.fps is None:
            # Attempt to get FPS if not set, or raise error
            # This is a good place for lazy FPS loading if needed by the worker.
            # For now, assume it should be set by the job creator (e.g., AnnolidWindow).
            raise ValueError(
                "FPS must be set on VideoProcessingJob to convert segment data.")

        return [
            TrackingSegment(
                video_path=self.video_path,
                fps=self.fps,
                annotated_frame=s_data['annotated_frame'],
                segment_start_frame=s_data['segment_start_frame'],
                segment_end_frame=s_data['segment_end_frame']
            ) for s_data in self.segments_data
        ]


# Example Usage (for testing or understanding)
if __name__ == "__main__":
    # Create a dummy video file for Path validation
    dummy_video_file = Path("dummy_video.mp4")
    if not dummy_video_file.exists():
        dummy_video_file.touch()

    # Segmented Job Example
    segment1_data = {
        "annotated_frame": 10,
        "segment_start_frame": 10,
        "segment_end_frame": 100
    }
    segment2_data = {
        "annotated_frame": 150,
        "segment_start_frame": 150,
        "segment_end_frame": 250
    }
    segmented_job = VideoProcessingJob(
        video_path=dummy_video_file,
        job_type=JobType.VIDEO_SEGMENTS,
        segments_data=[segment1_data, segment2_data],
        fps=30.0  # FPS is crucial for TrackingSegment calculations
    )
    print("Segmented Job:")
    print(segmented_job)
    for seg_obj in segmented_job.get_tracking_segments():
        print(f"  - {seg_obj}")
        # To test annotation path, create a dummy structure:
        # (dummy_video_file.with_suffix('')).mkdir(exist_ok=True)
        # (dummy_video_file.with_suffix('') / f"{dummy_video_file.stem}_{seg_obj.annotated_frame:09d}.json").touch()
        # print(f"    Annotation valid: {seg_obj.is_annotation_valid()}")

    # Whole Video Job Example
    whole_video_job = VideoProcessingJob(
        video_path=dummy_video_file,
        job_type=JobType.WHOLE_VIDEO,
        initial_annotated_frame_for_whole_video=5  # Example: user specified a start
    )
    print("\nWhole Video Job:")
    print(whole_video_job)

    # Clean up dummy file
    if dummy_video_file.exists():
        dummy_video_file.unlink()
    if (dummy_video_file.with_suffix('')).exists():
        import shutil
        shutil.rmtree(dummy_video_file.with_suffix(''))
