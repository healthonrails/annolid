import os
import random
import json
import itertools
from collections import defaultdict
from typing import List, Dict, Tuple
from dataclasses import dataclass
import logging
from hydra import initialize, compose
from hydra.core.config_store import ConfigStore
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
from moviepy.editor import VideoFileClip
import pandas as pd

# --- Configuration ---


@dataclass
class FilePaths:
    base_folder: str = "behavior_videos"
    output_video_folder: str = "behavior_video_clips"
    train_jsonl_path: str = "train_video_annotations.jsonl"
    test_jsonl_path: str = "test_video_annotations.jsonl"


@dataclass
class ProcessingConfig:
    train_split_ratio: float = 0.97
    gap_duration: float = 5.0  # seconds


@dataclass
class Config:
    file_paths: FilePaths = FilePaths()
    processing: ProcessingConfig = ProcessingConfig()


cs = ConfigStore.instance()
cs.store(name="config", node=Config)

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# --- Data Models ---


@dataclass
class BehaviorEvent:
    time: float
    behavior: str
    event: str


@dataclass
class AnnotationEntry:
    query: str
    response: str
    videos: List[str]


# --- Utility Functions ---


def extract_video_segment(
    video_file_path: str, start_time: float, end_time: float, output_path: str
) -> None:
    """Extracts a video segment from a file between start and end times."""
    try:
        ffmpeg_extract_subclip(
            video_file_path, start_time, end_time, targetname=output_path
        )
    except Exception as e:
        logging.error(
            f"Error extracting segment from {video_file_path} ({start_time}-{end_time}): {e}"
        )
        raise  # Re-raise the exception after logging


def create_annotation_entry(
    behavior: str,
    video_segment_path: str,
    prompt: str = "<video> What is the behavior in the video?",
) -> AnnotationEntry:
    """Creates an AnnotationEntry object for a video segment."""
    return AnnotationEntry(query=prompt, response=behavior, videos=[video_segment_path])


# --- Data Parsing ---


def parse_behavior_events(csv_path: str) -> pd.DataFrame:
    """Parses behavior events from a CSV file using pandas."""
    try:
        df = pd.read_csv(csv_path)
        if not all(
            col in df.columns for col in ["Recording time", "Behavior", "Event"]
        ):
            raise ValueError(f"CSV file {csv_path} is missing required columns.")
        return df
    except FileNotFoundError:
        logging.error(f"CSV file not found: {csv_path}")
        raise
    except ValueError as e:
        logging.error(f"Error parsing CSV file {csv_path}: {e}")
        raise


# --- Gap Finding ---


def find_gaps(
    df: pd.DataFrame, video_duration: float, gap_duration: float
) -> List[Tuple[float, float]]:
    """Finds gaps between behavior events to sample as 'Others'."""
    gaps: List[Tuple[float, float]] = []
    last_end_time = 0.0

    stop_events = df[df["Event"] == "state stop"].sort_values(by="Recording time")

    for index, start_row in (
        df[df["Event"] == "state start"].sort_values(by="Recording time").iterrows()
    ):
        start_time = start_row["Recording time"]

        if start_time - last_end_time >= gap_duration:
            gaps.append((last_end_time, start_time))

        # Find the corresponding stop event
        corresponding_stop = (
            stop_events[
                (stop_events["Behavior"] == start_row["Behavior"])
                & (stop_events["Recording time"] > start_time)
            ]
            .sort_values(by="Recording time")
            .iloc[:1]
        )

        if not corresponding_stop.empty:
            last_end_time = corresponding_stop["Recording time"].iloc[0]

    # Check for a gap after the last behavior
    if video_duration - last_end_time >= gap_duration:
        gaps.append((last_end_time, video_duration))

    return gaps


# --- Segment Sampling ---


def sample_segments_from_gaps(
    gaps: List[Tuple[float, float]],
    video_file_path: str,
    video_name: str,
    gap_duration: float,
    behavior_label: str = "Others",
) -> List[AnnotationEntry]:
    """Samples segments from the gaps and returns annotation entries for 'Others' behavior."""
    entries: List[AnnotationEntry] = []
    for start, end in gaps:
        num_segments = int((end - start) // gap_duration)
        for i in range(num_segments):
            segment_start = start + i * gap_duration
            segment_end = segment_start + gap_duration
            segment_path = os.path.join(
                Config.file_paths.output_video_folder,
                f"{video_name}_other_{segment_start:.2f}-{segment_end:.2f}.mp4",
            )
            extract_video_segment(
                video_file_path, segment_start, segment_end, segment_path
            )
            entries.append(create_annotation_entry(behavior_label, segment_path))
    return entries


def sample_limited_segments_from_gaps(
    gaps: List[Tuple[float, float]],
    video_file_path: str,
    video_name: str,
    gap_duration: float,
    max_count: int,
    behavior_label: str = "Others",
) -> List[AnnotationEntry]:
    """Samples a limited number of segments from the gaps."""
    entries: List[AnnotationEntry] = []
    for start, end in gaps:
        num_possible_segments = int((end - start) // gap_duration)
        num_to_sample = min(num_possible_segments, max_count - len(entries))
        if num_to_sample > 0:
            segment_starts = [
                start + i * gap_duration for i in range(num_possible_segments)
            ]
            sampled_starts = random.sample(segment_starts, num_to_sample)
            for segment_start in sorted(sampled_starts):
                segment_end = segment_start + gap_duration
                segment_path = os.path.join(
                    Config.file_paths.output_video_folder,
                    f"{video_name}_other_{segment_start:.2f}-{segment_end:.2f}.mp4",
                )
                extract_video_segment(
                    video_file_path, segment_start, segment_end, segment_path
                )
                entries.append(create_annotation_entry(behavior_label, segment_path))
                if len(entries) >= max_count:
                    return entries
    return entries


# --- Video Processing ---


def process_video_file(csv_path: str, video_file_path: str) -> List[AnnotationEntry]:
    """Processes a single video file, extracting behavior segments and 'Others'."""
    try:
        df_events = parse_behavior_events(csv_path)
        video_name = os.path.splitext(os.path.basename(video_file_path))[0].replace(
            " ", "_"
        )
        output_video_folder = Config.file_paths.output_video_folder
        os.makedirs(output_video_folder, exist_ok=True)

        with VideoFileClip(video_file_path) as video:
            video_duration = video.duration

        labeled_entries: List[AnnotationEntry] = []
        behavior_counts: Dict[str, int] = defaultdict(int)

        stop_events = df_events[df_events["Event"] == "state stop"].set_index(
            ["Behavior", "Recording time"]
        )

        for _, start_row in df_events[df_events["Event"] == "state start"].iterrows():
            start_time = start_row["Recording time"]
            behavior = start_row["Behavior"]
            try:
                relevant_stops = stop_events.loc[behavior]
                earliest_stop = (
                    relevant_stops[relevant_stops.index > start_time]
                    .sort_index()
                    .iloc[:1]
                )

                if not earliest_stop.empty:
                    end_time = earliest_stop.index.item()
                    segment_path = os.path.join(
                        output_video_folder,
                        f"{video_name}_{behavior.replace(' ', '_')}_{start_time:.2f}-{end_time:.2f}.mp4",
                    )
                    extract_video_segment(
                        video_file_path, start_time, end_time, segment_path
                    )
                    labeled_entries.append(
                        create_annotation_entry(behavior, segment_path)
                    )
                    behavior_counts[behavior] += 1
                else:
                    logging.warning(
                        f"No matching stop event found for behavior '{behavior}' at {start_time} in {csv_path}"
                    )
            except KeyError:
                logging.warning(
                    f"No stop events found for behavior '{behavior}' in {csv_path}"
                )

        max_behavior_count = max(behavior_counts.values(), default=0)
        gaps = find_gaps(df_events, video_duration, Config.processing.gap_duration)
        other_entries = sample_limited_segments_from_gaps(
            gaps,
            video_file_path,
            video_name,
            Config.processing.gap_duration,
            max_behavior_count,
        )
        return labeled_entries + other_entries

    except FileNotFoundError:
        logging.error(
            f"Video or CSV file not found for {video_file_path} or {csv_path}"
        )
        return []
    except Exception as e:
        logging.error(f"Error processing {video_file_path}: {e}")
        return []


# --- Data Splitting ---


def stratified_interleaved_split_and_save_annotations(
    entries: List[AnnotationEntry], train_path: str, test_path: str, train_ratio: float
) -> None:
    """Splits annotations into stratified, interleaved train and test sets."""
    behavior_groups = defaultdict(list)
    for entry in entries:
        behavior_groups[entry.response].append(entry)

    train_entries_grouped: List[List[AnnotationEntry]] = []
    test_entries_grouped: List[List[AnnotationEntry]] = []

    for group in behavior_groups.values():
        random.shuffle(group)
        split_index = int(len(group) * train_ratio)
        train_entries_grouped.append(group[:split_index])
        test_entries_grouped.append(group[split_index:])

    interleaved_train = list(
        itertools.chain.from_iterable(itertools.zip_longest(*train_entries_grouped))
    )
    interleaved_test = list(
        itertools.chain.from_iterable(itertools.zip_longest(*test_entries_grouped))
    )

    train_entries = [entry for entry in interleaved_train if entry is not None]
    test_entries = [entry for entry in interleaved_test if entry is not None]

    def write_jsonl(entries: List[AnnotationEntry], path: str) -> None:
        with open(path, "w") as f:
            for entry in entries:
                json.dump(entry.__dict__, f)  # Serialize dataclass to dict
                f.write("\n")

    write_jsonl(train_entries, train_path)
    write_jsonl(test_entries, test_path)


# --- Main Execution ---


def process_dataset() -> None:
    """Processes the entire dataset and creates train/test JSONL files."""
    all_entries: List[AnnotationEntry] = []
    base_folder = Config.file_paths.base_folder
    train_split_ratio = Config.processing.train_split_ratio

    for subdir in os.listdir(base_folder):
        subdir_path = os.path.join(base_folder, subdir)
        if os.path.isdir(subdir_path):
            for file in os.listdir(subdir_path):
                if file.endswith(".csv"):
                    csv_path = os.path.join(subdir_path, file)
                    video_file_path = csv_path.replace(".csv", ".mpg")
                    if os.path.exists(video_file_path):
                        entries = process_video_file(csv_path, video_file_path)
                        all_entries.extend(entries)

    stratified_interleaved_split_and_save_annotations(
        all_entries,
        Config.file_paths.train_jsonl_path,
        Config.file_paths.test_jsonl_path,
        train_split_ratio,
    )
    logging.info(
        "Conversion complete. Stratified interleaved training and testing datasets created."
    )


if __name__ == "__main__":
    initialize(config_path=".")
    Config.file_paths = compose(config_name="config").file_paths
    Config.processing = compose(config_name="config").processing
    process_dataset()
