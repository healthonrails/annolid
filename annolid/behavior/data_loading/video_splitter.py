import os
import random
import csv
import json
import itertools
from collections import defaultdict

# Constants for file paths and split ratio
BASE_FOLDER = "/content/behaivor_videos"
OUTPUT_VIDEO_FOLDER = "behavior_video_clips"
TRAIN_JSONL_PATH = "train_video_annotations.jsonl"
TEST_JSONL_PATH = "test_video_annotations.jsonl"
TRAIN_SPLIT_RATIO = 0.97

# Ensure output folder exists
os.makedirs(OUTPUT_VIDEO_FOLDER, exist_ok=True)


def extract_video_segment(video_file_path, start_time, end_time, output_path):
    """Extracts a video segment from a file between start and end times."""
    from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip

    ffmpeg_extract_subclip(
        video_file_path, start_time, end_time, targetname=output_path
    )


def create_annotation_entry(
    behavior, video_segment_path, prompt="<video> What is the behavior in the video?"
):
    """Creates a JSONL entry for a video segment."""
    return {"query": prompt, "response": behavior, "videos": [video_segment_path]}


def parse_behavior_events(csv_path):
    """Parses start and stop events from a behavior CSV file."""
    start_events, stop_events = [], []
    with open(csv_path, "r") as csv_file:
        reader = csv.DictReader(csv_file)
        for row in reader:
            time = float(row["Recording time"])
            behavior = row["Behavior"]
            event = row["Event"]
            if event == "state start":
                start_events.append({"time": time, "behavior": behavior})
            elif event == "state stop":
                stop_events.append({"time": time, "behavior": behavior})
    return start_events, stop_events


def find_gaps(start_events, stop_events, video_duration, gap_duration):
    """Finds gaps between behavior events to sample as 'Others'."""
    gaps = []
    last_end_time = 0

    for start_event in start_events:
        start_time = start_event["time"]
        if start_time - last_end_time >= gap_duration:
            gaps.append((last_end_time, start_time))

        matching_stop = next(
            (
                s
                for s in stop_events
                if s["behavior"] == start_event["behavior"] and s["time"] > start_time
            ),
            None,
        )
        if matching_stop:
            last_end_time = matching_stop["time"]
            stop_events.remove(matching_stop)

    if video_duration - last_end_time >= gap_duration:
        gaps.append((last_end_time, video_duration))

    return gaps


def sample_segments_from_gaps(
    gaps, video_file_path, video_name, gap_duration, behavior_label="Others"
):
    """Samples segments from the gaps and returns annotation entries for 'Others' behavior."""
    entries = []
    for start, end in gaps:
        num_segments = int((end - start) // gap_duration)
        for i in range(num_segments):
            segment_start = start + i * gap_duration
            segment_end = segment_start + gap_duration
            # Format the output file path with the video name and replace spaces with underscores
            segment_path = f"{OUTPUT_VIDEO_FOLDER}/{video_name}_other_{segment_start}-{segment_end}.mp4"
            extract_video_segment(
                video_file_path, segment_start, segment_end, segment_path
            )
            entries.append(create_annotation_entry(behavior_label, segment_path))
    return entries


def sample_limited_segments_from_gaps(
    gaps, video_file_path, video_name, gap_duration, max_count, behavior_label="Others"
):
    """Samples segments from the gaps with a limit on the number of segments for 'Others' behavior."""
    entries = []
    for start, end in gaps:
        num_segments = int((end - start) // gap_duration)
        for i in range(min(num_segments, max_count - len(entries))):
            segment_start = start + i * gap_duration
            segment_end = segment_start + gap_duration
            segment_path = f"{OUTPUT_VIDEO_FOLDER}/{video_name}_other_{segment_start}-{segment_end}.mp4"
            extract_video_segment(
                video_file_path, segment_start, segment_end, segment_path
            )
            entries.append(create_annotation_entry(behavior_label, segment_path))
            if len(entries) >= max_count:  # Stop if we reach the max limit
                break
        if len(entries) >= max_count:
            break
    return entries


def process_video_file(csv_path, video_file_path, gap_duration=5):
    """Processes a video file by extracting labeled and limited 'Others' segments."""
    from moviepy.editor import VideoFileClip

    start_events, stop_events = parse_behavior_events(csv_path)

    # Get a clean video name without spaces
    video_name = os.path.splitext(os.path.basename(video_file_path))[0].replace(
        " ", "_"
    )

    # Extract labeled segments and count occurrences of each behavior
    labeled_entries = []
    behavior_counts = defaultdict(int)
    for start_event in start_events:
        start_time = start_event["time"]
        behavior = start_event["behavior"]
        matching_stop = next(
            (
                stop
                for stop in stop_events
                if stop["behavior"] == behavior and stop["time"] > start_time
            ),
            None,
        )

        if matching_stop:
            end_time = matching_stop["time"]
            stop_events.remove(matching_stop)
            segment_path = f"{OUTPUT_VIDEO_FOLDER}/{video_name}_{behavior.replace(' ', '_')}_{start_time}-{end_time}.mp4"
            extract_video_segment(video_file_path, start_time, end_time, segment_path)
            labeled_entries.append(create_annotation_entry(behavior, segment_path))
            behavior_counts[behavior] += 1

    # Determine max count among labeled behaviors
    max_behavior_count = max(behavior_counts.values(), default=0)

    # Extract "Others" segments, limited by max_behavior_count
    with VideoFileClip(video_file_path) as video:
        video_duration = video.duration
    gaps = find_gaps(start_events, stop_events, video_duration, gap_duration)
    other_entries = sample_limited_segments_from_gaps(
        gaps, video_file_path, video_name, gap_duration, max_behavior_count
    )

    return labeled_entries + other_entries


def split_and_save_annotations(entries, train_path, test_path, train_ratio=0.97):
    """Splits annotations into train and test sets and saves them to JSONL files."""
    random.shuffle(entries)
    split_index = int(len(entries) * train_ratio)
    train_entries, test_entries = entries[:split_index], entries[split_index:]

    with open(train_path, "w") as f:
        for entry in train_entries:
            f.write(json.dumps(entry) + "\n")

    with open(test_path, "w") as f:
        for entry in test_entries:
            f.write(json.dumps(entry) + "\n")


def stratified_split_and_save_annotations(
    entries, train_path, test_path, train_ratio=0.97
):
    """Splits annotations into stratified train and test sets by behavior and saves them to JSONL files."""
    # Group entries by behavior
    behavior_groups = defaultdict(list)
    for entry in entries:
        # Assuming the 'response' field contains the behavior label
        behavior = entry["response"]
        behavior_groups[behavior].append(entry)

    # Stratified sampling for train and test sets
    train_entries, test_entries = [], []
    for behavior, group_entries in behavior_groups.items():
        random.shuffle(group_entries)
        split_index = int(len(group_entries) * train_ratio)
        train_entries.extend(group_entries[:split_index])
        test_entries.extend(group_entries[split_index:])

    # Save the stratified entries to JSONL files
    with open(train_path, "w") as f:
        for entry in train_entries:
            f.write(json.dumps(entry) + "\n")

    with open(test_path, "w") as f:
        for entry in test_entries:
            f.write(json.dumps(entry) + "\n")


def stratified_interleaved_split_and_save_annotations(
    entries, train_path, test_path, train_ratio=0.97
):
    """Splits annotations into stratified, interleaved train and test sets by behavior and saves them to JSONL files."""
    # Group entries by behavior
    behavior_groups = defaultdict(list)
    for entry in entries:
        # Assuming the 'response' field contains the behavior label
        behavior = entry["response"]
        behavior_groups[behavior].append(entry)

    # Initialize lists for train and test entries
    train_entries, test_entries = [], []

    # Stratified sampling with interleaving
    for behavior, group_entries in behavior_groups.items():
        random.shuffle(group_entries)  # Shuffle within each behavior group
        split_index = int(len(group_entries) * train_ratio)
        train_entries.append(group_entries[:split_index])
        test_entries.append(group_entries[split_index:])

    # Interleave entries from each behavior group
    interleaved_train = list(
        itertools.chain.from_iterable(itertools.zip_longest(*train_entries))
    )
    interleaved_test = list(
        itertools.chain.from_iterable(itertools.zip_longest(*test_entries))
    )

    # Remove None entries introduced by zip_longest
    interleaved_train = [entry for entry in interleaved_train if entry is not None]
    interleaved_test = [entry for entry in interleaved_test if entry is not None]

    # Save the interleaved entries to JSONL files
    with open(train_path, "w") as f:
        for entry in interleaved_train:
            f.write(json.dumps(entry) + "\n")

    with open(test_path, "w") as f:
        for entry in interleaved_test:
            f.write(json.dumps(entry) + "\n")


def process_dataset():
    """Processes the entire dataset and creates stratified interleaved train/test JSONL files."""
    all_entries = []
    for subdir in os.listdir(BASE_FOLDER):
        subdir_path = os.path.join(BASE_FOLDER, subdir)
        if os.path.isdir(subdir_path):
            for file in os.listdir(subdir_path):
                if file.endswith(".csv"):
                    csv_path = os.path.join(subdir_path, file)
                    video_file_path = csv_path.replace(".csv", ".mpg")
                    if os.path.exists(video_file_path):
                        entries = process_video_file(csv_path, video_file_path)
                        all_entries.extend(entries)

    # Use the interleaved function to split and save annotations
    stratified_interleaved_split_and_save_annotations(
        all_entries, TRAIN_JSONL_PATH, TEST_JSONL_PATH, TRAIN_SPLIT_RATIO
    )
    print(
        "Conversion complete. Stratified interleaved training and testing datasets created."
    )
