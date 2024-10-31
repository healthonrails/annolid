import csv
import json
import os
import random

# Constants for file paths and split ratio
BASE_FOLDER = "/content/behaivor_videos"
OUTPUT_VIDEO_FOLDER = "behavior_videos_clips"
TRAIN_JSONL_PATH = "train_video_annotations.jsonl"
TEST_JSONL_PATH = "test_video_annotations.jsonl"
TRAIN_SPLIT_RATIO = 0.8

# Ensure output folder exists
os.makedirs(OUTPUT_VIDEO_FOLDER, exist_ok=True)


def extract_video_segment(video_file_path, start_time, end_time, output_path):
    from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
    """Extracts a video segment using ffmpeg."""
    ffmpeg_extract_subclip(video_file_path, start_time,
                           end_time, targetname=output_path)


def create_annotation_entry(behavior, video_segment_path):
    """Creates a JSONL entry for a video segment."""
    return {
        "query": "<video> What is the behavior in the video?",
        "response": behavior,
        "videos": [video_segment_path]
    }


def process_csv_and_extract_segments(csv_path, video_file_path):
    """Processes a CSV file, extracts video segments, and returns annotation entries."""
    entries = []
    with open(csv_path, 'r') as csv_file:
        reader = csv.DictReader(csv_file)
        start_events = []
        stop_events = []

        for row in reader:
            recording_time = float(row["Recording time"])
            behavior = row["Behavior"]
            event = row["Event"]

            if event == "state start":
                start_events.append(
                    {"time": recording_time, "behavior": behavior})
            elif event == "state stop":
                stop_events.append(
                    {"time": recording_time, "behavior": behavior})

        for start_event in start_events:
            start_time = start_event["time"]
            behavior = start_event["behavior"]

            matching_stop_event = next((stop_event for stop_event in stop_events
                                       if stop_event["behavior"] == behavior and stop_event["time"] > start_time), None)

            if matching_stop_event:
                end_time = matching_stop_event["time"]
                # Avoid reusing stop events
                stop_events.remove(matching_stop_event)

                video_segment_path = f"{OUTPUT_VIDEO_FOLDER}/{behavior}_{start_time}-{end_time}.mp4"
                extract_video_segment(
                    video_file_path, start_time, end_time, video_segment_path)
                entries.append(create_annotation_entry(
                    behavior, video_segment_path))

    return entries


def process_dataset():
    """Processes the entire dataset and creates train/test JSONL files."""
    train_entries = []
    test_entries = []

    for subdir in os.listdir(BASE_FOLDER):
        subdir_path = os.path.join(BASE_FOLDER, subdir)
        if os.path.isdir(subdir_path):
            for file in os.listdir(subdir_path):
                if file.endswith('.csv'):
                    csv_path = os.path.join(subdir_path, file)
                    video_file_path = csv_path.replace('.csv', '.mpg')
                    if os.path.exists(video_file_path):
                        entries = process_csv_and_extract_segments(
                            csv_path, video_file_path)
                        random.shuffle(entries)
                        split_index = int(len(entries) * TRAIN_SPLIT_RATIO)
                        train_entries.extend(entries[:split_index])
                        test_entries.extend(entries[split_index:])

    # Write JSONL files
    with open(TRAIN_JSONL_PATH, 'w') as f:
        for entry in train_entries:
            f.write(json.dumps(entry) + "\n")

    with open(TEST_JSONL_PATH, 'w') as f:
        for entry in test_entries:
            f.write(json.dumps(entry) + "\n")

    print("Conversion complete. Training and testing datasets created.")
