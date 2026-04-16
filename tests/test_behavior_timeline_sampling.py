import pytest

from annolid.behavior.timeline_sampling import (
    compute_timeline_points,
    format_hhmmss,
    timeline_intervals_to_timestamp_rows,
    timestamp_rows_to_timeline_intervals,
)
from annolid.behavior import prompting as behavior_prompting


def test_compute_timeline_points_inclusive_endpoints():
    points = compute_timeline_points(
        start_seconds=0.0,
        end_seconds=5.0,
        step_seconds=1,
        fps=30.0,
        total_frames=300,
    )
    assert [idx for idx, _ in points] == [0, 30, 60, 90, 120, 150]
    assert [t for _, t in points] == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]


def test_format_hhmmss():
    assert format_hhmmss(0) == "00:00:00"
    assert format_hhmmss(5) == "00:00:05"
    assert format_hhmmss(65) == "00:01:05"
    assert format_hhmmss(3661) == "01:01:01"


def test_qwen_messages_supports_in_memory_base64_images():
    fake_base64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAAB"
    payload = behavior_prompting.qwen_messages([fake_base64], "describe")
    assert len(payload) == 1
    assert payload[0]["images"] == [fake_base64]


def test_build_behavior_classification_prompt_requires_defined_labels_and_json():
    prompt = behavior_prompting.build_behavior_classification_prompt(
        behavior_labels=["walking", "grooming", "rearing"],
        segment_label="00:00:00-00:00:01",
    )
    assert "You are an animal behavior observer." in prompt
    assert "Describe what the mouse is doing during 00:00:00-00:00:01." in prompt
    assert "Write 2–4 sentences that are concise but detailed." in prompt
    assert "- walking" in prompt
    assert "- grooming" in prompt
    assert "- rearing" in prompt
    assert '"label":"<one label from list>"' in prompt
    assert '"confidence":0.0' in prompt
    assert '"description":"2-4 sentence observable description"' in prompt


def test_build_behavior_classification_prompt_includes_optional_context():
    prompt = behavior_prompting.build_behavior_classification_prompt(
        behavior_labels=["walking", "grooming"],
        segment_label="00:00:00-00:00:01",
        video_description="Two mice in a resident-intruder arena.",
        instance_count=2,
        experiment_context="Resident intruder social assay.",
        behavior_definitions="Aggression bout includes slap in face, run away, and fight initiation.",
        focus_points="Count bouts and identify initiator/responder.",
    )
    assert "Video context: Two mice in a resident-intruder arena." in prompt
    assert "Track 2 instance(s)" in prompt
    assert "Experiment context: Resident intruder social assay." in prompt
    assert "Behavior definitions to apply:" in prompt
    assert "Aggression bout includes slap in face" in prompt
    assert (
        "Focus specifically on: Count bouts and identify initiator/responder." in prompt
    )
    assert "initiator/responder" in prompt


def test_timeline_intervals_to_timestamp_rows_merges_adjacent_intervals():
    intervals = [
        {"start_frame": 0, "end_frame": 29, "description": "walking"},
        {"start_frame": 30, "end_frame": 59, "description": "walking"},
        {"start_frame": 60, "end_frame": 89, "description": "standing"},
    ]
    rows = timeline_intervals_to_timestamp_rows(intervals, fps=30.0)
    assert len(rows) == 4
    assert rows[0][2:] == ("Subject 1", "walking", "state start")
    assert rows[1][2:] == ("Subject 1", "walking", "state stop")
    assert rows[2][2:] == ("Subject 1", "standing", "state start")
    assert rows[3][2:] == ("Subject 1", "standing", "state stop")
    assert rows[0][1] == 0.0
    assert rows[1][1] == pytest.approx(59 / 30.0)
    assert rows[2][1] == pytest.approx(2.0)
    assert rows[3][1] == pytest.approx(89 / 30.0)


def test_timestamp_rows_to_timeline_intervals_roundtrip_uses_annolid_columns():
    rows = [
        {
            "Trial time": 0.0,
            "Recording time": 0.0,
            "Subject": "Subject 1",
            "Behavior": "walking",
            "Event": "state start",
        },
        {
            "Trial time": 1.0,
            "Recording time": 1.0,
            "Subject": "Subject 1",
            "Behavior": "walking",
            "Event": "state stop",
        },
    ]
    intervals = timestamp_rows_to_timeline_intervals(rows, fps=30.0)
    assert len(intervals) == 1
    assert intervals[0]["start_frame"] == 0
    assert intervals[0]["end_frame"] == 30
    assert intervals[0]["description"] == "walking"
