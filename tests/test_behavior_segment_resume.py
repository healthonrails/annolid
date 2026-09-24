"""Regression coverage for resuming agent behavior observations."""

import json
from types import SimpleNamespace

import pytest

from annolid.behavior.segment_labeling import (
    behavior_segment_resume_key,
    load_resumable_behavior_segment_predictions,
)


def _load(tmp_path, payload):
    video = tmp_path / "mouse.mp4"
    (tmp_path / "mouse_behavior_segment_labels.json").write_text(json.dumps(payload))
    return load_resumable_behavior_segment_predictions(
        str(video),
        labels=["walking"],
        segment_frames=30,
        segment_seconds=1.0,
        sample_frames_per_segment=4,
    )


def test_resume_retries_unclassified_but_preserves_negative_observations(tmp_path):
    result = _load(
        tmp_path,
        {
            "skipped_predictions": [
                {"start_frame": 0, "end_frame": 29, "label": "unclassified"},
                {"start_frame": 30, "end_frame": 59, "label": "unsupported"},
                {"start_frame": 60, "end_frame": 89, "label": "background"},
            ],
        },
    )
    assert result["ok"]
    assert [(p["start_frame"], p["label"]) for p in result["skipped_predictions"]] == [
        (60, "no_behavior")
    ]


def test_resume_identity_includes_subject_and_preserves_frame_zero():
    first = {"start_frame": 0, "end_frame": 0, "subject": "resident"}
    second = {**first, "subject": "intruder"}
    assert behavior_segment_resume_key(first) == (0, 0, "resident")
    assert behavior_segment_resume_key(first) != behavior_segment_resume_key(second)
    assert behavior_segment_resume_key(
        {**first, "subject": None}, default_subject="resident"
    ) == behavior_segment_resume_key(first)


@pytest.mark.parametrize("records", [{}, "invalid", 42, None])
def test_resume_rejects_malformed_record_collections(tmp_path, records):
    assert not _load(tmp_path, {"predictions": records})["ok"]


def test_bot_preserves_unreadable_progress_log(tmp_path, monkeypatch):
    from annolid.gui.widgets.ai_chat_widget import AIChatWidget

    video = tmp_path / "mouse.mp4"
    log = tmp_path / "mouse_behavior_segment_labels.json"
    log.write_text('{"predictions": [')
    results = []
    mutations = []
    controller = SimpleNamespace(
        clear_behavior_data=lambda: mutations.append("clear"),
        clear_generic_marks=lambda **kw: mutations.append("marks"),
    )
    host = SimpleNamespace(
        video_file=str(video),
        num_frames=30,
        fps=30,
        behavior_controller=controller,
    )
    widget = SimpleNamespace(
        host_window_widget=host,
        _behavior_label_thread=None,
        _normalize_behavior_labels=lambda labels: labels,
        _resolve_segment_label_candidates=lambda labels, **kw: labels,
        _behavior_label_provider_request_interval=lambda *a: 0,
        _load_resumable_behavior_segment_predictions=load_resumable_behavior_segment_predictions,
        _set_bot_action_result=lambda action, result: results.append(result),
        status_label=SimpleNamespace(setText=lambda text: None),
    )
    monkeypatch.setattr(
        "cv2.VideoCapture", lambda path: SimpleNamespace(release=lambda: None)
    )
    AIChatWidget.bot_label_behavior_segments(
        widget,
        behavior_labels_csv="walking",
        segment_mode="uniform",
    )
    assert results[-1]["ok"] is False
    assert "Cannot resume" in results[-1]["error"]
    assert not mutations
    assert log.read_text() == '{"predictions": ['


def test_skipped_observations_keep_separate_subjects():
    from annolid.gui.widgets.ai_chat_widget import AIChatWidget

    widget = SimpleNamespace(
        _update_behavior_label_skipped_overlay_records=lambda ctx: None
    )
    context = {"default_subject": "resident"}
    record = {"start_frame": 0, "end_frame": 29, "label": "no_behavior"}
    for subject in (None, "intruder", "intruder"):
        AIChatWidget._commit_behavior_label_skipped_prediction(
            widget,
            context,
            {**record, "subject": subject},
        )
    assert [r["subject"] for r in context["skipped_predictions"]] == [
        "resident",
        "intruder",
    ]
