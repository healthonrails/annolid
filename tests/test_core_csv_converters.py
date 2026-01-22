import csv
import io

from annolid.core.io.behavior_csv import (
    behavior_events_from_csv,
    behavior_events_to_csv,
)
from annolid.core.io.tracking_csv import tracks_from_labelme_csv, tracks_to_labelme_csv


def _tracks_snapshot(tracks):
    snapshot = []
    for track in sorted(tracks, key=lambda t: t.track_id):
        obs_snapshot = []
        for obs in sorted(track.observations, key=lambda o: o.frame.frame_index):
            mask_key = None
            if obs.mask is not None:
                mask_key = (obs.mask.size, obs.mask.counts)
            obs_snapshot.append(
                (
                    obs.frame.frame_index,
                    tuple(round(v, 6) for v in obs.geometry.xyxy),
                    None if obs.score is None else round(obs.score, 6),
                    obs.label,
                    mask_key,
                )
            )
        snapshot.append(
            (
                track.track_id,
                track.label,
                track.meta.get("tracking_id"),
                tuple(obs_snapshot),
            )
        )
    return tuple(snapshot)


def test_tracks_csv_round_trip():
    buffer = io.StringIO()
    writer = csv.DictWriter(
        buffer,
        fieldnames=[
            "frame_number",
            "x1",
            "y1",
            "x2",
            "y2",
            "cx",
            "cy",
            "instance_name",
            "class_score",
            "segmentation",
            "tracking_id",
        ],
    )
    writer.writeheader()
    writer.writerow(
        {
            "frame_number": 0,
            "x1": 0,
            "y1": 0,
            "x2": 10,
            "y2": 10,
            "cx": 5,
            "cy": 5,
            "instance_name": "mouse1",
            "class_score": 0.9,
            "segmentation": "{'size': [10, 10], 'counts': 'abcd'}",
            "tracking_id": 1,
        }
    )
    writer.writerow(
        {
            "frame_number": 1,
            "x1": 1,
            "y1": 1,
            "x2": 11,
            "y2": 11,
            "cx": 6,
            "cy": 6,
            "instance_name": "mouse1",
            "class_score": 0.8,
            "segmentation": "",
            "tracking_id": 1,
        }
    )
    writer.writerow(
        {
            "frame_number": 0,
            "x1": 20,
            "y1": 20,
            "x2": 30,
            "y2": 30,
            "cx": 25,
            "cy": 25,
            "instance_name": "mouse2",
            "class_score": 1.0,
            "segmentation": "",
            "tracking_id": 0,
        }
    )

    buffer.seek(0)
    tracks = tracks_from_labelme_csv(buffer, video_name="test_video.mp4")
    assert len(tracks) == 2

    out = io.StringIO()
    tracks_to_labelme_csv(tracks, out)
    out.seek(0)
    reloaded = tracks_from_labelme_csv(out, video_name="test_video.mp4")
    assert _tracks_snapshot(tracks) == _tracks_snapshot(reloaded)


def _events_snapshot(events):
    snapshot = []
    for evt in sorted(events, key=lambda e: (e.frame.frame_index, e.behavior, e.event)):
        snapshot.append(
            (
                evt.frame.frame_index,
                None
                if evt.frame.timestamp_sec is None
                else round(evt.frame.timestamp_sec, 6),
                evt.subject,
                evt.behavior,
                evt.event,
                None if not evt.meta else evt.meta.get("trial_time_sec"),
            )
        )
    return tuple(snapshot)


def test_behavior_events_csv_round_trip():
    buffer = io.StringIO()
    buffer.write("Trial time,Recording time,Subject,Behavior,Event\n")
    buffer.write("0.5,1.0,mouse1,digging,start\n")
    buffer.write("0.7,1.2,mouse1,digging,end\n")
    buffer.seek(0)

    events = behavior_events_from_csv(buffer, fps=10.0, video_name="test_video.mp4")
    assert events[0].frame.frame_index == 10

    out = io.StringIO()
    behavior_events_to_csv(events, out)
    out.seek(0)
    reloaded = behavior_events_from_csv(out, fps=10.0, video_name="test_video.mp4")
    assert _events_snapshot(events) == _events_snapshot(reloaded)
