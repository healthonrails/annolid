from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import List, Tuple

import pytest

from annolid.behavior.cohort import (
    CohortRecordingSpec,
    analyze_behavior_cohort,
    apply_cohort_metadata_csv,
    compute_transition_matrix,
)
from annolid.behavior.time_budget import BehaviorInterval
from annolid.engine import cli as engine_cli


def _write_events_csv(path: Path, rows: List[Tuple[float, str, str, str]]) -> Path:
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["Recording time", "Subject", "Behavior", "Event"])
        for t, subj, beh, ev in rows:
            writer.writerow([f"{t:.3f}", subj, beh, ev])
    return path


def test_cohort_zero_occurrence_denominators_and_censored_latency(
    tmp_path: Path,
) -> None:
    # Control 1: 10s sniff (start=2s), 4s attack (start=15s) over 60s session
    c1 = _write_events_csv(
        tmp_path / "ctrl_01.csv",
        [
            (2.0, "resident", "sniff", "state start"),
            (12.0, "resident", "sniff", "state stop"),
            (15.0, "resident", "attack", "state start"),
            (19.0, "resident", "attack", "state stop"),
        ],
    )
    # Control 2: 20s sniff (start=5s), ZERO attacks (non-responder!) over 60s session
    c2 = _write_events_csv(
        tmp_path / "ctrl_02.csv",
        [
            (5.0, "resident", "sniff", "state start"),
            (25.0, "resident", "sniff", "state stop"),
        ],
    )
    # Treated 1: 6s sniff (start=1s), 10s attack (start=3s) over 60s session
    t1 = _write_events_csv(
        tmp_path / "trt_01.csv",
        [
            (1.0, "resident", "sniff", "state start"),
            (7.0, "resident", "sniff", "state stop"),
            (3.0, "resident", "attack", "state start"),
            (13.0, "resident", "attack", "state stop"),
        ],
    )

    specs = [
        CohortRecordingSpec("ctrl_01", c1, group="control", session_duration_s=60.0),
        CohortRecordingSpec("ctrl_02", c2, group="control", session_duration_s=60.0),
        CohortRecordingSpec("trt_01", t1, group="treated", session_duration_s=60.0),
    ]
    report = analyze_behavior_cohort(specs)

    assert report.recording_count == 3
    assert report.subject_unit_count == 3
    assert report.groups == ("control", "treated")
    assert report.behaviors == ("attack", "sniff")

    ctrl_2_attack = next(
        r
        for r in report.subject_rows
        if r.recording_id == "ctrl_02" and r.behavior == "attack"
    )
    assert ctrl_2_attack.occurrences == 0
    assert ctrl_2_attack.total_duration_s == 0.0
    assert ctrl_2_attack.latency_to_first_s is None
    assert ctrl_2_attack.censored is True
    assert ctrl_2_attack.censored_latency_s == pytest.approx(60.0)

    ctrl_attack_summary = next(
        s
        for s in report.group_summaries
        if s.group == "control" and s.behavior == "attack"
    )
    # Zero-occurrence ctrl_02 MUST remain in the denominator (n_subjects=2, n_responders=1)!
    assert ctrl_attack_summary.n_subjects == 2
    assert ctrl_attack_summary.n_responders == 1
    assert ctrl_attack_summary.responder_fraction == pytest.approx(0.5)
    # Mean duration across [4.0, 0.0] is 2.0s (not 4.0s!)
    assert ctrl_attack_summary.mean_total_duration_s == pytest.approx(2.0)
    # Responder-only latency averages [15.0] -> 15.0s; censored survival latency averages [15.0, 60.0] -> 37.5s
    assert ctrl_attack_summary.mean_latency_responders_s == pytest.approx(15.0)
    assert ctrl_attack_summary.mean_latency_censored_s == pytest.approx(37.5)


def test_blinded_dyadic_metadata_unblinding(tmp_path: Path) -> None:
    dyad_csv = _write_events_csv(
        tmp_path / "blind_vid_A9.csv",
        [
            (1.0, "mouse_1", "chase", "state start"),
            (5.0, "mouse_1", "chase", "state stop"),
            (2.0, "mouse_2", "flee", "state start"),
            (6.0, "mouse_2", "flee", "state stop"),
        ],
    )
    meta_csv = tmp_path / "unblind_key.csv"
    with meta_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(
            ["recording_id", "subject", "condition", "session_duration_s", "cage"]
        )
        writer.writerow(["blind_vid_A9", "mouse_1", "chr2_stim", "30.0", "C12"])
        writer.writerow(["blind_vid_A9", "mouse_2", "eyfp_ctrl", "30.0", "C12"])

    initial = [CohortRecordingSpec(recording_id="blind_vid_A9", events_csv=dyad_csv)]
    unblinded = apply_cohort_metadata_csv(initial, meta_csv)
    report = analyze_behavior_cohort(unblinded)

    assert report.groups == ("chr2_stim", "eyfp_ctrl")
    m1_chase = next(
        r
        for r in report.subject_rows
        if r.subject == "mouse_1" and r.behavior == "chase"
    )
    m2_flee = next(
        r
        for r in report.subject_rows
        if r.subject == "mouse_2" and r.behavior == "flee"
    )
    assert m1_chase.group == "chr2_stim"
    assert m2_flee.group == "eyfp_ctrl"
    assert m1_chase.metadata["cage"] == "C12"
    assert m1_chase.session_duration_s == pytest.approx(30.0)


def test_compute_transition_matrix_probabilities_gap_and_self_filter() -> None:
    seq = [
        BehaviorInterval("m1", "sniff", 0.0, 2.0),
        BehaviorInterval("m1", "sniff", 2.5, 4.0),  # self transition (gap 0.5s)
        BehaviorInterval("m1", "attack", 4.2, 6.0),  # sniff -> attack (gap 0.2s)
        BehaviorInterval("m1", "groom", 20.0, 25.0),  # attack -> groom (gap 14.0s)
    ]

    # 1. All transitions included
    edges_all = compute_transition_matrix({"grp": [seq]})
    by_pair = {(e.from_behavior, e.to_behavior): e for e in edges_all}
    assert by_pair[("sniff", "sniff")].probability == pytest.approx(0.5)
    assert by_pair[("sniff", "attack")].probability == pytest.approx(0.5)
    assert by_pair[("attack", "groom")].probability == pytest.approx(1.0)
    # Target base rate of attack is 1/3 -> lift = 0.5 / (1/3) = 1.5
    assert by_pair[("sniff", "attack")].lift == pytest.approx(1.5)

    # 2. Exclude self transitions + enforce max_transition_gap_s=2.0 (drops 14s gap to groom)
    edges_filtered = compute_transition_matrix(
        {"grp": [seq]},
        max_transition_gap_s=2.0,
        include_self_transitions=False,
    )
    assert len(edges_filtered) == 1
    assert edges_filtered[0].from_behavior == "sniff"
    assert edges_filtered[0].to_behavior == "attack"
    assert edges_filtered[0].probability == pytest.approx(1.0)


def test_behavior_cohort_cli_manifest_and_directory_unblinding(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    cohort_dir = tmp_path / "blinded_recordings"
    cohort_dir.mkdir()
    _write_events_csv(
        cohort_dir / "rec_a.csv",
        [
            (1.0, "m1", "sniff", "state start"),
            (4.0, "m1", "sniff", "state stop"),
            (4.5, "m1", "attack", "state start"),
            (8.5, "m1", "attack", "state stop"),
        ],
    )
    _write_events_csv(
        cohort_dir / "rec_b.csv",
        [
            (2.0, "m1", "sniff", "state start"),
            (7.0, "m1", "sniff", "state stop"),
        ],
    )
    # Co-locate metadata CSV inside the SAME directory to ensure directory globbing excludes it!
    meta_csv = cohort_dir / "metadata_unblind.csv"
    with meta_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["recording_id", "group", "session_duration_s"])
        writer.writerow(["rec_a", "stim", "60"])
        writer.writerow(["rec_b", "ctrl", "60"])

    out_json = tmp_path / "cohort_report.json"
    summary_csv = cohort_dir / "cohort_summary.csv"
    subjects_csv = tmp_path / "cohort_subjects.csv"
    transitions_csv = tmp_path / "cohort_transitions.csv"

    rc = engine_cli.main(
        [
            "behavior-cohort",
            str(cohort_dir),
            "--metadata-csv",
            str(meta_csv),
            "--json",
            "--out",
            str(out_json),
            "--summary-csv",
            str(summary_csv),
            "--subjects-csv",
            str(subjects_csv),
            "--transitions-csv",
            str(transitions_csv),
        ]
    )
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["recording_count"] == 2
    assert payload["groups"] == ["ctrl", "stim"]
    assert payload["provenance"]["schema"] == "behavior_cohort/v1"
    assert out_json.exists()
    assert summary_csv.exists()
    assert subjects_csv.exists()
    assert transitions_csv.exists()

    # Also test JSON manifest input
    manifest_json = tmp_path / "cohort_manifest.json"
    manifest_json.write_text(
        json.dumps(
            {
                "recordings": [
                    {
                        "recording_id": "rec_a",
                        "events_csv": str(cohort_dir / "rec_a.csv"),
                        "group": "stim",
                        "session_duration_s": 60.0,
                    },
                    {
                        "recording_id": "rec_b",
                        "events_csv": str(cohort_dir / "rec_b.csv"),
                        "group": "ctrl",
                        "session_duration_s": 60.0,
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    rc_manifest = engine_cli.main(["behavior-cohort", "--manifest", str(manifest_json)])
    assert rc_manifest == 0
    table_text = capsys.readouterr().out
    assert "Recordings: 2" in table_text
    assert "Top Behavioral Transitions:" in table_text
