"""Focused regression and capability tests for reporting, statistics, and analysis stack."""

from __future__ import annotations

import asyncio
import csv
import json
from pathlib import Path

import pandas as pd
import pytest

from annolid.agents.coding_agent.agent import AnalysisCodingAgent
from annolid.agents.report_agent.agent import ReportAgent
from annolid.annotation.pose_schema import PoseSchema
from annolid.behavior.cohort import analyze_behavior_cohort, CohortRecordingSpec
from annolid.behavior.irr import compute_irr_from_csvs
from annolid.behavior.time_budget import (
    BehaviorInterval,
    compute_binned_time_budget,
    extract_behavior_intervals,
    summarize_intervals,
)
from annolid.core.agent.config import ToolsConfig
from annolid.core.agent.gui_backend.commands import parse_direct_gui_command
from annolid.core.agent.tools.analytics_reporting import (
    AnnolidBehaviorCohortTool,
    AnnolidBehaviorIRRTool,
    AnnolidBehaviorQCTool,
    AnnolidKinematicsTool,
    AnnolidTimeBudgetTool,
    AnnolidTrackQCTool,
    AnnolidTrackingStatsTool,
    AnnolidUnifiedReportTool,
    AnnolidZoneAnalysisTool,
)
from annolid.core.agent.tools.annolid_run import AnnolidRunTool
from annolid.core.agent.tools.policy import resolve_allowed_tools
from annolid.domain.behavior_agent import (
    BehaviorSegment,
    TaskPlan,
    TrackArtifact,
)
from annolid.engine.cli import main as annolid_cli_main
from annolid.postprocessing import tracking_reports
from annolid.postprocessing.kinematics_stats import analyze_kinematics
from annolid.postprocessing.tracking_results_analyzer import TrackingResultsAnalyzer
from annolid.postprocessing.zone_analysis_engine import GenericZoneEngine
from annolid.postprocessing.zone_schema import (
    load_zone_shapes,
    zone_shape_covers_point,
    zone_shape_distance_to_point,
)
from annolid.services.body_part_segmentation import _load_skeleton_edges
from annolid.services.behavior_agent.bout_scoring import (
    normalize_aggression_sub_event_type,
)
from annolid.services.chat_tracking_stats import (
    run_chat_qc_slash_tool,
    run_chat_report_slash_tool,
    run_chat_stats_slash_tool,
)
from annolid.services.unified_reporting import build_unified_analysis_report


def _write_sample_tracking_csv(path: Path) -> Path:
    rows = [
        "frame_number,instance_name,x1,y1,x2,y2,cx,cy,class_score",
    ]
    # mouse_a moves steadily from x=20 to x=119 over 20 frames, then freezes at x=120 for 20 frames (at 20 fps -> 1.0s freeze)
    # mouse_b stays near (125, 50)
    for f in range(40):
        cx_a = 20.0 + f * 5.0 if f < 20 else 120.0
        cy_a = 50.0
        rows.append(
            f"{f},mouse_a,{cx_a - 5},{cy_a - 5},{cx_a + 5},{cy_a + 5},{cx_a},{cy_a},0.95"
        )
        if f != 10:  # 1-frame gap on mouse_b at frame 10
            cx_b = 125.0
            cy_b = 52.0
            rows.append(
                f"{f},mouse_b,{cx_b - 5},{cy_b - 5},{cx_b + 5},{cy_b + 5},{cx_b},{cy_b},0.92"
            )
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return path


def _write_events_csv(
    path: Path,
    intervals: list[tuple[str, str, float, float]],
    points: list[tuple[str, str, float]] = (),
) -> Path:
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["Recording time", "Subject", "Behavior", "Event"])
        events = []
        for subj, beh, s, e in intervals:
            events.append((s, subj, beh, "start"))
            events.append((e, subj, beh, "stop"))
        for subj, beh, t in points:
            events.append((t, subj, beh, "point"))
        events.sort(key=lambda x: x[0])
        for rec_t, subj, beh, ev in events:
            w.writerow([f"{rec_t:.3f}", subj, beh, ev])
    return path


def test_tracking_reports_zero_gap_return_and_csv_only_summary(tmp_path: Path) -> None:
    csv_path = tmp_path / "trial_tracking.csv"
    csv_path.write_text(
        "frame_number,instance_name,cx,cy\n"
        "0,mouse_1,10,10\n"
        "1,mouse_1,12,10\n"
        "2,mouse_1,14,10\n",
        encoding="utf-8",
    )
    gaps = tracking_reports.find_tracking_gaps(str(csv_path))
    assert gaps == {}
    md_path = tracking_reports.generate_reports(gaps, str(csv_path))
    assert md_path is not None
    assert Path(md_path).is_file()

    summary = tracking_reports.summarize_tracking_gaps(csv_path, fps=30.0)
    assert summary["schema"] == "tracking_gap_report/v1"
    assert summary["total_frames"] == 3
    assert summary["total_gap_events"] == 0
    assert summary["instances"][0]["coverage_ratio"] == 1.0


def test_circle_zone_geometry_occupancy_and_visit_dwell_columns(tmp_path: Path) -> None:
    zone_file = tmp_path / "arena_zones.json"
    zone_file.write_text(
        json.dumps(
            {
                "shapes": [
                    {
                        "label": "center_circle",
                        "shape_type": "circle",
                        "points": [[100.0, 100.0], [130.0, 100.0]],  # radius = 30 px
                        "flags": {"semantic_type": "zone", "zone_kind": "chamber"},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    specs = load_zone_shapes(json.loads(zone_file.read_text(encoding="utf-8")))
    assert len(specs) == 1
    spec = specs[0]
    assert spec.source_shape_type == "circle"
    assert len(spec.analysis_points) > 16
    assert zone_shape_covers_point(spec, (100.0, 100.0)) is True
    assert zone_shape_distance_to_point(spec, (100.0, 100.0)) == 0.0
    assert (zone_shape_distance_to_point(spec, (150.0, 100.0)) or 0.0) > 15.0

    df = pd.DataFrame(
        {
            "frame_number": [0, 1, 2, 3, 4],
            "instance_name": ["m1"] * 5,
            "cx": [100.0, 110.0, 120.0, 160.0, 105.0],
            "cy": [100.0, 100.0, 100.0, 100.0, 100.0],
        }
    )
    engine = GenericZoneEngine(specs, fps=10.0)
    res = engine.analyze_instance(df, "m1")
    row = res.to_summary_row(fps=10.0)
    assert row["occupancy_frames__center_circle"] == 4
    assert row["entry_count__center_circle"] == 2
    assert row["mean_visit_seconds__center_circle"] == pytest.approx(0.2)
    assert row["max_visit_seconds__center_circle"] == pytest.approx(0.3)


def test_tracking_results_analyzer_single_csv_and_symmetric_distances(
    tmp_path: Path,
) -> None:
    # Only *_tracking.csv exists (no *_tracked.csv and no .mp4 on disk)
    csv_path = _write_sample_tracking_csv(tmp_path / "dyad_tracking.csv")
    assert csv_path.is_file()
    analyzer = TrackingResultsAnalyzer(tmp_path / "dyad.mp4", fps=20.0)
    analyzer.read_csv_files()
    analyzer.merge_and_calculate_distance()
    dist_df = analyzer.distances_df
    assert dist_df is not None and not dist_df.empty
    # At frame 20, mouse_a=(120, 50) and mouse_b=(125, 52) -> sqrt(25 + 4) = sqrt(29)
    row_f20 = dist_df[dist_df["frame_number"] == 20].iloc[0]
    assert float(row_f20["distance"]) == pytest.approx(29.0**0.5, rel=1e-4)


def test_pose_schema_edges_and_bout_scoring_and_agents(tmp_path: Path) -> None:
    pose_path = tmp_path / "pose_schema.json"
    schema = PoseSchema(
        keypoints=["nose", "neck", "tail_base"],
        edges=[("nose", "neck"), ("neck", "tail_base")],
    )
    pose_path.write_text(json.dumps(schema.to_dict()), encoding="utf-8")
    edges = _load_skeleton_edges(pose_path)
    assert list(edges) == [("nose", "neck"), ("neck", "tail_base")]

    assert normalize_aggression_sub_event_type("slap_in_face") == "slap_face"
    assert normalize_aggression_sub_event_type("slap in face") == "slap_face"
    assert normalize_aggression_sub_event_type("run-away") == "run_away"

    plan = TaskPlan(
        assay_type="resident_intruder",
        objectives=["Quantify aggression bouts"],
        target_features=["aggression_bout"],
    )
    segments = [
        BehaviorSegment(
            segment_id="s1",
            label="aggression_bout",
            start_frame=10,
            end_frame=35,
            confidence=0.9,
        ),
        BehaviorSegment(
            segment_id="s2",
            label="aggression_bout",
            start_frame=50,
            end_frame=65,
            confidence=0.85,
        ),
    ]
    artifacts = [TrackArtifact(artifact_id="a1", frame_index=10, label="resident")]
    coding_agent = AnalysisCodingAgent(settings={"provider": "none"})
    coding_res = coding_agent.run(plan=plan, artifacts=artifacts, segments=segments)
    assert coding_res.execution_output.get("bout_count") == 2 or any(
        m.get("metric") == "bout_count" or "bout" in str(m)
        for m in coding_res.derived_metrics
    )

    report_agent = ReportAgent()
    rendered = report_agent.build(
        task_plan=plan,
        artifacts=artifacts,
        segments=segments,
        metrics=coding_res.derived_metrics,
        provenance={"run_id": "r1"},
    )
    assert rendered.html.startswith("<html>")
    assert "<table>" in rendered.html
    assert "| Segment ID |" in rendered.markdown or "| Metric |" in rendered.markdown


def test_time_budget_point_events_and_overlapping_session_union(tmp_path: Path) -> None:
    ev_csv = _write_events_csv(
        tmp_path / "events.csv",
        intervals=[
            ("mouse_1", "grooming", 0.0, 10.0),
            (
                "mouse_1",
                "rearing",
                5.0,
                15.0,
            ),  # overlaps [5, 10]; union is [0, 15] = 15s
        ],
        points=[("mouse_1", "jump", 7.5)],
    )
    from annolid.behavior.time_budget import _read_event_rows

    rows = _read_event_rows(ev_csv)
    intervals, warnings = extract_behavior_intervals(rows)
    assert not warnings
    assert len(intervals) == 3  # includes point event jump!

    summary = summarize_intervals(intervals)
    by_beh = {r.behavior: r for r in summary}
    assert by_beh["jump"].occurrences == 1
    assert by_beh["jump"].total_duration == 0.0
    # Union of intervals is 15.0s -> 10 / 15 = 66.6667% (never >100%)
    assert by_beh["grooming"].percent_of_session == pytest.approx(100.0 * 10.0 / 15.0)

    # Overlapping duplicate intervals inside a 10s bin must not exceed 100% of bin
    dup_intervals = [
        BehaviorInterval("m1", "sniff", 0.0, 8.0),
        BehaviorInterval("m1", "sniff", 4.0, 10.0),
    ]
    binned = compute_binned_time_budget(dup_intervals, 10.0)
    assert len(binned) == 1
    assert binned[0].duration == pytest.approx(10.0)
    assert binned[0].bin_end - binned[0].bin_start == pytest.approx(10.0)


def test_cohort_inferential_comparisons_survival_and_recording_unit(
    tmp_path: Path,
) -> None:
    specs = []
    # Control group: 4 recordings, low attack duration (~2s), late latency (~45s or censored)
    for i, (dur, lat) in enumerate(
        [(1.5, 42.0), (2.0, 48.0), (0.0, None), (2.5, 40.0)]
    ):
        p = tmp_path / f"ctrl_{i}.csv"
        ivs = [("resident", "attack", lat, lat + dur)] if lat is not None else []
        ivs.append(("intruder", "flee", 10.0, 12.0))
        _write_events_csv(p, ivs)
        specs.append(
            CohortRecordingSpec(
                recording_id=f"ctrl_{i}",
                events_csv=p,
                group="control",
                session_duration_s=60.0,
            )
        )
    # Drug group: 4 recordings, high attack duration (~18s), early latency (~4s)
    for i, (dur, lat) in enumerate(
        [(16.0, 3.0), (19.0, 4.0), (20.0, 5.0), (17.0, 3.5)]
    ):
        p = tmp_path / f"drug_{i}.csv"
        _write_events_csv(
            p,
            [("resident", "attack", lat, lat + dur), ("intruder", "flee", 8.0, 15.0)],
        )
        specs.append(
            CohortRecordingSpec(
                recording_id=f"drug_{i}",
                events_csv=p,
                group="drug",
                session_duration_s=60.0,
            )
        )

    # Subject unit emits pseudoreplication warning because each recording has 2 subjects
    rep_subj = analyze_behavior_cohort(specs, analysis_unit="subject")
    assert any("pseudoreplication" in w.lower() for w in rep_subj.warnings)

    # Recording unit aggregates per recording and computes inferential tests
    rep_rec = analyze_behavior_cohort(specs, analysis_unit="recording")
    assert rep_rec.analysis_unit == "recording"
    attack_cmp = next(c for c in rep_rec.comparisons if c.behavior == "attack")
    assert attack_cmp.p_parametric < 0.01
    assert attack_cmp.p_nonparametric < 0.05
    assert attack_cmp.hedges_g is not None and abs(attack_cmp.hedges_g) > 2.0
    assert attack_cmp.cliffs_delta is not None and abs(
        attack_cmp.cliffs_delta
    ) == pytest.approx(1.0)
    assert attack_cmp.logrank_p_value < 0.05
    assert attack_cmp.q_parametric_bh < 0.05


def test_irr_krippendorff_alpha_bootstrap_ci_and_disagreement_bouts(
    tmp_path: Path,
) -> None:
    r1 = _write_events_csv(
        tmp_path / "rater1.csv",
        [("mouse_1", "groom", 1.0, 5.0), ("mouse_1", "groom", 8.0, 10.0)],
    )
    r2 = _write_events_csv(
        tmp_path / "rater2.csv",
        [("mouse_1", "groom", 1.0, 4.0), ("mouse_1", "groom", 8.0, 10.0)],
    )
    irr = compute_irr_from_csvs(
        [r1, r2],
        rater_names=["Alice", "Bob"],
        bin_seconds=0.5,
        session_start=0.0,
        session_end=12.0,
    )
    assert irr.macro_krippendorff_alpha is not None
    assert irr.macro_krippendorff_alpha > 0.75
    row = irr.rows[0]
    assert row.cohens_kappa_ci95_low is not None
    assert row.cohens_kappa_ci95_high is not None
    assert len(irr.disagreements) == 1
    dis = irr.disagreements[0]
    assert dis.start_s == pytest.approx(4.0)
    assert dis.end_s == pytest.approx(5.0)
    assert dis.raters_active == ("Alice",)
    assert dis.raters_inactive == ("Bob",)


def test_kinematics_calibrated_metrics_and_unified_report_cli_and_tools(
    tmp_path: Path,
) -> None:
    csv_path = _write_sample_tracking_csv(tmp_path / "exp_tracking.csv")
    ev_path = _write_events_csv(
        tmp_path / "exp_events.csv",
        [("mouse_a", "investigate", 0.2, 1.5)],
    )
    kin = analyze_kinematics(
        csv_path,
        fps=20.0,
        px_per_cm=10.0,
        smooth_window=1,
        min_freeze_duration_s=0.5,
    )
    by_inst = {r.instance: r for r in kin.instances}
    # mouse_a moves 19 steps * 5 px = 95 px + 1 step * 5 px = 100 px -> 10.0 cm!
    assert by_inst["mouse_a"].total_distance_px == pytest.approx(100.0)
    assert by_inst["mouse_a"].total_distance_cm == pytest.approx(10.0)
    assert by_inst["mouse_a"].freezing_bout_count >= 1
    assert len(kin.pairwise) == 1

    # Unified report composer
    rep = build_unified_analysis_report(
        csv_path,
        events_csv=ev_path,
        fps=20.0,
        px_per_cm=10.0,
        output_dir=tmp_path / "unified_out",
        include_plots=False,
    )
    assert rep.html_path.is_file()
    assert rep.markdown_path.is_file()
    assert rep.json_path.is_file()
    html_text = rep.html_path.read_text(encoding="utf-8")
    assert "<svg" in html_text
    assert rep.payload["schema"] == "unified_analysis_report/v1"
    assert rep.payload["sections"]["kinematics"] is not None
    assert rep.payload["sections"]["behavior_time_budget"] is not None

    # CLI subcommands smoke verification
    rc_kin = annolid_cli_main(
        [
            "kinematics",
            str(csv_path),
            "--fps",
            "20",
            "--px-per-cm",
            "10",
            "--out-json",
            str(tmp_path / "kin_cli.json"),
            "--json",
        ]
    )
    assert rc_kin == 0
    assert (tmp_path / "kin_cli.json").is_file()

    rc_rep = annolid_cli_main(
        [
            "report",
            str(csv_path),
            "--events-csv",
            str(ev_path),
            "--fps",
            "20",
            "--output-dir",
            str(tmp_path / "cli_report"),
            "--no-plots",
            "--json",
        ]
    )
    assert rc_rep == 0
    assert (tmp_path / "cli_report" / "analysis_report.html").is_file()

    # AnnolidRunTool mutation refinement
    assert (
        AnnolidRunTool._is_mutating(["track-qc", "run/my_tracks.csv", "--json"])
        is False
    )
    assert (
        AnnolidRunTool._is_mutating(["time-budget", "events.csv", "-o", "out.csv"])
        is True
    )
    assert AnnolidRunTool._is_mutating(["update", "run"]) is True
    assert AnnolidRunTool._is_mutating(["report", "--help"]) is False

    # Core Analytics FunctionTools + Policy group:analysis + Slash commands
    for tool_cls, kwargs in [
        (AnnolidTrackQCTool, {"input_path": str(csv_path)}),
        (AnnolidKinematicsTool, {"source_path": str(csv_path), "fps": 20.0}),
        (AnnolidTimeBudgetTool, {"events_csv": str(ev_path), "bin_size": 1.0}),
        (AnnolidBehaviorQCTool, {"events_csv": str(ev_path)}),
        (
            AnnolidUnifiedReportTool,
            {
                "source_path": str(csv_path),
                "output_dir": str(tmp_path / "tool_report"),
                "include_plots": False,
            },
        ),
    ]:
        tool = tool_cls(allowed_dir=tmp_path)
        out = json.loads(asyncio.run(tool.execute(**kwargs)))
        assert out["ok"] is True

    # Ensure remaining tool classes instantiate and expose valid schemas
    for extra_cls in (
        AnnolidTrackingStatsTool,
        AnnolidZoneAnalysisTool,
        AnnolidBehaviorIRRTool,
        AnnolidBehaviorCohortTool,
    ):
        schema_def = extra_cls(allowed_dir=tmp_path).to_schema()
        assert schema_def["type"] == "function"

    cfg = ToolsConfig(profile="minimal", allow=["group:analysis", "group:ui"])
    resolved = resolve_allowed_tools(
        all_tool_names=[
            "annolid_generate_report",
            "annolid_kinematics_stats",
            "gui_analyze_tracking_stats",
            "gui_score_aggression_bouts",
            "exec",
        ],
        tools_cfg=cfg,
        provider="openrouter",
        model="gpt-5-mini",
    )
    assert "annolid_generate_report" in resolved.allowed_tools
    assert "annolid_kinematics_stats" in resolved.allowed_tools
    assert "gui_analyze_tracking_stats" in resolved.allowed_tools
    assert "gui_score_aggression_bouts" in resolved.allowed_tools
    assert "exec" not in resolved.allowed_tools

    # Slash command parsing & service handlers
    assert parse_direct_gui_command(f"/stats {csv_path} px_per_cm=10") == {
        "name": "stats_slash",
        "args": {"path": str(csv_path), "px_per_cm": 10.0},
    }
    assert parse_direct_gui_command(f"/qc {ev_path}")["name"] == "qc_slash"
    assert parse_direct_gui_command(f"/report {csv_path}")["name"] == "report_slash"

    assert (
        run_chat_stats_slash_tool(path=str(csv_path), fps=20.0, px_per_cm=10.0)["ok"]
        is True
    )
    assert run_chat_qc_slash_tool(path=str(ev_path))["mode"] == "behavior_qc"
    assert run_chat_qc_slash_tool(path=str(csv_path))["mode"] == "track_qc"
    assert (
        run_chat_report_slash_tool(
            path=str(csv_path), output_dir=str(tmp_path / "slash_rep")
        )["ok"]
        is True
    )


def test_scipy_and_matplotlib_free_exact_inferential_stats_and_svg_plots(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import numpy as np
    from annolid.behavior import cohort as cohort_mod
    from annolid.postprocessing import kinematics_stats as kin_mod

    # Build a 3-group cohort (vehicle, low_dose, high_dose) with ties + censorship
    specs_3g = []
    group_defs = {
        "vehicle": [(1.0, 45.0), (1.0, 48.0), (0.0, None), (2.0, 40.0)],
        "low_dose": [(6.0, 20.0), (7.5, 18.0), (6.0, 22.0), (8.0, 19.0)],
        "high_dose": [(18.0, 3.0), (21.0, 4.0), (19.5, 3.5), (22.0, 2.5)],
    }
    for grp, items in group_defs.items():
        for i, (dur, lat) in enumerate(items):
            p = tmp_path / f"{grp}_{i}.csv"
            ivs = [("mouse", "groom", lat, lat + dur)] if lat is not None else []
            _write_events_csv(p, ivs, points=[("mouse", "rear", 5.0)])
            specs_3g.append(
                CohortRecordingSpec(
                    recording_id=f"{grp}_{i}",
                    events_csv=p,
                    group=grp,
                    session_duration_s=60.0,
                )
            )

    # Compute baseline with SciPy (when installed in .venv)
    rep_with_scipy = analyze_behavior_cohort(specs_3g, analysis_unit="recording")
    cmp_scipy_3g = next(c for c in rep_with_scipy.comparisons if c.behavior == "groom")
    specs_2g = [s for s in specs_3g if s.group in {"vehicle", "high_dose"}]
    rep_2g_scipy = analyze_behavior_cohort(specs_2g, analysis_unit="recording")
    cmp_scipy_2g = next(c for c in rep_2g_scipy.comparisons if c.behavior == "groom")

    arr_signal = np.sin(np.linspace(0.0, 4.0, 25)) * 30.0 + 100.0
    smoothed_scipy = kin_mod._savgol_smooth_1d(arr_signal, 5, 2)

    # Now disable SciPy and Matplotlib completely and verify exact parity!
    monkeypatch.setattr(cohort_mod, "_scipy_stats", None)
    monkeypatch.setattr(kin_mod, "_savgol_filter", None)
    monkeypatch.setattr(kin_mod, "plt", None)

    rep_no_scipy_3g = analyze_behavior_cohort(specs_3g, analysis_unit="recording")
    cmp_pure_3g = next(c for c in rep_no_scipy_3g.comparisons if c.behavior == "groom")
    assert cmp_pure_3g.parametric_test == "one_way_anova"
    assert cmp_pure_3g.nonparametric_test == "kruskal_wallis"
    assert cmp_pure_3g.p_parametric < 1e-4
    assert cmp_pure_3g.p_nonparametric < 0.01
    if (
        cohort_mod.__dict__.get("_scipy_stats") is None
        and cmp_scipy_3g.parametric_stat > 0
    ):
        assert cmp_pure_3g.parametric_stat == pytest.approx(
            cmp_scipy_3g.parametric_stat, rel=1e-9
        )
        assert cmp_pure_3g.p_parametric == pytest.approx(
            cmp_scipy_3g.p_parametric, rel=1e-9
        )
        assert cmp_pure_3g.nonparametric_stat == pytest.approx(
            cmp_scipy_3g.nonparametric_stat, rel=1e-9
        )
        assert cmp_pure_3g.p_nonparametric == pytest.approx(
            cmp_scipy_3g.p_nonparametric, rel=1e-9
        )
        assert cmp_pure_3g.responder_p_value == pytest.approx(
            cmp_scipy_3g.responder_p_value, rel=1e-9
        )
        assert cmp_pure_3g.logrank_p_value == pytest.approx(
            cmp_scipy_3g.logrank_p_value, rel=1e-9
        )

    rep_no_scipy_2g = analyze_behavior_cohort(specs_2g, analysis_unit="recording")
    cmp_pure_2g = next(c for c in rep_no_scipy_2g.comparisons if c.behavior == "groom")
    assert cmp_pure_2g.p_parametric == pytest.approx(
        cmp_scipy_2g.p_parametric, rel=1e-9
    )
    assert cmp_pure_2g.p_nonparametric == pytest.approx(
        cmp_scipy_2g.p_nonparametric, rel=1e-9
    )
    assert cmp_pure_2g.responder_p_value == pytest.approx(
        cmp_scipy_2g.responder_p_value, rel=1e-9
    )

    smoothed_pure = kin_mod._savgol_smooth_1d(arr_signal, 5, 2)
    assert np.allclose(smoothed_pure, smoothed_scipy, rtol=1e-10, atol=1e-10)

    # Matplotlib-free standalone SVG plot generation + Unified Report embedding
    csv_path = _write_sample_tracking_csv(tmp_path / "svg_only_tracking.csv")
    uni = build_unified_analysis_report(
        csv_path,
        fps=20.0,
        output_dir=tmp_path / "svg_unified",
        include_plots=True,
    )
    kin_sec = uni.payload["sections"]["kinematics"]
    traj_svg = Path(kin_sec["artifacts"]["trajectory_plot"])
    budg_svg = Path(kin_sec["artifacts"]["activity_budget_plot"])
    assert traj_svg.suffix == ".svg" and traj_svg.is_file()
    assert budg_svg.suffix == ".svg" and budg_svg.is_file()
    assert "<svg" in traj_svg.read_text(encoding="utf-8")


def test_planar_perspective_homography_calibration_end_to_end(tmp_path: Path) -> None:
    from annolid.postprocessing.spatial_calibration import PlanarCalibration

    # Perspective trapezoid for a 50 cm x 50 cm physical arena viewed from an oblique angle:
    # Far wall (y=100): x spans [200, 400] -> 200 px wide corresponds to 50 cm (4 px/cm)
    # Near wall (y=500): x spans [100, 500] -> 400 px wide corresponds to 50 cm (8 px/cm)
    corners_px = [[200.0, 100.0], [400.0, 100.0], [500.0, 500.0], [100.0, 500.0]]
    cal = PlanarCalibration.from_rectangle_corners(
        corners_px, width_cm=50.0, height_cm=50.0
    )
    assert cal.mode == "homography"
    assert cal.reprojection_rmse_cm is not None and cal.reprojection_rmse_cm < 1e-9

    # Write a tracking CSV where mouse_far walks along the far wall (200->400 px = 200 px)
    # and mouse_near walks along the near wall (100->500 px = 400 px).
    # Both animals physically walk the exact same 50.0 cm arena width!
    tilt_csv = tmp_path / "tilted_tracking.csv"
    rows = ["frame_number,instance_name,x1,y1,x2,y2,cx,cy"]
    for i in range(11):
        frac = i / 10.0
        xf = 200.0 + frac * 200.0
        xn = 100.0 + frac * 400.0
        # 4x4 px box at far wall vs 8x8 px box at near wall
        rows.append(f"{i},mouse_far,{xf - 2},98,{xf + 2},102,{xf},100.0")
        rows.append(f"{i},mouse_near,{xn - 4},496,{xn + 4},504,{xn},500.0")
    tilt_csv.write_text("\n".join(rows) + "\n", encoding="utf-8")

    cal_json = tmp_path / "arena_homography.json"
    cal_json.write_text(
        json.dumps(
            {
                "src_points_px": corners_px,
                "width_cm": 50.0,
                "height_cm": 50.0,
            }
        ),
        encoding="utf-8",
    )

    kin = analyze_kinematics(
        tilt_csv,
        fps=10.0,
        calibration=cal_json,
        smooth_window=1,
    )
    by_inst = {r.instance: r for r in kin.instances}
    assert by_inst["mouse_far"].total_distance_px == pytest.approx(200.0)
    assert by_inst["mouse_near"].total_distance_px == pytest.approx(400.0)
    # Perspective homography recovers true 50.0 cm physical travel for BOTH!
    assert by_inst["mouse_far"].total_distance_cm == pytest.approx(50.0, rel=1e-4)
    assert by_inst["mouse_near"].total_distance_cm == pytest.approx(50.0, rel=1e-4)
    # Pairwise separation between opposite corners (y=0 cm vs y=50 cm, same x_cm) is 50.0 cm!
    assert kin.pairwise[0].mean_distance_cm == pytest.approx(50.0, rel=1e-4)
    assert by_inst["mouse_far"].median_area_cm2 is not None
    assert by_inst["mouse_far"].median_area_cm2 > 0.0

    # LabelMe _zones.json with physical width_cm/height_cm flags auto-calibrates Unified Report!
    zone_cal = tmp_path / "tilted_zones.json"
    zone_cal.write_text(
        json.dumps(
            {
                "shapes": [
                    {
                        "label": "open_field_floor",
                        "shape_type": "polygon",
                        "points": corners_px,
                        "flags": {
                            "semantic_type": "zone",
                            "width_cm": 50.0,
                            "height_cm": 50.0,
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    uni = build_unified_analysis_report(
        tilt_csv,
        zone_file=zone_cal,
        fps=10.0,
        output_dir=tmp_path / "tilt_report",
        include_plots=False,
    )
    assert uni.payload["sections"]["kinematics"]["calibration"]["mode"] == "homography"

    # CLI, FunctionTool, and /stats slash command with calibration JSON
    rc = annolid_cli_main(
        [
            "kinematics",
            str(tilt_csv),
            "--fps",
            "10",
            "--calibration",
            str(cal_json),
            "--smooth-window",
            "1",
            "--out-json",
            str(tmp_path / "tilt_cli.json"),
            "--json",
        ]
    )
    assert rc == 0
    cli_payload = json.loads((tmp_path / "tilt_cli.json").read_text(encoding="utf-8"))
    assert cli_payload["calibration"]["mode"] == "homography"

    tool_out = json.loads(
        asyncio.run(
            AnnolidKinematicsTool(allowed_dir=tmp_path).execute(
                source_path=str(tilt_csv),
                fps=10.0,
                calibration_json=str(cal_json),
            )
        )
    )
    assert tool_out["ok"] is True
    assert tool_out["calibration"]["mode"] == "homography"

    parsed = parse_direct_gui_command(f"/stats {tilt_csv} calibration={cal_json}")
    assert parsed["args"]["calibration"] == str(cal_json)
    slash_res = run_chat_stats_slash_tool(
        path=str(tilt_csv), fps=10.0, calibration=str(cal_json)
    )
    assert slash_res["ok"] is True
    assert slash_res["report"]["calibration"]["mode"] == "homography"
