"""Comprehensive tests for BORIS interoperability, analytics, CLI, and GUI workflows."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from qtpy import QtCore, QtGui, QtWidgets

from annolid.behavior.boris_io import (
    import_boris_project,
    materialize_boris_project,
    write_annolid_events_csv,
    write_boris_aggregated_csv,
    write_boris_project,
    write_boris_tabular_csv,
)
from annolid.behavior.cohort import (
    build_boris_transition_grid,
    compute_observation_transitions,
    export_transitions_graphviz_dot,
    render_transitions_svg,
)
from annolid.services.behavior_analytics import run_behavior_cohort
from annolid.behavior.ethogram import create_ethogram, render_ethogram_svg
from annolid.behavior.event_utils import (
    parse_event_mapping_rows,
    read_parsed_event_rows,
)
from annolid.behavior.project_schema import (
    BehaviorDefinition,
    CategoryDefinition,
    IndependentVariableDefinition,
    ModifierDefinition,
    ProjectSchema,
    SubjectDefinition,
    load_schema,
    save_schema,
    validate_schema,
)
from annolid.behavior.time_budget import (
    compute_behavior_cooccurrence,
    compute_binned_time_budget,
    compute_time_budget,
    extract_behavior_intervals,
)
from annolid.core.behavior.spec import load_behavior_spec
from annolid.engine.cli import main as cli_main
from annolid.gui.behavior_controller import BehaviorController
from annolid.gui.controllers.flags import FlagsController
from annolid.gui.mixins.behavior_interaction_mixin import BehaviorInteractionMixin
from annolid.gui.mixins.schema_behavior_loader_mixin import SchemaBehaviorLoaderMixin
from annolid.gui.widgets.behavior_controls import BehaviorControlsWidget
from annolid.gui.widgets.behavior_log import BehaviorEventLogWidget
from annolid.gui.widgets.flags import FlagTableWidget
from annolid.gui.widgets.project_dialog import ProjectDialog
from annolid.gui.widgets.timeline_panel import TimelinePanel
from annolid.gui.widgets.video_slider import VideoSlider
from annolid.infrastructure import AnnotationStore

_APP = None


def _ensure_qapp() -> QtWidgets.QApplication:
    global _APP
    app = QtWidgets.QApplication.instance()
    if app is None:
        app = QtWidgets.QApplication([])
    _APP = app
    return _APP


def _sample_boris_payload() -> dict:
    return {
        "project_name": "Resident Intruder Study",
        "project_date": "2026-05-01T10:00:00",
        "project_description": "BORIS v8 dyadic ethogram",
        "project_format_version": "7.0",
        "time_format": "s",
        "behavioral_categories": ["Agonistic", "Maintenance", "Visibility"],
        "subjects_conf": {
            "0": {"key": "1", "name": "resident", "description": "Cage owner"},
            "1": {"key": "2", "name": "intruder", "description": "Visitor"},
        },
        "behaviors_conf": {
            "0": {
                "type": "State event",
                "key": "a",
                "code": "attack",
                "description": "Clinch attack",
                "category": "Agonistic",
                "modifiers": {
                    "0": {
                        "name": "Target",
                        "type": 0,  # SINGLE_SELECTION
                        "values": ["flank (f)", "dorsum (d)"],
                    },
                    "1": {
                        "name": "Intensity",
                        "type": 1,  # MULTI_SELECTION
                        "values": ["mild", "severe"],
                    },
                },
                "excluded": "groom,out_of_sight",
            },
            "1": {
                "type": "State event",
                "key": "g",
                "code": "groom",
                "description": "Self-grooming",
                "category": "Maintenance",
                "modifiers": {},
                "excluded": "attack,out_of_sight",
            },
            "2": {
                "type": "Point event",
                "key": "b",
                "code": "bite",
                "description": "Discrete bite",
                "category": "Agonistic",
                "modifiers": {},
                "excluded": "",
            },
            "3": {
                "type": "State event",
                "key": "o",
                "code": "out_of_sight",
                "description": "Animal occluded",
                "category": "Visibility",
                "modifiers": {},
                "excluded": "attack,groom",
            },
        },
        "independent_variables": {
            "0": {
                "label": "Treatment",
                "description": "Drug vs vehicle",
                "type": "value from list",
                "default value": "ctrl",
                "possible values": "ctrl,drug",
            },
            "1": {
                "label": "Dose_mg_kg",
                "description": "Dosage",
                "type": "numeric",
                "default value": "0.0",
                "possible values": "",
            },
        },
        "observations": {
            "obs_ctrl_1": {
                "file": {"1": ["videos/ctrl_1.mp4"]},
                "type": "MEDIA",
                "date": "2026-05-01T10:15:00",
                "description": "Control mouse 1",
                "time offset": 0.0,
                "independent_variables": {"Treatment": "ctrl", "Dose_mg_kg": "0.0"},
                "media_info": {"fps": {"videos/ctrl_1.mp4": 30.0}},
                "events": [
                    [1.0, "resident", "attack", "flank|mild", "first lunge", "START"],
                    [2.0, "resident", "bite", "", "sharp bite"],  # 5-tuple point event
                    [4.0, "resident", "attack", "flank|mild", "", "STOP"],
                    [5.0, "resident", "groom", "", "", "START"],
                    [8.0, "resident", "groom", "", "", "STOP"],
                ],
            },
            "obs_ctrl_2": {
                "file": {"1": ["videos/ctrl_2.mp4"]},
                "type": "MEDIA",
                "independent_variables": {"Treatment": "ctrl"},
                "events": [
                    [1.5, "resident", "attack", "", "", "START"],
                    [3.0, "resident", "attack", "", "", "STOP"],
                ],
            },
            "obs_drug_1": {
                "file": {"1": ["videos/drug_1.mp4"]},
                "type": "MEDIA",
                "independent_variables": {"Treatment": "drug", "Dose_mg_kg": "5.0"},
                "events": [
                    [0.5, "resident", "attack", "dorsum|severe", "", "START"],
                    [8.5, "resident", "attack", "dorsum|severe", "", "STOP"],
                ],
            },
            "obs_drug_2": {
                "file": {"1": ["videos/drug_2.mp4"]},
                "type": "MEDIA",
                "independent_variables": {"Treatment": "drug", "Dose_mg_kg": "5.0"},
                "events": [
                    [0.5, "resident", "attack", "", "", "START"],
                    [9.5, "resident", "attack", "", "", "STOP"],
                ],
            },
        },
    }


# ---------------------------------------------------------------------------
# 1. Domain Schema & BORIS Interoperability
# ---------------------------------------------------------------------------
def test_project_schema_boris_extensions_and_subjects_as_modifiers(
    tmp_path: Path,
) -> None:
    schema = ProjectSchema(
        categories=[CategoryDefinition(id="soc", name="Social", color="#1565C0")],
        modifiers=[
            ModifierDefinition(
                id="loc_head",
                name="Head",
                group="BodyRegion",
                selection_mode="single",
                key_binding="h",
            )
        ],
        subjects=[
            SubjectDefinition(
                id="m1", name="mouse_1", key_binding="1", color="#ef4444"
            ),
            SubjectDefinition(
                id="m2", name="mouse_2", key_binding="2", color="#3b82f6"
            ),
        ],
        behaviors=[
            BehaviorDefinition(
                code="sniff",
                name="Sniff",
                category_id="soc",
                modifier_ids=["loc_head"],
                key_binding="s",
                is_state=True,
                exclusive_with=["attack"],
                color="#10b981",
            ),
            BehaviorDefinition(
                code="attack",
                name="Attack",
                category_id="soc",
                key_binding="a",
                is_state=True,
                color="#dc2626",
            ),
            BehaviorDefinition(
                code="usv_call",
                name="USV Call",
                key_binding="u",
                is_state=False,
            ),
        ],
        independent_variables=[
            IndependentVariableDefinition(
                id="genotype",
                name="Genotype",
                var_type="value_from_list",
                default_value="WT",
                allowed_values=["WT", "KO"],
            )
        ],
    )

    assert schema.symmetric_exclusions_for("attack") == {"sniff"}
    assert schema.symmetric_exclusions_for("sniff") == {"attack"}

    enriched = schema.with_subjects_as_modifiers(
        target_behaviors=["sniff", "attack"],
        selection_mode="single",
    )
    subj_mods = [m for m in enriched.modifiers if m.group == "Target Subject"]
    assert [m.name for m in subj_mods] == ["mouse_1", "mouse_2"]
    assert all(m.selection_mode == "single" for m in subj_mods)
    sniff_def = enriched.behavior_map()["sniff"]
    assert set(sniff_def.modifier_ids) >= {"loc_head", "subj_m1", "subj_m2"}

    # Round-trip JSON persistence
    schema_file = tmp_path / "project.annolid.json"
    save_schema(enriched, schema_file)
    reloaded = load_schema(schema_file)
    assert len(reloaded.independent_variables) == 1
    assert reloaded.independent_variables[0].allowed_values == ["WT", "KO"]
    assert reloaded.subjects[0].key_binding == "1"
    assert reloaded.behaviors[0].color == "#10b981"
    assert validate_schema(reloaded) == []

    # Shortcut collision detection
    reloaded.subjects[1].key_binding = "s"  # collides with sniff
    warnings = validate_schema(reloaded)
    assert any("Shortcut key" in w or "key" in w.lower() for w in warnings)


def test_boris_project_roundtrip_materialize_and_cohort_integration(
    tmp_path: Path,
) -> None:
    boris_file = tmp_path / "study.boris"
    boris_file.write_text(
        json.dumps(_sample_boris_payload(), indent=2), encoding="utf-8"
    )

    # Direct spec loader compatibility
    loaded_spec, loaded_path = load_behavior_spec(path=boris_file)
    assert loaded_path == boris_file
    assert set(loaded_spec.behavior_map()) == {
        "attack",
        "groom",
        "bite",
        "out_of_sight",
    }
    assert loaded_spec.behavior_map()["bite"].is_state is False

    imported = import_boris_project(boris_file)
    assert imported.project_name == "Resident Intruder Study"
    assert len(imported.observations) == 4
    assert len(imported.schema.independent_variables) == 2

    # Check parsed modifier groups & key bindings from "flank (f)"
    mod_by_name = {m.name: m for m in imported.schema.modifiers}
    assert mod_by_name["flank"].group == "Target"
    assert mod_by_name["flank"].selection_mode == "single"
    assert mod_by_name["flank"].key_binding == "f"
    assert mod_by_name["severe"].selection_mode == "multi"

    # Check 5-tuple point event + 6-tuple state events with comments
    obs1 = next(o for o in imported.observations if o.observation_id == "obs_ctrl_1")
    kinds = [(e.behavior, e.event_kind, e.modifiers, e.comment) for e in obs1.events]
    assert ("attack", "start", ("flank", "mild"), "first lunge") in kinds
    assert ("bite", "point", (), "sharp bite") in kinds

    # Export back to .boris and re-import
    roundtrip_file = tmp_path / "roundtrip.boris"
    write_boris_project(
        roundtrip_file,
        imported.schema,
        observations=imported.observations,
        project_name=imported.project_name,
    )
    reimported = import_boris_project(roundtrip_file)
    assert set(reimported.schema.behavior_map()) == set(imported.schema.behavior_map())
    assert reimported.schema.behavior_map()["bite"].is_state is False
    assert len(reimported.observations) == 4
    re_obs1 = next(
        o for o in reimported.observations if o.observation_id == "obs_ctrl_1"
    )
    assert len(re_obs1.events) == 5

    # Materialize to Annolid workspace + run cohort inferential analysis directly
    mat_dir = tmp_path / "materialized"
    artifacts = materialize_boris_project(
        boris_file, mat_dir, group_variable="Treatment"
    )
    assert artifacts.schema_path.is_file()
    assert artifacts.cohort_manifest_path is not None
    assert artifacts.cohort_manifest_path.is_file()
    assert len(artifacts.observation_csvs) == 4

    cohort_report = run_behavior_cohort(
        manifest_path=artifacts.cohort_manifest_path,
        analysis_unit="recording",
    )
    attack_cmp = [c for c in cohort_report.comparisons if c.behavior == "attack"]
    assert len(attack_cmp) == 1
    assert set(attack_cmp[0].groups) == {"ctrl", "drug"}


def test_boris_tabular_and_aggregated_csv_tsv_roundtrip(tmp_path: Path) -> None:
    imported = import_boris_project(
        _write_json(tmp_path / "src.boris", _sample_boris_payload())
    )
    obs1 = next(o for o in imported.observations if o.observation_id == "obs_ctrl_1")

    tab_csv = write_boris_tabular_csv(
        obs1.events,
        tmp_path / "obs1.tabular.csv",
        observation_id="obs_ctrl_1",
        media_file="videos/ctrl_1.mp4",
        schema=imported.schema,
    )
    # Also test BORIS preamble lines before tabular header
    preamble_tsv = tmp_path / "with_preamble.tsv"
    tab_tsv_rows = tab_csv.read_text(encoding="utf-8").replace(",", "\t")
    preamble_tsv.write_text(
        "Observation id\tobs_ctrl_1\nMedia file\tvideos/ctrl_1.mp4\n\n" + tab_tsv_rows,
        encoding="utf-8",
    )

    parsed_tab = read_parsed_event_rows(preamble_tsv)
    assert len(parsed_tab) == 5
    assert [r.event_kind for r in parsed_tab] == [
        "start",
        "point",
        "end",
        "start",
        "end",
    ]
    assert parsed_tab[1].behavior == "bite"
    assert parsed_tab[1].comment == "sharp bite"

    agg_csv = write_boris_aggregated_csv(
        obs1.events,
        tmp_path / "obs1.aggregated.csv",
        observation_id="obs_ctrl_1",
        schema=imported.schema,
    )
    parsed_agg = read_parsed_event_rows(agg_csv)
    # 2 state intervals (2*2=4 boundary rows) + 1 point row = 5 ParsedEventRows
    assert len(parsed_agg) == 5
    intervals, warnings = extract_behavior_intervals(
        [
            {
                "recording time": r.recording_time,
                "subject": r.subject,
                "behavior": r.behavior,
                "event": r.raw_event,
            }
            for r in parsed_agg
        ]
    )
    assert warnings == []
    by_beh = {(iv.behavior, iv.duration) for iv in intervals}
    assert ("attack", 3.0) in by_beh
    assert ("groom", 3.0) in by_beh
    assert ("bite", 0.0) in by_beh


def _write_json(path: Path, data: dict) -> Path:
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# 2. BORIS Time Budget, Out-of-Sight Subtraction, Co-occurrence, Transitions & SVG
# ---------------------------------------------------------------------------
def test_time_budget_out_of_sight_window_clipping_point_bins_and_cooccurrence() -> None:
    rows = [
        # Subject m1: total span 0..20s, out_of_sight for 5s (15..20s) -> effective 15s!
        {
            "recording time": 0.0,
            "subject": "m1",
            "behavior": "explore",
            "event": "start",
        },
        {
            "recording time": 6.0,
            "subject": "m1",
            "behavior": "explore",
            "event": "stop",
        },
        {"recording time": 3.0, "subject": "m1", "behavior": "sniff", "event": "start"},
        {"recording time": 8.0, "subject": "m1", "behavior": "sniff", "event": "stop"},
        {"recording time": 2.0, "subject": "m1", "behavior": "jump", "event": "point"},
        {"recording time": 5.0, "subject": "m1", "behavior": "jump", "event": "point"},
        {"recording time": 11.0, "subject": "m1", "behavior": "jump", "event": "point"},
        {
            "recording time": 15.0,
            "subject": "m1",
            "behavior": "out_of_sight",
            "event": "start",
        },
        {
            "recording time": 20.0,
            "subject": "m1",
            "behavior": "out_of_sight",
            "event": "stop",
        },
    ]
    summary, warnings = compute_time_budget(
        rows,
        session_duration=20.0,
        exclude_behaviors=["out_of_sight"],
        omit_excluded_rows=True,
    )
    assert warnings == []
    by_b = {r.behavior: r for r in summary}
    assert "out_of_sight" not in by_b

    # Explore lasted 6s out of (20s - 5s = 15s) visible time => 40.0%!
    assert math.isclose(by_b["explore"].total_duration, 6.0, rel_tol=1e-6)
    assert math.isclose(by_b["explore"].percent_of_session, 40.0, rel_tol=1e-5)

    # Jump had 3 point events at 2s, 5s, 11s -> IEIs = [3.0, 6.0], mean IEI = 4.5s, latency = 2.0s
    # Rate per min over 15s (0.25 min) = 3 / 0.25 = 12.0 / min
    jump = by_b["jump"]
    assert jump.occurrences == 3
    assert math.isclose(jump.total_duration, 0.0, abs_tol=1e-9)
    assert math.isclose(jump.latency_to_first, 2.0, rel_tol=1e-6)
    assert jump.inter_event_interval_mean is not None
    assert math.isclose(jump.inter_event_interval_mean, 4.5, rel_tol=1e-6)
    assert math.isclose(jump.rate_per_min, 12.0, rel_tol=1e-5)

    # Window clipping [2.0, 5.0] => explore clipped to 3.0s
    clipped_summary, _ = compute_time_budget(rows, start_time=2.0, end_time=5.0)
    clipped_explore = next(r for r in clipped_summary if r.behavior == "explore")
    assert math.isclose(clipped_explore.total_duration, 3.0, rel_tol=1e-6)
    assert math.isclose(clipped_explore.percent_of_session, 100.0, rel_tol=1e-6)

    # Binned time budget counts point events in occurrences
    intervals, _ = extract_behavior_intervals(rows)
    binned = compute_binned_time_budget(intervals, bin_size=10.0)
    jump_bin0 = next(
        b for b in binned if b.behavior == "jump" and math.isclose(b.bin_start, 0.0)
    )
    jump_bin1 = next(
        b for b in binned if b.behavior == "jump" and math.isclose(b.bin_start, 10.0)
    )
    assert jump_bin0.occurrences == 2  # at 2.0s and 5.0s
    assert jump_bin1.occurrences == 1  # at 11.0s

    # Co-occurrence between explore [0,6] and sniff [3,8] => overlap [3,6] = 3.0s, union [0,8] = 8.0s
    coocc = compute_behavior_cooccurrence(intervals)
    pair = next(
        c for c in coocc if {c.behavior_a, c.behavior_b} == {"explore", "sniff"}
    )
    assert pair.overlap_bouts == 1
    assert math.isclose(pair.overlap_duration, 3.0, rel_tol=1e-6)
    assert math.isclose(pair.jaccard_index, 3.0 / 8.0, rel_tol=1e-6)

    # Transitions + BORIS grids + DOT + SVG + Ethogram SVG
    edges = compute_observation_transitions(intervals)
    assert len(edges) >= 1
    behaviors_order, prob_grid = build_boris_transition_grid(edges, mode="probability")
    _, count_grid = build_boris_transition_grid(edges, mode="count")
    assert len(behaviors_order) >= 2
    assert sum(sum(r) for r in count_grid) >= 1.0
    assert all(0.0 <= cell <= 1.0 + 1e-6 for row in prob_grid for cell in row)

    dot_src = export_transitions_graphviz_dot(edges, min_probability=0.0)
    assert "digraph BehavioralTransitions" in dot_src
    assert "->" in dot_src

    trans_svg = render_transitions_svg(edges, min_probability=0.0)
    assert trans_svg.startswith("<svg") and trans_svg.endswith("</svg>")

    etho_svg = render_ethogram_svg(intervals, exclude_behaviors=["out_of_sight"])
    assert (
        etho_svg.startswith("<svg") and "<polygon" in etho_svg and "<rect" in etho_svg
    )


def test_ethogram_matplotlib_pairs_raw_start_stop_and_point_events(
    tmp_path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    csv_path = tmp_path / "events.csv"
    parsed = parse_event_mapping_rows(
        [
            {
                "Recording time": 1.0,
                "Subject": "m1",
                "Behavior": "groom",
                "Event": "start",
            },
            {
                "Recording time": 4.0,
                "Subject": "m1",
                "Behavior": "groom",
                "Event": "stop",
            },
            {
                "Recording time": 2.5,
                "Subject": "m1",
                "Behavior": "bite",
                "Event": "point",
            },
        ]
    )
    write_annolid_events_csv(parsed, csv_path)
    out_png = tmp_path / "ethogram.png"
    fig = create_ethogram(pd.read_csv(csv_path))
    fig.savefig(str(out_png))
    plt.close(fig)
    assert out_png.is_file() and out_png.stat().st_size > 500


# ---------------------------------------------------------------------------
# 3. CLI Subcommands (boris-import, boris-export, upgraded time-budget)
# ---------------------------------------------------------------------------
def test_cli_boris_import_export_and_time_budget_flags(tmp_path: Path) -> None:
    boris_path = _write_json(tmp_path / "project.boris", _sample_boris_payload())
    import_dir = tmp_path / "imported_cli"

    rc = cli_main(
        [
            "boris-import",
            str(boris_path),
            "--output-dir",
            str(import_dir),
            "--group-variable",
            "Treatment",
        ]
    )
    assert rc == 0
    assert (import_dir / "project.annolid.json").is_file()
    assert (import_dir / "cohort_manifest.json").is_file()
    obs_csv = import_dir / "obs_ctrl_1_events.csv"
    assert obs_csv.is_file()

    # Run upgraded time-budget CLI with BORIS options
    tb_csv = tmp_path / "tb.csv"
    coocc_csv = tmp_path / "coocc.csv"
    etho_svg = tmp_path / "etho.svg"
    trans_dot = tmp_path / "trans.dot"
    trans_svg = tmp_path / "trans.svg"
    rc_tb = cli_main(
        [
            "time-budget",
            str(obs_csv),
            "--schema",
            str(import_dir / "project.annolid.json"),
            "--exclude-behaviors",
            "out_of_sight",
            "--output-csv",
            str(tb_csv),
            "--cooccurrence-csv",
            str(coocc_csv),
            "--ethogram-svg",
            str(etho_svg),
            "--transitions-dot",
            str(trans_dot),
            "--transitions-svg",
            str(trans_svg),
        ]
    )
    assert rc_tb == 0
    assert tb_csv.is_file()
    assert etho_svg.is_file()
    assert trans_dot.is_file()
    assert trans_svg.is_file()

    # Export back via CLI boris-export in all 3 formats
    exported_boris = tmp_path / "reexported.boris"
    assert (
        cli_main(
            [
                "boris-export",
                str(obs_csv),
                "--schema",
                str(import_dir / "project.annolid.json"),
                "--format",
                "boris",
                "--output",
                str(exported_boris),
            ]
        )
        == 0
    )
    assert exported_boris.is_file()

    exported_agg = tmp_path / "reexported_agg.csv"
    assert (
        cli_main(
            [
                "boris-export",
                str(obs_csv),
                "--schema",
                str(import_dir / "project.annolid.json"),
                "--format",
                "aggregated-csv",
                "--output",
                str(exported_agg),
            ]
        )
        == 0
    )
    assert exported_agg.is_file()


# ---------------------------------------------------------------------------
# 4. Headless Qt GUI Regression Tests
# ---------------------------------------------------------------------------
def test_project_dialog_preserves_exclusions_subjects_as_modifiers_and_vars() -> None:
    _ensure_qapp()
    initial = ProjectSchema(
        subjects=[
            SubjectDefinition(id="res", name="resident", key_binding="1"),
            SubjectDefinition(id="int", name="intruder", key_binding="2"),
        ],
        behaviors=[
            BehaviorDefinition(
                code="attack",
                name="Attack",
                key_binding="a",
                is_state=True,
                exclusive_with=["groom"],
            ),
            BehaviorDefinition(
                code="groom",
                name="Groom",
                key_binding="g",
                is_state=True,
                exclusive_with=["attack"],
            ),
            BehaviorDefinition(
                code="bite",
                name="Bite",
                key_binding="b",
                is_state=False,
            ),
        ],
        independent_variables=[
            IndependentVariableDefinition(
                id="lighting",
                name="Lighting",
                var_type="value_from_list",
                default_value="red",
                allowed_values=["red", "white"],
            )
        ],
    )

    dlg = ProjectDialog(initial)
    try:
        # Regression check: get_schema() MUST preserve exclusive_with without wiping it!
        collected = dlg.get_schema()
        assert collected.behavior_map()["attack"].exclusive_with == ["groom"]
        assert collected.behavior_map()["groom"].exclusive_with == ["attack"]
        assert collected.behavior_map()["bite"].is_state is False
        assert len(collected.independent_variables) == 1
        assert collected.independent_variables[0].allowed_values == ["red", "white"]

        # Test "Add Subjects as Modifiers" button
        dlg._add_subjects_as_modifiers()
        after_subj_mods = dlg.get_schema()
        mod_names = {
            m.name for m in after_subj_mods.modifiers if m.group == "Target Subject"
        }
        assert mod_names == {"resident", "intruder"}
    finally:
        dlg.deleteLater()


def test_behavior_controls_single_vs_multi_modifiers_ethogram_pad_and_comments() -> (
    None
):
    _ensure_qapp()
    controls = BehaviorControlsWidget()
    try:
        controls.set_subjects(
            [
                SubjectDefinition(id="m1", name="mouse_1", key_binding="1"),
                SubjectDefinition(id="m2", name="mouse_2", key_binding="2"),
            ]
        )
        controls.set_behaviors(
            [
                BehaviorDefinition(
                    code="attack", name="Attack", key_binding="a", is_state=True
                ),
                BehaviorDefinition(
                    code="bite", name="Bite", key_binding="b", is_state=False
                ),
            ]
        )
        controls.set_modifiers(
            [
                ModifierDefinition(
                    id="loc_flank",
                    name="Flank",
                    group="Target",
                    selection_mode="single",
                    key_binding="f",
                ),
                ModifierDefinition(
                    id="loc_back",
                    name="Back",
                    group="Target",
                    selection_mode="single",
                    key_binding="d",
                ),
                ModifierDefinition(
                    id="int_mild",
                    name="Mild",
                    group="Intensity",
                    selection_mode="multi",
                ),
                ModifierDefinition(
                    id="int_high",
                    name="High",
                    group="Intensity",
                    selection_mode="multi",
                ),
            ]
        )

        # Single-selection group enforces radio exclusivity
        controls._modifier_buttons["loc_flank"].click()
        assert controls.checked_modifiers() == {"loc_flank"}
        controls._modifier_buttons["loc_back"].click()
        assert controls.checked_modifiers() == {"loc_back"}

        # Multi-selection group allows simultaneous checks
        controls._modifier_buttons["int_mild"].click()
        controls._modifier_buttons["int_high"].click()
        assert set(controls.checked_modifiers()) == {"loc_back", "int_mild", "int_high"}

        # Ethogram Pad click emits behaviorTriggerRequested
        triggered: list[str] = []
        controls.behaviorTriggerRequested.connect(triggered.append)
        controls._behavior_buttons["bite"].click()
        assert triggered == ["bite"]

        # Active state badge shows ● on active state button
        controls.set_active_behaviors({"attack"})
        assert "●" in controls._behavior_buttons["attack"].text()
        assert "●" not in controls._behavior_buttons["bite"].text()

        # Comment auto-clear on take_event_comment
        controls.set_comment("clear contact")
        assert controls.take_event_comment() == "clear contact"
        assert controls.current_comment() == ""
    finally:
        controls.deleteLater()


class _CodingHarness(
    QtWidgets.QMainWindow, BehaviorInteractionMixin, SchemaBehaviorLoaderMixin
):
    """Lightweight window harness exercising live BORIS coding, flags, log, and loader."""

    def __init__(self, schema: ProjectSchema, tmp_dir: Path) -> None:
        super().__init__()
        self.settings = QtCore.QSettings(
            str(tmp_dir / "annolid_boris_harness.ini"),
            QtCore.QSettings.IniFormat,
        )
        self.project_schema = schema
        self.project_schema_path = None
        self.fps = 25.0
        self.num_frames = 500
        self.frame_number = 0
        self.event_type = None
        self._active_subject_name = None
        self._behavior_modifier_state: dict[str, set[str]] = {}
        self._playback_speed = 1.0
        self.isPlaying = False
        self.canvas = SimpleNamespace(
            selectedShapes=[],
            setBehaviorText=lambda *_a: None,
        )
        self.flag_widget = FlagTableWidget()
        self.seekbar = VideoSlider(self)
        self.seekbar.setMinimum(0)
        self.seekbar.setMaximum(499)

        self.behavior_controller = BehaviorController(lambda _b: "blue")
        self.behavior_controller.configure_from_schema(schema)
        self.behavior_controller.attach_slider(self.seekbar)
        self.behavior_controls_widget = BehaviorControlsWidget(self)
        self.behavior_log_widget = BehaviorEventLogWidget(self)
        self.flags_controller = FlagsController(
            window=self,
            widget=self.flag_widget,
            config_path=tmp_dir / "behaviors.yaml",
        )
        self.pinned_flags = self.flags_controller.pinned_flags

        self.behavior_controls_widget.subjectChanged.connect(
            self._on_active_subject_changed
        )
        self.behavior_controls_widget.modifierToggled.connect(self._on_modifier_toggled)
        self.behavior_controls_widget.behaviorTriggerRequested.connect(
            self.trigger_ethogram_behavior
        )

        self._populate_behavior_controls_from_schema(schema)
        self._sync_behavior_flags_from_schema(schema)

    def loadFlags(self, flags: dict[str, bool]) -> None:  # noqa: N802
        self.flags_controller.load_flags(flags)

    def _update_pinned_flag(self, behavior: str, checked: bool) -> None:
        self.flags_controller.pinned_flags[behavior] = bool(checked)

    def set_frame_number(self, frame: int) -> None:
        self.frame_number = int(frame)
        self._refresh_ethogram_active_badges(self.frame_number)
        self._sync_boris_media_and_streams(self.frame_number)

    def _refresh_behavior_log(self) -> None:
        self.behavior_log_widget.set_events(
            list(self.behavior_controller.iter_events()),
            fps=self.fps,
        )


def test_gui_live_keyboard_coding_multi_subject_auto_exclusion_and_point_persistence(
    tmp_path: Path,
) -> None:
    _ensure_qapp()
    schema = ProjectSchema(
        subjects=[
            SubjectDefinition(id="m1", name="mouse_1", key_binding="1"),
            SubjectDefinition(id="m2", name="mouse_2", key_binding="2"),
        ],
        modifiers=[
            ModifierDefinition(
                id="flank",
                name="flank",
                group="Target",
                selection_mode="single",
                key_binding="f",
            )
        ],
        behaviors=[
            BehaviorDefinition(
                code="attack",
                name="Attack",
                modifier_ids=["flank"],
                key_binding="a",
                is_state=True,
                exclusive_with=["groom"],
            ),
            BehaviorDefinition(
                code="groom",
                name="Groom",
                key_binding="g",
                is_state=True,
                exclusive_with=["attack"],
            ),
            BehaviorDefinition(
                code="bite",
                name="Bite",
                key_binding="b",
                is_state=False,
            ),
        ],
    )

    store = AnnotationStore(tmp_path / "annotations.jsonl")
    win = _CodingHarness(schema, tmp_path)
    win.behavior_controller.attach_annotation_store(store)
    timeline_panel = TimelinePanel(win)
    timeline_panel.set_behavior_controller(win.behavior_controller)
    timeline_panel.set_time_range(0, 499)
    try:
        # Modifiers must NOT all be pre-checked on initial schema load!
        assert win._selected_modifiers_for_behavior("attack") == []

        # Enable Live Keyboard Coding
        win.behavior_controls_widget.set_coding_mode_enabled(True)

        def _press(char: str) -> None:
            ev = QtGui.QKeyEvent(QtCore.QEvent.KeyPress, 0, QtCore.Qt.NoModifier, char)
            assert win._handle_boris_coding_keypress(ev) is True

        # Frame 10: select mouse_1 ('1'), start 'attack' ('a'), toggle modifier ('f')
        win.set_frame_number(10)
        _press("1")
        assert win._current_subject_name() == "mouse_1"
        win.behavior_controls_widget.set_comment("onset lunge")
        _press("a")
        assert win.behavior_controller.is_behavior_active(
            10, "attack", subject="mouse_1"
        )
        assert not win.behavior_controller.is_behavior_active(
            10, "attack", subject="mouse_2"
        )

        # Simultaneous multi-subject state at the exact same frame 10 for mouse_2!
        _press("2")
        assert win._current_subject_name() == "mouse_2"
        _press("g")  # start groom for mouse_2 at frame 10
        assert win.behavior_controller.is_behavior_active(
            10, "groom", subject="mouse_2"
        )
        assert win.behavior_controller.is_behavior_active(
            10, "attack", subject="mouse_1"
        )

        # Frame 25: switch back to mouse_1, fire point event 'bite' ('b')
        win.set_frame_number(25)
        _press("1")
        _press("b")

        # Frame 40: start mutually-exclusive 'groom' ('g') for mouse_1
        # -> auto-stops mouse_1's 'attack' at frame 40 WITHOUT stopping mouse_2's 'groom'!
        win.set_frame_number(40)
        _press("g")
        assert not win.behavior_controller.is_behavior_active(
            45, "attack", subject="mouse_1"
        )
        assert win.behavior_controller.is_behavior_active(
            45, "groom", subject="mouse_1"
        )
        assert win.behavior_controller.is_behavior_active(
            45, "groom", subject="mouse_2"
        )

        # Frame 60: stop groom for mouse_1 and mouse_2
        win.set_frame_number(60)
        _press("g")  # stops mouse_1 groom
        _press("2")
        _press("g")  # stops mouse_2 groom

        # Verify BehaviorEventLogWidget columns (8=Duration, 9=Status, 10=Comment)
        log_table = win.behavior_log_widget._table
        assert log_table.columnCount() == 11
        assert log_table.horizontalHeaderItem(8).text() == "Duration (s)"
        assert log_table.horizontalHeaderItem(9).text() == "Status"
        assert log_table.horizontalHeaderItem(10).text() == "Comment"

        # Point event 'bite' has Duration "0.00" and clean "Confirmed" status (no pairing issue)
        bite_rows = [
            r
            for r in range(log_table.rowCount())
            if log_table.item(r, 1).text() == "bite"
        ]
        assert len(bite_rows) == 1
        assert log_table.item(bite_rows[0], 8).text() == "0.00"
        assert log_table.item(bite_rows[0], 9).text() == "Confirmed"

        # Verify TimelinePanel builds both state intervals and point events cleanly across modes
        timeline_panel.refresh_from_behavior_controller()
        point_items = [e for e in timeline_panel._model.events if e.kind == "point"]
        assert len(point_items) == 1
        assert point_items[0].behavior == "bite"

        timeline_panel._row_mode_combo.setCurrentText("Behavior+Instance")
        timeline_panel.refresh_from_behavior_controller()
        bi_points = [e for e in timeline_panel._model.events if e.kind == "point"]
        assert len(bi_points) == 1 and bi_points[0].subject == "mouse_1"

        # Verify AnnotationStore persistence & reload preserves point events + comments
        reloaded_ctrl = BehaviorController(lambda _b: "blue")
        reloaded_ctrl.configure_from_schema(schema)
        reloaded_ctrl.load_events_from_store(store)
        reloaded_events = list(reloaded_ctrl.iter_events())
        assert any(e.behavior == "bite" and e.event == "point" for e in reloaded_events)
        assert any(
            e.behavior == "attack" and e.comment == "onset lunge"
            for e in reloaded_events
        )

        # Verify SchemaBehaviorLoaderMixin._load_behavior loads BORIS Aggregated CSV directly
        agg_file = tmp_path / "exported.aggregated.csv"
        detailed = reloaded_ctrl.export_detailed_rows()
        write_boris_aggregated_csv(
            parse_event_mapping_rows(detailed),
            agg_file,
            schema=schema,
        )
        win._load_behavior(str(agg_file))
        after_agg = list(win.behavior_controller.iter_events())
        assert any(e.behavior == "bite" and e.event == "point" for e in after_agg)
        assert win.behavior_controller.is_behavior_active(
            20, "attack", subject="mouse_1"
        )
    finally:
        timeline_panel.deleteLater()
        win.deleteLater()


# ---------------------------------------------------------------------------
# 5. Sub-Frame Timestamp Preservation, FPS Re-Indexing & Synchronized Media/Data
# ---------------------------------------------------------------------------
def test_subframe_event_preservation_rapid_bouts_and_rebind_fps(
    tmp_path: Path,
) -> None:
    store = AnnotationStore(tmp_path / "subframe.jsonl")
    ctrl = BehaviorController(lambda _b: "green")
    ctrl.attach_annotation_store(store)

    # Load 3 lick point events at 1.002s, 1.014s (both round to frame 25 @ 25fps!) and 1.080s (frame 27)
    # plus a rapid sub-frame state bout [2.002s, 2.015s] (both round to frame 50 @ 25fps!)
    rows = parse_event_mapping_rows(
        [
            {
                "Recording time": 1.002,
                "Subject": "m1",
                "Behavior": "lick",
                "Event": "point",
            },
            {
                "Recording time": 1.014,
                "Subject": "m1",
                "Behavior": "lick",
                "Event": "point",
            },
            {
                "Recording time": 1.080,
                "Subject": "m1",
                "Behavior": "lick",
                "Event": "point",
            },
            {
                "Recording time": 2.002,
                "Subject": "m1",
                "Behavior": "whisk",
                "Event": "start",
            },
            {
                "Recording time": 2.015,
                "Subject": "m1",
                "Behavior": "whisk",
                "Event": "stop",
            },
        ]
    )
    ctrl.load_events_from_rows(rows, time_to_frame=lambda t: int(round(t * 25.0)))

    # Zero sub-frame events lost despite colliding on frame 25!
    all_evts = list(ctrl.iter_events())
    assert len(all_evts) == 5
    licks = [e for e in all_evts if e.behavior == "lick"]
    assert [e.frame for e in licks] == [25, 25, 27]
    assert [e.timestamp for e in licks] == [1.002, 1.014, 1.08]
    assert licks[0].storage_key != licks[1].storage_key

    # Rapid sub-frame bout [2.002s, 2.015s] paired cleanly as (50, 50)
    assert ctrl.timeline.get_behavior_ranges("whisk", subject="m1") == [(50, 50)]

    # Reload from AnnotationStore preserves all 5 sub-frame events!
    reloaded = BehaviorController(lambda _b: "green")
    reloaded.load_events_from_store(store)
    assert reloaded.events_count == 5

    # Rebind FPS from 25 Hz -> 100 Hz separates 1.002s (frame 100) and 1.014s (frame 101)!
    assert ctrl.rebind_fps(100.0, max_frame=999) is True
    rebound_licks = [e for e in ctrl.iter_events() if e.behavior == "lick"]
    assert [e.frame for e in rebound_licks] == [100, 101, 108]
    assert [e.timestamp for e in rebound_licks] == [1.002, 1.014, 1.08]
    assert ctrl.timeline.get_behavior_ranges("whisk", subject="m1") == [(200, 202)]


def test_external_data_streams_envelope_decimation_boris_plot_data_and_cli(
    tmp_path: Path,
) -> None:
    import numpy as np

    from annolid.behavior.boris_io import BorisObservation
    from annolid.behavior.external_data import (
        ExternalDataStreamSpec,
        load_external_data_stream,
        render_external_data_svg,
        summarize_external_stream_by_behavior,
    )

    # Create a 2,000-sample 100 Hz photometry CSV (0.00s..19.99s) with baseline=5.0 and a sharp spike +12.0 at t=5.00s
    photom_csv = tmp_path / "photometry.csv"
    t_vals = np.arange(2000, dtype=np.float64) * 0.01
    y_vals = np.full_like(t_vals, 5.0)
    y_vals[200:400] = 7.0  # elevated during [2.00s, 3.99s] -> +2.0 above baseline
    y_vals[500] = 17.0  # 10ms transient spike (+12.0 above baseline) at t=5.00s
    lines = ["time_s,dff"] + [f"{t:.3f},{y:.4f}" for t, y in zip(t_vals, y_vals)]
    photom_csv.write_text("\n".join(lines) + "\n", encoding="utf-8")

    spec = ExternalDataStreamSpec(
        file_path="photometry.csv",
        columns="1,2",
        variable_name="GCaMP_dFF",
        color="#16a34a",
        time_offset_s=0.5,  # shifts stream forward by +0.5s in video time!
        time_interval_s=20.0,
        subtract_first_value=True,
    )
    stream = load_external_data_stream(spec, base_dir=tmp_path)
    assert stream.sample_count == 2000
    # At video time 0.5s (raw 0.0s), baseline-subtracted value is 0.0
    assert math.isclose(stream.value_at_time(0.5) or 0.0, 0.0, abs_tol=1e-6)
    # At video time 3.5s (raw 3.0s), value is 7.0 - 5.0 = 2.0
    assert math.isclose(stream.value_at_time(3.5) or 0.0, 2.0, abs_tol=1e-5)

    # Min-max envelope decimation down to 80 points MUST preserve the 1-sample +12.0 spike!
    dec_t, dec_v = stream.window_slice(10.0, 25.0, max_points=80)
    assert dec_v.size <= 80
    assert math.isclose(float(np.max(dec_v)), 12.0, rel_tol=1e-6)

    # Behavior-aligned bout summary over video interval [2.5s, 4.5s] (= raw [2.0s, 4.0s])
    intervals, _ = extract_behavior_intervals(
        [
            {
                "recording time": 2.5,
                "subject": "m1",
                "behavior": "attack",
                "event": "start",
            },
            {
                "recording time": 4.5,
                "subject": "m1",
                "behavior": "attack",
                "event": "stop",
            },
        ]
    )
    bout_stats = summarize_external_stream_by_behavior(stream, intervals)
    assert len(bout_stats) == 1
    assert bout_stats[0].behavior == "attack"
    assert bout_stats[0].mean_value > 1.9

    svg = render_external_data_svg([stream], intervals=intervals, playhead_time=3.5)
    assert svg.startswith("<svg") and "<polyline" in svg

    # Multi-camera + plot_data .boris round-trip
    schema = ProjectSchema(
        behaviors=[BehaviorDefinition(code="attack", name="Attack", is_state=True)]
    )
    obs = BorisObservation(
        observation_id="multi_cam_obs",
        media_files=("top.mp4", "side.mp4"),
        media_by_player={"1": ("top.mp4",), "2": ("side.mp4",)},
        media_offsets={"1": 0.0, "2": -0.75},
        external_data=(spec,),
    )
    boris_out = write_boris_project(tmp_path / "sync.boris", schema, observations=[obs])
    reimp = import_boris_project(boris_out)
    robs = reimp.observations[0]
    assert robs.media_by_player == {"1": ("top.mp4",), "2": ("side.mp4",)}
    assert math.isclose(robs.media_offsets["2"], -0.75, rel_tol=1e-6)
    assert len(robs.external_data) == 1
    assert robs.external_data[0].variable_name == "GCaMP_dFF"
    assert robs.external_data[0].subtract_first_value is True

    # CLI time-budget with --external-data and --external-data-svg
    ev_csv = write_annolid_events_csv(
        parse_event_mapping_rows(
            [
                {
                    "Recording time": 2.5,
                    "Subject": "m1",
                    "Behavior": "attack",
                    "Event": "start",
                },
                {
                    "Recording time": 4.5,
                    "Subject": "m1",
                    "Behavior": "attack",
                    "Event": "stop",
                },
            ]
        ),
        tmp_path / "ev.csv",
    )
    ext_svg = tmp_path / "ext_cli.svg"
    tb_json = tmp_path / "tb_ext.json"
    assert (
        cli_main(
            [
                "time-budget",
                str(ev_csv),
                "--external-data",
                str(photom_csv),
                "--external-offset",
                "0.5",
                "--external-data-svg",
                str(ext_svg),
                "--json-out",
                str(tb_json),
            ]
        )
        == 0
    )
    assert ext_svg.is_file()
    payload = json.loads(tb_json.read_text(encoding="utf-8"))
    assert len(payload["external_data_summary"]) == 1


def test_boris_sync_dock_multi_camera_offset_and_external_plot_scrubbing(
    tmp_path: Path,
) -> None:
    import cv2
    import numpy as np

    from annolid.behavior.boris_io import BorisObservation
    from annolid.behavior.external_data import ExternalDataStreamSpec

    _ensure_qapp()

    # Write a tiny 20-frame 10.0 fps secondary MJPG .avi where frame k has R = k * 10
    sec_video = tmp_path / "side_cam.avi"
    writer = cv2.VideoWriter(
        str(sec_video),
        cv2.VideoWriter_fourcc(*"MJPG"),
        10.0,
        (64, 48),
    )
    for k in range(20):
        bgr = np.zeros((48, 64, 3), dtype=np.uint8)
        bgr[:, :, 2] = min(255, k * 10)
        writer.write(bgr)
    writer.release()
    assert sec_video.is_file()

    ext_csv = tmp_path / "heart_rate.csv"
    ext_csv.write_text(
        "time,bpm\n0.0,320\n1.0,360\n2.0,410\n3.0,380\n", encoding="utf-8"
    )

    schema = ProjectSchema(
        subjects=[SubjectDefinition(id="m1", name="mouse_1", key_binding="1")],
        behaviors=[
            BehaviorDefinition(code="run", name="Run", key_binding="r", is_state=True)
        ],
    )
    win = _CodingHarness(schema, tmp_path)
    try:
        obs = BorisObservation(
            observation_id="obs_sync",
            media_by_player={"1": ("primary.mp4",), "2": ("side_cam.avi",)},
            media_offsets={"1": 0.0, "2": 0.5},  # +0.5s camera 2 offset!
            external_data=(
                ExternalDataStreamSpec(
                    file_path="heart_rate.csv",
                    columns="time,bpm",
                    variable_name="HeartRate",
                    time_interval_s=4.0,
                ),
            ),
        )
        win._apply_boris_observation_sync_assets(obs, tmp_path)
        sync_w = win.boris_sync_widget
        assert len(sync_w.secondary_tiles) == 1
        assert len(sync_w.external_streams) == 1

        # Primary win is 25.0 fps -> frame 25 = 1.00s primary time.
        # Secondary side_cam has +0.5s offset & 10.0 fps -> (1.0 + 0.5) * 10 = frame 15!
        win.set_frame_number(25)
        sync_w.sync_to_time(1.0, immediate=True)
        assert sync_w.secondary_tiles[0].displayed_frame == 15

        # Offscreen render of QPainter plot_canvas succeeds without error
        sync_w.resize(640, 420)
        pix = sync_w.plot_canvas.grab()
        assert not pix.isNull()

        # Click-scrubbing plot_canvas emits timeJumpRequested and jumps primary window
        jumped_times: list[float] = []
        sync_w.timeJumpRequested.connect(jumped_times.append)
        press_ev = QtGui.QMouseEvent(
            QtCore.QEvent.MouseButtonPress,
            QtCore.QPointF(sync_w.plot_canvas.width() * 0.75, 80.0),
            QtCore.Qt.LeftButton,
            QtCore.Qt.LeftButton,
            QtCore.Qt.NoModifier,
        )
        sync_w.plot_canvas.mousePressEvent(press_ev)
        rel_ev = QtGui.QMouseEvent(
            QtCore.QEvent.MouseButtonRelease,
            QtCore.QPointF(sync_w.plot_canvas.width() * 0.75, 80.0),
            QtCore.Qt.LeftButton,
            QtCore.Qt.NoButton,
            QtCore.Qt.NoModifier,
        )
        sync_w.plot_canvas.mouseReleaseEvent(rel_ev)
        assert len(jumped_times) == 1 and jumped_times[0] > 1.0

        # Export back to BORIS structures
        by_p, offs, ext_specs = sync_w.export_boris_media_and_plots("primary.mp4")
        assert set(by_p.keys()) == {"1", "2"}
        assert math.isclose(offs["2"], 0.5, rel_tol=1e-6)
        assert len(ext_specs) == 1 and ext_specs[0].variable_name == "HeartRate"
    finally:
        if hasattr(win, "boris_sync_widget") and win.boris_sync_widget is not None:
            win.boris_sync_widget.shutdown()
        win.deleteLater()


def test_safe_boris_converters_execution_roundtrip_and_security_rejection(
    tmp_path: Path,
) -> None:
    from annolid.behavior.external_data import (
        UnsafeConverterError,
        evaluate_safe_boris_converter,
        load_external_data_stream,
    )

    # 1. Built-in and multi-statement BORIS INPUT/OUTPUT converters
    assert math.isclose(
        evaluate_safe_boris_converter("HHMMSS_2_seconds", "00:01:30.500"),
        90.5,
        rel_tol=1e-6,
    )
    snippet = (
        "h, m, s = INPUT.split(':')\nOUTPUT = int(h) * 3600 + int(m) * 60 + float(s)"
    )
    assert math.isclose(
        evaluate_safe_boris_converter(snippet, "01:02:03.25"),
        3723.25,
        rel_tol=1e-6,
    )
    assert math.isclose(
        evaluate_safe_boris_converter("OUTPUT = float(INPUT) * 0.001 + 2.0", "500"),
        2.5,
        rel_tol=1e-6,
    )

    # 2. Adversarial / unsafe AST constructs MUST raise UnsafeConverterError
    malicious_payloads = [
        "import os\nOUTPUT = 1.0",
        "__import__('os').system('echo pwned')",
        "OUTPUT = INPUT.__class__.__mro__",
        "OUTPUT = open('/etc/passwd')",
        "OUTPUT = [x for x in INPUT]",
        "OUTPUT = 2 ** 5000",
    ]
    for bad in malicious_payloads:
        try:
            evaluate_safe_boris_converter(bad, "10")
        except UnsafeConverterError:
            pass
        else:
            raise AssertionError(f"Expected UnsafeConverterError for payload: {bad!r}")

    # 3. End-to-end .boris top-level 'converters' + plot_data['converters'] round-trip
    iso_csv = tmp_path / "iso_sensor.csv"
    iso_csv.write_text(
        "ts_iso,mv\n"
        "2026-05-01T10:00:00.000Z,1000\n"
        "2026-05-01T10:00:01.500Z,2500\n"
        "2026-05-01T10:00:03.000Z,4000\n",
        encoding="utf-8",
    )
    top_converters = {
        "mv_to_v": {
            "name": "mv_to_v",
            "description": "Millivolts to Volts",
            "code": "OUTPUT = float(INPUT) / 1000.0",
        }
    }
    boris_file = tmp_path / "with_converters.boris"
    raw_boris = _sample_boris_payload()
    raw_boris["converters"] = top_converters
    raw_boris["observations"]["obs_ctrl_1"]["plot_data"] = {
        "0": {
            "file_path": "iso_sensor.csv",
            "columns": "1,2",
            "variable_name": "Sensor_V",
            "converters": {"1": "ISO8601_2_seconds", "2": "mv_to_v"},
            "substract_first_value": "True",
            "time_offset": 0.0,
            "time_interval": 10,
            "color": "#2563eb",
        }
    }
    boris_file.write_text(json.dumps(raw_boris), encoding="utf-8")

    imported = import_boris_project(boris_file)
    assert "mv_to_v" in imported.converters
    obs1 = next(o for o in imported.observations if o.observation_id == "obs_ctrl_1")
    stream = load_external_data_stream(obs1.external_data[0], base_dir=tmp_path)
    # Subtracting first value: [1.0, 2.5, 4.0] -> [0.0, 1.5, 3.0] over 3.0s span!
    assert math.isclose(
        float(stream.times_raw[-1] - stream.times_raw[0]), 3.0, rel_tol=1e-6
    )
    assert [round(float(v), 3) for v in stream.values] == [0.0, 1.5, 3.0]

    # Re-exporting preserves top-level 'converters' automatically from spec.converter_definitions
    reexported = tmp_path / "reexported_conv.boris"
    write_boris_project(reexported, imported.schema, observations=[obs1])
    re_json = json.loads(reexported.read_text(encoding="utf-8"))
    assert "mv_to_v" in re_json["converters"]


def test_synchronized_multi_camera_audio_mixing_offset_headroom_and_gui_controls(
    tmp_path: Path,
) -> None:
    import cv2
    import numpy as np

    from annolid.core.media.audio import (
        AudioBuffer,
        AudioTrackMixSpec,
        mix_synchronized_audio_buffers,
    )
    from annolid.gui.widgets.boris_sync_dock import BorisSyncWidget

    _ensure_qapp()

    # Primary 24 kHz constant +0.8 signal (2.0s); Secondary 48 kHz constant +0.8 signal (1.0s)
    prim_buf = AudioBuffer(
        audio_data=np.full(24_000 * 2, 0.8, dtype=np.float32),
        sample_rate=24_000,
        fps=30.0,
    )
    sec_buf = AudioBuffer(
        audio_data=np.full(48_000 * 1, 0.8, dtype=np.float32),
        sample_rate=48_000,
        fps=30.0,
    )

    # Secondary offset = -0.5s at primary_start=0.0s => 0.5s (24,000 samples @ 48kHz) leading delay,
    # then [0.5s, 1.5s] overlaps primary+secondary (0.8 + 0.8 = 1.6 -> headroom-limited to <= 1.0!)
    mixed, sr = mix_synchronized_audio_buffers(
        [
            AudioTrackMixSpec(buffer=prim_buf, offset_s=0.0, gain=1.0),
            AudioTrackMixSpec(buffer=sec_buf, offset_s=-0.5, gain=1.0),
        ],
        primary_start_sec=0.0,
        target_sample_rate=48_000,
    )
    assert sr == 48_000
    assert mixed.shape[0] == 96_000
    assert float(np.max(np.abs(mixed))) <= 1.0 + 1e-6
    # Before 0.5s (sample 10,000), only primary contributes (0.8 / 1.6 = 0.5);
    # during [0.5s, 1.5s] (sample 48,000), both contribute (1.6 / 1.6 = 1.0)!
    assert math.isclose(float(mixed[10_000]), 0.5, abs_tol=1e-3)
    assert math.isclose(float(mixed[48_000]), 1.0, abs_tol=1e-3)

    # 2.0x playback speed halves wall-clock output length
    fast_mix, _ = mix_synchronized_audio_buffers(
        [AudioTrackMixSpec(buffer=sec_buf)],
        primary_start_sec=0.0,
        target_sample_rate=48_000,
        playback_speed=2.0,
    )
    assert fast_mix.shape[0] == 24_000

    # GUI BorisSyncWidget per-camera Mute/Gain/Solo + live dBFS meter
    sec_vid = tmp_path / "mic_cam.avi"
    writer = cv2.VideoWriter(
        str(sec_vid), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, (32, 24)
    )
    for _ in range(10):
        writer.write(np.zeros((24, 32, 3), dtype=np.uint8))
    writer.release()

    sync_w = BorisSyncWidget()
    try:
        tile = sync_w.add_secondary_video(
            sec_vid, player_id="2", offset_s=-0.25, audio_buffer=sec_buf
        )
        assert tile.is_audio_muted is True
        assert sync_w.custom_audio_mix_active() is False

        # Unmute secondary camera & mute primary to solo Camera #2 at 50% gain
        tile.set_audio_muted(False)
        tile.set_audio_gain_percent(50)
        sync_w.primary_audio_chk.setChecked(False)
        sync_w.sync_to_time(0.5, immediate=True)

        assert sync_w.custom_audio_mix_active() is True
        assert "dBFS" in tile.status_label.text()

        solo_mix, solo_sr = sync_w.build_mixed_audio(
            prim_buf, primary_start_sec=0.0, target_sample_rate=48_000
        )
        assert solo_sr == 48_000
        # Leading 0.25s (12,000 samples) is silent due to -0.25s offset, then 0.8 * 0.5 = 0.4!
        assert math.isclose(float(solo_mix[1_000]), 0.0, abs_tol=1e-6)
        assert math.isclose(float(solo_mix[24_000]), 0.4, abs_tol=1e-3)
    finally:
        sync_w.shutdown()
        sync_w.deleteLater()
