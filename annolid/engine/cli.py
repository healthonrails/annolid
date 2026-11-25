from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional


class _AnnolidHelpFormatter(
    argparse.ArgumentDefaultsHelpFormatter,
    argparse.RawDescriptionHelpFormatter,
):
    pass


_ROOT_COMMAND_GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "Models",
        ("list-models", "dinov3-models", "train", "predict", "colab", "sim2real"),
    ),
    (
        "Tracking",
        (
            "track-qc",
            "repair-tracks",
            "gap-report",
            "tracking-stats",
            "kinematics",
            "zone-analysis",
            "segment-body-parts",
            "body-part-ethogram",
        ),
    ),
    (
        "Behavior",
        (
            "time-budget",
            "behavior-qc",
            "behavior-irr",
            "behavior-cohort",
            "behavior-search",
            "boris-import",
            "boris-export",
            "report",
        ),
    ),
    (
        "Agent",
        (
            "agent",
            "agent-onboard",
            "agent-status",
            "agent-behavior",
            "agent-capabilities",
            "agent-skill-pool",
            "agent-security-check",
            "agent-security-audit",
            "agent-secrets-audit",
            "agent-secrets-set",
            "agent-secrets-remove",
            "agent-secrets-migrate",
            "agent-config-migrate",
            "agent-box-auth-url",
            "agent-box-auth-exchange",
            "agent-box-token-refresh",
            "agent-update",
            "agent-eval",
            "agent-meta-learning-status",
            "agent-meta-learning-history",
            "agent-meta-learning-maintenance-status",
            "agent-meta-learning-maintenance-next-window",
            "agent-meta-learning-maintenance-run",
            "agent-skills-import",
            "agent-cron-list",
            "agent-cron-add",
            "agent-cron-remove",
            "agent-cron-enable",
            "agent-cron-run",
        ),
    ),
    (
        "Data",
        (
            "collect-labels",
            "index-to-yolo",
            "import-deeplabcut-training-data",
            "import-pose",
            "export-pose",
            "dino-kpseg-embeddings",
            "dino-kpseg-audit",
            "dino-kpseg-split",
            "dino-kpseg-precompute",
        ),
    ),
    (
        "Utilities",
        (
            "runtime",
            "validate-agent-output",
            "validate-agent-tools",
            "citations-list",
            "citations-upsert",
            "citations-remove",
            "citations-format",
            "memory",
        ),
    ),
)


def _root_help_epilog() -> str:
    return (
        "Quick start:\n"
        "  annolid-run list-models\n"
        "  annolid-run dinov3-models --list\n"
        "  annolid-run dinov3-models --model facebook/dinov3-vits16-pretrain-lvd1689m\n"
        "  annolid-run train <model> --help-model\n"
        "  annolid-run predict <model> --help-model\n"
        "  annolid-run agent-status\n"
        "  annolid-run agent-behavior --video /path/to/video.mp4\n"
        "  annolid-run agent-capabilities\n"
        "  annolid-run help train\n\n"
        "Common areas:\n"
        "  Models: train, predict, list-models, dinov3-models\n"
        "  Tracking: track-qc, repair-tracks, gap-report, tracking-stats, kinematics, zone-analysis, segment-body-parts, body-part-ethogram\n"
        "  Behavior: time-budget, behavior-qc, behavior-irr, behavior-cohort, behavior-search, boris-import, boris-export, report\n"
        "  Agent: agent, agent-behavior, agent-status, agent-capabilities, agent-security-*, agent-cron-*\n"
        "  Data: collect-labels, index-to-yolo, import-deeplabcut-training-data, import-pose, export-pose\n"
        "  Utilities: citations-*, validate-agent-output, validate-agent-tools"
    )


def _mode_help_epilog(mode: str) -> str:
    label = str(mode or "").strip().lower()
    if label == "train":
        return (
            "Examples:\n"
            "  annolid-run train dino_kpseg --help-model\n"
            "  annolid-run help train dino_kpseg\n"
            "  annolid-run list-models"
        )
    return (
        "Examples:\n"
        "  annolid-run predict dino_kpseg --help-model\n"
        "  annolid-run help predict dino_kpseg\n"
        "  annolid-run list-models"
    )


_PLUGIN_HELP_GROUPS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "Required inputs",
        (
            "--data",
            "--source",
            "--weights",
            "--run-config",
        ),
    ),
    (
        "Outputs and run location",
        (
            "--output",
            "--runs-root",
            "--run-name",
            "--project",
            "--name",
        ),
    ),
    (
        "Model and runtime",
        (
            "--model-name",
            "--device",
            "--imgsz",
            "--short-side",
            "--layers",
        ),
    ),
    (
        "Training controls",
        (
            "--epochs",
            "--batch",
            "--lr",
            "--accumulate",
            "--grad-clip",
            "--threshold",
            "--conf",
            "--iou",
        ),
    ),
    (
        "Data and augmentation",
        (
            "--data-format",
            "--instance-mode",
            "--bbox-scale",
            "--augment",
            "--no-augment",
            "--hflip",
            "--degrees",
            "--translate",
            "--scale",
        ),
    ),
    (
        "Saving and reporting",
        (
            "--save",
            "--save-txt",
            "--plots",
            "--log-every-steps",
            "--tb-projector",
            "--tb-add-graph",
        ),
    ),
)


def _summarize_action_help(action: argparse.Action) -> str:
    help_text = str(getattr(action, "help", "") or "").strip()
    if not help_text:
        return "See full flag list below."
    return help_text.rstrip(".")


def _action_option_strings(action: argparse.Action) -> tuple[str, ...]:
    return tuple(str(token) for token in getattr(action, "option_strings", ()) if token)


def _build_plugin_help_summary(
    parser: argparse.ArgumentParser,
    *,
    mode: str,
    help_sections: tuple[tuple[str, tuple[str, ...]], ...] = (),
) -> str:
    optional_actions = [
        action
        for action in getattr(parser, "_actions", [])
        if getattr(action, "option_strings", None)
        and "--help" not in set(action.option_strings)
    ]
    lines: list[str] = []
    seen_dests: set[str] = set()
    section_groups = help_sections or _PLUGIN_HELP_GROUPS
    for group_name, preferred_options in section_groups:
        matched: list[argparse.Action] = []
        for option in preferred_options:
            for action in optional_actions:
                if action.dest in seen_dests:
                    continue
                if option in _action_option_strings(action):
                    matched.append(action)
                    seen_dests.add(str(action.dest))
                    break
        if not matched:
            continue
        lines.append(f"{group_name}:")
        for action in matched:
            primary = next(
                (opt for opt in _action_option_strings(action) if opt.startswith("--")),
                _action_option_strings(action)[0],
            )
            lines.append(f"  {primary:<24} {_summarize_action_help(action)}")
        lines.append("")
    if not lines:
        return ""
    header = (
        "Quick reference:\n"
        "  Start with the flags in these groups, then use the full flag list below for advanced tuning.\n\n"
    )
    return header + "\n".join(lines).rstrip()


def _plugin_help_description(
    *,
    mode: str,
    model_name: str,
    plugin_description: str,
    parser: argparse.ArgumentParser,
    help_sections: tuple[tuple[str, tuple[str, ...]], ...] = (),
) -> str:
    mode_label = "training" if mode == "train" else "inference"
    lines = [
        f"Model-specific {mode_label} help for `{model_name}`.",
    ]
    detail = str(plugin_description or "").strip()
    if detail:
        lines.append(detail)
    lines.append("Flags below are provided by the plugin.")
    summary = _build_plugin_help_summary(
        parser,
        mode=mode,
        help_sections=help_sections,
    )
    if summary:
        lines.extend(["", summary])
    return "\n".join(lines)


def _plugin_examples(
    plugin: object,
    *,
    mode: str,
    model_name: str,
) -> str:
    if mode == "train":
        examples = tuple(
            str(item)
            for item in getattr(plugin, "train_examples", ())
            if str(item).strip()
        )
        if examples:
            return "Examples:\n" + "\n".join(f"  {item}" for item in examples)
        return (
            "Examples:\n"
            f"  annolid-run train {model_name} --help-model\n"
            f"  annolid-run help train {model_name}\n"
            f"  annolid-run train {model_name} --run-config <path>"
        )
    examples = tuple(
        str(item)
        for item in getattr(plugin, "predict_examples", ())
        if str(item).strip()
    )
    if examples:
        return "Examples:\n" + "\n".join(f"  {item}" for item in examples)
    return (
        "Examples:\n"
        f"  annolid-run predict {model_name} --help-model\n"
        f"  annolid-run help predict {model_name}\n"
        f"  annolid-run predict {model_name} --source <video-or-image>"
    )


def _collect_root_command_help(
    parser: argparse.ArgumentParser,
) -> dict[str, str]:
    command_help: dict[str, str] = {}
    for action in getattr(parser, "_actions", []):
        if not isinstance(action, argparse._SubParsersAction):
            continue
        for choice_action in getattr(action, "_choices_actions", []):
            command_help[str(choice_action.dest)] = str(
                choice_action.help or ""
            ).strip()
    command_help.setdefault(
        "runtime",
        "Inspect and install optional on-demand runtime packages and profiles.",
    )
    command_help.setdefault(
        "colab",
        "Authenticate with Google Colab and run remote GPU/TPU training or inference.",
    )
    command_help.setdefault(
        "sim2real",
        "Match example videos to Mouse/Fly 3D simulations, generate datasets, iterate on Colab, and write reports.",
    )
    command_help.setdefault("memory", "Memory inspection and maintenance commands.")
    return command_help


def _format_root_help(parser: argparse.ArgumentParser) -> str:
    command_help = _collect_root_command_help(parser)
    lines = [
        "usage: annolid-run <command> [options]",
        "",
        "Annolid command-line interface for models, agent workflows, datasets, validation, and maintenance tasks.",
        "",
        "Quick start:",
        "  annolid-run list-models",
        "  annolid-run runtime status",
        "  annolid-run dinov3-models --list",
        "  annolid-run train <model> --help-model",
        "  annolid-run predict <model> --help-model",
        "  annolid-run agent-status",
        "  annolid-run agent-capabilities",
        "  annolid-run help train",
        "",
    ]
    for group_name, commands in _ROOT_COMMAND_GROUPS:
        lines.append(f"{group_name}:")
        for command in commands:
            help_text = command_help.get(command, "")
            lines.append(f"  {command:<34} {help_text}")
        lines.append("")
    lines.extend(
        [
            "Options:",
            "  -h, --help                         Show this grouped root help.",
            "",
            "Examples:",
            "  annolid-run help",
            "  annolid-run help predict",
            "  annolid-run memory stats --scope global",
        ]
    )
    return "\n".join(lines).rstrip() + "\n"


def _cmd_list_models(_: argparse.Namespace) -> int:
    from annolid.engine.registry import (
        get_load_failures,
        list_models,
        load_builtin_models,
    )

    failures = load_builtin_models()
    if failures:
        details = get_load_failures()
        for name in failures:
            print(
                f"[annolid-run] Failed to import {name}: {details.get(name, '')}",
                file=sys.stderr,
            )
    rows = [
        {
            "name": m.name,
            "train": m.supports_train,
            "predict": m.supports_predict,
            "capabilities": list(m.required_capabilities),
            "description": m.description,
        }
        for m in list_models(load_builtins=False)
    ]
    print(json.dumps(rows, indent=2))
    return 0


def _cmd_dinov3_models(args: argparse.Namespace) -> int:
    from annolid.features.dino_models import (
        DEFAULT_DINOV3_MODEL_ID,
        download_dino_model,
        iter_dino_feature_models,
        resolve_dino_model_id,
    )

    if bool(getattr(args, "list", False)):
        rows = [
            {
                "display_name": model.display_name,
                "model_id": model.model_id,
                "family": model.family,
                "gated": model.gated,
                "aliases": list(model.aliases),
                "note": model.note,
            }
            for model in iter_dino_feature_models(
                dinov3_only=bool(getattr(args, "dinov3_only", False))
            )
        ]
        print(json.dumps(rows, indent=2))
        return 0

    requested = list(getattr(args, "model", None) or [])
    if not requested:
        requested = [DEFAULT_DINOV3_MODEL_ID]

    results: list[dict[str, str]] = []
    exit_code = 0
    for raw_model in requested:
        model_id = resolve_dino_model_id(raw_model)
        try:
            snapshot_path = download_dino_model(
                model_id,
                cache_dir=getattr(args, "cache_dir", None),
                local_files_only=bool(getattr(args, "local_files_only", False)),
            )
        except Exception as exc:
            exit_code = 1
            results.append(
                {
                    "model_id": model_id,
                    "status": "error",
                    "error": str(exc),
                }
            )
            print(f"[annolid-run] {exc}", file=sys.stderr)
            continue
        results.append(
            {
                "model_id": model_id,
                "status": "available",
                "path": str(snapshot_path),
            }
        )
    print(json.dumps(results, indent=2))
    return exit_code


def _cmd_time_budget(args: argparse.Namespace) -> int:
    from annolid.services.behavior_analytics import run_behavior_time_budget_cli

    return int(
        run_behavior_time_budget_cli(
            events_csv=args.events_csv,
            output_csv=getattr(args, "output", None),
            json_out=getattr(args, "json_out", None),
            json_stdout=bool(getattr(args, "json", False)),
            strict=bool(getattr(args, "strict", False)),
            bin_size=getattr(args, "bin_size", None),
            session_duration=getattr(args, "session_duration", None),
            start_time=getattr(args, "start_time", None),
            end_time=getattr(args, "end_time", None),
            exclude_behaviors=getattr(args, "exclude_behaviors", None),
            schema_path=getattr(args, "schema", None),
            bout_profile=getattr(args, "bout_profile", None),
            bout_name=str(getattr(args, "bout_name", "custom_bout") or "custom_bout"),
            bout_behaviors=getattr(args, "bout_behaviors", None),
            bout_initiation_behaviors=getattr(args, "bout_initiation_behaviors", None),
            bout_gap_seconds=getattr(args, "bout_gap_seconds", None),
            cooccurrence_csv=getattr(args, "cooccurrence_csv", None),
            ethogram_svg=getattr(args, "ethogram_svg", None),
            transitions_dot=getattr(args, "transitions_dot", None),
            transitions_svg=getattr(args, "transitions_svg", None),
            external_data_csvs=getattr(args, "external_data", None),
            external_offset_s=float(getattr(args, "external_offset", 0.0) or 0.0),
            external_data_svg=getattr(args, "external_data_svg", None),
        )
    )


def _cmd_behavior_qc(args: argparse.Namespace) -> int:
    from annolid.services.behavior_analytics import run_behavior_qc_cli

    return int(
        run_behavior_qc_cli(
            events_csv=args.events_csv,
            schema_path=getattr(args, "schema", None),
            strict=bool(getattr(args, "strict", False)),
            strict_schema=bool(getattr(args, "strict_schema", False)),
            json_stdout=bool(getattr(args, "json", False)),
            out_json=getattr(args, "out", None),
            csv_out=getattr(args, "csv_out", None),
        )
    )


def _cmd_behavior_irr(args: argparse.Namespace) -> int:
    from annolid.services.behavior_analytics import run_behavior_irr_cli

    raw_rater_names = getattr(args, "rater_names", None)
    rater_names = (
        [item.strip() for item in str(raw_rater_names).split(",") if item.strip()]
        if raw_rater_names
        else None
    )
    return int(
        run_behavior_irr_cli(
            csv_paths=list(args.events_csvs),
            rater_names=rater_names,
            schema_path=getattr(args, "schema", None),
            bin_seconds=float(args.bin_seconds),
            onset_tolerance_seconds=float(args.onset_tolerance_seconds),
            subject=getattr(args, "subject", None),
            pool_subjects=bool(getattr(args, "pool_subjects", False)),
            session_start=getattr(args, "session_start", None),
            session_end=getattr(args, "session_end", None),
            session_duration=getattr(args, "session_duration", None),
            include_empty_schema_behaviors=bool(
                getattr(args, "include_empty_schema_behaviors", False)
            ),
            min_kappa=getattr(args, "min_kappa", None),
            json_stdout=bool(getattr(args, "json", False)),
            out_json=getattr(args, "out", None),
            csv_out=getattr(args, "csv_out", None),
            disagreements_csv=getattr(args, "disagreements_csv", None),
        )
    )


def _cmd_behavior_cohort(args: argparse.Namespace) -> int:
    from annolid.services.behavior_analytics import run_behavior_cohort_cli

    return int(
        run_behavior_cohort_cli(
            manifest_path=getattr(args, "manifest", None),
            events_csvs=list(getattr(args, "events_csvs", None) or ()),
            metadata_csv=getattr(args, "metadata_csv", None),
            schema_path=getattr(args, "schema", None),
            session_duration=getattr(args, "session_duration", None),
            max_transition_gap=getattr(args, "max_transition_gap", None),
            exclude_self_transitions=bool(
                getattr(args, "exclude_self_transitions", False)
            ),
            include_schema_behaviors=bool(
                getattr(args, "include_schema_behaviors", False)
            ),
            analysis_unit=str(getattr(args, "unit", "subject") or "subject"),
            strict=bool(getattr(args, "strict", False)),
            json_stdout=bool(getattr(args, "json", False)),
            out_json=getattr(args, "out", None),
            summary_csv=getattr(args, "summary_csv", None),
            subjects_csv=getattr(args, "subjects_csv", None),
            transitions_csv=getattr(args, "transitions_csv", None),
            comparisons_csv=getattr(args, "comparisons_csv", None),
            transitions_dot=getattr(args, "transitions_dot", None),
            transitions_svg=getattr(args, "transitions_svg", None),
        )
    )


def _cmd_behavior_search(args: argparse.Namespace) -> int:
    from annolid.services.search import run_behavior_search_cli

    raw_behaviors = getattr(args, "behaviors", None)
    behaviors = (
        [item.strip() for item in str(raw_behaviors).split(",") if item.strip()]
        if raw_behaviors
        else None
    )
    raw_threshold = getattr(args, "threshold", None)
    payload, rc = run_behavior_search_cli(
        video_path=str(args.video),
        query=getattr(args, "query", None),
        query_frame=getattr(args, "query_frame", None),
        query_image=getattr(args, "query_image", None),
        behaviors=behaviors,
        backend=str(getattr(args, "backend", "embeddinggemma2_onnx")),
        model_id=getattr(args, "model_id", None),
        quantization=str(getattr(args, "quantization", "q4")),
        matryoshka_dim=getattr(args, "matryoshka_dim", None),
        soft_tokens=getattr(args, "soft_tokens", None),
        stride=int(getattr(args, "stride", 5)),
        max_frames=getattr(args, "max_frames", None),
        clip_frames=int(getattr(args, "clip_frames", 1)),
        batch_size=int(getattr(args, "batch_size", 4)),
        threshold=float(raw_threshold) if raw_threshold is not None else None,
        top_k=int(getattr(args, "top_k", 50)),
        overlay_shapes=bool(getattr(args, "overlay_shapes", False)),
        annotation_dir=getattr(args, "annotation_dir", None),
        index_cache=getattr(args, "index_cache", None),
        auto_disk_cache=bool(getattr(args, "auto_cache", False)),
        out_path=getattr(args, "out", None),
        emit_json=bool(getattr(args, "json", False)),
    )
    if bool(getattr(args, "json", False)):
        print(json.dumps(payload, indent=2))
    elif behaviors:
        frames = payload.get("frames") or []
        print(
            f"Scored {len(frames)} sampled frame(s) against {len(behaviors)} behavior(s) "
            f"(backend={payload.get('backend')})."
        )
        for row in frames[:10]:
            top_list = row.get("behaviors") or []
            summary = ", ".join(
                f"{b['label']}={float(b['similarity']):.3f}" for b in top_list
            )
            print(f"  Frame {int(row['frame_index']):09d}: {summary or '(none)'}")
    else:
        matches = payload.get("matches") or []
        bouts = payload.get("bouts") or []
        score_summary = payload.get("score_summary") or {}
        thr_note = ""
        if score_summary:
            eff_thr = float(score_summary.get("effective_threshold", 0.0))
            thr_mode = str(score_summary.get("threshold_mode") or "fixed")
            thr_note = f", {thr_mode}≥{eff_thr:.3f}"
        print(
            f"Found {len(matches)} match(es) across {len(bouts)} bout(s) "
            f"(mode={payload.get('query_mode')}, backend={payload.get('backend')}{thr_note})."
        )
        if not matches and score_summary and int(score_summary.get("count", 0)) > 0:
            best_f = score_summary.get("best_frame_index")
            max_sim = float(score_summary.get("max_similarity", 0.0))
            min_sim = float(score_summary.get("min_similarity", 0.0))
            med_sim = float(score_summary.get("median_similarity", 0.0))
            best_desc = f"Frame {int(best_f):09d}" if best_f is not None else "n/a"
            print(
                f"  Observed similarity range: {min_sim:.3f}–{max_sim:.3f} "
                f"(median={med_sim:.3f}; best {best_desc} sim={max_sim:.3f})."
            )
        for m in matches[:10]:
            print(
                f"  Frame {int(m['frame_index']):09d} • sim={float(m['similarity']):.3f}"
            )
    return int(rc)


def _cmd_boris_import(args: argparse.Namespace) -> int:
    from annolid.services.behavior_analytics import run_boris_import_cli

    raw_obs = getattr(args, "observations", None)
    obs_filter = (
        [item.strip() for item in str(raw_obs).split(",") if item.strip()]
        if raw_obs
        else None
    )
    return int(
        run_boris_import_cli(
            boris_path=args.boris_file,
            output_dir=args.output_dir,
            observations=obs_filter,
            apply_time_offset=bool(getattr(args, "apply_time_offset", False)),
            group_variable=getattr(args, "group_variable", None),
            json_stdout=bool(getattr(args, "json", False)),
        )
    )


def _cmd_boris_export(args: argparse.Namespace) -> int:
    from annolid.services.behavior_analytics import run_boris_export_cli

    fmt = str(getattr(args, "format", "") or "").strip().lower()
    out_arg = getattr(args, "output", None)
    out_boris = out_arg
    tab_csv = getattr(args, "tabular_csv", None)
    agg_csv = getattr(args, "aggregated_csv", None)
    if out_arg and fmt == "tabular-csv":
        tab_csv = tab_csv or out_arg
        out_boris = None
    elif out_arg and fmt == "aggregated-csv":
        agg_csv = agg_csv or out_arg
        out_boris = None

    return int(
        run_boris_export_cli(
            schema_path=getattr(args, "schema", None),
            events_csvs=list(getattr(args, "events_csvs", None) or ()),
            output_boris=out_boris,
            tabular_csv=tab_csv,
            aggregated_csv=agg_csv,
            project_name=str(
                getattr(args, "project_name", "Annolid Behavioral Project")
                or "Annolid Behavioral Project"
            ),
            json_stdout=bool(getattr(args, "json", False)),
        )
    )


def _cmd_kinematics(args: argparse.Namespace) -> int:
    from annolid.services.kinematics_analytics import run_kinematics_cli

    return int(
        run_kinematics_cli(
            source_path=args.source_path,
            fps=getattr(args, "fps", None),
            px_per_cm=getattr(args, "px_per_cm", None),
            calibration_path=getattr(args, "calibration", None),
            smooth_window=int(getattr(args, "smooth_window", 5)),
            freeze_speed_threshold=getattr(args, "freeze_speed_threshold", None),
            min_freeze_duration_s=float(getattr(args, "min_freeze_duration", 0.5)),
            close_contact_distance_px=float(getattr(args, "close_contact_px", 80.0)),
            output_dir=getattr(args, "output_dir", None),
            out_json=getattr(args, "out_json", None),
            events_csv=getattr(args, "events_csv", None),
            include_plots=not bool(getattr(args, "no_plots", False)),
            json_stdout=bool(getattr(args, "json", False)),
        )
    )


def _cmd_zone_analysis(args: argparse.Namespace) -> int:
    from annolid.services.zone_analytics import run_zone_analysis_cli

    return int(
        run_zone_analysis_cli(
            video_path=args.video_path,
            zone_file=getattr(args, "zone_file", None),
            zone_policy_file=getattr(args, "zone_policy", None),
            fps=getattr(args, "fps", None),
            assay_profile=str(getattr(args, "profile", "generic") or "generic"),
            mode=str(getattr(args, "mode", "zone_metrics") or "zone_metrics"),
            anchor_part=getattr(args, "anchor_part", None),
            latency_reference_frame=getattr(args, "latency_reference_frame", None),
            output_dir=getattr(args, "output_dir", None),
            out_json=getattr(args, "out_json", None),
            events_csv=getattr(args, "events_csv", None),
            json_stdout=bool(getattr(args, "json", False)),
        )
    )


def _cmd_body_part_ethogram(args: argparse.Namespace) -> int:
    from annolid.services.kinematics_analytics import run_body_part_ethogram_cli

    return int(
        run_body_part_ethogram_cli(
            annotation_dir=args.annotation_dir,
            fps=float(getattr(args, "fps", 30.0) or 30.0),
            contact_margin_px=float(getattr(args, "contact_margin_px", 6.0)),
            min_bout_frames=int(getattr(args, "min_bout_frames", 1)),
            max_uncertain_frames=int(getattr(args, "max_uncertain_frames", 10)),
            camera_view=str(getattr(args, "camera_view", "auto") or "auto"),
            output_prefix=getattr(args, "output_prefix", None),
            events_csv=getattr(args, "events_csv", None),
            out_json=getattr(args, "out_json", None),
            json_stdout=bool(getattr(args, "json", False)),
        )
    )


def _cmd_gap_report(args: argparse.Namespace) -> int:
    from annolid.services.track_analytics import run_gap_report_cli

    return int(
        run_gap_report_cli(
            source_path=args.source_path,
            fps=getattr(args, "fps", None),
            write_reports=not bool(getattr(args, "no_write_reports", False)),
            out_json=getattr(args, "out_json", None),
            json_stdout=bool(getattr(args, "json", False)),
        )
    )


def _cmd_tracking_stats(args: argparse.Namespace) -> int:
    from annolid.services.track_analytics import run_tracking_stats_cli

    return int(
        run_tracking_stats_cli(
            root_dir=args.root_dir,
            output_dir=getattr(args, "output_dir", None),
            include_plots=not bool(getattr(args, "no_plots", False)),
            out_json=getattr(args, "out_json", None),
            json_stdout=bool(getattr(args, "json", False)),
        )
    )


def _cmd_unified_report(args: argparse.Namespace) -> int:
    from annolid.services.unified_reporting import run_unified_report_cli

    return int(
        run_unified_report_cli(
            source_path=args.source_path,
            output_dir=getattr(args, "output_dir", None),
            events_csv=getattr(args, "events_csv", None),
            zone_file=getattr(args, "zone_file", None),
            calibration_path=getattr(args, "calibration", None),
            schema_path=getattr(args, "schema", None),
            pose_schema_path=getattr(args, "pose_schema", None),
            fps=getattr(args, "fps", None),
            px_per_cm=getattr(args, "px_per_cm", None),
            title=getattr(args, "title", None),
            include_plots=not bool(getattr(args, "no_plots", False)),
            json_stdout=bool(getattr(args, "json", False)),
        )
    )


def _cmd_track_qc(args: argparse.Namespace) -> int:
    from annolid.services.track_analytics import run_track_qc_cli

    return int(
        run_track_qc_cli(
            input_path=args.input_path,
            schema_path=getattr(args, "schema", None),
            pose_schema_path=getattr(args, "pose_schema", None),
            expected_instances=getattr(args, "expected_instances", None),
            fps=getattr(args, "fps", None),
            max_speed_px_per_frame=getattr(args, "max_speed_px", None),
            speed_mad_z=float(getattr(args, "speed_mad_z", 6.0)),
            min_jump_px=float(getattr(args, "min_jump_px", 25.0)),
            max_area_ratio=float(getattr(args, "max_area_ratio", 2.5)),
            overlap_iou_threshold=float(getattr(args, "overlap_iou", 0.25)),
            close_contact_px=float(getattr(args, "close_contact_px", 80.0)),
            max_edge_stretch_ratio=float(getattr(args, "max_edge_stretch", 2.5)),
            min_score=float(getattr(args, "min_score", 0.35)),
            bout_merge_gap_frames=int(getattr(args, "bout_gap_frames", 5)),
            top_k_review=getattr(args, "top_k", 25),
            strict=bool(getattr(args, "strict", False)),
            json_stdout=bool(getattr(args, "json", False)),
            out_json=getattr(args, "out", None),
            issues_csv=getattr(args, "issues_csv", None),
            review_csv=getattr(args, "review_csv", None),
        )
    )


def _cmd_repair_tracks(args: argparse.Namespace) -> int:
    from annolid.services.track_analytics import run_track_repair_cli

    return int(
        run_track_repair_cli(
            annotation_dir=args.annotation_dir,
            mode=str(getattr(args, "mode", "temporal") or "temporal"),
            apply_changes=bool(getattr(args, "apply", False)),
            report_path=getattr(args, "report_out", None),
            json_out=getattr(args, "json_out", None),
            json_stdout=bool(getattr(args, "json", False)),
            start_frame=int(getattr(args, "start_frame", 0)),
            expected_instance_count=getattr(args, "expected_instances", None),
            max_gap_frames=int(getattr(args, "max_gap_frames", 5)),
            max_match_distance=float(getattr(args, "max_match_distance", 80.0)),
            policy_path=getattr(args, "policy", None),
            zone_file=getattr(args, "zones", None),
        )
    )


def _cmd_segment_body_parts(args: argparse.Namespace) -> int:
    from annolid.services.body_part_segmentation import run_body_part_segmentation_cli

    part_full_arg: Optional[bool] = None
    if bool(getattr(args, "partition_full_instance", False)):
        part_full_arg = True
    elif bool(getattr(args, "partial_parts", False)):
        part_full_arg = False

    return int(
        run_body_part_segmentation_cli(
            video_path=args.video_path,
            annotation_dir=getattr(args, "annotation_dir", None),
            mode=str(getattr(args, "mode", "auto") or "auto"),
            feature_backend=str(getattr(args, "backbone", "fast") or "fast"),
            model_name=str(
                getattr(
                    args,
                    "dino_model",
                    "facebook/dinov3-vits16-pretrain-lvd1689m",
                )
            ),
            crop_size=int(getattr(args, "crop_size", 224)),
            crop_padding_ratio=float(getattr(args, "crop_padding", 0.12)),
            num_parts=int(getattr(args, "num_parts", 3)),
            part_names=getattr(args, "parts", None),
            keypoint_granularity=str(
                getattr(args, "keypoint_granularity", "coarse") or "coarse"
            ),
            keypoint_part_map=getattr(args, "keypoint_part_map", None),
            partition_full_instance=part_full_arg,
            share_seed_across_instances=not bool(
                getattr(args, "no_share_seeds", False)
            ),
            max_part_area_growth=float(getattr(args, "max_area_growth", 2.5)),
            min_part_area_px=int(getattr(args, "min_part_area", 6)),
            boundary_refine=not bool(getattr(args, "no_boundary_refine", False)),
            crf_refine=bool(getattr(args, "crf_refine", False)),
            polygon_epsilon=float(getattr(args, "polygon_epsilon", 1.0)),
            split_merged_parent_masks=not bool(
                getattr(args, "no_split_merged_parents", False)
            ),
            recover_missing_merged_instances=not bool(
                getattr(args, "no_recover_merged_instances", False)
            ),
            qualify_label_with_instance=bool(getattr(args, "qualify_labels", False)),
            persist_json=bool(getattr(args, "persist_json", False)),
            device=getattr(args, "device", None),
            pose_schema_path=getattr(args, "pose_schema", None),
            start_frame=int(getattr(args, "start_frame", 0)),
            end_frame=getattr(args, "end_frame", None),
            step=int(getattr(args, "step", 1)),
            export_csv=bool(getattr(args, "export_csv", False)),
            export_ethogram=bool(getattr(args, "export_ethogram", False)),
            camera_view=str(getattr(args, "camera_view", "auto") or "auto"),
            json_stdout=bool(getattr(args, "json", False)),
            out_json=getattr(args, "out", None),
        )
    )


def _cmd_validate_agent_output(args: argparse.Namespace) -> int:
    """Validate Annolid agent NDJSON output against the canonical JSON Schema."""
    from annolid.core.output.validate import (
        AgentOutputValidationError,
        validate_agent_record,
    )

    ndjson_path = Path(args.ndjson).expanduser().resolve()
    if not ndjson_path.exists():
        print(f"[annolid-run] NDJSON file not found: {ndjson_path}", file=sys.stderr)
        return 2

    max_errors = int(args.max_errors)
    if max_errors < 1:
        max_errors = 1

    total = 0
    errors = 0
    with ndjson_path.open("r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            raw = line.strip()
            if not raw:
                continue
            total += 1
            try:
                record = json.loads(raw)
                if not isinstance(record, dict):
                    raise AgentOutputValidationError("Record must be a JSON object.")
                validate_agent_record(record)
            except (json.JSONDecodeError, AgentOutputValidationError) as exc:
                errors += 1
                print(
                    f"[annolid-run] Invalid record at {ndjson_path}:{lineno}: {exc}",
                    file=sys.stderr,
                )
                if errors >= max_errors:
                    return 1

    summary = {"ndjson": str(ndjson_path), "records": total, "errors": errors}
    print(json.dumps(summary, indent=2))
    return 0 if errors == 0 else 1


def _cmd_agent_validate_tools(_: argparse.Namespace) -> int:
    from annolid.services.agent_tooling import validate_agent_tools

    summary, exit_code = validate_agent_tools()
    print(json.dumps(summary, indent=2))
    return exit_code


def _cmd_collect_labels(args: argparse.Namespace) -> int:
    from annolid.datasets.labelme_collection import (
        DEFAULT_LABEL_INDEX_NAME,
        DEFAULT_LABEL_INDEX_DIRNAME,
        generate_labelme_spec_and_splits,
        index_labelme_dataset,
        normalize_labelme_sources,
        resolve_label_index_path,
    )

    dataset_root = Path(args.dataset_root).expanduser().resolve()
    index_file_default = str(
        Path(DEFAULT_LABEL_INDEX_DIRNAME) / DEFAULT_LABEL_INDEX_NAME
    )
    index_file = resolve_label_index_path(
        Path(getattr(args, "index_file", index_file_default)),
        dataset_root=dataset_root,
    )

    sources, missing_sources = normalize_labelme_sources([Path(p) for p in args.source])
    if missing_sources:
        print(
            "Warning: missing sources:\n" + "\n".join(f"- {p}" for p in missing_sources)
        )
    if not sources:
        print("No existing sources provided.")
        return 1

    summary = index_labelme_dataset(
        sources=sources,
        index_file=index_file,
        recursive=bool(args.recursive),
        include_empty=bool(args.include_empty),
        dedupe=not bool(args.allow_duplicates),
    )
    if bool(args.write_spec):
        raw_names = str(args.keypoint_names or "").strip()
        keypoint_names = [n.strip() for n in raw_names.split(",") if n.strip()] or None
        spec_result = generate_labelme_spec_and_splits(
            sources=sources,
            dataset_root=dataset_root,
            recursive=bool(args.recursive),
            include_empty=bool(args.include_empty),
            split_dir=str(args.split_dir or DEFAULT_LABEL_INDEX_DIRNAME),
            val_size=float(args.val_size),
            test_size=float(args.test_size),
            seed=int(args.seed),
            group_by=str(args.group_by),
            group_regex=(str(args.group_regex) if args.group_regex else None),
            keypoint_names=keypoint_names,
            kpt_dims=int(args.kpt_dims),
            infer_flip_idx=bool(args.infer_flip_idx),
            max_keypoint_files=int(args.max_keypoint_files),
            min_keypoint_count=int(args.min_keypoint_count),
            spec_path=Path(args.spec_path) if args.spec_path else None,
            source="annolid_cli",
        )
        summary.update(
            {
                "spec_path": spec_result.get("spec_path"),
                "split_counts": spec_result.get("split_counts"),
            }
        )
    print(json.dumps(summary, indent=2))
    return 0


def _cmd_index_to_yolo(args: argparse.Namespace) -> int:
    from annolid.services.export import build_yolo_dataset_from_index

    summary = build_yolo_dataset_from_index(
        index_file=Path(args.index_file),
        output_dir=Path(args.output_dir),
        dataset_name=str(args.dataset_name),
        val_size=float(args.val_size),
        test_size=float(args.test_size),
        link_mode=str(args.link_mode),
        task=str(args.task),
        include_empty=bool(args.include_empty),
        keep_staging=bool(args.keep_staging),
    )
    print(json.dumps(summary, indent=2))
    return 0


def _cmd_import_deeplabcut_training_data(args: argparse.Namespace) -> int:
    from annolid.domain import DeepLabCutTrainingImportConfig
    from annolid.datasets.labelme_collection import index_labelme_dataset
    from annolid.services.export import import_deeplabcut_dataset

    source_dir = Path(args.source_dir).expanduser().resolve()
    labeled_data = Path(args.labeled_data)
    labeled_data = (
        labeled_data if labeled_data.is_absolute() else (source_dir / labeled_data)
    )

    summary = import_deeplabcut_dataset(
        DeepLabCutTrainingImportConfig(
            source_dir=source_dir,
            labeled_data_root=Path(args.labeled_data),
            instance_label=str(args.instance_label),
            overwrite=bool(args.overwrite),
            recursive=not bool(args.no_recursive),
        ),
        write_pose_schema=bool(args.write_pose_schema),
        pose_schema_out=Path(args.pose_schema_out) if args.pose_schema_out else None,
        pose_schema_preset=str(args.pose_schema_preset),
        instance_separator=str(args.instance_separator or "_"),
    )

    if bool(args.write_index):
        index_file = Path(args.index_file)
        if not index_file.is_absolute():
            index_file = source_dir / index_file
        index_summary = index_labelme_dataset(
            sources=[labeled_data],
            index_file=index_file,
            recursive=True,
            include_empty=False,
            dedupe=True,
        )
        summary["index_file"] = str(index_file)
        summary["index_summary"] = index_summary

    print(json.dumps(summary, indent=2))
    return 0


def _cmd_import_pose(args: argparse.Namespace) -> int:
    from annolid.datasets.labelme_collection import index_labelme_dataset
    from annolid.services.export import import_pose_dataset

    summary = import_pose_dataset(
        args.source,
        args.output_dir,
        source_format=str(args.format or "auto"),
        video_path=args.video,
        pose_schema_path=args.pose_schema,
        config_path=args.config,
        save_frames=not bool(args.no_save_frames),
        video_index=args.video_index,
        min_likelihood=args.min_likelihood,
        is_multi_animal=args.multi_animal,
        default_instance=str(args.instance_label or "mouse"),
        prefer_user_instances=not bool(args.keep_both_user_and_predicted),
        include_predicted=not bool(args.user_only),
        include_occluded=not bool(args.skip_occluded),
        write_pose_schema=not bool(args.no_pose_schema),
        pose_schema_preset=str(args.pose_schema_preset or "auto"),
        filename_prefix=args.filename_prefix,
        prefix_labels_with_instance=bool(args.prefix_instance_labels),
        instance_separator=str(args.instance_separator or "_"),
        overwrite=not bool(args.no_overwrite),
        recursive=not bool(args.no_recursive),
    )

    if bool(args.write_index):
        out_root = Path(
            summary.get("output_dir")
            or summary.get("labeled_data_root")
            or args.output_dir
        )
        index_file = (
            Path(args.index_file)
            if args.index_file
            else (out_root / "annolid_dataset.jsonl")
        )
        if not index_file.is_absolute():
            index_file = out_root / index_file
        index_summary = index_labelme_dataset(
            sources=[out_root],
            index_file=index_file,
            recursive=True,
            include_empty=False,
            dedupe=True,
        )
        summary["index_file"] = str(index_file)
        summary["index_summary"] = index_summary

    print(json.dumps(summary, indent=2))
    return 0


def _cmd_export_pose(args: argparse.Namespace) -> int:
    from annolid.services.export import export_pose_dataset

    summary = export_pose_dataset(
        args.source,
        args.output,
        target_format=str(args.format),
        source_format=str(args.source_format or "auto"),
        pose_schema=args.pose_schema,
        video_path=args.video,
        scorer=str(args.scorer or "annolid"),
        multi_animal=args.multi_animal,
        video_name=args.video_name,
        copy_images=not bool(args.no_copy_images),
        embed_images=bool(args.embed_images),
        write_config_yaml=not bool(args.no_config_yaml),
        recursive=bool(args.recursive),
    )
    print(json.dumps(summary, indent=2))
    return 0


def _parse_float_csv(
    raw: Optional[str], default: tuple[float, ...]
) -> tuple[float, ...]:
    if not raw:
        return default
    vals: list[float] = []
    for tok in str(raw).split(","):
        tok = tok.strip()
        if not tok:
            continue
        vals.append(float(tok))
    return tuple(vals) if vals else default


def _cmd_extract_frames(args: argparse.Namespace) -> int:
    from annolid.data.frame_sampling import extract_frames_smart

    res = extract_frames_smart(
        args.video,
        output_dir=args.output,
        algo=str(args.algo or "kmeans"),
        num_frames=int(args.num_frames),
        start_seconds=args.start_seconds,
        end_seconds=args.end_seconds,
        image_format=str(args.image_format or "png"),
        seed=int(args.seed),
    )
    if bool(getattr(args, "json", False)):
        print(json.dumps(res.to_dict(), indent=2))
    else:
        print(
            f"Extracted {len(res.saved_paths)} frame(s) ({res.algo}) from "
            f"{res.video_path} -> {res.output_dir}"
        )
    return 0


def _cmd_extract_outliers(args: argparse.Namespace) -> int:
    from annolid.data.frame_sampling import extract_pose_outlier_frames

    res = extract_pose_outlier_frames(
        args.source,
        args.video,
        args.output,
        source_format=str(args.source_format or "auto"),
        pose_schema=args.pose_schema,
        num_frames=int(args.num_frames),
        min_frame_gap=int(args.min_frame_gap),
        min_confidence=float(args.min_confidence),
        jump_threshold_px=args.jump_threshold,
        jump_mad_multiplier=float(args.jump_mad),
        max_edge_stretch_ratio=float(args.max_edge_stretch),
        min_edge_shrink_ratio=float(args.min_edge_shrink),
        expected_instances=args.expected_instances,
        include_predictions=not bool(args.no_predictions),
    )
    if bool(getattr(args, "json", False)):
        print(json.dumps(res.to_dict(), indent=2))
    else:
        print(
            f"Extracted {res.frames_extracted} outlier frame(s) "
            f"(out of {res.outliers_found} flagged / {res.frames_evaluated} evaluated) "
            f"-> {res.output_dir}"
        )
        print(f"Manifest: {res.manifest_path}")
    return 0


def _cmd_eval_pose(args: argparse.Namespace) -> int:
    from annolid.postprocessing.pose_evaluation import (
        evaluate_pose_predictions,
        format_pose_eval_table,
        write_pose_eval_report_bundle,
    )

    pck_px = _parse_float_csv(args.pck_px, (2.0, 4.0, 8.0, 16.0))
    pck_norm = _parse_float_csv(args.pck_norm, (0.05, 0.10, 0.20))
    report = evaluate_pose_predictions(
        args.gt_source,
        args.pred_source,
        gt_format=str(args.gt_format or "auto"),
        pred_format=str(args.pred_format or "auto"),
        pose_schema=args.pose_schema,
        pcutoff=float(args.pcutoff),
        pck_thresholds_px=pck_px,
        pck_norm_thresholds=pck_norm,
        oks_sigma=float(args.oks_sigma),
        top_k_worst_frames=int(args.top_k_worst),
    )
    artifacts: dict[str, str] = {}
    if args.output_dir:
        artifacts = write_pose_eval_report_bundle(report, args.output_dir)

    if bool(getattr(args, "json", False)):
        payload = report.to_dict()
        if artifacts:
            payload["artifacts"] = artifacts
        print(json.dumps(payload, indent=2))
    else:
        print(format_pose_eval_table(report))
        if artifacts:
            print(f"\nReport written to: {artifacts.get('html') or args.output_dir}")
    return 0


def _cmd_filter_pose(args: argparse.Namespace) -> int:
    from annolid.postprocessing.pose_filtering import (
        PoseFilterConfig,
        filter_pose_source,
    )

    cfg = PoseFilterConfig(
        min_confidence=args.min_confidence,
        max_jump_px=args.max_jump_px,
        jump_mad_multiplier=args.jump_mad,
        enforce_bone_bounds=bool(args.enforce_bone_bounds),
        max_edge_stretch_ratio=float(args.max_edge_stretch),
        min_edge_shrink_ratio=float(args.min_edge_shrink),
        interpolate=str(args.interpolate or "linear"),
        max_gap_frames=int(args.max_gap),
        method=str(args.method or "savgol"),
        window_length=int(args.window),
        polyorder=int(args.polyorder),
        fps=float(args.fps),
    )
    report = filter_pose_source(
        args.source,
        output_path=args.output,
        config=cfg,
        source_format=str(args.source_format or "auto"),
        target_format=str(args.format or "auto"),
        pose_schema=args.pose_schema,
        video_path=args.video,
        scorer=str(args.scorer or "annolid"),
        in_place=bool(args.in_place),
    )
    if bool(getattr(args, "json", False)):
        print(json.dumps(report.to_dict(), indent=2))
    else:
        print(
            f"Filtered {report.frames_processed} frame(s) ({report.method}, interp={report.interpolate}) -> {report.output_path}\n"
            f"  Masked: low_conf={report.masked_low_confidence}, jumps={report.masked_jump_outliers}, bones={report.masked_bone_outliers}\n"
            f"  Interpolated gaps: {report.gaps_interpolated} | Smoothed points: {report.points_smoothed} "
            f"(mean_disp={report.mean_displacement_px:.2f}px, max_disp={report.max_displacement_px:.2f}px)"
        )
    return 0


def _cmd_swap_pose(args: argparse.Namespace) -> int:
    from annolid.postprocessing.pose_filtering import swap_pose_instances_from_frame

    res = swap_pose_instances_from_frame(
        args.source_dir,
        start_frame=int(args.start_frame),
        end_frame=args.end_frame,
        instance_a=str(args.instance_a),
        instance_b=str(args.instance_b),
        pose_schema=args.pose_schema,
        dry_run=bool(args.dry_run),
    )
    if bool(getattr(args, "json", False)):
        print(json.dumps(res.to_dict(), indent=2))
    else:
        mode = "DRY-RUN" if res.dry_run else "APPLIED"
        print(
            f"[{mode}] Swapped '{res.instance_a}' <-> '{res.instance_b}' from frame {res.start_frame} "
            f"(modified {res.frames_modified}/{res.frames_scanned} frames, "
            f"{res.points_swapped} keypoints + {res.polygons_swapped} polygons)."
        )
    return 0


def _cmd_pose_kinematics(args: argparse.Namespace) -> int:
    from annolid.postprocessing.pose_kinematics import (
        analyze_pose_kinematics,
        format_pose_kinematics_table,
        write_pose_kinematics_bundle,
    )

    report = analyze_pose_kinematics(
        args.source,
        source_format=str(args.source_format or "auto"),
        pose_schema=args.pose_schema,
        fps=float(args.fps),
        px_per_cm=args.px_per_cm,
        calibration=args.calibration,
        min_confidence=args.min_confidence,
        close_contact_threshold_px=float(args.close_contact_px),
    )
    artifacts: dict[str, str] = {}
    if args.output_dir:
        artifacts = write_pose_kinematics_bundle(report, args.output_dir)

    if bool(getattr(args, "json", False)):
        payload = report.to_dict()
        if artifacts:
            payload["artifacts"] = artifacts
        print(json.dumps(payload, indent=2))
    else:
        print(format_pose_kinematics_table(report))
        if artifacts:
            print(f"\nKinematics bundle written to: {args.output_dir}")
    return 0


def _cmd_render_pose_video(args: argparse.Namespace) -> int:
    from annolid.postprocessing.pose_video_renderer import (
        PoseVideoRenderConfig,
        render_pose_video,
    )

    cfg = PoseVideoRenderConfig(
        color_mode=str(args.color_mode or "instance"),
        min_confidence=args.min_confidence,
        thickness=int(args.thickness),
        point_radius=int(args.point_radius),
        trail_length=int(args.trail_length),
        draw_labels=bool(args.draw_labels),
        draw_instance_names=not bool(args.no_instance_names),
        draw_frame_index=not bool(args.no_frame_index),
        max_bone_dist=args.max_bone_dist,
        start_frame=int(args.start_frame),
        end_frame=args.end_frame,
        fps=args.fps,
    )
    res = render_pose_video(
        args.video,
        args.pose_source,
        args.output,
        source_format=str(args.source_format or "auto"),
        pose_schema=args.pose_schema,
        comparison_source=args.comparison,
        config=cfg,
    )
    if bool(getattr(args, "json", False)):
        print(json.dumps(res.to_dict(), indent=2))
    else:
        print(
            f"Rendered {res.frames_rendered} frame(s) ({res.frames_with_pose} with pose, "
            f"mode={res.color_mode}) -> {res.output_video_path}"
        )
    return 0


def _cmd_dino_kpseg_embeddings(args: argparse.Namespace) -> int:
    from annolid.segmentation.dino_kpseg.tensorboard_embeddings import main as tb_main

    argv: list[str] = []
    argv.extend(["--data", str(args.data)])
    argv.extend(["--split", str(args.split)])
    if args.weights:
        argv.extend(["--weights", str(args.weights)])
    if args.model_name:
        argv.extend(["--model-name", str(args.model_name)])
    if args.short_side is not None:
        argv.extend(["--short-side", str(int(args.short_side))])
    if args.layers:
        argv.extend(["--layers", str(args.layers)])
    if args.device:
        argv.extend(["--device", str(args.device)])
    argv.extend(["--radius-px", str(float(args.radius_px))])
    argv.extend(["--mask-type", str(args.mask_type)])
    if args.heatmap_sigma is not None:
        argv.extend(["--heatmap-sigma", str(float(args.heatmap_sigma))])
    argv.extend(["--instance-mode", str(args.instance_mode)])
    argv.extend(["--bbox-scale", str(float(args.bbox_scale))])
    if bool(args.no_cache):
        argv.append("--no-cache")
    argv.extend(["--max-images", str(int(args.max_images))])
    argv.extend(["--max-patches", str(int(args.max_patches))])
    argv.extend(["--per-image-per-keypoint", str(int(args.per_image_per_keypoint))])
    argv.extend(["--pos-threshold", str(float(args.pos_threshold))])
    if bool(args.add_negatives):
        argv.append("--add-negatives")
    argv.extend(["--neg-threshold", str(float(args.neg_threshold))])
    argv.extend(["--negatives-per-image", str(int(args.negatives_per_image))])
    argv.extend(["--crop-px", str(int(args.crop_px))])
    argv.extend(["--sprite-border-px", str(int(args.sprite_border_px))])
    argv.extend(["--seed", str(int(args.seed))])
    if args.output:
        argv.extend(["--output", str(args.output)])
    if args.runs_root:
        argv.extend(["--runs-root", str(args.runs_root)])
    if args.run_name:
        argv.extend(["--run-name", str(args.run_name)])
    return int(tb_main(argv))


def _cmd_dino_kpseg_audit(args: argparse.Namespace) -> int:
    from annolid.segmentation.dino_kpseg.dataset_tools import audit_yolo_pose_dataset

    report = audit_yolo_pose_dataset(
        Path(args.data).expanduser().resolve(),
        split=str(args.split),
        max_images=(int(args.max_images) if args.max_images is not None else None),
        seed=int(args.seed),
        instance_mode=str(args.instance_mode),
        bbox_scale=float(args.bbox_scale),
    )
    if args.out:
        out_path = Path(args.out).expanduser().resolve()
        out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    else:
        print(json.dumps(report, indent=2))
    return 0


def _cmd_dino_kpseg_split(args: argparse.Namespace) -> int:
    from annolid.segmentation.dino_kpseg.dataset_tools import stratified_split

    summary = stratified_split(
        Path(args.data).expanduser().resolve(),
        output_dir=Path(args.output),
        val_size=float(args.val_size),
        seed=int(args.seed),
        group_by=str(args.group_by),
        group_regex=str(args.group_regex) if args.group_regex else None,
        include_val=bool(args.include_val),
    )
    print(json.dumps(summary, indent=2))
    return 0


def _cmd_dino_kpseg_precompute(args: argparse.Namespace) -> int:
    from annolid.segmentation.dino_kpseg.dataset_tools import precompute_features
    from annolid.segmentation.dino_kpseg.cli_utils import parse_layers

    layers = parse_layers(str(args.layers))
    summary = precompute_features(
        data_yaml=Path(args.data).expanduser().resolve(),
        model_name=str(args.model_name),
        short_side=int(args.short_side),
        layers=layers,
        device=(str(args.device).strip() if args.device else None),
        split=str(args.split),
        instance_mode=str(args.instance_mode),
        bbox_scale=float(args.bbox_scale),
        cache_dir=(
            Path(args.cache_dir).expanduser().resolve() if args.cache_dir else None
        ),
        cache_dtype=str(args.cache_dtype),
    )
    print(json.dumps(summary, indent=2))
    return 0


def _cmd_citations_list(args: argparse.Namespace) -> int:
    from annolid.utils.citations import entry_to_dict, load_bibtex, search_entries

    bib_file = Path(args.bib_file).expanduser().resolve()
    entries = load_bibtex(bib_file)
    query = str(args.query or "").strip()
    field = str(args.field).strip() if args.field else None
    limit = max(1, int(args.limit))
    rows = (
        search_entries(entries, query, field=field, limit=limit)
        if query
        else list(entries[:limit])
    )
    payload: dict[str, object] = {
        "bib_file": str(bib_file),
        "total_entries": len(entries),
        "returned": len(rows),
    }
    if bool(args.verbose):
        payload["entries"] = [entry_to_dict(entry) for entry in rows]
    else:
        payload["entries"] = [
            {
                "key": entry.key,
                "entry_type": entry.entry_type,
                "title": entry.fields.get("title"),
                "author": entry.fields.get("author"),
                "year": entry.fields.get("year"),
            }
            for entry in rows
        ]
    print(json.dumps(payload, indent=2))
    return 0


def _parse_citation_fields(args: argparse.Namespace) -> dict[str, str]:
    fields: dict[str, str] = {}
    for raw in list(args.field or []):
        token = str(raw or "").strip()
        if not token or "=" not in token:
            raise SystemExit(f"Invalid --field value {raw!r}; expected name=value.")
        name, value = token.split("=", 1)
        field_name = name.strip().lower()
        if not field_name:
            raise SystemExit(f"Invalid --field value {raw!r}; empty field name.")
        fields[field_name] = value.strip()

    for name in ("title", "author", "year", "journal", "booktitle", "doi", "url"):
        value = getattr(args, name, None)
        if value is not None and str(value).strip():
            fields[name] = str(value).strip()
    return fields


def _cmd_citations_upsert(args: argparse.Namespace) -> int:
    from annolid.utils.citations import BibEntry, load_bibtex, save_bibtex, upsert_entry

    bib_file = Path(args.bib_file).expanduser().resolve()
    key = str(args.key).strip()
    if not key:
        raise SystemExit("--key must be non-empty.")
    entry_type = str(args.entry_type or "").strip().lower()
    if not entry_type:
        raise SystemExit("--entry-type must be non-empty.")

    fields = _parse_citation_fields(args)
    if not fields:
        raise SystemExit(
            "At least one citation field is required. Use --field name=value "
            "or shortcuts like --title/--author/--year."
        )

    entries = load_bibtex(bib_file)
    updated, created = upsert_entry(
        entries, BibEntry(entry_type=entry_type, key=key, fields=fields)
    )
    save_bibtex(bib_file, updated, sort_keys=not bool(args.no_sort))

    print(
        json.dumps(
            {
                "bib_file": str(bib_file),
                "key": key,
                "created": bool(created),
                "total_entries": len(updated),
            },
            indent=2,
        )
    )
    return 0


def _cmd_citations_remove(args: argparse.Namespace) -> int:
    from annolid.utils.citations import load_bibtex, remove_entry, save_bibtex

    bib_file = Path(args.bib_file).expanduser().resolve()
    key = str(args.key).strip()
    if not key:
        raise SystemExit("--key must be non-empty.")
    entries = load_bibtex(bib_file)
    updated, removed = remove_entry(entries, key)
    if removed:
        save_bibtex(bib_file, updated, sort_keys=not bool(args.no_sort))
    print(
        json.dumps(
            {
                "bib_file": str(bib_file),
                "key": key,
                "removed": bool(removed),
                "total_entries": len(updated if removed else entries),
            },
            indent=2,
        )
    )
    return 0 if removed else 1


def _cmd_citations_format(args: argparse.Namespace) -> int:
    from annolid.utils.citations import load_bibtex, save_bibtex

    bib_file = Path(args.bib_file).expanduser().resolve()
    entries = load_bibtex(bib_file)
    save_bibtex(bib_file, entries, sort_keys=not bool(args.no_sort))
    print(
        json.dumps(
            {
                "bib_file": str(bib_file),
                "entries": len(entries),
                "sorted": not bool(args.no_sort),
            },
            indent=2,
        )
    )
    return 0


def _cmd_agent(args: argparse.Namespace) -> int:
    import time

    from annolid.services.agent import AgentPipelineRequest, run_agent_pipeline

    last_print = 0.0

    def _progress(frame_idx: int, written: int, total: int | None) -> None:
        nonlocal last_print
        if args.no_progress:
            return
        now = time.monotonic()
        if now - last_print < float(args.progress_interval):
            return
        last_print = now
        if total and total > 0:
            pct = int(round((written / max(total, 1)) * 100))
            print(
                f"[annolid-run] agent progress: frame={frame_idx} records={written} ({pct}%)",
                file=sys.stderr,
            )
        else:
            print(
                f"[annolid-run] agent progress: frame={frame_idx} records={written}",
                file=sys.stderr,
            )

    result = run_agent_pipeline(
        AgentPipelineRequest(
            video_path=str(args.video),
            behavior_spec_path=(str(args.schema) if args.schema else None),
            results_dir=(str(args.results_dir) if args.results_dir else None),
            out_ndjson_name=str(args.ndjson_name),
            max_frames=args.max_frames,
            stride=int(args.stride),
            include_llm_summary=bool(args.include_llm_summary),
            llm_summary_prompt=str(args.llm_summary_prompt),
            vision_adapter=str(args.vision_adapter or "none"),
            vision_pretrained=bool(args.vision_pretrained),
            vision_score_threshold=float(args.vision_score_threshold),
            vision_device=(str(args.vision_device) if args.vision_device else None),
            vision_weights=(str(args.vision_weights) if args.vision_weights else None),
            llm_adapter=str(args.llm_adapter or "none"),
            llm_profile=(str(args.llm_profile) if args.llm_profile else None),
            llm_provider=(str(args.llm_provider) if args.llm_provider else None),
            llm_model=(str(args.llm_model) if args.llm_model else None),
            llm_persist=bool(args.llm_persist),
            reuse_cache=not bool(args.no_cache),
        ),
        progress_callback=_progress,
    )

    summary = {
        "results_dir": str(result.results_dir),
        "ndjson_path": str(result.ndjson_path),
        "store_path": str(result.store_path),
        "records_written": result.records_written,
        "total_frames": result.total_frames,
        "stopped": result.stopped,
        "cached": result.cached,
    }
    print(json.dumps(summary, indent=2))
    return 0


def _cmd_agent_behavior(args: argparse.Namespace) -> int:
    from annolid.services.behavior_agent.runtime import (
        run_default_behavior_agent_pipeline,
    )

    video_path = Path(args.video).expanduser().resolve()
    result = run_default_behavior_agent_pipeline(
        video_path=video_path,
        results_dir=(str(args.results_dir) if args.results_dir else None),
        artifacts_ndjson=(
            str(args.artifacts_ndjson) if args.artifacts_ndjson else None
        ),
        run_id=(str(args.run_id) if args.run_id else None),
        episode_id=(str(args.episode_id) if args.episode_id else None),
        context_prompt=str(args.context_prompt or ""),
        assay=str(args.assay or ""),
        default_assay=str(args.default_assay),
        model_policy=str(args.model_policy),
        bout_frame_gap=int(args.bout_frame_gap),
        use_memory=not bool(args.no_memory),
        use_analysis=not bool(args.no_analysis),
    )
    summary = {
        "run_id": result.run_id,
        "manifest_path": str(result.manifest_path),
        "episode_id": result.episode.episode_id,
        "video_path": result.episode.video_path,
        "task_plan_assay": result.task_plan_assay,
        "artifact_count": result.artifact_count,
        "segment_count": result.segment_count,
        "validation_errors": list(result.validation_errors),
        "bout_counts": [row.to_dict() for row in result.bout_counts],
    }
    print(json.dumps(summary, indent=2))
    if bool(args.fail_on_validation_error) and result.validation_errors:
        return 1
    return 0


def _cmd_agent_onboard(args: argparse.Namespace) -> int:
    from annolid.services.agent_cron import onboard_agent_workspace

    overwrite = bool(
        getattr(args, "overwrite", False) or getattr(args, "update", False)
    )
    summary = onboard_agent_workspace(
        workspace=getattr(args, "workspace", None),
        overwrite=overwrite,
        dry_run=bool(getattr(args, "dry_run", False)),
        backup=not bool(getattr(args, "no_backup", False)),
        backup_dir=getattr(args, "backup_dir", None),
        prune_bootstrap=bool(getattr(args, "prune_bootstrap", False)),
    )
    print(json.dumps(summary, indent=2))
    return 0


def _cmd_agent_status(_: argparse.Namespace) -> int:
    from annolid.services.agent_cron import get_agent_status

    summary = get_agent_status()
    print(json.dumps(summary, indent=2))
    return 0


def _cmd_agent_tool_pool(args: argparse.Namespace) -> int:
    from annolid.services.agent_tooling import describe_agent_tool_pool

    summary = describe_agent_tool_pool(
        workspace=getattr(args, "workspace", None),
        provider=getattr(args, "provider", None),
        model=getattr(args, "model", None),
        config_path=getattr(args, "config", None),
    )
    print(json.dumps(summary, indent=2))
    return 0


def _cmd_agent_skill_pool(args: argparse.Namespace) -> int:
    from annolid.services.agent_tooling import describe_agent_skill_pool

    summary = describe_agent_skill_pool(
        workspace=getattr(args, "workspace", None),
        task_hint=getattr(args, "task_hint", None),
        top_k=max(1, int(getattr(args, "top_k", 5) or 5)),
    )
    print(json.dumps(summary, indent=2))
    return 0


def _cmd_agent_capabilities(args: argparse.Namespace) -> int:
    from annolid.services.agent_tooling import describe_agent_capabilities

    summary = describe_agent_capabilities(
        workspace=getattr(args, "workspace", None),
        provider=getattr(args, "provider", None),
        model=getattr(args, "model", None),
        task_hint=getattr(args, "task_hint", None),
        top_k=max(1, int(getattr(args, "top_k", 5) or 5)),
        config_path=getattr(args, "config", None),
    )
    print(json.dumps(summary, indent=2))
    return 0


def _cmd_agent_turn_snapshots(args: argparse.Namespace) -> int:
    from annolid.core.agent.session_manager import AgentSessionManager
    from annolid.core.agent.utils import get_agent_workspace_path

    workspace = get_agent_workspace_path(getattr(args, "workspace", None))
    manager = AgentSessionManager(workspace=workspace)
    session_id = str(getattr(args, "session_id", "") or "default")
    rows = manager.read_snapshots(
        session_id,
        limit=max(1, int(getattr(args, "limit", 20) or 20)),
    )
    payload = {
        "workspace": str(workspace),
        "session_id": session_id,
        "count": len(rows),
        "snapshots": rows,
    }
    print(json.dumps(payload, indent=2))
    return 0


def _mode_octal(path: Path) -> Optional[str]:
    try:
        return oct(path.stat().st_mode & 0o777)
    except OSError:
        return None


def _is_private_dir_mode(path: Path) -> bool:
    try:
        mode = path.stat().st_mode & 0o777
    except OSError:
        return False
    return (mode & 0o077) == 0


def _is_private_file_mode(path: Path) -> bool:
    try:
        mode = path.stat().st_mode & 0o777
    except OSError:
        return False
    # Owner read/write only.
    return mode == 0o600


def _find_persisted_secret_keys(data: object, prefix: str = "") -> list[str]:
    secret_names = {
        "api_key",
        "apikey",
        "access_token",
        "client_secret",
        "refresh_token",
        "token",
        "secret",
        "password",
    }
    if isinstance(data, dict):
        hits: list[str] = []
        for key, value in data.items():
            key_text = str(key or "").strip().lower()
            path = f"{prefix}.{key}" if prefix else str(key)
            if key_text in secret_names:
                hits.append(path)
            hits.extend(_find_persisted_secret_keys(value, path))
        return hits
    if isinstance(data, list):
        hits: list[str] = []
        for idx, item in enumerate(data):
            item_prefix = f"{prefix}[{idx}]" if prefix else f"[{idx}]"
            hits.extend(_find_persisted_secret_keys(item, item_prefix))
        return hits
    return []


def _find_agent_config_plaintext_secret_paths(
    data: object, prefix: str = ""
) -> list[str]:
    secret_names = {
        "access_token",
        "accesstoken",
        "api_key",
        "apikey",
        "client_secret",
        "clientsecret",
        "bridge_token",
        "bridgetoken",
        "password",
        "refresh_token",
        "refreshtoken",
        "secret",
        "token",
        "verify_token",
        "verifytoken",
    }
    if isinstance(data, dict):
        hits: list[str] = []
        for key, value in data.items():
            key_text = str(key or "").strip().lower()
            path = f"{prefix}.{key}" if prefix else str(key)
            if path == "secrets" or path.startswith("secrets."):
                continue
            if key_text in secret_names and str(value or "").strip():
                hits.append(path)
            hits.extend(_find_agent_config_plaintext_secret_paths(value, path))
        return hits
    if isinstance(data, list):
        hits: list[str] = []
        for idx, item in enumerate(data):
            item_prefix = f"{prefix}[{idx}]" if prefix else f"[{idx}]"
            hits.extend(_find_agent_config_plaintext_secret_paths(item, item_prefix))
        return hits
    return []


def _cmd_agent_security_check(_: argparse.Namespace) -> int:
    from annolid.services.agent_admin import run_agent_security_check

    payload = run_agent_security_check()
    print(json.dumps(payload, indent=2))
    return 0 if payload.get("status") == "ok" else 1


def _cmd_agent_security_audit(args: argparse.Namespace) -> int:
    from annolid.services.agent_admin import run_agent_security_audit

    payload = run_agent_security_audit(
        config_path=getattr(args, "config", None),
        fix=bool(getattr(args, "fix", False)),
    )
    print(json.dumps(payload, indent=2))
    return 0 if payload.get("status") == "ok" else 1


def _cmd_agent_secrets_audit(args: argparse.Namespace) -> int:
    from annolid.services.agent_admin import audit_agent_secrets

    payload = audit_agent_secrets(config_path=getattr(args, "config", None))
    print(json.dumps(payload, indent=2))
    return 0 if payload.get("status") == "ok" else 1


def _cmd_agent_secrets_set(args: argparse.Namespace) -> int:
    from annolid.services.agent_admin import set_agent_secret

    payload = set_agent_secret(
        path=str(args.path or ""),
        env=getattr(args, "env", None),
        local=getattr(args, "local", None),
        value=getattr(args, "value", None),
        config_path=getattr(args, "config", None),
    )
    print(json.dumps(payload, indent=2))
    return 0


def _cmd_agent_secrets_remove(args: argparse.Namespace) -> int:
    from annolid.services.agent_admin import remove_agent_secret

    payload = remove_agent_secret(
        path=str(args.path or ""),
        config_path=getattr(args, "config", None),
        delete_local_value=bool(args.delete_local_value),
    )
    print(json.dumps(payload, indent=2))
    return 0


def _cmd_agent_secrets_migrate(args: argparse.Namespace) -> int:
    from annolid.services.agent_admin import migrate_agent_secrets

    payload, exit_code = migrate_agent_secrets(
        config_path=getattr(args, "config", None),
        apply=bool(args.apply),
    )
    print(json.dumps(payload, indent=2))
    return exit_code


def _cmd_agent_config_migrate(args: argparse.Namespace) -> int:
    from annolid.services.agent_admin import migrate_agent_config_integrations

    payload, exit_code = migrate_agent_config_integrations(
        config_path=getattr(args, "config", None),
        apply=bool(getattr(args, "apply", False)),
    )
    print(json.dumps(payload, indent=2))
    return exit_code


def _cmd_agent_box_auth_url(args: argparse.Namespace) -> int:
    from annolid.services.agent_box import get_box_oauth_authorize_url
    import webbrowser

    payload = get_box_oauth_authorize_url(
        config_path=getattr(args, "config", None),
        client_id=getattr(args, "client_id", None),
        redirect_uri=getattr(args, "redirect_uri", None),
        state=getattr(args, "state", None),
        authorize_base_url=getattr(args, "authorize_base_url", None),
    )
    if bool(getattr(args, "open_browser", False)):
        authorize_url = str(payload.get("authorize_url") or "").strip()
        opened = False
        if authorize_url:
            try:
                opened = bool(webbrowser.open_new_tab(authorize_url))
            except Exception:
                opened = False
        payload["opened_in_browser"] = opened
    print(json.dumps(payload, indent=2))
    return 0


def _cmd_agent_box_auth_exchange(args: argparse.Namespace) -> int:
    from annolid.services.agent_box import exchange_box_oauth_code

    payload, exit_code = exchange_box_oauth_code(
        code=str(getattr(args, "code", "") or ""),
        redirect_uri=getattr(args, "redirect_uri", None),
        config_path=getattr(args, "config", None),
        client_id=getattr(args, "client_id", None),
        client_secret=getattr(args, "client_secret", None),
        token_url=getattr(args, "token_url", None),
        authorize_base_url=getattr(args, "authorize_base_url", None),
        persist=bool(getattr(args, "persist", False)),
        enable_box=bool(getattr(args, "enable_box", True)),
    )
    print(json.dumps(payload, indent=2))
    return exit_code


def _cmd_agent_box_token_refresh(args: argparse.Namespace) -> int:
    from annolid.services.agent_box import refresh_box_oauth_token

    payload, exit_code = refresh_box_oauth_token(
        config_path=getattr(args, "config", None),
        client_id=getattr(args, "client_id", None),
        client_secret=getattr(args, "client_secret", None),
        refresh_token=getattr(args, "refresh_token", None),
        token_url=getattr(args, "token_url", None),
        persist=bool(getattr(args, "persist", False)),
    )
    print(json.dumps(payload, indent=2))
    return exit_code


def _cmd_agent_update(args: argparse.Namespace) -> int:
    from annolid.services.agent_update import run_legacy_agent_update

    payload, exit_code = run_legacy_agent_update(
        channel=str(args.channel or "stable"),
        timeout_s=float(args.timeout_s),
        apply=bool(args.apply),
        execute=bool(args.execute),
        skip_doctor=bool(args.skip_doctor),
        require_signature=bool(args.require_signature),
    )
    print(json.dumps(payload, indent=2))
    return exit_code


def _cmd_agent_eval(args: argparse.Namespace) -> int:
    from annolid.services.agent_eval import run_agent_eval

    payload, exit_code = run_agent_eval(
        traces=args.traces,
        candidate_responses=args.candidate_responses,
        baseline_responses=args.baseline_responses,
        out=args.out,
        max_regressions=int(args.max_regressions),
    )
    print(json.dumps(payload, indent=2))
    return exit_code


def _cmd_agent_eval_build_regression(args: argparse.Namespace) -> int:
    from annolid.services.agent_eval import build_agent_regression_eval

    payload = build_agent_regression_eval(
        workspace=getattr(args, "workspace", None),
        out=args.out,
        min_abs_rating=int(args.min_abs_rating),
    )
    print(json.dumps(payload, indent=2))
    return 0


def _cmd_agent_eval_gate(args: argparse.Namespace) -> int:
    from annolid.services.agent_eval import evaluate_agent_eval_gate

    payload, exit_code = evaluate_agent_eval_gate(
        report=args.report,
        changed_files=args.changed_files,
        max_regressions=int(args.max_regressions),
        min_pass_rate=float(args.min_pass_rate),
    )
    print(json.dumps(payload, indent=2))
    return exit_code


def _cmd_agent_skills_refresh(args: argparse.Namespace) -> int:
    from annolid.services.agent_workspace import refresh_agent_skills

    payload = refresh_agent_skills(workspace=getattr(args, "workspace", None))
    print(json.dumps(payload, indent=2))
    return 0


def _cmd_agent_skills_inspect(args: argparse.Namespace) -> int:
    from annolid.services.agent_workspace import inspect_agent_skills

    payload = inspect_agent_skills(workspace=getattr(args, "workspace", None))
    print(json.dumps(payload, indent=2))
    return 0


def _cmd_agent_skills_shadow(args: argparse.Namespace) -> int:
    from annolid.services.agent_workspace import shadow_agent_skills

    payload = shadow_agent_skills(
        workspace=getattr(args, "workspace", None),
        candidate_pack=args.candidate_pack,
    )
    print(json.dumps(payload, indent=2))
    return 0


def _cmd_agent_skills_import(args: argparse.Namespace) -> int:
    from annolid.services.agent_workspace import import_agent_skills_pack

    payload = import_agent_skills_pack(
        workspace=getattr(args, "workspace", None),
        source_dir=str(getattr(args, "source_dir")),
        overwrite=bool(getattr(args, "overwrite", False)),
    )
    print(json.dumps(payload, indent=2))
    return 0


def _cmd_agent_feedback_add(args: argparse.Namespace) -> int:
    from annolid.services.agent_workspace import add_agent_feedback

    payload = add_agent_feedback(
        workspace=getattr(args, "workspace", None),
        session_id=str(getattr(args, "session_id", "") or "default"),
        rating=int(args.rating),
        comment=str(getattr(args, "comment", "") or ""),
        trace_id=str(getattr(args, "trace_id", "") or ""),
        expected_substring=str(getattr(args, "expected_substring", "") or ""),
    )
    print(json.dumps(payload, indent=2))
    return 0


def _cmd_agent_memory_flush(args: argparse.Namespace) -> int:
    from annolid.services.agent_workspace import flush_agent_memory

    payload = flush_agent_memory(
        workspace=getattr(args, "workspace", None),
        session_id=getattr(args, "session_id", None),
        note=getattr(args, "note", None),
    )
    print(json.dumps(payload, indent=2))
    return 0


def _cmd_agent_memory_inspect(args: argparse.Namespace) -> int:
    from annolid.services.agent_workspace import inspect_agent_memory

    payload = inspect_agent_memory(workspace=getattr(args, "workspace", None))
    print(json.dumps(payload, indent=2))
    return 0


def _cmd_agent_meta_learning_status(args: argparse.Namespace) -> int:
    from annolid.services.agent_workspace import inspect_agent_meta_learning

    payload = inspect_agent_meta_learning(
        workspace=getattr(args, "workspace", None),
        limit=int(getattr(args, "limit", 20)),
        brief=bool(getattr(args, "brief", False)),
    )
    print(json.dumps(payload, indent=2))
    return 0


def _cmd_agent_meta_learning_history(args: argparse.Namespace) -> int:
    from annolid.services.agent_workspace import inspect_agent_meta_learning_history

    payload = inspect_agent_meta_learning_history(
        workspace=getattr(args, "workspace", None),
        limit=int(getattr(args, "limit", 20)),
        full=bool(getattr(args, "full", False)),
    )
    print(json.dumps(payload, indent=2))
    return 0


def _cmd_agent_meta_learning_maintenance_run(args: argparse.Namespace) -> int:
    from annolid.services.agent_workspace import run_agent_meta_learning_maintenance

    payload = run_agent_meta_learning_maintenance(
        workspace=getattr(args, "workspace", None),
        force=bool(getattr(args, "force", False)),
        max_jobs=getattr(args, "max_jobs", None),
    )
    print(json.dumps(payload, indent=2))
    return 0


def _cmd_agent_meta_learning_maintenance_status(args: argparse.Namespace) -> int:
    from annolid.services.agent_workspace import (
        inspect_agent_meta_learning_maintenance_status,
    )

    payload = inspect_agent_meta_learning_maintenance_status(
        workspace=getattr(args, "workspace", None),
    )
    print(json.dumps(payload, indent=2))
    return 0


def _cmd_agent_meta_learning_maintenance_next_window(args: argparse.Namespace) -> int:
    from annolid.services.agent_workspace import inspect_agent_meta_learning_next_window

    payload = inspect_agent_meta_learning_next_window(
        workspace=getattr(args, "workspace", None),
    )
    print(json.dumps(payload, indent=2))
    return 0


def _cmd_agent_acp_bridge(args: argparse.Namespace) -> int:
    from annolid.services.agent_bridge import run_agent_acp_bridge

    return run_agent_acp_bridge(workspace=getattr(args, "workspace", None))


def _cmd_update_check(args: argparse.Namespace) -> int:
    from annolid.services.agent_update import check_for_agent_update

    payload = check_for_agent_update(
        project=str(args.project or "annolid"),
        channel=str(args.channel or "stable"),
        timeout_s=float(args.timeout_s),
        require_signature=bool(args.require_signature),
    )
    print(json.dumps(payload, indent=2))
    return 0


def _cmd_update_run(args: argparse.Namespace) -> int:
    from annolid.services.agent_update import run_agent_update

    payload, exit_code = run_agent_update(
        project=str(args.project or "annolid"),
        channel=str(args.channel or "stable"),
        timeout_s=float(args.timeout_s),
        require_signature=bool(args.require_signature),
        execute=bool(args.execute),
        skip_post_check=bool(args.skip_post_check),
        canary_metrics=args.canary_metrics,
        canary_min_samples=int(args.canary_min_samples),
        canary_max_failure_rate=float(args.canary_max_failure_rate),
        canary_max_regressions=int(args.canary_max_regressions),
    )
    print(json.dumps(payload, indent=2))
    return exit_code


def _cmd_update_rollback(args: argparse.Namespace) -> int:
    from annolid.services.agent_update import rollback_agent_update

    payload, exit_code = rollback_agent_update(
        install_mode=str(args.install_mode or "package"),
        project=str(args.project or "annolid"),
        previous_version=str(args.previous_version or ""),
        execute=bool(args.execute),
    )
    print(json.dumps(payload, indent=2))
    return exit_code


def _cmd_runtime_cli(argv: list[str]) -> int:
    from annolid.infrastructure.capabilities import (
        CAPABILITIES,
        INSTALL_PROFILES,
        _build_smart_install_plan,
        _find_matching_host_python,
        _find_uv_binary,
        _missing_dependencies,
        capability_status_dict,
        check_capability,
        detect_accelerator_summary,
        export_wheelhouse_bundle,
        install_capability_stream,
        install_profile_stream,
        list_capabilities,
        resolve_wheelhouse_dir,
        user_runtime_site_dir,
        verify_wheelhouse_bundle,
    )

    parser = argparse.ArgumentParser(
        prog="annolid-run runtime",
        description=(
            "Inspect, install, and bundle optional Annolid runtime packages on demand or offline."
        ),
        formatter_class=_AnnolidHelpFormatter,
    )
    sub = parser.add_subparsers(dest="runtime_cmd")

    status_p = sub.add_parser(
        "status",
        help="Show accelerator info and installed/missing optional capabilities.",
    )
    status_p.add_argument(
        "--json",
        action="store_true",
        help="Emit structured JSON output.",
    )

    profiles_p = sub.add_parser(
        "profiles",
        help="List named install profiles (minimal, gui, tracking, workstation, full).",
    )
    profiles_p.add_argument(
        "--json",
        action="store_true",
        help="Emit structured JSON output.",
    )

    install_p = sub.add_parser(
        "install",
        help="Install one or more optional capabilities or a named profile.",
    )
    install_p.add_argument(
        "targets",
        nargs="*",
        help="Capability or profile names (e.g. tracking sam3 webengine pdf).",
    )
    install_p.add_argument(
        "--profile",
        choices=tuple(sorted(INSTALL_PROFILES)),
        default=None,
        help="Named install profile to install.",
    )
    install_p.add_argument(
        "--wheelhouse",
        default=None,
        metavar="DIR",
        help="Local offline wheelhouse folder containing .whl packages.",
    )
    install_p.add_argument(
        "--offline",
        action="store_true",
        default=None,
        help="Disable PyPI network access (--no-index) and install strictly from --wheelhouse.",
    )
    install_p.add_argument(
        "--user-runtime",
        action="store_true",
        default=False,
        help="Allow installing into isolated ~/.annolid/runtime when running from a frozen bundle.",
    )
    install_p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print resolved install commands without executing them.",
    )

    bundle_p = sub.add_parser(
        "bundle",
        help="Export a SHA256-verified offline wheelhouse bundle for air-gapped machines.",
    )
    bundle_p.add_argument(
        "targets",
        nargs="*",
        help="Capability or profile names to bundle (e.g. tracking sam3 workstation).",
    )
    bundle_p.add_argument(
        "--profile",
        choices=tuple(sorted(INSTALL_PROFILES)),
        default=None,
        help="Named install profile to bundle.",
    )
    bundle_p.add_argument(
        "--out",
        "-o",
        required=True,
        metavar="DIR",
        help="Output directory for downloaded .whl packages and wheelhouse-manifest.json.",
    )
    bundle_p.add_argument(
        "--python-version",
        default=None,
        metavar="X.Y",
        help="Optional target Python major.minor version (e.g. 3.12, 3.14).",
    )
    bundle_p.add_argument(
        "--platform",
        default=None,
        metavar="TAG",
        help="Optional PEP 425 platform tag (e.g. manylinux2014_x86_64, win_amd64, macosx_11_0_arm64).",
    )
    bundle_p.add_argument(
        "--json",
        action="store_true",
        help="Emit manifest JSON to stdout.",
    )

    verify_p = sub.add_parser(
        "verify-bundle",
        help="Verify .whl files and SHA256 checksums in an offline wheelhouse directory.",
    )
    verify_p.add_argument(
        "wheelhouse",
        metavar="DIR",
        help="Path to the offline wheelhouse directory to verify.",
    )
    verify_p.add_argument(
        "--json",
        action="store_true",
        help="Emit verification report as JSON.",
    )

    effective_argv = argv if argv else ["status"]
    if effective_argv in (["-h"], ["--help"]):
        parser.print_help()
        return 0
    args = parser.parse_args(effective_argv)
    cmd = str(args.runtime_cmd or "status")

    def _parse_targets_and_profiles(
        raw_targets: list[str],
        explicit_profile: str | None,
    ) -> tuple[list[str], list[str]] | None:
        req_profiles: list[str] = []
        req_caps: list[str] = []
        if explicit_profile:
            req_profiles.append(str(explicit_profile).strip().lower())
        for raw_target in raw_targets:
            for token in str(raw_target).split(","):
                key = token.strip().lower()
                if not key:
                    continue
                if key in CAPABILITIES:
                    if key not in req_caps:
                        req_caps.append(key)
                elif key in INSTALL_PROFILES:
                    if key not in req_profiles:
                        req_profiles.append(key)
                else:
                    known_caps = ", ".join(list_capabilities())
                    known_profiles = ", ".join(sorted(INSTALL_PROFILES))
                    print(
                        f"[annolid-run] Unknown capability or profile {key!r}.\n"
                        f"  Capabilities: {known_caps}\n"
                        f"  Profiles:     {known_profiles}",
                        file=sys.stderr,
                    )
                    return None
        for pname in req_profiles:
            for cap in INSTALL_PROFILES[pname]["capabilities"]:  # type: ignore[index]
                cap_key = str(cap)
                if cap_key not in req_caps:
                    req_caps.append(cap_key)
        return req_profiles, req_caps

    if cmd == "status":
        env_info = detect_accelerator_summary()
        statuses = [check_capability(name) for name in list_capabilities()]
        if bool(args.json):
            payload = {
                "environment": env_info,
                "capabilities": [capability_status_dict(st) for st in statuses],
            }
            print(json.dumps(payload, indent=2))
            return 0
        header_line = (
            f"Environment: Python {env_info['python_version']} "
            f"({env_info['python_executable']}) | "
            f"accelerator={env_info['accelerator']} | "
            f"writable={env_info['writable']}"
        )
        if env_info.get("offline_mode"):
            header_line += " | offline=True"
        if env_info.get("wheelhouse_dir"):
            header_line += f" | wheelhouse={env_info['wheelhouse_dir']}"
        print(header_line)
        if env_info.get("frozen"):
            print(
                f"Frozen User Runtime: {env_info['user_runtime_dir']} "
                f"(backend={env_info['user_runtime_backend']})"
            )
        print("")
        print(f"{'CAPABILITY':<18} {'STATUS':<11} {'~MB':>5}  DESCRIPTION")
        print("-" * 80)
        for st in statuses:
            state_label = "installed" if st.available else "missing"
            print(
                f"{st.name:<18} {state_label:<11} {st.approx_download_mb:>5}  {st.description}"
            )
        return 0

    if cmd == "profiles":
        rows = []
        for pname, pmeta in INSTALL_PROFILES.items():
            caps = [str(c) for c in pmeta.get("capabilities", ())]  # type: ignore[union-attr]
            missing_caps = [c for c in caps if not check_capability(c).available]
            rows.append(
                {
                    "profile": pname,
                    "ready": len(missing_caps) == 0,
                    "capabilities": caps,
                    "missing_capabilities": missing_caps,
                    "extras": list(pmeta.get("extras", ())),  # type: ignore[arg-type]
                    "description": str(pmeta.get("description", "")),
                }
            )
        if bool(args.json):
            print(json.dumps(rows, indent=2))
            return 0
        print(f"{'PROFILE':<14} {'READY':<8} CAPABILITIES / DESCRIPTION")
        print("-" * 80)
        for row in rows:
            ready_label = "yes" if row["ready"] else "no"
            caps_str = (
                ", ".join(row["capabilities"])
                if row["capabilities"]
                else "(core GUI only)"
            )
            print(f"{row['profile']:<14} {ready_label:<8} {caps_str}")
            print(f"{'':<23} {row['description']}")
        return 0

    if cmd == "install":
        parsed = _parse_targets_and_profiles(list(args.targets or ()), args.profile)
        if parsed is None:
            return 2
        requested_profiles, requested_caps = parsed
        if not requested_caps and not requested_profiles:
            install_p.print_help()
            return 2

        if bool(args.dry_run):
            all_missing = []
            seen_pkgs: set[str] = set()
            for cap_key in requested_caps:
                for dep in _missing_dependencies(CAPABILITIES[cap_key]):
                    if dep.package_spec not in seen_pkgs:
                        seen_pkgs.add(dep.package_spec)
                        all_missing.append(dep)
            if not all_missing:
                print("All requested capabilities are already installed.")
                return 0
            try:
                use_user_rt = bool(getattr(sys, "frozen", False))
                target_site = user_runtime_site_dir() if use_user_rt else None
                if target_site is not None and not (
                    _find_uv_binary() or _find_matching_host_python()
                ):
                    wh_resolved = resolve_wheelhouse_dir(args.wheelhouse)
                    if wh_resolved is None:
                        raise RuntimeError(
                            "Frozen bundle requires --wheelhouse, 'uv', or matching host Python."
                        )
                    specs_str = " ".join(dep.package_spec for dep in all_missing)
                    print(
                        f"[wheelhouse] extract ({specs_str}) from {wh_resolved} -> {target_site}"
                    )
                    return 0
                _, steps = _build_smart_install_plan(
                    tuple(all_missing),
                    wheelhouse=args.wheelhouse,
                    offline=args.offline,
                    target_site_dir=target_site,
                )
            except RuntimeError as exc:
                print(f"[annolid-run] {exc}", file=sys.stderr)
                return 1
            for primary_cmd, _fallback in steps:
                print(" ".join(primary_cmd))
            return 0

        def _echo(line: str) -> None:
            sys.stdout.write(line if line.endswith("\n") else line + "\n")
            sys.stdout.flush()

        try:
            installed_specs: list[str] = []
            if requested_profiles and not args.targets and len(requested_profiles) == 1:
                specs = install_profile_stream(
                    requested_profiles[0],
                    wheelhouse=args.wheelhouse,
                    offline=args.offline,
                    allow_frozen_user_runtime=bool(args.user_runtime),
                    on_output=_echo,
                )
                installed_specs.extend(specs)
            else:
                for cap_key in requested_caps:
                    specs = install_capability_stream(
                        cap_key,
                        auto_install=True,
                        wheelhouse=args.wheelhouse,
                        offline=args.offline,
                        allow_frozen_user_runtime=bool(args.user_runtime),
                        on_output=_echo,
                    )
                    for spec in specs:
                        if spec not in installed_specs:
                            installed_specs.append(spec)
        except RuntimeError as exc:
            print(f"[annolid-run] {exc}", file=sys.stderr)
            return 1

        if not installed_specs:
            print("All requested runtime capabilities are already installed.")
        else:
            print(f"Installed {len(installed_specs)} package specification(s).")
        return 0

    if cmd == "bundle":
        parsed = _parse_targets_and_profiles(list(args.targets or ()), args.profile)
        if parsed is None:
            return 2
        requested_profiles, requested_caps = parsed
        if not requested_caps:
            bundle_p.print_help()
            return 2

        def _bundle_echo(line: str) -> None:
            out_stream = sys.stderr if bool(args.json) else sys.stdout
            out_stream.write(line if line.endswith("\n") else line + "\n")
            out_stream.flush()

        try:
            manifest = export_wheelhouse_bundle(
                args.out,
                capability_names=requested_caps,
                profile_names=requested_profiles,
                python_version=args.python_version,
                platform_tag=args.platform,
                on_output=_bundle_echo,
            )
        except (RuntimeError, ValueError, KeyError) as exc:
            print(f"[annolid-run] {exc}", file=sys.stderr)
            return 1

        if bool(args.json):
            print(json.dumps(manifest, indent=2))
        else:
            print(
                f"Exported offline bundle ({manifest['wheel_count']} wheels) to {args.out}"
            )
        return 0

    if cmd == "verify-bundle":
        try:
            report = verify_wheelhouse_bundle(args.wheelhouse)
        except RuntimeError as exc:
            if bool(args.json):
                print(json.dumps({"valid": False, "error": str(exc)}, indent=2))
            else:
                print(
                    f"[annolid-run] Bundle verification failed: {exc}", file=sys.stderr
                )
            return 1

        if bool(args.json):
            print(json.dumps(report, indent=2))
        else:
            manifest_note = (
                "manifest SHA256 verified"
                if report.get("manifest_present")
                else "wheel files present (no manifest)"
            )
            print(
                f"Verified wheelhouse: {report['wheelhouse_dir']} "
                f"({report['wheel_count']} wheels, {manifest_note})"
            )
        return 0

    parser.print_help()
    return 2


def _dispatch_operator_commands(argv: list[str]) -> Optional[int]:
    if not argv:
        return None

    # annolid-run runtime ...
    if argv[0] == "runtime":
        return _cmd_runtime_cli(argv[1:])

    # annolid-run colab ...
    if argv[0] == "colab":
        from annolid.colab.cli import main as colab_cli_main

        return int(colab_cli_main(argv[1:]))

    # annolid-run sim2real ...
    if argv[0] == "sim2real":
        from annolid.sim2real.cli import main as sim2real_cli_main

        return int(sim2real_cli_main(argv[1:]))

    # annolid-run memory ...
    if argv[0] == "memory":
        from annolid.interfaces.memory.cli import main as memory_cli_main

        return int(memory_cli_main(argv[1:]))

    # annolid-run agent skills refresh
    if (
        len(argv) >= 3
        and argv[0] == "agent"
        and argv[1] == "skills"
        and argv[2] == "refresh"
    ):
        p = argparse.ArgumentParser(prog="annolid-run agent skills refresh")
        p.add_argument("--workspace", default=None)
        args = p.parse_args(argv[3:])
        return _cmd_agent_skills_refresh(args)

    # annolid-run agent skills inspect
    if (
        len(argv) >= 3
        and argv[0] == "agent"
        and argv[1] == "skills"
        and argv[2] == "inspect"
    ):
        p = argparse.ArgumentParser(prog="annolid-run agent skills inspect")
        p.add_argument("--workspace", default=None)
        args = p.parse_args(argv[3:])
        return _cmd_agent_skills_inspect(args)

    # annolid-run agent skills shadow
    if (
        len(argv) >= 3
        and argv[0] == "agent"
        and argv[1] == "skills"
        and argv[2] == "shadow"
    ):
        p = argparse.ArgumentParser(prog="annolid-run agent skills shadow")
        p.add_argument("--workspace", default=None)
        p.add_argument("--candidate-pack", required=True)
        args = p.parse_args(argv[3:])
        return _cmd_agent_skills_shadow(args)

    # annolid-run agent skills import
    if (
        len(argv) >= 3
        and argv[0] == "agent"
        and argv[1] == "skills"
        and argv[2] == "import"
    ):
        p = argparse.ArgumentParser(prog="annolid-run agent skills import")
        p.add_argument("--workspace", default=None)
        p.add_argument("--source-dir", required=True)
        p.add_argument("--overwrite", action="store_true")
        args = p.parse_args(argv[3:])
        return _cmd_agent_skills_import(args)

    # annolid-run agent memory flush
    if (
        len(argv) >= 3
        and argv[0] == "agent"
        and argv[1] == "memory"
        and argv[2] == "flush"
    ):
        p = argparse.ArgumentParser(prog="annolid-run agent memory flush")
        p.add_argument("--workspace", default=None)
        p.add_argument("--session-id", default=None)
        p.add_argument("--note", default=None)
        args = p.parse_args(argv[3:])
        return _cmd_agent_memory_flush(args)

    # annolid-run agent memory inspect
    if (
        len(argv) >= 3
        and argv[0] == "agent"
        and argv[1] == "memory"
        and argv[2] == "inspect"
    ):
        p = argparse.ArgumentParser(prog="annolid-run agent memory inspect")
        p.add_argument("--workspace", default=None)
        args = p.parse_args(argv[3:])
        return _cmd_agent_memory_inspect(args)

    # annolid-run agent meta-learning status
    if (
        len(argv) >= 3
        and argv[0] == "agent"
        and argv[1] == "meta-learning"
        and argv[2] == "status"
    ):
        p = argparse.ArgumentParser(prog="annolid-run agent meta-learning status")
        p.add_argument("--workspace", default=None)
        p.add_argument("--limit", type=int, default=20)
        p.add_argument(
            "--brief",
            action="store_true",
            help="Show concise summary output without event and file details.",
        )
        args = p.parse_args(argv[3:])
        return _cmd_agent_meta_learning_status(args)

    # annolid-run agent meta-learning history
    if (
        len(argv) >= 3
        and argv[0] == "agent"
        and argv[1] == "meta-learning"
        and argv[2] == "history"
    ):
        p = argparse.ArgumentParser(prog="annolid-run agent meta-learning history")
        p.add_argument("--workspace", default=None)
        p.add_argument("--limit", type=int, default=20)
        p.add_argument("--full", action="store_true")
        args = p.parse_args(argv[3:])
        return _cmd_agent_meta_learning_history(args)

    # annolid-run agent meta-learning maintenance status
    if (
        len(argv) >= 4
        and argv[0] == "agent"
        and argv[1] == "meta-learning"
        and argv[2] == "maintenance"
        and argv[3] == "status"
    ):
        p = argparse.ArgumentParser(
            prog="annolid-run agent meta-learning maintenance status"
        )
        p.add_argument("--workspace", default=None)
        args = p.parse_args(argv[4:])
        return _cmd_agent_meta_learning_maintenance_status(args)

    # annolid-run agent meta-learning maintenance next-window
    if (
        len(argv) >= 4
        and argv[0] == "agent"
        and argv[1] == "meta-learning"
        and argv[2] == "maintenance"
        and argv[3] == "next-window"
    ):
        p = argparse.ArgumentParser(
            prog="annolid-run agent meta-learning maintenance next-window"
        )
        p.add_argument("--workspace", default=None)
        args = p.parse_args(argv[4:])
        return _cmd_agent_meta_learning_maintenance_next_window(args)

    # annolid-run agent meta-learning maintenance run
    if (
        len(argv) >= 4
        and argv[0] == "agent"
        and argv[1] == "meta-learning"
        and argv[2] == "maintenance"
        and argv[3] == "run"
    ):
        p = argparse.ArgumentParser(
            prog="annolid-run agent meta-learning maintenance run"
        )
        p.add_argument("--workspace", default=None)
        p.add_argument("--force", action="store_true")
        p.add_argument("--max-jobs", type=int, default=None)
        args = p.parse_args(argv[4:])
        return _cmd_agent_meta_learning_maintenance_run(args)

    # annolid-run agent acp bridge
    if (
        len(argv) >= 3
        and argv[0] == "agent"
        and argv[1] == "acp"
        and argv[2] == "bridge"
    ):
        p = argparse.ArgumentParser(prog="annolid-run agent acp bridge")
        p.add_argument("--workspace", default=None)
        args = p.parse_args(argv[3:])
        return _cmd_agent_acp_bridge(args)

    # annolid-run agent eval run
    if len(argv) >= 3 and argv[0] == "agent" and argv[1] == "eval" and argv[2] == "run":
        p = argparse.ArgumentParser(prog="annolid-run agent eval run")
        p.add_argument("--traces", required=True)
        p.add_argument("--candidate-responses", required=True)
        p.add_argument("--baseline-responses", default=None)
        p.add_argument("--out", required=True)
        p.add_argument("--max-regressions", type=int, default=0)
        args = p.parse_args(argv[3:])
        return _cmd_agent_eval(args)

    # annolid-run agent eval build-regression
    if (
        len(argv) >= 3
        and argv[0] == "agent"
        and argv[1] == "eval"
        and argv[2] == "build-regression"
    ):
        p = argparse.ArgumentParser(prog="annolid-run agent eval build-regression")
        p.add_argument("--workspace", default=None)
        p.add_argument("--out", required=True)
        p.add_argument("--min-abs-rating", type=int, default=1)
        args = p.parse_args(argv[3:])
        return _cmd_agent_eval_build_regression(args)

    # annolid-run agent eval gate
    if (
        len(argv) >= 3
        and argv[0] == "agent"
        and argv[1] == "eval"
        and argv[2] == "gate"
    ):
        p = argparse.ArgumentParser(prog="annolid-run agent eval gate")
        p.add_argument("--report", default=None)
        p.add_argument("--changed-files", default=None)
        p.add_argument("--max-regressions", type=int, default=0)
        p.add_argument("--min-pass-rate", type=float, default=0.0)
        args = p.parse_args(argv[3:])
        return _cmd_agent_eval_gate(args)

    # annolid-run agent feedback add
    if (
        len(argv) >= 3
        and argv[0] == "agent"
        and argv[1] == "feedback"
        and argv[2] == "add"
    ):
        p = argparse.ArgumentParser(prog="annolid-run agent feedback add")
        p.add_argument("--workspace", default=None)
        p.add_argument("--session-id", default="default")
        p.add_argument("--trace-id", default=None)
        p.add_argument("--rating", type=int, choices=(-1, 0, 1), required=True)
        p.add_argument("--comment", default="")
        p.add_argument("--expected-substring", default="")
        args = p.parse_args(argv[3:])
        return _cmd_agent_feedback_add(args)

    # annolid-run agent secrets audit
    if (
        len(argv) >= 3
        and argv[0] == "agent"
        and argv[1] == "secrets"
        and argv[2] == "audit"
    ):
        p = argparse.ArgumentParser(prog="annolid-run agent secrets audit")
        p.add_argument("--config", default=None)
        args = p.parse_args(argv[3:])
        return _cmd_agent_secrets_audit(args)

    # annolid-run agent secrets set
    if (
        len(argv) >= 3
        and argv[0] == "agent"
        and argv[1] == "secrets"
        and argv[2] == "set"
    ):
        p = argparse.ArgumentParser(prog="annolid-run agent secrets set")
        p.add_argument("--config", default=None)
        p.add_argument("--path", required=True)
        p.add_argument("--env", default=None)
        p.add_argument("--local", default=None)
        p.add_argument("--value", default="")
        args = p.parse_args(argv[3:])
        return _cmd_agent_secrets_set(args)

    # annolid-run agent secrets remove
    if (
        len(argv) >= 3
        and argv[0] == "agent"
        and argv[1] == "secrets"
        and argv[2] == "remove"
    ):
        p = argparse.ArgumentParser(prog="annolid-run agent secrets remove")
        p.add_argument("--config", default=None)
        p.add_argument("--path", required=True)
        p.add_argument("--delete-local-value", action="store_true")
        args = p.parse_args(argv[3:])
        return _cmd_agent_secrets_remove(args)

    # annolid-run agent secrets migrate
    if (
        len(argv) >= 3
        and argv[0] == "agent"
        and argv[1] == "secrets"
        and argv[2] == "migrate"
    ):
        p = argparse.ArgumentParser(prog="annolid-run agent secrets migrate")
        p.add_argument("--config", default=None)
        p.add_argument("--apply", action="store_true")
        args = p.parse_args(argv[3:])
        return _cmd_agent_secrets_migrate(args)

    # annolid-run agent config migrate
    if (
        len(argv) >= 3
        and argv[0] == "agent"
        and argv[1] == "config"
        and argv[2] == "migrate"
    ):
        p = argparse.ArgumentParser(prog="annolid-run agent config migrate")
        p.add_argument("--config", default=None)
        p.add_argument("--apply", action="store_true")
        args = p.parse_args(argv[3:])
        return _cmd_agent_config_migrate(args)

    # annolid-run agent security audit
    if (
        len(argv) >= 3
        and argv[0] == "agent"
        and argv[1] == "security"
        and argv[2] == "audit"
    ):
        p = argparse.ArgumentParser(prog="annolid-run agent security audit")
        p.add_argument("--config", default=None)
        p.add_argument("--fix", action="store_true")
        args = p.parse_args(argv[3:])
        return _cmd_agent_security_audit(args)

    # annolid-run update check|run|rollback
    if len(argv) >= 2 and argv[0] == "update":
        action = argv[1]
        if action == "check":
            p = argparse.ArgumentParser(prog="annolid-run update check")
            p.add_argument("--project", default="annolid")
            p.add_argument(
                "--channel", choices=("stable", "beta", "dev"), default="stable"
            )
            p.add_argument("--timeout-s", type=float, default=4.0)
            p.add_argument("--require-signature", action="store_true")
            args = p.parse_args(argv[2:])
            return _cmd_update_check(args)
        if action == "run":
            p = argparse.ArgumentParser(prog="annolid-run update run")
            p.add_argument("--project", default="annolid")
            p.add_argument(
                "--channel", choices=("stable", "beta", "dev"), default="stable"
            )
            p.add_argument("--timeout-s", type=float, default=4.0)
            p.add_argument("--require-signature", action="store_true")
            p.add_argument("--execute", action="store_true")
            p.add_argument("--skip-post-check", action="store_true")
            p.add_argument("--canary-metrics", default=None)
            p.add_argument("--canary-min-samples", type=int, default=20)
            p.add_argument("--canary-max-failure-rate", type=float, default=0.05)
            p.add_argument("--canary-max-regressions", type=int, default=0)
            args = p.parse_args(argv[2:])
            return _cmd_update_run(args)
        if action == "rollback":
            p = argparse.ArgumentParser(prog="annolid-run update rollback")
            p.add_argument("--project", default="annolid")
            p.add_argument(
                "--install-mode", choices=("package", "source"), default="package"
            )
            p.add_argument("--previous-version", required=True)
            p.add_argument("--execute", action="store_true")
            args = p.parse_args(argv[2:])
            return _cmd_update_rollback(args)
        raise SystemExit(f"Unknown update action: {action}")

    return None


def _cmd_agent_cron_list(args: argparse.Namespace) -> int:
    from annolid.services.agent_cron import list_agent_cron_jobs

    rows = list_agent_cron_jobs(include_all=bool(args.all))
    print(json.dumps(rows, indent=2))
    return 0


def _cmd_agent_cron_add(args: argparse.Namespace) -> int:
    from annolid.services.agent_cron import add_agent_cron_job

    payload = add_agent_cron_job(
        name=str(args.name),
        message=str(args.message),
        deliver=bool(args.deliver),
        channel=(str(args.channel) if args.channel else None),
        to=(str(args.to) if args.to else None),
        every=args.every,
        cron_expr=args.cron_expr,
        at=args.at,
        tz=args.tz,
    )
    print(json.dumps(payload, indent=2))
    return 0


def _cmd_agent_cron_remove(args: argparse.Namespace) -> int:
    from annolid.services.agent_cron import remove_agent_cron_job

    payload, exit_code = remove_agent_cron_job(job_id=str(args.job_id))
    print(json.dumps(payload, indent=2))
    return exit_code


def _cmd_agent_cron_enable(args: argparse.Namespace) -> int:
    from annolid.services.agent_cron import set_agent_cron_job_enabled

    payload, exit_code = set_agent_cron_job_enabled(
        job_id=str(args.job_id),
        enabled=not bool(args.disable),
    )
    print(json.dumps(payload, indent=2))
    return exit_code


def _cmd_agent_cron_run(args: argparse.Namespace) -> int:
    from annolid.services.agent_cron import run_agent_cron_job

    payload, exit_code = run_agent_cron_job(
        job_id=str(args.job_id),
        force=bool(args.force),
    )
    print(json.dumps(payload, indent=2))
    return exit_code


def _build_root_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="annolid-run",
        description=(
            "Annolid command-line interface for models, agent workflows, datasets, "
            "validation, and maintenance tasks."
        ),
        epilog=_root_help_epilog(),
        formatter_class=_AnnolidHelpFormatter,
    )
    sub = p.add_subparsers(dest="command", required=True)

    list_p = sub.add_parser("list-models", help="List available model plugins.")
    list_p.set_defaults(_handler=_cmd_list_models)

    dinov3_p = sub.add_parser(
        "dinov3-models",
        help="List or pre-download DINOv3 feature backbones.",
    )
    dinov3_p.add_argument(
        "--list",
        action="store_true",
        help="List supported DINO-family feature backbones as JSON.",
    )
    dinov3_p.add_argument(
        "--dinov3-only",
        action="store_true",
        help="With --list, hide compatible non-DINOv3 backbones.",
    )
    dinov3_p.add_argument(
        "--model",
        action="append",
        help=(
            "Model id, display name, or alias to download. May be repeated. "
            "Defaults to the recommended DINOv3 ViT-S/16 checkpoint."
        ),
    )
    dinov3_p.add_argument(
        "--cache-dir",
        default=None,
        help="Optional Hugging Face cache directory. DINOV3_LOCATION is also honored.",
    )
    dinov3_p.add_argument(
        "--local-files-only",
        action="store_true",
        help="Verify the model is already cached without network access.",
    )
    dinov3_p.set_defaults(_handler=_cmd_dinov3_models)

    val_p = sub.add_parser(
        "validate-agent-output",
        help="Validate agent NDJSON output against the canonical JSON Schema.",
    )
    val_p.add_argument("--ndjson", required=True, help="Path to the NDJSON file.")
    val_p.add_argument(
        "--max-errors",
        type=int,
        default=1,
        help="Stop after this many errors (default: 1).",
    )
    val_p.set_defaults(_handler=_cmd_validate_agent_output)

    validate_tools_p = sub.add_parser(
        "validate-agent-tools",
        help="Run lightweight validation checks for agent tool modules.",
    )
    validate_tools_p.set_defaults(_handler=_cmd_agent_validate_tools)

    citations_list_p = sub.add_parser(
        "citations-list",
        help="List/search BibTeX entries from a .bib file.",
    )
    citations_list_p.add_argument(
        "--bib-file", required=True, help="Path to a BibTeX .bib file."
    )
    citations_list_p.add_argument(
        "--query",
        default=None,
        help="Optional case-insensitive substring query.",
    )
    citations_list_p.add_argument(
        "--field",
        default=None,
        help="Optional field name to constrain query, e.g. title or author.",
    )
    citations_list_p.add_argument(
        "--limit",
        type=int,
        default=50,
        help="Maximum number of entries returned (default: 50).",
    )
    citations_list_p.add_argument(
        "--verbose",
        action="store_true",
        help="Include complete entry fields in output.",
    )
    citations_list_p.set_defaults(_handler=_cmd_citations_list)

    citations_upsert_p = sub.add_parser(
        "citations-upsert",
        help="Create or update a BibTeX entry by key.",
    )
    citations_upsert_p.add_argument(
        "--bib-file", required=True, help="Path to a BibTeX .bib file."
    )
    citations_upsert_p.add_argument("--key", required=True, help="Citation key.")
    citations_upsert_p.add_argument(
        "--entry-type", default="article", help="BibTeX entry type (default: article)."
    )
    citations_upsert_p.add_argument(
        "--field",
        action="append",
        default=[],
        help="Field assignment in name=value format (repeatable).",
    )
    citations_upsert_p.add_argument("--title", default=None)
    citations_upsert_p.add_argument("--author", default=None)
    citations_upsert_p.add_argument("--year", default=None)
    citations_upsert_p.add_argument("--journal", default=None)
    citations_upsert_p.add_argument("--booktitle", default=None)
    citations_upsert_p.add_argument("--doi", default=None)
    citations_upsert_p.add_argument("--url", default=None)
    citations_upsert_p.add_argument(
        "--no-sort",
        action="store_true",
        help="Do not sort entries by key when writing output.",
    )
    citations_upsert_p.set_defaults(_handler=_cmd_citations_upsert)

    citations_remove_p = sub.add_parser(
        "citations-remove",
        help="Remove a BibTeX entry by key.",
    )
    citations_remove_p.add_argument(
        "--bib-file", required=True, help="Path to a BibTeX .bib file."
    )
    citations_remove_p.add_argument("--key", required=True, help="Citation key.")
    citations_remove_p.add_argument(
        "--no-sort",
        action="store_true",
        help="Do not sort entries by key when writing output.",
    )
    citations_remove_p.set_defaults(_handler=_cmd_citations_remove)

    citations_format_p = sub.add_parser(
        "citations-format",
        help="Rewrite a BibTeX file using canonical formatting.",
    )
    citations_format_p.add_argument(
        "--bib-file", required=True, help="Path to a BibTeX .bib file."
    )
    citations_format_p.add_argument(
        "--no-sort",
        action="store_true",
        help="Preserve original order instead of sorting by key.",
    )
    citations_format_p.set_defaults(_handler=_cmd_citations_format)

    time_budget_p = sub.add_parser(
        "time-budget",
        help="Compute behavior time-budget and bout summaries from an event CSV.",
    )
    time_budget_p.add_argument(
        "events_csv",
        help="Path to a behavior event CSV export.",
    )
    time_budget_p.add_argument(
        "-o",
        "--output",
        "--output-csv",
        dest="output",
        default=None,
        help="Optional CSV path to write the time-budget summary.",
    )
    time_budget_p.add_argument(
        "--json-out",
        default=None,
        help="Optional JSON path to write the structured time-budget report.",
    )
    time_budget_p.add_argument(
        "--json",
        action="store_true",
        help="Print structured JSON report to stdout instead of a formatted table.",
    )
    time_budget_p.add_argument(
        "--strict",
        action="store_true",
        help="Fail with exit code 2 if unmatched start/end events are encountered.",
    )
    time_budget_p.add_argument(
        "--bin-size",
        type=float,
        default=None,
        help="Optional bin width (seconds) for a binned secondary time-budget table.",
    )
    time_budget_p.add_argument(
        "--session-duration",
        type=float,
        default=None,
        help="Optional explicit session duration (seconds) for % of session denominator.",
    )
    time_budget_p.add_argument(
        "--schema",
        default=None,
        help="Optional project schema file (project.annolid.json or YAML).",
    )
    time_budget_p.add_argument(
        "--bout-profile",
        choices=("aggression",),
        default=None,
        help="Optional bout preset (e.g., aggression).",
    )
    time_budget_p.add_argument(
        "--bout-name",
        default="custom_bout",
        help="Name for a custom bout definition (used with --bout-behaviors).",
    )
    time_budget_p.add_argument(
        "--bout-behaviors",
        default=None,
        help="Comma-separated behaviors to group into bouts.",
    )
    time_budget_p.add_argument(
        "--bout-initiation-behaviors",
        default=None,
        help="Comma-separated behaviors counted as bout initiations.",
    )
    time_budget_p.add_argument(
        "--bout-gap-seconds",
        type=float,
        default=None,
        help="Maximum gap (seconds) between intervals within the same bout.",
    )
    time_budget_p.add_argument(
        "--start-time",
        type=float,
        default=None,
        help="Optional analysis window start time (seconds) to clip intervals.",
    )
    time_budget_p.add_argument(
        "--end-time",
        type=float,
        default=None,
        help="Optional analysis window end time (seconds) to clip intervals.",
    )
    time_budget_p.add_argument(
        "--exclude-behaviors",
        default=None,
        help="Comma-separated behaviors (e.g. out_of_sight) subtracted from per-subject session duration.",
    )
    time_budget_p.add_argument(
        "--cooccurrence-csv",
        default=None,
        help="Optional CSV path to write pairwise behavioral co-occurrence statistics.",
    )
    time_budget_p.add_argument(
        "--ethogram-svg",
        default=None,
        help="Optional SVG path to render a multi-subject State/Point ethogram.",
    )
    time_budget_p.add_argument(
        "--transitions-dot",
        default=None,
        help="Optional Graphviz DOT path for sequential behavioral transitions.",
    )
    time_budget_p.add_argument(
        "--transitions-svg",
        default=None,
        help="Optional standalone SVG path for sequential behavioral transitions.",
    )
    time_budget_p.add_argument(
        "--external-data",
        action="append",
        default=None,
        help="Optional external physiological/hardware CSV/TSV file(s) to align against behaviors.",
    )
    time_budget_p.add_argument(
        "--external-offset",
        type=float,
        default=0.0,
        help="Time offset in seconds added to external data timestamps (default: 0.0).",
    )
    time_budget_p.add_argument(
        "--external-data-svg",
        default=None,
        help="Optional SVG path to render synchronized external traces with shaded behavior bouts.",
    )
    time_budget_p.set_defaults(_handler=_cmd_time_budget)

    qc_p = sub.add_parser(
        "behavior-qc",
        help="Validate behavior event CSVs for unpaired states and schema exclusions.",
    )
    qc_p.add_argument(
        "events_csv",
        help="Path to a behavior event CSV export.",
    )
    qc_p.add_argument(
        "--schema",
        default=None,
        help="Optional project schema file (project.annolid.json or YAML).",
    )
    qc_p.add_argument(
        "--strict",
        action="store_true",
        help="Treat warnings as failures (exit code 2).",
    )
    qc_p.add_argument(
        "--strict-schema",
        action="store_true",
        help="Treat undeclared behaviors, subjects, or modifiers as errors.",
    )
    qc_p.add_argument(
        "--json",
        action="store_true",
        help="Print structured JSON report to stdout.",
    )
    qc_p.add_argument(
        "--out",
        default=None,
        help="Optional JSON path to write the QC report.",
    )
    qc_p.add_argument(
        "--csv-out",
        default=None,
        help="Optional CSV path to write detected QC issues.",
    )
    qc_p.set_defaults(_handler=_cmd_behavior_qc)

    irr_p = sub.add_parser(
        "behavior-irr",
        help="Compute Inter-Rater Reliability (Cohen's & Fleiss' kappa) across 2+ CSVs.",
    )
    irr_p.add_argument(
        "events_csvs",
        nargs="+",
        help="Paths to 2 or more behavior event CSV files to compare.",
    )
    irr_p.add_argument(
        "--rater-names",
        default=None,
        help="Optional comma-separated rater labels matching input CSV order.",
    )
    irr_p.add_argument(
        "--schema",
        default=None,
        help="Optional project schema file (project.annolid.json or YAML).",
    )
    irr_p.add_argument(
        "--bin-seconds",
        type=float,
        default=0.1,
        help="Time-unit rasterization bin width in seconds (default: 0.1).",
    )
    irr_p.add_argument(
        "--onset-tolerance-seconds",
        type=float,
        default=0.5,
        help="Maximum timestamp offset (seconds) for matching event onsets (default: 0.5).",
    )
    irr_p.add_argument(
        "--subject",
        default=None,
        help="Optional subject name filter.",
    )
    irr_p.add_argument(
        "--pool-subjects",
        action="store_true",
        help="Pool events across all subjects per behavior.",
    )
    irr_p.add_argument(
        "--session-start",
        type=float,
        default=None,
        help="Explicit evaluation window start time in seconds.",
    )
    irr_p.add_argument(
        "--session-end",
        type=float,
        default=None,
        help="Explicit evaluation window end time in seconds.",
    )
    irr_p.add_argument(
        "--session-duration",
        type=float,
        default=None,
        help="Explicit evaluation window duration in seconds from session start.",
    )
    irr_p.add_argument(
        "--include-empty-schema-behaviors",
        action="store_true",
        help="Include zero-occurrence behaviors from --schema in IRR rows.",
    )
    irr_p.add_argument(
        "--min-kappa",
        type=float,
        default=None,
        help="Optional quality gate: fail with exit code 2 if macro Cohen's kappa is below this threshold.",
    )
    irr_p.add_argument(
        "--json",
        action="store_true",
        help="Print structured JSON report to stdout.",
    )
    irr_p.add_argument(
        "--out",
        default=None,
        help="Optional JSON path to write the full IRR report.",
    )
    irr_p.add_argument(
        "--csv-out",
        default=None,
        help="Optional CSV path to write per-behavior IRR metrics.",
    )
    irr_p.add_argument(
        "--disagreements-csv",
        default=None,
        help="Optional CSV path to write contiguous rater disagreement windows for adjudication.",
    )
    irr_p.set_defaults(_handler=_cmd_behavior_irr)

    cohort_p = sub.add_parser(
        "behavior-cohort",
        help="Aggregate multi-animal cohort behavior endpoints and Markov transition matrices.",
    )
    cohort_p.add_argument(
        "events_csvs",
        nargs="*",
        help="Optional behavior event CSV files or directories (may be combined with --manifest).",
    )
    cohort_p.add_argument(
        "--manifest",
        default=None,
        help="Optional CSV or JSON cohort manifest mapping recordings to experimental groups.",
    )
    cohort_p.add_argument(
        "--metadata-csv",
        default=None,
        help="Optional metadata CSV to unblind/join experimental group assignments onto recordings.",
    )
    cohort_p.add_argument(
        "--schema",
        default=None,
        help="Optional project schema file (project.annolid.json or YAML).",
    )
    cohort_p.add_argument(
        "--session-duration",
        type=float,
        default=None,
        help="Optional explicit session duration override in seconds for % session and censored latency.",
    )
    cohort_p.add_argument(
        "--max-transition-gap",
        type=float,
        default=None,
        help="Optional maximum inter-event gap (seconds) counted as a Markov transition.",
    )
    cohort_p.add_argument(
        "--exclude-self-transitions",
        action="store_true",
        help="Exclude repeated B->B self-transitions when computing transition probabilities.",
    )
    cohort_p.add_argument(
        "--include-schema-behaviors",
        action="store_true",
        help="Include zero-occurrence behaviors declared in --schema across cohort tables.",
    )
    cohort_p.add_argument(
        "--unit",
        choices=("subject", "recording"),
        default="subject",
        help="Experimental unit for inferential comparisons ('subject' or 'recording' to prevent dyadic pseudoreplication).",
    )
    cohort_p.add_argument(
        "--strict",
        action="store_true",
        help="Fail with exit code 2 if unmatched state start/stop events are encountered.",
    )
    cohort_p.add_argument(
        "--json",
        action="store_true",
        help="Print structured JSON cohort report to stdout.",
    )
    cohort_p.add_argument(
        "--out",
        default=None,
        help="Optional JSON path to write the full cohort report and provenance.",
    )
    cohort_p.add_argument(
        "--summary-csv",
        default=None,
        help="Optional CSV path to write group-level mean/SD/SEM/median/IQR summaries.",
    )
    cohort_p.add_argument(
        "--subjects-csv",
        default=None,
        help="Optional CSV path to write tidy per-(recording, subject, behavior) rows.",
    )
    cohort_p.add_argument(
        "--transitions-csv",
        default=None,
        help="Optional CSV path to write first-order Markov transition probabilities and lift.",
    )
    cohort_p.add_argument(
        "--comparisons-csv",
        default=None,
        help="Optional CSV path to write inferential between-group hypothesis test and effect-size rows.",
    )
    cohort_p.add_argument(
        "--transitions-dot",
        default=None,
        help="Optional Graphviz DOT path to export behavioral transition pathways.",
    )
    cohort_p.add_argument(
        "--transitions-svg",
        default=None,
        help="Optional standalone SVG path to render behavioral transition pathways.",
    )
    cohort_p.set_defaults(_handler=_cmd_behavior_cohort)

    bsearch_p = sub.add_parser(
        "behavior-search",
        help="Search video frames or clips for behaviors by text or exemplar frame using ONNX EmbeddingGemma 2.",
    )
    bsearch_p.add_argument("video", help="Path to the source video file.")
    bsearch_p.add_argument(
        "--query",
        default=None,
        help="Natural-language behavior description (e.g. 'mouse grooming').",
    )
    bsearch_p.add_argument(
        "--query-frame",
        type=int,
        default=None,
        help="Reference frame index for visual or hybrid frame similarity search.",
    )
    bsearch_p.add_argument(
        "--query-image",
        default=None,
        help="Optional external exemplar image path for visual similarity search.",
    )
    bsearch_p.add_argument(
        "--behaviors",
        default=None,
        help="Comma-separated behavior labels for zero-shot multi-behavior scoring.",
    )
    bsearch_p.add_argument(
        "--backend",
        choices=("embeddinggemma2_onnx", "dinov3", "qwen3vl"),
        default="embeddinggemma2_onnx",
        help="Embedding backend (default: embeddinggemma2_onnx).",
    )
    bsearch_p.add_argument(
        "--model-id",
        default=None,
        help="Optional Hugging Face repo id or local ONNX directory override.",
    )
    bsearch_p.add_argument(
        "--quantization",
        choices=("q4", "quantized", "q4f16", "fp32", "fp16"),
        default="q4",
        help="ONNX EmbeddingGemma 2 quantization preset (default: q4).",
    )
    bsearch_p.add_argument(
        "--matryoshka-dim",
        type=int,
        choices=(128, 256, 512, 768),
        default=None,
        help="Optional Matryoshka truncated embedding dimension (128, 256, 512, or 768).",
    )
    bsearch_p.add_argument(
        "--soft-tokens",
        type=int,
        choices=(70, 140, 280, 560),
        default=None,
        help="ONNX EmbeddingGemma 2 visual soft-token budget per frame (default: 140; 70 is fastest on CPU).",
    )
    bsearch_p.add_argument(
        "--stride",
        type=int,
        default=5,
        help="Sample every Nth frame across the video (default: 5).",
    )
    bsearch_p.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional cap on the number of sampled frames/windows.",
    )
    bsearch_p.add_argument(
        "--clip-frames",
        type=int,
        default=1,
        help="Frames per temporal clip window (1 = single frame, 4/8 = short behavior clip).",
    )
    bsearch_p.add_argument(
        "--batch-size",
        type=int,
        default=4,
        help="Batch size for single-frame ONNX embedding extraction (default: 4).",
    )
    bsearch_p.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Minimum cosine similarity to keep a match; omit for adaptive auto-threshold.",
    )
    bsearch_p.add_argument(
        "--top-k",
        type=int,
        default=50,
        help="Maximum number of top matches to return (default: 50).",
    )
    bsearch_p.add_argument(
        "--overlay-shapes",
        action="store_true",
        help="Overlay existing LabelMe/AnnotationStore shapes before embedding.",
    )
    bsearch_p.add_argument(
        "--annotation-dir",
        default=None,
        help="Optional annotation directory for --overlay-shapes.",
    )
    bsearch_p.add_argument(
        "--index-cache",
        default=None,
        help="Optional .npz file path to save/reuse video frame embeddings across queries.",
    )
    bsearch_p.add_argument(
        "--auto-cache",
        action="store_true",
        help="Automatically save/reuse bounded .npz frame embedding caches under ~/.cache/annolid/embedding_search.",
    )
    bsearch_p.add_argument(
        "--out",
        default=None,
        help="Optional JSON file path to write full search results and bouts.",
    )
    bsearch_p.add_argument(
        "--json",
        action="store_true",
        help="Print structured JSON search results to stdout.",
    )
    bsearch_p.set_defaults(_handler=_cmd_behavior_search)

    boris_import_p = sub.add_parser(
        "boris-import",
        help="Import a BORIS (.boris) project into Annolid schema, event CSVs, and cohort manifest.",
    )
    boris_import_p.add_argument(
        "boris_file",
        help="Path to a .boris JSON project file.",
    )
    boris_import_p.add_argument(
        "-o",
        "--output-dir",
        required=True,
        help="Target directory for project.annolid.json, <obs>_events.csv, and cohort_manifest.json.",
    )
    boris_import_p.add_argument(
        "--observations",
        default=None,
        help="Optional comma-separated observation IDs to import.",
    )
    boris_import_p.add_argument(
        "--apply-time-offset",
        action="store_true",
        help="Add each observation's BORIS time_offset to event timestamps.",
    )
    boris_import_p.add_argument(
        "--group-variable",
        default=None,
        help="Optional BORIS independent variable name used as experimental group in cohort_manifest.json.",
    )
    boris_import_p.add_argument(
        "--json",
        action="store_true",
        help="Print structured JSON import summary to stdout.",
    )
    boris_import_p.set_defaults(_handler=_cmd_boris_import)

    boris_export_p = sub.add_parser(
        "boris-export",
        help="Export Annolid schema and event CSVs to BORIS (.boris) and/or BORIS CSV/TSV tables.",
    )
    boris_export_p.add_argument(
        "events_csvs",
        nargs="*",
        help="Zero or more behavior event CSV/TSV files to include as observations.",
    )
    boris_export_p.add_argument(
        "--schema",
        default=None,
        help="Optional project.annolid.json (or YAML) schema file.",
    )
    boris_export_p.add_argument(
        "-o",
        "--output",
        default=None,
        help="Output path (.boris by default, or CSV/TSV when --format is tabular-csv/aggregated-csv).",
    )
    boris_export_p.add_argument(
        "--format",
        choices=("boris", "tabular-csv", "aggregated-csv"),
        default="boris",
        help="Primary format written to -o/--output (default: boris).",
    )
    boris_export_p.add_argument(
        "--tabular-csv",
        default=None,
        help="Optional output path for BORIS Tabular Events CSV/TSV (START/STOP/POINT).",
    )
    boris_export_p.add_argument(
        "--aggregated-csv",
        default=None,
        help="Optional output path for BORIS Aggregated Events CSV/TSV (Start/Stop/Duration).",
    )
    boris_export_p.add_argument(
        "--project-name",
        default="Annolid Behavioral Project",
        help="Project title stored in the exported .boris file.",
    )
    boris_export_p.add_argument(
        "--json",
        action="store_true",
        help="Print structured JSON export summary to stdout.",
    )
    boris_export_p.set_defaults(_handler=_cmd_boris_export)

    kinematics_p = sub.add_parser(
        "kinematics",
        help="Compute calibrated trajectory kinematics, freezing bouts, and pairwise inter-animal metrics.",
    )
    kinematics_p.add_argument(
        "source_path",
        help="Path to a tracking CSV (*_tracking.csv / *_tracked.csv) or LabelMe frame JSON directory.",
    )
    kinematics_p.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Recording frame rate in Hz (inferred from timestamps when omitted).",
    )
    kinematics_p.add_argument(
        "--px-per-cm",
        type=float,
        default=None,
        help="Spatial calibration factor (pixels per centimeter) for physical distance and cm/s velocity.",
    )
    kinematics_p.add_argument(
        "--smooth-window",
        type=int,
        default=5,
        help="Rolling median/mean smoothing window in frames (default: 5).",
    )
    kinematics_p.add_argument(
        "--freeze-speed-threshold",
        type=float,
        default=None,
        help="Speed threshold (px/s) below which an instance is scored as freezing.",
    )
    kinematics_p.add_argument(
        "--min-freeze-duration",
        type=float,
        default=0.5,
        help="Minimum contiguous immobility duration in seconds for a freezing bout (default: 0.5).",
    )
    kinematics_p.add_argument(
        "--close-contact-px",
        type=float,
        default=80.0,
        help="Inter-animal centroid distance threshold (px) for close-contact percentage (default: 80.0).",
    )
    kinematics_p.add_argument(
        "--output-dir",
        default=None,
        help="Optional directory to write kinematics CSVs, JSON, and plot PNGs/SVGs.",
    )
    kinematics_p.add_argument(
        "--calibration",
        default=None,
        help="Optional planar perspective homography / arena corner calibration JSON file.",
    )
    kinematics_p.add_argument(
        "--out-json",
        default=None,
        help="Optional path to write the structured kinematics JSON report.",
    )
    kinematics_p.add_argument(
        "--events-csv",
        default=None,
        help="Optional path to export freezing and high-activity bouts as a Behavior Event CSV.",
    )
    kinematics_p.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip matplotlib PNG plot generation when --output-dir is set.",
    )
    kinematics_p.add_argument(
        "--json",
        action="store_true",
        help="Print structured JSON kinematics report to stdout.",
    )
    kinematics_p.set_defaults(_handler=_cmd_kinematics)

    zone_p = sub.add_parser(
        "zone-analysis",
        help="Compute ROI zone occupancy, entries, latencies, place preference, and social proximity summaries.",
    )
    zone_p.add_argument(
        "video_path",
        help="Path to the video file, tracking CSV, or frame annotation directory.",
    )
    zone_p.add_argument(
        "--zone-file",
        default=None,
        help="Optional *_zones.json file defining polygon, rectangle, or circle arena zones.",
    )
    zone_p.add_argument(
        "--zone-policy",
        default=None,
        help="Optional zone correction policy JSON/YAML file.",
    )
    zone_p.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Optional frame rate override (Hz).",
    )
    zone_p.add_argument(
        "--profile",
        default="generic",
        help="Assay profile name (e.g. generic, open_field, elevated_plus_maze, three_chamber_social).",
    )
    zone_p.add_argument(
        "--mode",
        choices=(
            "legacy_csv",
            "zone_metrics",
            "assay_summary",
            "social_summary",
            "zone_corrected_tracked_csv",
        ),
        default="zone_metrics",
        help="Zone export mode (default: zone_metrics).",
    )
    zone_p.add_argument(
        "--anchor-part",
        default=None,
        help="Optional anatomical keypoint anchor (e.g. 'nose') for zone occupancy.",
    )
    zone_p.add_argument(
        "--latency-reference-frame",
        type=int,
        default=None,
        help="Optional reference frame index for first-entry latency calculations.",
    )
    zone_p.add_argument(
        "--output-dir",
        default=None,
        help="Optional directory for exported zone CSV and Markdown artifacts.",
    )
    zone_p.add_argument(
        "--out-json",
        default=None,
        help="Optional path to write the structured zone analysis JSON summary.",
    )
    zone_p.add_argument(
        "--events-csv",
        default=None,
        help="Optional path to export contiguous zone visits as a Behavior Event CSV.",
    )
    zone_p.add_argument(
        "--json",
        action="store_true",
        help="Print structured JSON summary to stdout.",
    )
    zone_p.set_defaults(_handler=_cmd_zone_analysis)

    bp_ethogram_p = sub.add_parser(
        "body-part-ethogram",
        help="Compute standalone anatomical body-part posture and social-contact ethograms.",
    )
    bp_ethogram_p.add_argument(
        "annotation_dir",
        help="Directory containing LabelMe JSONs or AnnotationStore with body-part polygons.",
    )
    bp_ethogram_p.add_argument(
        "--fps",
        type=float,
        default=30.0,
        help="Frame rate in Hz (default: 30.0).",
    )
    bp_ethogram_p.add_argument(
        "--contact-margin-px",
        type=float,
        default=6.0,
        help="Boundary proximity threshold (px) for directed part-to-part contact bouts (default: 6.0).",
    )
    bp_ethogram_p.add_argument(
        "--min-bout-frames",
        type=int,
        default=1,
        help="Minimum contiguous frames for a contact bout (default: 1).",
    )
    bp_ethogram_p.add_argument(
        "--max-uncertain-frames",
        type=int,
        default=10,
        help="Maximum active-learning uncertain frames to report (default: 10).",
    )
    bp_ethogram_p.add_argument(
        "--camera-view",
        choices=("auto", "top", "oblique", "side", "bottom"),
        default="auto",
        help="Camera viewpoint hint for posture disentanglement (default: auto).",
    )
    bp_ethogram_p.add_argument(
        "--output-prefix",
        default=None,
        help="Optional path prefix for exporting kinematics, bouts, and uncertainty CSVs.",
    )
    bp_ethogram_p.add_argument(
        "--events-csv",
        default=None,
        help="Optional path to export social-contact and posture bouts as a Behavior Event CSV.",
    )
    bp_ethogram_p.add_argument(
        "--out-json",
        default=None,
        help="Optional path to write the structured body-part ethogram JSON summary.",
    )
    bp_ethogram_p.add_argument(
        "--json",
        action="store_true",
        help="Print structured JSON summary to stdout.",
    )
    bp_ethogram_p.set_defaults(_handler=_cmd_body_part_ethogram)

    gap_report_p = sub.add_parser(
        "gap-report",
        help="Scan a video, tracking CSV, or frame JSON directory for per-instance tracking gaps.",
    )
    gap_report_p.add_argument(
        "source_path",
        help="Path to the video file, tracking CSV, or LabelMe frame JSON directory.",
    )
    gap_report_p.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Optional frame rate override (Hz) for gap durations in seconds.",
    )
    gap_report_p.add_argument(
        "--no-write-reports",
        action="store_true",
        help="Do not write sibling Markdown and CSV gap report files.",
    )
    gap_report_p.add_argument(
        "--out-json",
        default=None,
        help="Optional path to write the structured tracking gap JSON report.",
    )
    gap_report_p.add_argument(
        "--json",
        action="store_true",
        help="Print structured JSON gap report to stdout.",
    )
    gap_report_p.set_defaults(_handler=_cmd_gap_report)

    tracking_stats_p = sub.add_parser(
        "tracking-stats",
        help="Aggregate *_tracking_stats.json telemetry across videos into CSVs and summary plots.",
    )
    tracking_stats_p.add_argument(
        "root_dir",
        help="Root directory to recursively scan for *_tracking_stats.json files.",
    )
    tracking_stats_p.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for CSVs and plots (defaults to <root_dir>/tracking_stats_dashboard).",
    )
    tracking_stats_p.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip matplotlib PNG plot generation.",
    )
    tracking_stats_p.add_argument(
        "--out-json",
        default=None,
        help="Optional path to write the tracking stats summary JSON.",
    )
    tracking_stats_p.add_argument(
        "--json",
        action="store_true",
        help="Print structured JSON summary to stdout.",
    )
    tracking_stats_p.set_defaults(_handler=_cmd_tracking_stats)

    report_p = sub.add_parser(
        "report",
        help="Generate a unified self-contained HTML5 + inline SVG + Markdown + JSON scientific report.",
    )
    report_p.add_argument(
        "source_path",
        help="Path to a video, tracking CSV, frame JSON directory, or workspace directory.",
    )
    report_p.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for analysis_report.{html,md,json}.",
    )
    report_p.add_argument(
        "--events-csv",
        default=None,
        help="Optional Behavior Event CSV for Time Budget and Ethogram QC sections.",
    )
    report_p.add_argument(
        "--zone-file",
        default=None,
        help="Optional *_zones.json file for Spatial Zone & Place-Preference section.",
    )
    report_p.add_argument(
        "--schema",
        default=None,
        help="Optional project schema file (project.annolid.json or YAML).",
    )
    report_p.add_argument(
        "--pose-schema",
        default=None,
        help="Optional PoseSchema file for anatomical skeleton checks.",
    )
    report_p.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Optional frame rate override in Hz.",
    )
    report_p.add_argument(
        "--px-per-cm",
        type=float,
        default=None,
        help="Optional spatial calibration factor (pixels per cm).",
    )
    report_p.add_argument(
        "--calibration",
        default=None,
        help="Optional planar perspective homography / arena corner calibration JSON file.",
    )
    report_p.add_argument(
        "--title",
        default=None,
        help="Optional custom report title.",
    )
    report_p.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip optional matplotlib PNG plots (deterministic inline SVGs are always included).",
    )
    report_p.add_argument(
        "--json",
        action="store_true",
        help="Print structured JSON report manifest to stdout.",
    )
    report_p.set_defaults(_handler=_cmd_unified_report)

    track_qc_p = sub.add_parser(
        "track-qc",
        help="Audit multi-animal tracks and generate a prioritized active-learning review queue.",
    )
    track_qc_p.add_argument(
        "input_path",
        help="Path to a LabelMe frame JSON directory or Annolid tracking CSV (*_tracking.csv / *_tracked.csv).",
    )
    track_qc_p.add_argument(
        "--schema",
        default=None,
        help="Optional project schema file (project.annolid.json or YAML).",
    )
    track_qc_p.add_argument(
        "--pose-schema",
        default=None,
        help="Optional PoseSchema JSON/YAML file for anatomical keypoint bone-length checks.",
    )
    track_qc_p.add_argument(
        "--expected-instances",
        type=int,
        default=None,
        help="Expected number of tracked animal instances per frame.",
    )
    track_qc_p.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Optional recording frame rate (Hz) recorded in report metadata.",
    )
    track_qc_p.add_argument(
        "--max-speed-px",
        type=float,
        default=None,
        help="Explicit centroid velocity threshold (px/frame); defaults to adaptive MAD threshold.",
    )
    track_qc_p.add_argument(
        "--speed-mad-z",
        type=float,
        default=6.0,
        help="Robust MAD z-score multiplier for adaptive velocity spike detection (default: 6.0).",
    )
    track_qc_p.add_argument(
        "--min-jump-px",
        type=float,
        default=25.0,
        help="Minimum displacement floor in pixels for jump and swap checks (default: 25.0).",
    )
    track_qc_p.add_argument(
        "--max-area-ratio",
        type=float,
        default=2.5,
        help="Frame-to-frame mask/bbox area fold-change threshold (default: 2.5).",
    )
    track_qc_p.add_argument(
        "--overlap-iou",
        type=float,
        default=0.25,
        help="Pairwise mask/bbox IoU threshold for close-contact overlap warnings (default: 0.25).",
    )
    track_qc_p.add_argument(
        "--close-contact-px",
        type=float,
        default=80.0,
        help="Centroid proximity threshold (px) for close-contact swap checks (default: 80.0).",
    )
    track_qc_p.add_argument(
        "--max-edge-stretch",
        type=float,
        default=2.5,
        help="Skeleton edge stretch multiplier relative to median bone length (default: 2.5).",
    )
    track_qc_p.add_argument(
        "--min-score",
        type=float,
        default=0.35,
        help="Minimum prediction confidence before flagging low_confidence (default: 0.35).",
    )
    track_qc_p.add_argument(
        "--bout-gap-frames",
        type=int,
        default=5,
        help="Maximum frame gap for merging nearby anomalies into a ReviewBout (default: 5).",
    )
    track_qc_p.add_argument(
        "--top-k",
        type=int,
        default=25,
        help="Maximum number of prioritized ReviewBout windows to emit (default: 25).",
    )
    track_qc_p.add_argument(
        "--strict",
        action="store_true",
        help="Treat warnings as failures (exit code 2).",
    )
    track_qc_p.add_argument(
        "--json",
        action="store_true",
        help="Print structured JSON report to stdout.",
    )
    track_qc_p.add_argument(
        "--out",
        default=None,
        help="Optional JSON path to write the full track QC report and provenance.",
    )
    track_qc_p.add_argument(
        "--issues-csv",
        default=None,
        help="Optional CSV path to write individual detected tracking anomalies.",
    )
    track_qc_p.add_argument(
        "--review-csv",
        default=None,
        help="Optional CSV path to write the ranked active-learning review queue.",
    )
    track_qc_p.set_defaults(_handler=_cmd_track_qc)

    repair_tracks_p = sub.add_parser(
        "repair-tracks",
        help="Audit or repair multi-animal identity switches in LabelMe JSON directories (dry-run by default).",
    )
    repair_tracks_p.add_argument(
        "annotation_dir",
        help="Directory containing per-frame LabelMe JSON annotations.",
    )
    repair_tracks_p.add_argument(
        "--mode",
        choices=("temporal", "governor"),
        default="temporal",
        help="Repair engine: 'temporal' multi-cue continuity or 'governor' zone/evidence policy (default: temporal).",
    )
    repair_tracks_p.add_argument(
        "--apply",
        action="store_true",
        help="Write repaired identities back to frame JSON files (defaults to safe dry-run).",
    )
    repair_tracks_p.add_argument(
        "--start-frame",
        type=int,
        default=0,
        help="Reference anchor start frame for temporal repair (default: 0).",
    )
    repair_tracks_p.add_argument(
        "--expected-instances",
        type=int,
        default=None,
        help="Expected number of tracked animal instances in temporal mode.",
    )
    repair_tracks_p.add_argument(
        "--max-gap-frames",
        type=int,
        default=5,
        help="Maximum occlusion gap frames to fill/carry forward in temporal mode (default: 5).",
    )
    repair_tracks_p.add_argument(
        "--max-match-distance",
        type=float,
        default=80.0,
        help="Maximum centroid matching distance in pixels for temporal repair (default: 80.0).",
    )
    repair_tracks_p.add_argument(
        "--policy",
        default=None,
        help="Path to a GovernorPolicy JSON/YAML file (required when --mode governor).",
    )
    repair_tracks_p.add_argument(
        "--zones",
        default=None,
        help="Optional zone shapes JSON file for --mode governor.",
    )
    repair_tracks_p.add_argument(
        "--report-out",
        default=None,
        help="Optional path for the detailed engine audit JSON report.",
    )
    repair_tracks_p.add_argument(
        "--json-out",
        default=None,
        help="Optional path to write the CLI repair summary JSON.",
    )
    repair_tracks_p.add_argument(
        "--json",
        action="store_true",
        help="Print structured JSON summary to stdout.",
    )
    repair_tracks_p.set_defaults(_handler=_cmd_repair_tracks)

    seg_parts_p = sub.add_parser(
        "segment-body-parts",
        help="Segment anatomical body parts strictly inside tracked whole-animal instances.",
    )
    seg_parts_p.add_argument(
        "video_path",
        help="Path to the source video file.",
    )
    seg_parts_p.add_argument(
        "--annotations",
        "--result-dir",
        dest="annotation_dir",
        default=None,
        help="Tracked LabelMe JSON / AnnotationStore folder (defaults to <video_stem>/).",
    )
    seg_parts_p.add_argument(
        "--mode",
        choices=("auto", "seeded", "keypoint_guided", "unsupervised"),
        default="auto",
        help="Segmentation strategy: manual part seeds, tracked keypoints, unsupervised K-parts, or auto.",
    )
    seg_parts_p.add_argument(
        "--backbone",
        choices=("fast", "dino", "hybrid"),
        default="fast",
        help="Crop feature backend: lightweight Lab+gradient ('fast'), DINO patch features ('dino'), or fused ('hybrid').",
    )
    seg_parts_p.add_argument(
        "--dino-model",
        default="facebook/dinov3-vits16-pretrain-lvd1689m",
        help="DINO feature model ID or alias when --backbone is 'dino' or 'hybrid'.",
    )
    seg_parts_p.add_argument(
        "--crop-size",
        type=int,
        default=224,
        help="Maximum working side (px) for tight per-instance ROI crops (default: 224).",
    )
    seg_parts_p.add_argument(
        "--crop-padding",
        type=float,
        default=0.12,
        help="Fractional bounding-box context padding around each tracked instance mask.",
    )
    seg_parts_p.add_argument(
        "--num-parts",
        type=int,
        default=3,
        help="Number of intra-instance body parts in unsupervised mode (default: 3 -> head, torso, tail).",
    )
    seg_parts_p.add_argument(
        "--parts",
        default=None,
        help="Optional comma-separated body-part names for unsupervised mode (e.g. 'head,torso,tail').",
    )
    seg_parts_p.add_argument(
        "--keypoint-granularity",
        choices=("coarse", "fine", "per_keypoint"),
        default="coarse",
        help="Landmark-to-part grouping granularity in keypoint_guided mode.",
    )
    seg_parts_p.add_argument(
        "--keypoint-part-map",
        default=None,
        help="Optional JSON object or 'kp=part,kp2=part2' overrides for keypoint_guided mode.",
    )
    seg_parts_p.add_argument(
        "--pose-schema",
        default=None,
        help="Optional PoseSchema YAML/JSON path providing intra-part skeleton edges.",
    )
    part_group = seg_parts_p.add_mutually_exclusive_group()
    part_group.add_argument(
        "--partition-full-instance",
        action="store_true",
        help="Force every pixel of each parent instance mask to be assigned to a body part.",
    )
    part_group.add_argument(
        "--partial-parts",
        action="store_true",
        help="Allow unseeded remainder pixels inside the parent instance mask to stay unassigned.",
    )
    seg_parts_p.add_argument(
        "--no-share-seeds",
        action="store_true",
        help="Do not share seeded body-part prototypes onto unseeded animal instances.",
    )
    seg_parts_p.add_argument(
        "--max-area-growth",
        type=float,
        default=2.5,
        help="Maximum part area-ratio growth factor relative to seed area ratio.",
    )
    seg_parts_p.add_argument(
        "--min-part-area",
        type=int,
        default=6,
        help="Minimum connected-component area in pixels for emitted body parts.",
    )
    seg_parts_p.add_argument(
        "--no-boundary-refine",
        action="store_true",
        help="Disable crop-local edge-preserving guided probability filtering.",
    )
    seg_parts_p.add_argument(
        "--crf-refine",
        action="store_true",
        help="Enable additional crop-local CRF / bilateral mask boundary refinement.",
    )
    seg_parts_p.add_argument(
        "--polygon-epsilon",
        type=float,
        default=1.0,
        help="Douglas-Peucker contour approximation epsilon in pixels.",
    )
    seg_parts_p.add_argument(
        "--no-split-merged-parents",
        action="store_true",
        help="Disable automatic geodesic splitting of bloated merged parent masks during close-contact occlusion.",
    )
    seg_parts_p.add_argument(
        "--no-recover-merged-instances",
        action="store_true",
        help="When splitting a merged parent mask, do not emit the missing partner's parent polygon and parts.",
    )
    seg_parts_p.add_argument(
        "--qualify-labels",
        action="store_true",
        help="Write polygon shape labels as '<instance>:<part>' in addition to part metadata.",
    )
    seg_parts_p.add_argument(
        "--persist-json",
        action="store_true",
        help="Write full per-frame LabelMe JSON files alongside AnnotationStore records.",
    )
    seg_parts_p.add_argument(
        "--device",
        default=None,
        help="Optional Torch compute device override (e.g. 'cpu', 'cuda', 'mps').",
    )
    seg_parts_p.add_argument(
        "--start-frame",
        type=int,
        default=0,
        help="First frame index to segment (default: 0).",
    )
    seg_parts_p.add_argument(
        "--end-frame",
        type=int,
        default=None,
        help="Inclusive last frame index to segment (defaults to end of tracked video).",
    )
    seg_parts_p.add_argument(
        "--step",
        type=int,
        default=1,
        help="Frame stride (default: 1).",
    )
    seg_parts_p.add_argument(
        "--export-csv",
        action="store_true",
        help="Regenerate <result_dir>_tracking.csv including hierarchical '<instance>:<part>' rows.",
    )
    seg_parts_p.add_argument(
        "--export-ethogram",
        action="store_true",
        help="Export part-level kinematics, directed social-contact bouts, and active-learning uncertainty reports.",
    )
    seg_parts_p.add_argument(
        "--camera-view",
        choices=("auto", "top", "oblique", "side", "bottom"),
        default="auto",
        help="Camera viewpoint hint for ethogram posture disentanglement (default: auto).",
    )
    seg_parts_p.add_argument(
        "--out",
        default=None,
        help="Optional JSON path to write the segmentation summary report.",
    )
    seg_parts_p.add_argument(
        "--json",
        action="store_true",
        help="Print structured JSON summary report to stdout.",
    )
    seg_parts_p.set_defaults(_handler=_cmd_segment_body_parts)

    collect_p = sub.add_parser(
        "collect-labels",
        help="Index LabelMe PNG/JSON pairs by absolute path (no copying).",
    )
    collect_p.add_argument(
        "--source",
        action="append",
        required=True,
        help="Source folder (repeatable) containing per-frame JSON/PNG pairs.",
    )
    collect_p.add_argument(
        "--dataset-root",
        required=True,
        help="Directory that will store the index file.",
    )
    collect_p.add_argument("--recursive", action="store_true", default=True)
    collect_p.add_argument(
        "--no-recursive",
        dest="recursive",
        action="store_false",
        help="Only scan the top-level of each source directory.",
    )
    collect_p.add_argument(
        "--include-empty",
        action="store_true",
        help="Collect JSON files even when they contain no shapes.",
    )
    collect_p.add_argument(
        "--index-file",
        default="logs/label_index/annolid_dataset.jsonl",
        help="Index file path relative to --dataset-root (default: logs/label_index/annolid_dataset.jsonl).",
    )
    collect_p.add_argument(
        "--allow-duplicates",
        action="store_true",
        help="Append even if the image path already exists in the index.",
    )
    collect_p.add_argument(
        "--write-spec",
        action="store_true",
        help="Generate a LabelMe spec.yaml with train/val/test splits.",
    )
    collect_p.add_argument(
        "--spec-path",
        default=None,
        help="Output spec.yaml path (default: <dataset-root>/labelme_spec.yaml).",
    )
    collect_p.add_argument("--val-size", type=float, default=0.1)
    collect_p.add_argument("--test-size", type=float, default=0.0)
    collect_p.add_argument("--seed", type=int, default=0)
    collect_p.add_argument(
        "--group-by",
        choices=("parent", "grandparent", "stem_prefix", "regex", "none"),
        default="parent",
        help="How to group images before splitting (default: parent).",
    )
    collect_p.add_argument("--group-regex", default=None)
    collect_p.add_argument(
        "--split-dir",
        default="logs/label_index",
        help="Directory (relative to dataset root) to store split JSONL files.",
    )
    collect_p.add_argument(
        "--keypoint-names",
        default=None,
        help="Comma-separated keypoint names (overrides inference).",
    )
    collect_p.add_argument(
        "--kpt-dims",
        type=int,
        default=3,
        choices=(2, 3),
        help="Keypoint dims for LabelMe spec (default: 3).",
    )
    collect_p.add_argument(
        "--infer-flip-idx",
        action="store_true",
        help="Infer flip_idx from keypoint names.",
    )
    collect_p.add_argument(
        "--max-keypoint-files",
        type=int,
        default=500,
        help="Max JSON files to scan when inferring keypoint names.",
    )
    collect_p.add_argument(
        "--min-keypoint-count",
        type=int,
        default=1,
        help="Minimum occurrences for inferred keypoint names.",
    )
    collect_p.set_defaults(_handler=_cmd_collect_labels)

    yolo_p = sub.add_parser(
        "index-to-yolo",
        help="Convert an Annolid dataset index JSONL into a YOLO dataset.",
    )
    yolo_p.add_argument(
        "--index-file",
        required=True,
        help="Path to annolid_dataset.jsonl (JSONL index).",
    )
    yolo_p.add_argument(
        "--output-dir", required=True, help="Directory to write the YOLO dataset."
    )
    yolo_p.add_argument(
        "--dataset-name", default="YOLO_dataset", help="Output dataset folder name."
    )
    yolo_p.add_argument("--val-size", type=float, default=0.1)
    yolo_p.add_argument("--test-size", type=float, default=0.1)
    yolo_p.add_argument(
        "--link-mode", choices=("hardlink", "copy", "symlink"), default="hardlink"
    )
    yolo_p.add_argument(
        "--task",
        choices=("auto", "segmentation", "pose"),
        default="auto",
        help="How to handle mixed polygon+point annotations (default: auto).",
    )
    yolo_p.add_argument(
        "--include-empty",
        action="store_true",
        help="Include JSON records with no shapes.",
    )
    yolo_p.add_argument(
        "--keep-staging", action="store_true", help="Keep temporary staging files."
    )
    yolo_p.set_defaults(_handler=_cmd_index_to_yolo)

    imp_p = sub.add_parser(
        "import-deeplabcut-training-data",
        help="Convert DeepLabCut labeled-data (CollectedData_*.csv) into LabelMe JSON next to images, then (optionally) index it.",
    )
    imp_p.add_argument(
        "--source-dir",
        required=True,
        help="Dataset root, e.g. /Users/.../mouse_training_data",
    )
    imp_p.add_argument(
        "--labeled-data",
        default="labeled-data",
        help="labeled-data root relative to --source-dir (default: labeled-data)",
    )
    imp_p.add_argument(
        "--instance-label",
        default="mouse",
        help="Instance label prefix to use for point shapes (default: mouse)",
    )
    imp_p.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing per-image LabelMe JSON files.",
    )
    imp_p.add_argument(
        "--no-recursive",
        action="store_true",
        help="Do not search for CollectedData_*.csv recursively under --labeled-data.",
    )
    imp_p.add_argument(
        "--write-pose-schema",
        action="store_true",
        help="Write pose_schema.json derived from DeepLabCut bodyparts.",
    )
    imp_p.add_argument(
        "--pose-schema-out",
        default="labeled-data/pose_schema.json",
        help="Path (relative to --source-dir) for the pose schema (default: labeled-data/pose_schema.json).",
    )
    imp_p.add_argument(
        "--pose-schema-preset",
        default="mouse",
        help="Edge preset to use when building the schema (default: mouse).",
    )
    imp_p.add_argument(
        "--instance-separator",
        default="_",
        help="Separator used for instance prefixes (default: _).",
    )
    imp_p.add_argument(
        "--index-file",
        default="logs/label_index/annolid_dataset.jsonl",
        help="Index file path relative to --source-dir (default: logs/label_index/annolid_dataset.jsonl).",
    )
    imp_p.add_argument(
        "--no-index",
        dest="write_index",
        action="store_false",
        help="Skip writing annolid_dataset.jsonl",
    )
    imp_p.set_defaults(write_index=True, _handler=_cmd_import_deeplabcut_training_data)

    import_pose_p = sub.add_parser(
        "import-pose",
        help="Import DeepLabCut (CSV/H5/training), Lightning Pose, SLEAP (.slp/.pkg.slp/.analysis.h5), or COCO pose files into LabelMe JSON + PoseSchema.",
    )
    import_pose_p.add_argument(
        "source",
        help="Input pose file or folder (.csv, .h5, .slp, .pkg.slp, .json, video, or DLC project folder).",
    )
    import_pose_p.add_argument(
        "--output-dir",
        "-o",
        default=None,
        help="Output directory for LabelMe JSON files (defaults to <source_stem>_labelme/).",
    )
    import_pose_p.add_argument(
        "--format",
        default="auto",
        help="Source format: auto, deeplabcut-csv, deeplabcut-h5, deeplabcut-training, lightning-pose, sleap, sleap-slp, sleap-h5, coco (default: auto).",
    )
    import_pose_p.add_argument(
        "--video",
        default=None,
        help="Optional video file path for frame dimensions or frame extraction.",
    )
    import_pose_p.add_argument(
        "--config",
        default=None,
        help="Optional DeepLabCut config.yaml for skeleton edges and bodypart ordering.",
    )
    import_pose_p.add_argument(
        "--pose-schema",
        default=None,
        help="Optional existing PoseSchema JSON/YAML for canonical keypoint name mapping.",
    )
    import_pose_p.add_argument(
        "--pose-schema-preset",
        default="auto",
        help="Skeleton edge inference preset when source has no explicit edges (auto, mouse, none; default: auto).",
    )
    import_pose_p.add_argument(
        "--no-pose-schema",
        action="store_true",
        help="Skip writing pose_schema.json.",
    )
    import_pose_p.add_argument(
        "--no-save-frames",
        action="store_true",
        help="Do not extract embedded SLEAP frames or video frames.",
    )
    import_pose_p.add_argument(
        "--video-index",
        type=int,
        default=None,
        help="Optional single video index to extract from a multi-video SLEAP .slp file.",
    )
    import_pose_p.add_argument(
        "--min-likelihood",
        type=float,
        default=None,
        help="Drop keypoints with likelihood/score below this threshold.",
    )
    import_pose_p.add_argument(
        "--multi-animal",
        action="store_true",
        default=None,
        help="Hint multi-animal layout (auto-detected by default).",
    )
    import_pose_p.add_argument(
        "--instance-label",
        default="mouse",
        help="Default instance label for single-animal inputs (default: mouse).",
    )
    import_pose_p.add_argument(
        "--instance-separator",
        default="_",
        help="Instance prefix separator (default: _).",
    )
    import_pose_p.add_argument(
        "--prefix-instance-labels",
        action="store_true",
        help="Prefix point shape labels with '<instance><sep><keypoint>'.",
    )
    import_pose_p.add_argument(
        "--filename-prefix",
        default=None,
        help="Optional filename prefix for output JSON files.",
    )
    import_pose_p.add_argument(
        "--user-only",
        action="store_true",
        help="For SLEAP .slp files, import user-labeled instances only (skip predictions).",
    )
    import_pose_p.add_argument(
        "--keep-both-user-and-predicted",
        action="store_true",
        help="For SLEAP .slp files, keep predicted instances alongside user labels on the same frame.",
    )
    import_pose_p.add_argument(
        "--skip-occluded",
        action="store_true",
        help="Drop occluded keypoints (finite coords with visible=False).",
    )
    import_pose_p.add_argument(
        "--no-overwrite",
        action="store_true",
        help="Do not overwrite existing LabelMe JSON files.",
    )
    import_pose_p.add_argument(
        "--no-recursive",
        action="store_true",
        help="Do not recursively scan subfolders for DLC CollectedData files.",
    )
    import_pose_p.add_argument(
        "--write-index",
        action="store_true",
        help="Also write an Annolid dataset JSONL index after importing.",
    )
    import_pose_p.add_argument(
        "--index-file",
        default=None,
        help="Optional index file path when --write-index is set.",
    )
    import_pose_p.set_defaults(_handler=_cmd_import_pose)

    export_pose_p = sub.add_parser(
        "export-pose",
        help="Export Annolid LabelMe / AnnotationStore / Tracking CSV / COCO pose data to DeepLabCut (CSV or training project), Lightning Pose, SimBA, or SLEAP (.slp / .analysis.h5).",
    )
    export_pose_p.add_argument(
        "source",
        help="Input LabelMe directory, single LabelMe JSON, Annolid *_tracking.csv, COCO JSON, or pose file.",
    )
    export_pose_p.add_argument(
        "--output",
        "-o",
        required=True,
        help="Output file path (.csv, .h5, .slp, .pkg.slp) or project directory (for deeplabcut-project).",
    )
    export_pose_p.add_argument(
        "--format",
        required=True,
        choices=(
            "deeplabcut-csv",
            "deeplabcut-h5",
            "deeplabcut-project",
            "lightning-pose",
            "simba",
            "sleap-h5",
            "sleap-slp",
            "labelme",
            "coco",
            "coco-json",
        ),
        help="Target pose format.",
    )
    export_pose_p.add_argument(
        "--source-format",
        default="auto",
        help="Input format hint (default: auto).",
    )
    export_pose_p.add_argument(
        "--pose-schema",
        default=None,
        help="Optional PoseSchema JSON/YAML for keypoint ordering, edges, and symmetry pairs.",
    )
    export_pose_p.add_argument(
        "--video",
        default=None,
        help="Optional source video path recorded in SLEAP metadata.",
    )
    export_pose_p.add_argument(
        "--scorer",
        default="annolid",
        help="Scorer name for DeepLabCut / Lightning Pose exports (default: annolid).",
    )
    export_pose_p.add_argument(
        "--multi-animal",
        action="store_true",
        default=None,
        help="Force 4-level multi-animal DLC header (auto-detected by default).",
    )
    export_pose_p.add_argument(
        "--video-name",
        default=None,
        help="Subfolder name under labeled-data/ for deeplabcut-project export.",
    )
    export_pose_p.add_argument(
        "--no-copy-images",
        action="store_true",
        help="Do not copy sidecar images into labeled-data/ for deeplabcut-project export.",
    )
    export_pose_p.add_argument(
        "--embed-images",
        action="store_true",
        help="Embed sidecar frame images inside SLEAP .pkg.slp export.",
    )
    export_pose_p.add_argument(
        "--no-config-yaml",
        action="store_true",
        help="Skip writing config.yaml for deeplabcut-project export.",
    )
    export_pose_p.add_argument(
        "--recursive",
        action="store_true",
        help="Recursively scan subdirectories for LabelMe JSON files.",
    )
    export_pose_p.set_defaults(_handler=_cmd_export_pose)

    extract_frames_p = sub.add_parser(
        "extract-frames",
        help="Extract training frames from a video using k-means visual diversity, uniform, random, optical flow, or keyframes.",
    )
    extract_frames_p.add_argument("video", help="Source video path.")
    extract_frames_p.add_argument(
        "--output",
        "-o",
        default=None,
        help="Output directory for extracted frames.",
    )
    extract_frames_p.add_argument(
        "--algo",
        choices=("kmeans", "uniform", "random", "flow", "keyframes"),
        default="kmeans",
        help="Frame selection algorithm (default: kmeans).",
    )
    extract_frames_p.add_argument(
        "--num-frames",
        type=int,
        default=50,
        help="Number of frames to extract (-1 for all frames; default: 50).",
    )
    extract_frames_p.add_argument(
        "--start-seconds",
        type=float,
        default=None,
        help="Optional clip start time in seconds.",
    )
    extract_frames_p.add_argument(
        "--end-seconds",
        type=float,
        default=None,
        help="Optional clip end time in seconds.",
    )
    extract_frames_p.add_argument(
        "--image-format",
        default="png",
        help="Output image format (default: png).",
    )
    extract_frames_p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Deterministic random seed for kmeans/random sampling.",
    )
    extract_frames_p.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON summary.",
    )
    extract_frames_p.set_defaults(_handler=_cmd_extract_frames)

    extract_outliers_p = sub.add_parser(
        "extract-outliers",
        help="Extract high-risk pose outlier frames + editable LabelMe JSONs for active-learning refinement.",
    )
    extract_outliers_p.add_argument(
        "source",
        help="Pose predictions source (LabelMe dir, SLEAP .slp/.h5, DLC CSV/H5, or tracking CSV).",
    )
    extract_outliers_p.add_argument(
        "--video",
        required=True,
        help="Source video file to extract outlier frames from.",
    )
    extract_outliers_p.add_argument(
        "--output",
        "-o",
        required=True,
        help="Output directory for outlier PNGs, LabelMe JSONs, and manifest.",
    )
    extract_outliers_p.add_argument(
        "--source-format",
        default="auto",
        help="Input pose format hint (default: auto).",
    )
    extract_outliers_p.add_argument(
        "--pose-schema",
        default=None,
        help="Optional PoseSchema JSON/YAML for bone-length anomaly checks.",
    )
    extract_outliers_p.add_argument(
        "--num-frames",
        type=int,
        default=20,
        help="Maximum outlier frames to extract (default: 20).",
    )
    extract_outliers_p.add_argument(
        "--min-frame-gap",
        type=int,
        default=10,
        help="Minimum frame spacing between selected outlier frames (default: 10).",
    )
    extract_outliers_p.add_argument(
        "--min-confidence",
        type=float,
        default=0.6,
        help="Keypoint confidence threshold below which points are flagged (default: 0.6).",
    )
    extract_outliers_p.add_argument(
        "--jump-threshold",
        type=float,
        default=None,
        help="Optional explicit per-frame keypoint jump threshold in pixels.",
    )
    extract_outliers_p.add_argument(
        "--jump-mad",
        type=float,
        default=5.0,
        help="Adaptive MAD multiplier when --jump-threshold is omitted (default: 5.0).",
    )
    extract_outliers_p.add_argument(
        "--max-edge-stretch",
        type=float,
        default=1.8,
        help="Bone stretch ratio threshold vs median length (default: 1.8).",
    )
    extract_outliers_p.add_argument(
        "--min-edge-shrink",
        type=float,
        default=0.35,
        help="Bone collapse ratio threshold vs median length (default: 0.35).",
    )
    extract_outliers_p.add_argument(
        "--expected-instances",
        type=int,
        default=None,
        help="Optional expected animal count per frame.",
    )
    extract_outliers_p.add_argument(
        "--no-predictions",
        action="store_true",
        help="Write empty LabelMe templates instead of seeding with predicted keypoints.",
    )
    extract_outliers_p.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON summary.",
    )
    extract_outliers_p.set_defaults(_handler=_cmd_extract_outliers)

    eval_pose_p = sub.add_parser(
        "eval-pose",
        help="Evaluate pose predictions vs ground truth (DLC RMSE/p-cutoff, SLEAP percentiles, PCK, OKS mAP, L/R swaps).",
    )
    eval_pose_p.add_argument(
        "gt_source",
        help="Ground-truth pose source (LabelMe dir, SLEAP, DLC, COCO JSON).",
    )
    eval_pose_p.add_argument(
        "pred_source",
        help="Predicted pose source (LabelMe dir, SLEAP, DLC, COCO JSON, Tracking CSV).",
    )
    eval_pose_p.add_argument(
        "--gt-format",
        default="auto",
        help="Ground-truth format hint (default: auto).",
    )
    eval_pose_p.add_argument(
        "--pred-format",
        default="auto",
        help="Prediction format hint (default: auto).",
    )
    eval_pose_p.add_argument(
        "--pose-schema",
        default=None,
        help="Optional PoseSchema JSON/YAML for canonical keypoints and L/R symmetry pairs.",
    )
    eval_pose_p.add_argument(
        "--pcutoff",
        type=float,
        default=0.6,
        help="Confidence cutoff for DLC-style RMSE@pcutoff (default: 0.6).",
    )
    eval_pose_p.add_argument(
        "--pck-px",
        default="2,4,8,16",
        help="Comma-separated pixel thresholds for PCK (default: 2,4,8,16).",
    )
    eval_pose_p.add_argument(
        "--pck-norm",
        default="0.05,0.1,0.2",
        help="Comma-separated scale-normalized thresholds for PCK (default: 0.05,0.1,0.2).",
    )
    eval_pose_p.add_argument(
        "--oks-sigma",
        type=float,
        default=0.05,
        help="Uniform OKS sigma for arbitrary animal skeletons (default: 0.05).",
    )
    eval_pose_p.add_argument(
        "--top-k-worst",
        type=int,
        default=50,
        help="Number of worst frames to retain in the report (default: 50).",
    )
    eval_pose_p.add_argument(
        "--output-dir",
        "-o",
        default=None,
        help="Optional output directory for JSON, CSV, and HTML+SVG report bundle.",
    )
    eval_pose_p.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON summary.",
    )
    eval_pose_p.set_defaults(_handler=_cmd_eval_pose)

    filter_pose_p = sub.add_parser(
        "filter-pose",
        help="Filter, clean outliers, and interpolate gaps in pose trajectories (median, Savitzky-Golay, One-Euro).",
    )
    filter_pose_p.add_argument(
        "source",
        help="Input pose source (LabelMe dir, SLEAP .slp/.h5, DLC CSV/H5, COCO JSON).",
    )
    filter_pose_p.add_argument(
        "--output",
        "-o",
        default=None,
        help="Output path or directory (defaults to <source>_filtered).",
    )
    filter_pose_p.add_argument(
        "--source-format",
        default="auto",
        help="Input format hint (default: auto).",
    )
    filter_pose_p.add_argument(
        "--format",
        default="auto",
        help="Output format (auto, labelme, deeplabcut-csv, deeplabcut-h5, sleap-h5, sleap-slp, coco).",
    )
    filter_pose_p.add_argument(
        "--pose-schema",
        default=None,
        help="Optional PoseSchema JSON/YAML for anatomical bone-length checks.",
    )
    filter_pose_p.add_argument(
        "--video",
        default=None,
        help="Optional source video path for SLEAP metadata.",
    )
    filter_pose_p.add_argument(
        "--scorer",
        default="annolid",
        help="Scorer name when exporting DLC CSV (default: annolid).",
    )
    filter_pose_p.add_argument(
        "--method",
        choices=("savgol", "median", "one_euro", "moving_average", "none"),
        default="savgol",
        help="Temporal smoothing filter (default: savgol).",
    )
    filter_pose_p.add_argument(
        "--window",
        type=int,
        default=5,
        help="Odd filter window length in frames (default: 5).",
    )
    filter_pose_p.add_argument(
        "--polyorder",
        type=int,
        default=2,
        help="Savitzky-Golay polynomial order (default: 2).",
    )
    filter_pose_p.add_argument(
        "--interpolate",
        choices=("linear", "cubic", "nearest", "none"),
        default="linear",
        help="Gap interpolation strategy (default: linear).",
    )
    filter_pose_p.add_argument(
        "--max-gap",
        type=int,
        default=5,
        help="Maximum gap length in frames to interpolate (default: 5).",
    )
    filter_pose_p.add_argument(
        "--min-confidence",
        type=float,
        default=None,
        help="Mask keypoints below this confidence threshold before interpolation.",
    )
    filter_pose_p.add_argument(
        "--max-jump-px",
        type=float,
        default=None,
        help="Mask single-frame keypoint spikes exceeding this pixel jump.",
    )
    filter_pose_p.add_argument(
        "--jump-mad",
        type=float,
        default=None,
        help="Mask keypoint spikes exceeding median + jump_mad * MAD.",
    )
    filter_pose_p.add_argument(
        "--enforce-bone-bounds",
        action="store_true",
        help="Mask keypoints involved in stretched or collapsed skeleton bones.",
    )
    filter_pose_p.add_argument(
        "--max-edge-stretch",
        type=float,
        default=1.8,
        help="Max bone length ratio vs median (default: 1.8).",
    )
    filter_pose_p.add_argument(
        "--min-edge-shrink",
        type=float,
        default=0.35,
        help="Min bone length ratio vs median (default: 0.35).",
    )
    filter_pose_p.add_argument(
        "--fps",
        type=float,
        default=30.0,
        help="Video FPS for One-Euro filter (default: 30).",
    )
    filter_pose_p.add_argument(
        "--in-place",
        action="store_true",
        help="Overwrite a LabelMe source directory in place.",
    )
    filter_pose_p.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON summary.",
    )
    filter_pose_p.set_defaults(_handler=_cmd_filter_pose)

    swap_pose_p = sub.add_parser(
        "swap-pose",
        help="Swap two animal identities (all pose keypoints and polygons) from start_frame forward.",
    )
    swap_pose_p.add_argument(
        "source_dir", help="LabelMe / AnnotationStore frame directory."
    )
    swap_pose_p.add_argument(
        "--start-frame",
        type=int,
        required=True,
        help="First frame index (inclusive) to swap identities.",
    )
    swap_pose_p.add_argument(
        "--end-frame",
        type=int,
        default=None,
        help="Optional last frame index (inclusive) to swap.",
    )
    swap_pose_p.add_argument(
        "--instance-a",
        required=True,
        help="First instance label or group_id to swap.",
    )
    swap_pose_p.add_argument(
        "--instance-b",
        required=True,
        help="Second instance label or group_id to swap.",
    )
    swap_pose_p.add_argument(
        "--pose-schema",
        default=None,
        help="Optional PoseSchema JSON/YAML for instance separator.",
    )
    swap_pose_p.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview affected frames and shapes without writing changes.",
    )
    swap_pose_p.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON summary.",
    )
    swap_pose_p.set_defaults(_handler=_cmd_swap_pose)

    pose_kin_p = sub.add_parser(
        "pose-kinematics",
        help="Compute calibrated per-keypoint speeds, bone lengths, joint angles, egocentric pose, and dyadic keypoint metrics.",
    )
    pose_kin_p.add_argument(
        "source",
        help="Input pose source (LabelMe dir, SLEAP .slp/.h5, DLC CSV/H5, COCO JSON).",
    )
    pose_kin_p.add_argument(
        "--source-format",
        default="auto",
        help="Input format hint (default: auto).",
    )
    pose_kin_p.add_argument(
        "--pose-schema",
        default=None,
        help="Optional PoseSchema JSON/YAML for skeleton edges and keypoint order.",
    )
    pose_kin_p.add_argument(
        "--fps",
        type=float,
        default=30.0,
        help="Video frames per second (default: 30).",
    )
    pose_kin_p.add_argument(
        "--px-per-cm",
        type=float,
        default=None,
        help="Optional orthographic spatial scale in pixels per centimeter.",
    )
    pose_kin_p.add_argument(
        "--calibration",
        default=None,
        help="Optional planar perspective calibration JSON file.",
    )
    pose_kin_p.add_argument(
        "--min-confidence",
        type=float,
        default=None,
        help="Ignore keypoints below this confidence score.",
    )
    pose_kin_p.add_argument(
        "--close-contact-px",
        type=float,
        default=35.0,
        help="Close-contact distance threshold in pixels for dyadic keypoint pairs (default: 35).",
    )
    pose_kin_p.add_argument(
        "--output-dir",
        "-o",
        default=None,
        help="Optional output directory for JSON and CSV tables (including egocentric_pose.csv).",
    )
    pose_kin_p.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON summary.",
    )
    pose_kin_p.set_defaults(_handler=_cmd_pose_kinematics)

    render_pose_p = sub.add_parser(
        "render-pose-video",
        help="Render an offline labeled skeleton video (with color modes, p-cutoff, motion trails, and comparison overlay).",
    )
    render_pose_p.add_argument("video", help="Input video file.")
    render_pose_p.add_argument(
        "pose_source",
        help="Pose annotations or predictions (LabelMe dir, SLEAP, DLC, COCO, Tracking CSV).",
    )
    render_pose_p.add_argument(
        "--output",
        "-o",
        required=True,
        help="Output labeled video path (e.g., labeled_pose.mp4).",
    )
    render_pose_p.add_argument(
        "--source-format",
        default="auto",
        help="Pose format hint (default: auto).",
    )
    render_pose_p.add_argument(
        "--pose-schema",
        default=None,
        help="Optional PoseSchema JSON/YAML for skeleton bones.",
    )
    render_pose_p.add_argument(
        "--comparison",
        default=None,
        help="Optional secondary pose source (e.g., unfiltered raw predictions) drawn as hollow rings.",
    )
    render_pose_p.add_argument(
        "--color-mode",
        choices=("instance", "bodypart", "confidence"),
        default="instance",
        help="Color scheme for skeletons and keypoints (default: instance).",
    )
    render_pose_p.add_argument(
        "--min-confidence",
        type=float,
        default=None,
        help="Hide keypoints/bones below this confidence score (p-cutoff).",
    )
    render_pose_p.add_argument(
        "--trail-length",
        type=int,
        default=0,
        help="Number of recent frames for fading motion trails (default: 0 = disabled).",
    )
    render_pose_p.add_argument(
        "--draw-labels",
        action="store_true",
        help="Draw keypoint names next to markers.",
    )
    render_pose_p.add_argument(
        "--no-instance-names",
        action="store_true",
        help="Hide animal/instance badges above skeletons.",
    )
    render_pose_p.add_argument(
        "--no-frame-index",
        action="store_true",
        help="Hide frame number and timestamp HUD.",
    )
    render_pose_p.add_argument(
        "--thickness",
        type=int,
        default=2,
        help="Skeleton bone line thickness in pixels (default: 2).",
    )
    render_pose_p.add_argument(
        "--point-radius",
        type=int,
        default=4,
        help="Keypoint marker radius in pixels (default: 4).",
    )
    render_pose_p.add_argument(
        "--max-bone-dist",
        type=float,
        default=None,
        help="Optional max pixel length beyond which bones are skipped.",
    )
    render_pose_p.add_argument(
        "--start-frame",
        type=int,
        default=0,
        help="First frame index to render (default: 0).",
    )
    render_pose_p.add_argument(
        "--end-frame",
        type=int,
        default=None,
        help="Optional last frame index (inclusive) to render.",
    )
    render_pose_p.add_argument(
        "--fps",
        type=float,
        default=None,
        help="Optional output video FPS override.",
    )
    render_pose_p.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON summary.",
    )
    render_pose_p.set_defaults(_handler=_cmd_render_pose_video)

    tb_p = sub.add_parser(
        "dino-kpseg-embeddings",
        help="Write TensorBoard projector embeddings for DinoKPSEG (DINOv3 patch features).",
    )
    tb_p.add_argument("--data", required=True, help="Path to YOLO pose data.yaml")
    tb_p.add_argument("--split", choices=("train", "val"), default="val")
    tb_p.add_argument(
        "--weights",
        default=None,
        help="Optional DinoKPSEG checkpoint (.pt); enables pred overlays and keypoint names.",
    )
    tb_p.add_argument(
        "--model-name", default="facebook/dinov3-vits16-pretrain-lvd1689m"
    )
    tb_p.add_argument("--short-side", type=int, default=768)
    tb_p.add_argument("--layers", type=str, default="-1")
    tb_p.add_argument("--device", default=None)
    tb_p.add_argument("--radius-px", type=float, default=6.0)
    tb_p.add_argument("--mask-type", choices=("disk", "gaussian"), default="gaussian")
    tb_p.add_argument("--heatmap-sigma", type=float, default=None)
    tb_p.add_argument(
        "--instance-mode", choices=("auto", "union", "per_instance"), default="auto"
    )
    tb_p.add_argument("--bbox-scale", type=float, default=1.25)
    tb_p.add_argument("--no-cache", action="store_true")
    tb_p.add_argument("--max-images", type=int, default=64)
    tb_p.add_argument("--max-patches", type=int, default=4000)
    tb_p.add_argument("--per-image-per-keypoint", type=int, default=3)
    tb_p.add_argument("--pos-threshold", type=float, default=0.35)
    tb_p.add_argument("--add-negatives", action="store_true")
    tb_p.add_argument("--neg-threshold", type=float, default=0.02)
    tb_p.add_argument("--negatives-per-image", type=int, default=6)
    tb_p.add_argument("--crop-px", type=int, default=96)
    tb_p.add_argument("--sprite-border-px", type=int, default=3)
    tb_p.add_argument("--seed", type=int, default=0)
    tb_p.add_argument("--output", default=None, help="Run output directory (optional)")
    tb_p.add_argument("--runs-root", default=None, help="Runs root (optional)")
    tb_p.add_argument(
        "--run-name", default=None, help="Optional run name (default: timestamp)"
    )
    tb_p.set_defaults(_handler=_cmd_dino_kpseg_embeddings)

    audit_p = sub.add_parser(
        "dino-kpseg-audit",
        help="Audit a YOLO pose dataset for DinoKPSEG and emit a report.",
    )
    audit_p.add_argument("--data", required=True, help="Path to YOLO pose data.yaml")
    audit_p.add_argument("--split", choices=("train", "val", "both"), default="both")
    audit_p.add_argument("--max-images", type=int, default=None)
    audit_p.add_argument("--seed", type=int, default=0)
    audit_p.add_argument(
        "--instance-mode", choices=("auto", "union", "per_instance"), default="auto"
    )
    audit_p.add_argument("--bbox-scale", type=float, default=1.25)
    audit_p.add_argument("--out", default=None, help="Optional output JSON path")
    audit_p.set_defaults(_handler=_cmd_dino_kpseg_audit)

    split_p = sub.add_parser(
        "dino-kpseg-split",
        help="Create a stratified train/val split for a YOLO pose dataset.",
    )
    split_p.add_argument("--data", required=True, help="Path to YOLO pose data.yaml")
    split_p.add_argument(
        "--output", required=True, help="Output directory for split lists"
    )
    split_p.add_argument("--val-size", type=float, default=0.1)
    split_p.add_argument("--seed", type=int, default=0)
    split_p.add_argument(
        "--group-by",
        choices=("parent", "grandparent", "stem_prefix", "regex"),
        default="parent",
    )
    split_p.add_argument("--group-regex", default=None)
    split_p.add_argument("--include-val", action="store_true")
    split_p.set_defaults(_handler=_cmd_dino_kpseg_split)

    pre_p = sub.add_parser(
        "dino-kpseg-precompute",
        help="Precompute and cache DINOv3 features for a DinoKPSEG dataset.",
    )
    pre_p.add_argument("--data", required=True, help="Path to YOLO pose data.yaml")
    pre_p.add_argument("--model-name", required=True)
    pre_p.add_argument("--short-side", type=int, default=768)
    pre_p.add_argument("--layers", type=str, default="-1")
    pre_p.add_argument("--device", default=None)
    pre_p.add_argument("--split", choices=("train", "val", "both"), default="both")
    pre_p.add_argument(
        "--instance-mode", choices=("auto", "union", "per_instance"), default="auto"
    )
    pre_p.add_argument("--bbox-scale", type=float, default=1.25)
    pre_p.add_argument("--cache-dir", default=None)
    pre_p.add_argument(
        "--cache-dtype", choices=("float16", "float32"), default="float16"
    )
    pre_p.set_defaults(_handler=_cmd_dino_kpseg_precompute)

    agent_p = sub.add_parser(
        "agent",
        help="Run the unified agent pipeline and write per-video results.",
    )
    agent_p.add_argument("--video", required=True, help="Input video path.")
    agent_p.add_argument(
        "--schema",
        default=None,
        help="Optional behavior spec path (project.annolid.json/yaml).",
    )
    agent_p.add_argument(
        "--results-dir",
        default=None,
        help="Output directory for results (default: <video_stem>/).",
    )
    agent_p.add_argument(
        "--ndjson-name",
        default="agent.ndjson",
        help="NDJSON file name under the results dir (default: agent.ndjson).",
    )
    agent_p.add_argument("--max-frames", type=int, default=None)
    agent_p.add_argument("--stride", type=int, default=1)
    agent_p.add_argument("--include-llm-summary", action="store_true")
    agent_p.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable artifact cache reuse for agent runs.",
    )
    agent_p.add_argument(
        "--llm-summary-prompt",
        default="Summarize the behaviors defined in this behavior spec.",
    )
    agent_p.add_argument(
        "--vision-adapter",
        choices=("none", "maskrcnn", "dino_kpseg"),
        default="none",
        help="Optional vision adapter (default: none).",
    )
    agent_p.add_argument(
        "--vision-weights",
        default=None,
        help="Weights path for dino_kpseg vision adapter.",
    )
    agent_p.add_argument(
        "--vision-pretrained",
        action="store_true",
        help="Use pretrained weights for the vision adapter.",
    )
    agent_p.add_argument(
        "--vision-score-threshold",
        type=float,
        default=0.5,
        help="Score threshold for vision detections.",
    )
    agent_p.add_argument(
        "--vision-device",
        default=None,
        help="Override device for vision adapter (e.g., cpu, cuda).",
    )
    agent_p.add_argument(
        "--llm-adapter",
        choices=("none", "llm_chat"),
        default="none",
        help="Optional LLM adapter (default: none).",
    )
    agent_p.add_argument("--llm-profile", default=None)
    agent_p.add_argument("--llm-provider", default=None)
    agent_p.add_argument("--llm-model", default=None)
    agent_p.add_argument("--llm-persist", action="store_true")
    agent_p.add_argument(
        "--progress-interval",
        type=float,
        default=2.0,
        help="Progress print interval in seconds.",
    )
    agent_p.add_argument("--no-progress", action="store_true")
    agent_p.set_defaults(_handler=_cmd_agent)

    agent_behavior_p = sub.add_parser(
        "agent-behavior",
        help="Run typed behavior-agent orchestration and write immutable artifacts.",
    )
    agent_behavior_p.add_argument("--video", required=True, help="Input video path.")
    agent_behavior_p.add_argument(
        "--results-dir",
        default=None,
        help="Root output directory (default: <video_stem>/).",
    )
    agent_behavior_p.add_argument(
        "--artifacts-ndjson",
        default=None,
        help="Optional TrackArtifact NDJSON input for replayable perception.",
    )
    agent_behavior_p.add_argument("--run-id", default=None)
    agent_behavior_p.add_argument("--episode-id", default=None)
    agent_behavior_p.add_argument(
        "--model-policy",
        default="annolid_behavior_agent_v1",
    )
    agent_behavior_p.add_argument(
        "--context-prompt",
        default="",
        help="Optional task prompt context for assay inference.",
    )
    agent_behavior_p.add_argument(
        "--assay",
        default="",
        help="Optional assay hint context for task inference.",
    )
    agent_behavior_p.add_argument(
        "--default-assay",
        default="unknown",
        help="Fallback assay type when keyword inference does not match.",
    )
    agent_behavior_p.add_argument(
        "--bout-frame-gap",
        type=int,
        default=20,
        help="Frame-gap threshold used to group aggression sub-events into bouts.",
    )
    agent_behavior_p.add_argument("--no-memory", action="store_true")
    agent_behavior_p.add_argument("--no-analysis", action="store_true")
    agent_behavior_p.add_argument("--fail-on-validation-error", action="store_true")
    agent_behavior_p.set_defaults(_handler=_cmd_agent_behavior)

    onboard_p = sub.add_parser(
        "agent-onboard",
        help="Initialize an Annolid agent workspace with bootstrap templates.",
    )
    onboard_p.add_argument(
        "--workspace",
        default=None,
        help="Workspace path (default: ~/.annolid/workspace).",
    )
    onboard_p.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing bootstrap files (legacy alias for --update).",
    )
    onboard_p.add_argument(
        "--update",
        action="store_true",
        help="Update existing bootstrap files to current template versions.",
    )
    onboard_p.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview onboarding/update actions without writing files.",
    )
    onboard_p.add_argument(
        "--no-backup",
        action="store_true",
        help="When updating, skip backup copies before overwriting files.",
    )
    onboard_p.add_argument(
        "--backup-dir",
        default=None,
        help="Optional backup directory for overwritten files during --update.",
    )
    onboard_p.add_argument(
        "--prune-bootstrap",
        action="store_true",
        help="Remove stale bootstrap-managed files not present in current templates.",
    )
    onboard_p.set_defaults(_handler=_cmd_agent_onboard)

    status_p = sub.add_parser(
        "agent-status",
        help="Show Annolid agent workspace/cron status.",
    )
    status_p.set_defaults(_handler=_cmd_agent_status)

    capabilities_p = sub.add_parser(
        "agent-capabilities",
        help="Show combined agent tool and skill capability summary.",
    )
    capabilities_p.add_argument(
        "--workspace",
        default=None,
        help="Workspace path used for both skill and tool discovery.",
    )
    capabilities_p.add_argument(
        "--provider",
        default=None,
        help="Provider override for tool policy resolution.",
    )
    capabilities_p.add_argument(
        "--model",
        default=None,
        help="Model override for tool policy resolution.",
    )
    capabilities_p.add_argument(
        "--task-hint",
        default=None,
        help="Optional task text used to score likely skills.",
    )
    capabilities_p.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Maximum suggested skills returned for the task hint.",
    )
    capabilities_p.add_argument(
        "--config",
        default=None,
        help="Optional path to agent config JSON.",
    )
    capabilities_p.set_defaults(_handler=_cmd_agent_capabilities)

    skill_pool_p = sub.add_parser(
        "agent-skill-pool",
        help="Show effective skill pool summary and optional task-based suggestions.",
    )
    skill_pool_p.add_argument(
        "--workspace",
        default=None,
        help="Workspace path for skill discovery (default: ~/.annolid/workspace).",
    )
    skill_pool_p.add_argument(
        "--task-hint",
        default=None,
        help="Optional task text used to score and suggest relevant skills.",
    )
    skill_pool_p.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Maximum suggested skills returned when --task-hint is provided.",
    )
    skill_pool_p.set_defaults(_handler=_cmd_agent_skill_pool)

    tool_pool_p = sub.add_parser(
        "agent-tool-pool",
        help="Show effective registered/allowed/denied agent tools for current policy.",
    )
    tool_pool_p.add_argument(
        "--workspace",
        default=None,
        help="Workspace path for tool initialization (default: ~/.annolid/workspace).",
    )
    tool_pool_p.add_argument(
        "--provider",
        default=None,
        help="Provider override for policy resolution.",
    )
    tool_pool_p.add_argument(
        "--model",
        default=None,
        help="Model override for policy resolution.",
    )
    tool_pool_p.add_argument(
        "--config",
        default=None,
        help="Optional path to agent config JSON.",
    )
    tool_pool_p.set_defaults(_handler=_cmd_agent_tool_pool)

    snapshots_p = sub.add_parser(
        "agent-turn-snapshots",
        help="Show recent lightweight runtime turn snapshots for a session.",
    )
    snapshots_p.add_argument(
        "--workspace",
        default=None,
        help="Workspace path (default: ~/.annolid/workspace).",
    )
    snapshots_p.add_argument(
        "--session-id",
        default="default",
        help="Session identifier used by the agent runtime.",
    )
    snapshots_p.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Maximum number of recent snapshots to return (default: 20).",
    )
    snapshots_p.set_defaults(_handler=_cmd_agent_turn_snapshots)

    meta_status_p = sub.add_parser(
        "agent-meta-learning-status",
        help="Show meta-learning events, failure patterns, and evolved skills.",
    )
    meta_status_p.add_argument(
        "--workspace",
        default=None,
        help="Workspace path (default: ~/.annolid/workspace).",
    )
    meta_status_p.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Maximum number of recent rows/items to include.",
    )
    meta_status_p.add_argument(
        "--brief",
        action="store_true",
        help="Show concise summary output without event and file details.",
    )
    meta_status_p.set_defaults(_handler=_cmd_agent_meta_learning_status)
    meta_history_p = sub.add_parser(
        "agent-meta-learning-history",
        help="Show detailed meta-learning evolution history.",
    )
    meta_history_p.add_argument(
        "--workspace",
        default=None,
        help="Workspace path (default: ~/.annolid/workspace).",
    )
    meta_history_p.add_argument(
        "--limit",
        type=int,
        default=20,
        help="Maximum number of recent evolution events to include.",
    )
    meta_history_p.add_argument(
        "--full",
        action="store_true",
        help="Include larger skill content excerpts in history output.",
    )
    meta_history_p.set_defaults(_handler=_cmd_agent_meta_learning_history)
    meta_maintenance_status_p = sub.add_parser(
        "agent-meta-learning-maintenance-status",
        help="Show idle-window maintenance scheduler status.",
    )
    meta_maintenance_status_p.add_argument(
        "--workspace",
        default=None,
        help="Workspace path (default: ~/.annolid/workspace).",
    )
    meta_maintenance_status_p.set_defaults(
        _handler=_cmd_agent_meta_learning_maintenance_status
    )
    meta_maintenance_next_window_p = sub.add_parser(
        "agent-meta-learning-maintenance-next-window",
        help="Estimate when the next maintenance/evolution window will open.",
    )
    meta_maintenance_next_window_p.add_argument(
        "--workspace",
        default=None,
        help="Workspace path (default: ~/.annolid/workspace).",
    )
    meta_maintenance_next_window_p.set_defaults(
        _handler=_cmd_agent_meta_learning_maintenance_next_window
    )
    meta_maintenance_p = sub.add_parser(
        "agent-meta-learning-maintenance-run",
        help="Run idle-window meta-learning maintenance/evolution jobs.",
    )
    meta_maintenance_p.add_argument(
        "--workspace",
        default=None,
        help="Workspace path (default: ~/.annolid/workspace).",
    )
    meta_maintenance_p.add_argument(
        "--force",
        action="store_true",
        help="Run maintenance regardless of current idle-window state.",
    )
    meta_maintenance_p.add_argument(
        "--max-jobs",
        type=int,
        default=None,
        help="Maximum pending jobs to process in this run.",
    )
    meta_maintenance_p.set_defaults(_handler=_cmd_agent_meta_learning_maintenance_run)
    skills_import_p = sub.add_parser(
        "agent-skills-import",
        help="Import and adapt external skill packs into workspace skills.",
    )
    skills_import_p.add_argument(
        "--workspace",
        default=None,
        help="Workspace path (default: ~/.annolid/workspace).",
    )
    skills_import_p.add_argument(
        "--source-dir",
        required=True,
        help="Directory containing */SKILL.md skills (e.g. MetaClaw memory_data/skills).",
    )
    skills_import_p.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing workspace skills with same name.",
    )
    skills_import_p.set_defaults(_handler=_cmd_agent_skills_import)

    security_check_p = sub.add_parser(
        "agent-security-check",
        help="Check agent key/config security posture (permissions + secret persistence).",
    )
    security_check_p.set_defaults(_handler=_cmd_agent_security_check)

    security_audit_p = sub.add_parser(
        "agent-security-audit",
        help="Audit agent security posture and optionally fix local permission issues.",
    )
    security_audit_p.add_argument("--config", default=None)
    security_audit_p.add_argument(
        "--fix",
        action="store_true",
        help="Apply safe permission fixes for local agent state files.",
    )
    security_audit_p.set_defaults(_handler=_cmd_agent_security_audit)

    secrets_audit_p = sub.add_parser(
        "agent-secrets-audit",
        help="Audit agent config plaintext secrets and secret references.",
    )
    secrets_audit_p.add_argument("--config", default=None)
    secrets_audit_p.set_defaults(_handler=_cmd_agent_secrets_audit)

    secrets_set_p = sub.add_parser(
        "agent-secrets-set",
        help="Attach a secret reference to an agent config path.",
    )
    secrets_set_p.add_argument("--config", default=None)
    secrets_set_p.add_argument("--path", required=True)
    secrets_set_p.add_argument("--env", default=None)
    secrets_set_p.add_argument("--local", default=None)
    secrets_set_p.add_argument(
        "--value",
        default="",
        help="Secret value to store when using --local.",
    )
    secrets_set_p.set_defaults(_handler=_cmd_agent_secrets_set)

    secrets_remove_p = sub.add_parser(
        "agent-secrets-remove",
        help="Remove a secret reference from an agent config path.",
    )
    secrets_remove_p.add_argument("--config", default=None)
    secrets_remove_p.add_argument("--path", required=True)
    secrets_remove_p.add_argument("--delete-local-value", action="store_true")
    secrets_remove_p.set_defaults(_handler=_cmd_agent_secrets_remove)

    secrets_migrate_p = sub.add_parser(
        "agent-secrets-migrate",
        help="Migrate plaintext agent config secrets into the local secret store.",
    )
    secrets_migrate_p.add_argument("--config", default=None)
    secrets_migrate_p.add_argument(
        "--apply",
        action="store_true",
        help="Write local secret refs and scrub plaintext values.",
    )
    secrets_migrate_p.set_defaults(_handler=_cmd_agent_secrets_migrate)

    config_migrate_p = sub.add_parser(
        "agent-config-migrate",
        help=(
            "Migrate legacy Google integration settings "
            "to the current googleAuth/googleDrive structure."
        ),
    )
    config_migrate_p.add_argument("--config", default=None)
    config_migrate_p.add_argument(
        "--apply",
        action="store_true",
        help="Write migrated config to disk.",
    )
    config_migrate_p.set_defaults(_handler=_cmd_agent_config_migrate)

    box_auth_url_p = sub.add_parser(
        "agent-box-auth-url",
        help="Build a Box OAuth authorization URL for first-run consent.",
    )
    box_auth_url_p.add_argument("--config", default=None)
    box_auth_url_p.add_argument("--client-id", default=None)
    box_auth_url_p.add_argument(
        "--authorize-base-url",
        default=None,
        help=(
            "Box org host for auth, for example "
            "https://my_org_xxx.account.box.com or https://ent.box.com."
        ),
    )
    box_auth_url_p.add_argument(
        "--open-browser",
        action="store_true",
        help="Open the generated Box auth URL in your default browser.",
    )
    box_auth_url_p.add_argument(
        "--redirect-uri",
        default=None,
        help="OAuth redirect URI configured in your Box app.",
    )
    box_auth_url_p.add_argument("--state", default=None)
    box_auth_url_p.set_defaults(_handler=_cmd_agent_box_auth_url)

    box_auth_exchange_p = sub.add_parser(
        "agent-box-auth-exchange",
        help="Exchange a Box OAuth authorization code for access and refresh tokens.",
    )
    box_auth_exchange_p.add_argument("--config", default=None)
    box_auth_exchange_p.add_argument("--client-id", default=None)
    box_auth_exchange_p.add_argument("--client-secret", default=None)
    box_auth_exchange_p.add_argument(
        "--authorize-base-url",
        default=None,
        help="Persist the Box org auth host alongside the exchanged tokens.",
    )
    box_auth_exchange_p.add_argument(
        "--redirect-uri",
        default=None,
        help="OAuth redirect URI used when requesting the authorization code.",
    )
    box_auth_exchange_p.add_argument("--code", required=True)
    box_auth_exchange_p.add_argument("--token-url", default=None)
    box_auth_exchange_p.add_argument(
        "--persist",
        action="store_true",
        help="Persist received tokens to tools.box.* in agent config.",
    )
    box_auth_exchange_p.add_argument(
        "--disable-box",
        dest="enable_box",
        action="store_false",
        help="Do not enable tools.box when persisting.",
    )
    box_auth_exchange_p.set_defaults(
        enable_box=True, _handler=_cmd_agent_box_auth_exchange
    )

    box_refresh_p = sub.add_parser(
        "agent-box-token-refresh",
        help="Refresh Box access token using OAuth refresh_token grant.",
    )
    box_refresh_p.add_argument("--config", default=None)
    box_refresh_p.add_argument("--client-id", default=None)
    box_refresh_p.add_argument("--client-secret", default=None)
    box_refresh_p.add_argument("--refresh-token", default=None)
    box_refresh_p.add_argument("--token-url", default=None)
    box_refresh_p.add_argument(
        "--persist",
        action="store_true",
        help="Persist refreshed access/refresh tokens to tools.box.* in agent config.",
    )
    box_refresh_p.set_defaults(_handler=_cmd_agent_box_token_refresh)

    update_p = sub.add_parser(
        "agent-update",
        help="Check for agent updates and optionally apply them.",
    )
    update_p.add_argument(
        "--channel",
        choices=("stable", "beta", "dev"),
        default="stable",
        help="Update channel policy (default: stable).",
    )
    update_p.add_argument(
        "--timeout-s",
        type=float,
        default=4.0,
        help="PyPI metadata request timeout in seconds (default: 4.0).",
    )
    update_p.add_argument(
        "--apply",
        action="store_true",
        help="Prepare an update run (dry-run unless --execute is also set).",
    )
    update_p.add_argument(
        "--execute",
        action="store_true",
        help="Execute update commands (only used with --apply).",
    )
    update_p.add_argument(
        "--skip-doctor",
        action="store_true",
        help="Skip post-update validation commands.",
    )
    update_p.add_argument(
        "--require-signature",
        action="store_true",
        help="Require a valid signed update manifest before staging/apply.",
    )
    update_p.set_defaults(_handler=_cmd_agent_update)

    eval_p = sub.add_parser(
        "agent-eval",
        help="Replay traces, score outcomes, compare baseline vs candidate.",
    )
    eval_p.add_argument("--traces", required=True, help="Eval traces (.json/.jsonl).")
    eval_p.add_argument(
        "--candidate-responses",
        required=True,
        help="Candidate responses (.json/.jsonl).",
    )
    eval_p.add_argument(
        "--baseline-responses",
        default=None,
        help="Optional baseline responses (.json/.jsonl).",
    )
    eval_p.add_argument("--out", required=True, help="Output report JSON path.")
    eval_p.add_argument(
        "--max-regressions",
        type=int,
        default=0,
        help="Fail if regressions exceed this threshold (default: 0).",
    )
    eval_p.set_defaults(_handler=_cmd_agent_eval)

    cron_list_p = sub.add_parser(
        "agent-cron-list", help="List scheduled Annolid agent cron jobs."
    )
    cron_list_p.add_argument(
        "--all", action="store_true", help="Include disabled jobs."
    )
    cron_list_p.set_defaults(_handler=_cmd_agent_cron_list)

    cron_add_p = sub.add_parser(
        "agent-cron-add", help="Add a scheduled Annolid agent cron job."
    )
    cron_add_p.add_argument("--name", required=True, help="Job name.")
    cron_add_p.add_argument("--message", required=True, help="Message payload.")
    cron_add_p.add_argument(
        "--every", type=int, default=None, help="Run every N seconds."
    )
    cron_add_p.add_argument(
        "--cron",
        dest="cron_expr",
        default=None,
        help="Cron expression, e.g. '0 9 * * *'.",
    )
    cron_add_p.add_argument(
        "--tz",
        default=None,
        help="IANA timezone for --cron, e.g. America/New_York.",
    )
    cron_add_p.add_argument(
        "--at",
        default=None,
        help="Run once at ISO datetime, e.g. 2026-02-11T10:00:00.",
    )
    cron_add_p.add_argument(
        "--deliver",
        action="store_true",
        help="Mark response as deliverable to channel recipient.",
    )
    cron_add_p.add_argument("--channel", default=None, help="Channel name.")
    cron_add_p.add_argument("--to", default=None, help="Recipient ID.")
    cron_add_p.set_defaults(_handler=_cmd_agent_cron_add)

    cron_rm_p = sub.add_parser(
        "agent-cron-remove", help="Remove an Annolid agent cron job by ID."
    )
    cron_rm_p.add_argument("job_id", help="Job ID.")
    cron_rm_p.set_defaults(_handler=_cmd_agent_cron_remove)

    cron_enable_p = sub.add_parser(
        "agent-cron-enable",
        help="Enable or disable an Annolid agent cron job.",
    )
    cron_enable_p.add_argument("job_id", help="Job ID.")
    cron_enable_p.add_argument(
        "--disable", action="store_true", help="Disable instead of enable."
    )
    cron_enable_p.set_defaults(_handler=_cmd_agent_cron_enable)

    cron_run_p = sub.add_parser(
        "agent-cron-run", help="Manually run an Annolid agent cron job."
    )
    cron_run_p.add_argument("job_id", help="Job ID.")
    cron_run_p.add_argument(
        "--force", action="store_true", help="Run even if currently disabled."
    )
    cron_run_p.set_defaults(_handler=_cmd_agent_cron_run)

    train_p = sub.add_parser("train", help="Train a model.")
    train_p.description = (
        "Train an Annolid model plugin. Use `list-models` to discover model names, "
        "then use `--help-model` for plugin-specific arguments."
    )
    train_p.epilog = _mode_help_epilog("train")
    train_p.formatter_class = _AnnolidHelpFormatter
    train_p.add_argument("model", help="Model plugin name (see list-models).")
    train_p.add_argument(
        "--help-model", action="store_true", help="Show model-specific help."
    )
    train_p.set_defaults(_handler="train")

    pred_p = sub.add_parser("predict", help="Run inference.")
    pred_p.description = (
        "Run inference with an Annolid model plugin. Use `list-models` to discover "
        "model names, then use `--help-model` for plugin-specific arguments."
    )
    pred_p.epilog = _mode_help_epilog("predict")
    pred_p.formatter_class = _AnnolidHelpFormatter
    pred_p.add_argument("model", help="Model plugin name (see list-models).")
    pred_p.add_argument(
        "--help-model", action="store_true", help="Show model-specific help."
    )
    pred_p.set_defaults(_handler="predict")

    return p


def _dispatch_model_subcommand(
    *,
    base_args: argparse.Namespace,
    argv: list[str],
) -> int:
    from annolid.engine.registry import get_model

    model_name = str(base_args.model)
    plugin = get_model(model_name)

    mode = base_args._handler
    if mode == "train":
        if not plugin.__class__.supports_train():
            raise SystemExit(f"Model {model_name!r} does not support training.")
        p = argparse.ArgumentParser(
            prog=f"annolid-run train {model_name}",
            epilog=_plugin_examples(plugin, mode="train", model_name=model_name),
            formatter_class=_AnnolidHelpFormatter,
        )
        plugin.add_train_args(p)
        p.description = _plugin_help_description(
            mode="train",
            model_name=model_name,
            plugin_description=str(getattr(plugin, "description", "") or ""),
            parser=p,
            help_sections=plugin.get_help_sections("train"),
        )
        if base_args.help_model:
            p.print_help()
            return 0
        from annolid.engine.run_config import expand_argv_with_run_config

        resolved_argv = expand_argv_with_run_config(
            parser=p,
            argv=argv,
            model_name=model_name,
            mode="train",
        )
        args = p.parse_args(list(resolved_argv))
        get_caps = getattr(plugin, "get_required_capabilities", None)
        if callable(get_caps):
            from annolid.infrastructure.capabilities import ensure_capability

            for cap_name in get_caps("train"):
                ensure_capability(
                    cap_name,
                    reason=f"annolid-run train {model_name}",
                )
        return int(plugin.train(args))

    if mode == "predict":
        if not plugin.__class__.supports_predict():
            raise SystemExit(f"Model {model_name!r} does not support inference.")
        p = argparse.ArgumentParser(
            prog=f"annolid-run predict {model_name}",
            epilog=_plugin_examples(plugin, mode="predict", model_name=model_name),
            formatter_class=_AnnolidHelpFormatter,
        )
        plugin.add_predict_args(p)
        p.description = _plugin_help_description(
            mode="predict",
            model_name=model_name,
            plugin_description=str(getattr(plugin, "description", "") or ""),
            parser=p,
            help_sections=plugin.get_help_sections("predict"),
        )
        if base_args.help_model:
            p.print_help()
            return 0
        from annolid.engine.run_config import expand_argv_with_run_config

        resolved_argv = expand_argv_with_run_config(
            parser=p,
            argv=argv,
            model_name=model_name,
            mode="predict",
        )
        args = p.parse_args(list(resolved_argv))
        get_caps = getattr(plugin, "get_required_capabilities", None)
        if callable(get_caps):
            from annolid.infrastructure.capabilities import ensure_capability

            for cap_name in get_caps("predict"):
                ensure_capability(
                    cap_name,
                    reason=f"annolid-run predict {model_name}",
                )
        return int(plugin.predict(args))

    raise SystemExit(f"Unknown mode: {mode}")


def _configure_cli_logging() -> None:
    from annolid.utils.logger import configure_logging

    try:
        configure_logging(enable_file_logging=True)
    except TypeError:
        configure_logging()


def _normalize_help_argv(argv: list[str]) -> list[str]:
    if not argv:
        return argv
    first = str(argv[0] or "").strip().lower()
    if first != "help":
        return argv
    if len(argv) == 1:
        return ["--help"]
    topic = [str(part) for part in argv[1:] if str(part).strip()]
    if not topic:
        return ["--help"]
    if topic[0] in {"annolid-run", "annolid", "cli"}:
        if len(topic) == 1:
            return ["--help"]
        topic = topic[1:]
    if len(topic) >= 2 and topic[0] in {"train", "predict"}:
        return [topic[0], topic[1], "--help-model"]
    return [*topic, "--help"]


def main(argv: Optional[list[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    argv = _normalize_help_argv(argv)
    if not argv or argv == ["--help"]:
        p = _build_root_parser()
        print(_format_root_help(p), end="")
        return 0
    _configure_cli_logging()
    operator_rc = _dispatch_operator_commands(argv)
    if operator_rc is not None:
        return int(operator_rc)
    p = _build_root_parser()
    args, rest = p.parse_known_args(argv)

    handler = getattr(args, "_handler", None)
    if callable(handler):
        return int(handler(args))

    return _dispatch_model_subcommand(base_args=args, argv=rest)


if __name__ == "__main__":
    raise SystemExit(main())
