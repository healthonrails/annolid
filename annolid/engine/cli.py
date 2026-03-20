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
    ("Models", ("list-models", "train", "predict")),
    (
        "Agent",
        (
            "agent",
            "agent-onboard",
            "agent-status",
            "agent-security-check",
            "agent-security-audit",
            "agent-secrets-audit",
            "agent-secrets-set",
            "agent-secrets-remove",
            "agent-secrets-migrate",
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
            "dino-kpseg-embeddings",
            "dino-kpseg-audit",
            "dino-kpseg-split",
            "dino-kpseg-precompute",
        ),
    ),
    (
        "Utilities",
        (
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
        "  annolid-run train <model> --help-model\n"
        "  annolid-run predict <model> --help-model\n"
        "  annolid-run agent-status\n"
        "  annolid-run help train\n\n"
        "Common areas:\n"
        "  Models: train, predict, list-models\n"
        "  Agent: agent, agent-status, agent-security-*, agent-cron-*\n"
        "  Data: collect-labels, index-to-yolo, import-deeplabcut-training-data\n"
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
        "  annolid-run train <model> --help-model",
        "  annolid-run predict <model> --help-model",
        "  annolid-run agent-status",
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
            "description": m.description,
        }
        for m in list_models(load_builtins=False)
    ]
    print(json.dumps(rows, indent=2))
    return 0


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


def _cmd_agent_onboard(args: argparse.Namespace) -> int:
    from annolid.services.agent_cron import onboard_agent_workspace

    summary = onboard_agent_workspace(
        workspace=getattr(args, "workspace", None),
        overwrite=bool(args.overwrite),
    )
    print(json.dumps(summary, indent=2))
    return 0


def _cmd_agent_status(_: argparse.Namespace) -> int:
    from annolid.services.agent_cron import get_agent_status

    summary = get_agent_status()
    print(json.dumps(summary, indent=2))
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
    secret_names = {"api_key", "apikey", "access_token", "token", "secret", "password"}
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
        "bridge_token",
        "bridgetoken",
        "password",
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


def _dispatch_operator_commands(argv: list[str]) -> Optional[int]:
    if not argv:
        return None

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
        help="Overwrite existing bootstrap files.",
    )
    onboard_p.set_defaults(_handler=_cmd_agent_onboard)

    status_p = sub.add_parser(
        "agent-status",
        help="Show Annolid agent workspace/cron status.",
    )
    status_p.set_defaults(_handler=_cmd_agent_status)

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
        args = p.parse_args(argv)
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
