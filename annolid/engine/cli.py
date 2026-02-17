from __future__ import annotations

import argparse
import asyncio
import datetime
import json
import sys
from pathlib import Path
from typing import Optional

from annolid.engine.registry import (
    get_load_failures,
    get_model,
    list_models,
    load_builtin_models,
)
from annolid.utils.logger import configure_logging


def _cmd_list_models(_: argparse.Namespace) -> int:
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
    import tempfile

    summary: dict[str, object] = {"status": "ok", "checks": []}

    def _record(name: str, *, ok: bool, detail: str) -> None:
        summary["checks"].append({"name": name, "ok": ok, "detail": detail})
        if not ok:
            summary["status"] = "error"

    try:
        from annolid.core.agent.tools.artifacts import FileArtifactStore, content_hash

        with tempfile.TemporaryDirectory() as tmpdir:
            store = FileArtifactStore(base_dir=Path(tmpdir), run_id="validate")
            meta_path = store.resolve("agent_cache.json", kind="cache")
            payload = {"hello": "world"}
            store.write_meta(meta_path, {"content_hash": content_hash(payload)})
            ok = store.should_reuse_cache(meta_path, content_hash(payload))
            _record("artifacts", ok=bool(ok), detail="cache metadata round-trip")
    except Exception as exc:
        _record("artifacts", ok=False, detail=str(exc))

    try:
        from annolid.core.agent.tools.sampling import (
            FPSampler,
            RandomSampler,
            UniformSampler,
        )

        uniform = UniformSampler(step=2).sample_indices(10)
        fps = FPSampler(target_fps=5).sample_indices(30, fps=30)
        random = RandomSampler(count=2, seed=1).sample_indices(5)
        ok = bool(uniform) and bool(fps) and bool(random)
        _record("sampling", ok=ok, detail="uniform/fps/random sampling")
    except Exception as exc:
        _record("sampling", ok=False, detail=str(exc))

    try:
        from annolid.core.agent.tools.registry import ToolRegistry
        from annolid.core.agent.tools.base import Tool, ToolContext
        from annolid.core.agent.tools.function_registry import FunctionToolRegistry
        from annolid.core.agent.tools.function_builtin import (
            register_nanobot_style_tools,
        )
        from annolid.core.agent.tools.utility import register_builtin_utility_tools

        class _DummyTool(Tool[int, int]):
            name = "dummy"

            def run(self, ctx: ToolContext, payload: int) -> int:
                _ = ctx
                return payload + 1

        registry = ToolRegistry()
        registry.register("dummy", _DummyTool)
        register_builtin_utility_tools(registry)
        fn_registry = FunctionToolRegistry()
        asyncio.run(register_nanobot_style_tools(fn_registry))
        instance = registry.create("dummy")
        ok = (
            isinstance(instance, _DummyTool)
            and registry.has("calculator")
            and fn_registry.has("read_file")
            and fn_registry.has("exec")
        )
        _record(
            "registry",
            ok=ok,
            detail="register/create tool + utility + nanobot-style function tools",
        )
    except Exception as exc:
        _record("registry", ok=False, detail=str(exc))

    try:
        from annolid.core.agent.tools.vector_index import NumpyEmbeddingIndex
        from annolid.core.types import FrameRef

        index = NumpyEmbeddingIndex(
            embeddings=[[0.1, 0.0], [0.0, 1.0]],
            frames=[FrameRef(frame_index=0), FrameRef(frame_index=1)],
        )
        results = index.search([0.2, 0.1], top_k=1)
        ok = bool(results)
        _record("vector_index", ok=ok, detail="numpy cosine search")
    except ImportError as exc:
        _record("vector_index", ok=True, detail=f"skipped: {exc}")
    except Exception as exc:
        _record("vector_index", ok=False, detail=str(exc))

    print(json.dumps(summary, indent=2))
    return 0 if summary.get("status") == "ok" else 1


def _cmd_collect_labels(args: argparse.Namespace) -> int:
    from annolid.datasets.labelme_collection import (
        DEFAULT_LABEL_INDEX_NAME,
        DEFAULT_LABEL_INDEX_DIRNAME,
        generate_labelme_spec_and_splits,
        index_labelme_dataset,
        normalize_labelme_sources,
    )

    dataset_root = Path(args.dataset_root).expanduser().resolve()
    index_file_default = str(
        Path(DEFAULT_LABEL_INDEX_DIRNAME) / DEFAULT_LABEL_INDEX_NAME
    )
    index_file = Path(getattr(args, "index_file", index_file_default))
    if not index_file.is_absolute():
        index_file = dataset_root / index_file

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
    from annolid.datasets.builders.label_index_yolo import build_yolo_from_label_index

    summary = build_yolo_from_label_index(
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
    from annolid.datasets.importers.deeplabcut_training_data import (
        DeepLabCutTrainingImportConfig,
        import_deeplabcut_training_data,
    )
    from annolid.datasets.labelme_collection import index_labelme_dataset

    source_dir = Path(args.source_dir).expanduser().resolve()
    labeled_data = Path(args.labeled_data)
    labeled_data = (
        labeled_data if labeled_data.is_absolute() else (source_dir / labeled_data)
    )

    summary = import_deeplabcut_training_data(
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

    from annolid.core.agent.runner import AgentRunConfig
    from annolid.core.agent.service import run_agent_to_results

    vision_model = None
    if str(args.vision_adapter or "").strip().lower() == "maskrcnn":
        from annolid.core.models.adapters.maskrcnn_torchvision import (
            TorchvisionMaskRCNNAdapter,
        )

        vision_model = TorchvisionMaskRCNNAdapter(
            pretrained=bool(args.vision_pretrained),
            score_threshold=float(args.vision_score_threshold),
            device=str(args.vision_device) if args.vision_device else None,
        )
    elif str(args.vision_adapter or "").strip().lower() == "dino_kpseg":
        from annolid.core.models.adapters.dino_kpseg_adapter import DinoKPSEGAdapter

        if not args.vision_weights:
            raise ValueError("DinoKPSEG adapter requires --vision-weights.")
        vision_model = DinoKPSEGAdapter(
            weight_path=str(args.vision_weights),
            device=str(args.vision_device) if args.vision_device else None,
            score_threshold=float(args.vision_score_threshold),
        )

    llm_model = None
    if str(args.llm_adapter or "").strip().lower() == "llm_chat":
        from annolid.core.models.adapters.llm_chat import LLMChatAdapter

        llm_model = LLMChatAdapter(
            profile=str(args.llm_profile) if args.llm_profile else None,
            provider=str(args.llm_provider) if args.llm_provider else None,
            model=str(args.llm_model) if args.llm_model else None,
            persist=bool(args.llm_persist),
        )

    config = AgentRunConfig(
        max_frames=args.max_frames,
        stride=int(args.stride),
        include_llm_summary=bool(args.include_llm_summary),
        llm_summary_prompt=str(args.llm_summary_prompt),
    )

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

    result = run_agent_to_results(
        video_path=args.video,
        behavior_spec_path=args.schema,
        results_dir=args.results_dir,
        out_ndjson_name=str(args.ndjson_name),
        vision_model=vision_model,
        llm_model=llm_model,
        config=config,
        progress_callback=_progress,
        reuse_cache=not bool(args.no_cache),
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


def _default_agent_cron_store_path() -> Path:
    from annolid.core.agent.utils import get_agent_data_path

    return get_agent_data_path() / "cron" / "jobs.json"


def _cmd_agent_onboard(args: argparse.Namespace) -> int:
    from annolid.core.agent import bootstrap_workspace
    from annolid.core.agent.utils import get_agent_workspace_path

    workspace = get_agent_workspace_path(args.workspace)
    outcomes = bootstrap_workspace(workspace, overwrite=bool(args.overwrite))
    summary = {
        "workspace": str(workspace),
        "overwrite": bool(args.overwrite),
        "files": outcomes,
    }
    print(json.dumps(summary, indent=2))
    return 0


def _cmd_agent_status(_: argparse.Namespace) -> int:
    from annolid.core.agent.cron import CronService
    from annolid.core.agent.utils import get_agent_data_path, get_agent_workspace_path

    data_dir = get_agent_data_path()
    workspace = get_agent_workspace_path()
    store_path = _default_agent_cron_store_path()
    cron = CronService(store_path=store_path)
    cron_status = cron.status()
    summary = {
        "data_dir": str(data_dir),
        "workspace": str(workspace),
        "workspace_templates": {
            "AGENTS.md": (workspace / "AGENTS.md").exists(),
            "SOUL.md": (workspace / "SOUL.md").exists(),
            "USER.md": (workspace / "USER.md").exists(),
            "TOOLS.md": (workspace / "TOOLS.md").exists(),
            "HEARTBEAT.md": (workspace / "HEARTBEAT.md").exists(),
            "memory/MEMORY.md": (workspace / "memory" / "MEMORY.md").exists(),
            "memory/HISTORY.md": (workspace / "memory" / "HISTORY.md").exists(),
        },
        "cron_store_path": str(store_path),
        "cron": cron_status,
    }
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


def _cmd_agent_security_check(_: argparse.Namespace) -> int:
    from annolid.core.agent.utils import get_agent_data_path
    from annolid.utils.llm_settings import (
        has_provider_api_key,
        settings_path,
    )

    data_dir = get_agent_data_path()
    settings_file = settings_path()
    settings_dir = settings_file.parent

    persisted_payload: dict = {}
    parse_error: Optional[str] = None
    if settings_file.exists():
        try:
            persisted_payload = json.loads(settings_file.read_text(encoding="utf-8"))
            if not isinstance(persisted_payload, dict):
                persisted_payload = {}
                parse_error = "llm_settings.json content is not a JSON object."
        except Exception as exc:
            parse_error = str(exc)

    persisted_secret_keys = _find_persisted_secret_keys(persisted_payload)
    # Use persisted payload for inspection so this command does not mutate
    # permissions via load_llm_settings() side effects before reporting.
    settings = persisted_payload if isinstance(persisted_payload, dict) else {}

    checks = {
        "settings_dir_exists": settings_dir.exists(),
        "settings_file_exists": settings_file.exists(),
        "settings_dir_private": _is_private_dir_mode(settings_dir),
        "settings_file_private": _is_private_file_mode(settings_file),
        "persisted_secrets_found": bool(persisted_secret_keys),
        "settings_json_parse_ok": parse_error is None,
    }
    status = "ok"
    if not all(
        [
            checks["settings_dir_exists"],
            checks["settings_file_exists"],
            checks["settings_dir_private"],
            checks["settings_file_private"],
            checks["settings_json_parse_ok"],
        ]
    ):
        status = "warning"
    if checks["persisted_secrets_found"]:
        status = "warning"

    summary = {
        "status": status,
        "data_dir": str(data_dir),
        "llm_settings_path": str(settings_file),
        "llm_settings_dir_mode": _mode_octal(settings_dir),
        "llm_settings_file_mode": _mode_octal(settings_file),
        "checks": checks,
        "persisted_secret_keys": persisted_secret_keys,
        "provider_key_presence": {
            "openai": bool(has_provider_api_key(settings, "openai")),
            "gemini": bool(has_provider_api_key(settings, "gemini")),
        },
    }
    if parse_error is not None:
        summary["settings_json_error"] = parse_error

    print(json.dumps(summary, indent=2))
    return 0 if status == "ok" else 1


def _cmd_agent_cron_list(args: argparse.Namespace) -> int:
    from annolid.core.agent.cron import CronService

    service = CronService(store_path=_default_agent_cron_store_path())
    jobs = service.list_jobs(include_disabled=bool(args.all))
    rows = []
    for j in jobs:
        rows.append(
            {
                "id": j.id,
                "name": j.name,
                "enabled": j.enabled,
                "schedule": {
                    "kind": j.schedule.kind,
                    "at_ms": j.schedule.at_ms,
                    "every_ms": j.schedule.every_ms,
                    "expr": j.schedule.expr,
                    "tz": j.schedule.tz,
                },
                "payload": {
                    "message": j.payload.message,
                    "deliver": j.payload.deliver,
                    "channel": j.payload.channel,
                    "to": j.payload.to,
                },
                "state": {
                    "next_run_at_ms": j.state.next_run_at_ms,
                    "last_run_at_ms": j.state.last_run_at_ms,
                    "last_status": j.state.last_status,
                    "last_error": j.state.last_error,
                },
            }
        )
    print(json.dumps(rows, indent=2))
    return 0


def _cmd_agent_cron_add(args: argparse.Namespace) -> int:
    from annolid.core.agent.cron import CronPayload, CronSchedule, CronService

    if args.every is None and args.cron_expr is None and args.at is None:
        raise SystemExit("Specify one of --every, --cron, or --at.")

    if args.at is not None:
        try:
            dt = datetime.datetime.fromisoformat(str(args.at))
        except ValueError as exc:
            raise SystemExit(f"Invalid --at value: {args.at}") from exc
        schedule = CronSchedule(kind="at", at_ms=int(dt.timestamp() * 1000))
        delete_after_run = True
    elif args.every is not None:
        every = int(args.every)
        if every <= 0:
            raise SystemExit("--every must be > 0")
        schedule = CronSchedule(kind="every", every_ms=every * 1000)
        delete_after_run = False
    else:
        schedule = CronSchedule(kind="cron", expr=str(args.cron_expr))
        delete_after_run = False

    payload = CronPayload(
        kind="agent_turn",
        message=str(args.message),
        deliver=bool(args.deliver),
        channel=(str(args.channel) if args.channel else None),
        to=(str(args.to) if args.to else None),
    )
    service = CronService(store_path=_default_agent_cron_store_path())
    job = service.add_job(
        name=str(args.name),
        schedule=schedule,
        payload=payload,
        delete_after_run=delete_after_run,
    )
    print(
        json.dumps(
            {
                "id": job.id,
                "name": job.name,
                "enabled": job.enabled,
                "next_run_at_ms": job.state.next_run_at_ms,
            },
            indent=2,
        )
    )
    return 0


def _cmd_agent_cron_remove(args: argparse.Namespace) -> int:
    from annolid.core.agent.cron import CronService

    service = CronService(store_path=_default_agent_cron_store_path())
    ok = service.remove_job(str(args.job_id))
    print(json.dumps({"removed": bool(ok), "job_id": str(args.job_id)}, indent=2))
    return 0 if ok else 1


def _cmd_agent_cron_enable(args: argparse.Namespace) -> int:
    from annolid.core.agent.cron import CronService

    service = CronService(store_path=_default_agent_cron_store_path())
    job = service.enable_job(str(args.job_id), enabled=not bool(args.disable))
    if job is None:
        print(json.dumps({"updated": False, "job_id": str(args.job_id)}, indent=2))
        return 1
    print(
        json.dumps(
            {"updated": True, "job_id": job.id, "enabled": bool(job.enabled)}, indent=2
        )
    )
    return 0


def _cmd_agent_cron_run(args: argparse.Namespace) -> int:
    from annolid.core.agent.cron import CronService

    service = CronService(store_path=_default_agent_cron_store_path())

    async def _run() -> bool:
        return await service.run_job(str(args.job_id), force=bool(args.force))

    ok = bool(asyncio.run(_run()))
    print(json.dumps({"ran": ok, "job_id": str(args.job_id)}, indent=2))
    return 0 if ok else 1


def _build_root_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="annolid-run",
        description="Unified training/inference CLI (plugin-based).",
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
        default="annolid_logs/annolid_dataset.jsonl",
        help="Index file path relative to --dataset-root (default: annolid_logs/annolid_dataset.jsonl).",
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
        default="annolid_logs",
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
        default="annolid_logs/annolid_dataset.jsonl",
        help="Index file path relative to --source-dir (default: annolid_logs/annolid_dataset.jsonl).",
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

    security_check_p = sub.add_parser(
        "agent-security-check",
        help="Check agent key/config security posture (permissions + secret persistence).",
    )
    security_check_p.set_defaults(_handler=_cmd_agent_security_check)

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
    train_p.add_argument("model", help="Model plugin name (see list-models).")
    train_p.add_argument(
        "--help-model", action="store_true", help="Show model-specific help."
    )
    train_p.set_defaults(_handler="train")

    pred_p = sub.add_parser("predict", help="Run inference.")
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
    model_name = str(base_args.model)
    plugin = get_model(model_name)

    mode = base_args._handler
    if mode == "train":
        if not plugin.__class__.supports_train():
            raise SystemExit(f"Model {model_name!r} does not support training.")
        p = argparse.ArgumentParser(prog=f"annolid-run train {model_name}")
        plugin.add_train_args(p)
        if base_args.help_model:
            p.print_help()
            return 0
        args = p.parse_args(argv)
        return int(plugin.train(args))

    if mode == "predict":
        if not plugin.__class__.supports_predict():
            raise SystemExit(f"Model {model_name!r} does not support inference.")
        p = argparse.ArgumentParser(prog=f"annolid-run predict {model_name}")
        plugin.add_predict_args(p)
        if base_args.help_model:
            p.print_help()
            return 0
        args = p.parse_args(argv)
        return int(plugin.predict(args))

    raise SystemExit(f"Unknown mode: {mode}")


def main(argv: Optional[list[str]] = None) -> int:
    configure_logging()
    argv = list(sys.argv[1:] if argv is None else argv)
    p = _build_root_parser()
    args, rest = p.parse_known_args(argv)

    handler = getattr(args, "_handler", None)
    if callable(handler):
        return int(handler(args))

    return _dispatch_model_subcommand(base_args=args, argv=rest)


if __name__ == "__main__":
    raise SystemExit(main())
