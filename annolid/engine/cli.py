from __future__ import annotations

import argparse
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

        class _DummyTool(Tool[int, int]):
            name = "dummy"

            def run(self, ctx: ToolContext, payload: int) -> int:
                _ = ctx
                return payload + 1

        registry = ToolRegistry()
        registry.register("dummy", _DummyTool)
        instance = registry.create("dummy")
        ok = isinstance(instance, _DummyTool)
        _record("registry", ok=ok, detail="register/create tool")
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
    argv = list(sys.argv[1:] if argv is None else argv)
    p = _build_root_parser()
    args, rest = p.parse_known_args(argv)

    handler = getattr(args, "_handler", None)
    if callable(handler):
        return int(handler(args))

    return _dispatch_model_subcommand(base_args=args, argv=rest)


if __name__ == "__main__":
    raise SystemExit(main())
