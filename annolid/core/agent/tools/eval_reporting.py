from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence

import numpy as np
from sklearn import metrics
import yaml

from annolid.segmentation.dino_kpseg.eval import build_paper_report

from .common import (
    _normalize_allowed_read_roots,
    _resolve_read_path,
    _resolve_write_path,
)
from .function_base import FunctionTool

try:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
except Exception:  # pragma: no cover - optional dependency guard
    COCO = None
    COCOeval = None


def _format_float(value: object, *, precision: int = 4) -> str:
    try:
        if value is None:
            return "NA"
        return f"{float(value):.{int(precision)}f}"
    except Exception:
        return "NA"


def _paper_markdown_table(*, rows: Sequence[tuple[str, str, str]]) -> str:
    lines = [
        "| Metric | Value | 95% CI |",
        "|---|---:|---:|",
    ]
    for metric, value, ci in rows:
        lines.append(f"| {metric} | {value} | {ci} |")
    return "\n".join(lines)


def _paper_latex_table(*, rows: Sequence[tuple[str, str, str]]) -> str:
    lines = [
        "\\begin{tabular}{lcc}",
        "\\hline",
        "Metric & Value & 95\\% CI \\\\",
        "\\hline",
    ]
    for metric, value, ci in rows:
        metric_text = str(metric).replace("_", "\\_")
        value_text = str(value).replace("%", "\\%")
        ci_text = str(ci).replace("%", "\\%")
        lines.append(f"{metric_text} & {value_text} & {ci_text} \\\\")
    lines.extend(["\\hline", "\\end{tabular}"])
    return "\n".join(lines)


def _paper_csv(*, row: Dict[str, object]) -> str:
    fieldnames = list(row.keys())
    stream = StringIO()
    writer = csv.DictWriter(stream, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerow({k: ("" if row.get(k) is None else row.get(k)) for k in fieldnames})
    return stream.getvalue()


def _quality_status_from_checks(
    checks: Sequence[dict[str, object]],
) -> str:
    for item in checks:
        if str(item.get("status") or "").strip().lower() == "fail":
            return "fail"
    for item in checks:
        if str(item.get("status") or "").strip().lower() == "warn":
            return "warn"
    return "pass"


def _quality_checks_markdown(
    checks: Sequence[dict[str, object]],
) -> str:
    if not checks:
        return "No quality checks were recorded."
    lines = [
        "| Check | Status | Details |",
        "|---|---|---|",
    ]
    for item in checks:
        check_id = str(item.get("id") or "check")
        status = str(item.get("status") or "pass").strip().upper()
        message = str(item.get("message") or "")
        lines.append(f"| `{check_id}` | {status} | {message} |")
    return "\n".join(lines)


def _report_quality_check(
    *,
    check_id: str,
    status: str,
    message: str,
) -> dict[str, object]:
    normalized = str(status or "pass").strip().lower()
    if normalized not in {"pass", "warn", "fail"}:
        normalized = "warn"
    return {
        "id": str(check_id or "check"),
        "status": normalized,
        "message": str(message or ""),
    }


def _resolve_citation_report_path(
    *,
    source_path: Path,
    citation_report_path: Optional[Path],
) -> Optional[Path]:
    if citation_report_path is not None:
        return citation_report_path
    root = source_path if source_path.is_dir() else source_path.parent
    report_dir = root / ".annolid_cache" / "citation_verification"
    if not report_dir.exists():
        return None
    batch_candidates = sorted(
        report_dir.glob("*_batch.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    if batch_candidates:
        return batch_candidates[0]
    all_candidates = sorted(
        report_dir.glob("*.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return all_candidates[0] if all_candidates else None


def _load_citation_report_summary(path: Path) -> dict[str, object]:
    payload = _read_json(path)
    summary = payload.get("summary") if isinstance(payload.get("summary"), dict) else {}
    counts = summary.get("counts") if isinstance(summary.get("counts"), dict) else {}
    total = _safe_int(summary.get("total"))
    if total is None:
        total = sum(int(_safe_int(v) or 0) for v in counts.values())
    return {
        "path": str(path),
        "total": max(0, int(total or 0)),
        "counts": {
            "verified": int(_safe_int(counts.get("verified")) or 0),
            "suspicious": int(_safe_int(counts.get("suspicious")) or 0),
            "hallucinated": int(_safe_int(counts.get("hallucinated")) or 0),
            "skipped": int(_safe_int(counts.get("skipped")) or 0),
        },
        "integrity_score": _safe_float(summary.get("integrity_score")),
    }


def _apply_citation_quality_gate(
    *,
    report: Dict[str, object],
    source_path: Path,
    citation_gate: bool,
    citation_gate_required: bool,
    citation_report_path: Optional[Path],
    citation_hallucinated_max: int,
    citation_suspicious_rate_warn: float,
    citation_integrity_min_warn: float,
) -> Dict[str, object]:
    def _citation_gate_markdown_block(
        *,
        status: str,
        details: str,
        thresholds: Optional[dict[str, object]] = None,
    ) -> str:
        lines = [
            "## Citation Quality Gate",
            "",
            f"- Status: `{str(status or '').upper()}`",
            f"- Details: {details}",
        ]
        if isinstance(thresholds, dict):
            lines.extend(
                [
                    "- Thresholds:",
                    (
                        f"  - hallucinated_max={int(thresholds.get('hallucinated_max') or 0)}"
                    ),
                    (
                        "  - suspicious_rate_warn="
                        f"{float(thresholds.get('suspicious_rate_warn') or 0.0):.3f}"
                    ),
                    (
                        "  - integrity_min_warn="
                        f"{float(thresholds.get('integrity_min_warn') or 0.0):.3f}"
                    ),
                ]
            )
        return "\n".join(lines).strip()

    def _append_markdown_gate(report_obj: Dict[str, object], block: str) -> None:
        markdown = str(report_obj.get("report_markdown") or "").strip()
        if not markdown:
            report_obj["report_markdown"] = f"{block}\n"
            return
        if "## Citation Quality Gate" in markdown:
            return
        report_obj["report_markdown"] = f"{markdown}\n\n{block}\n"

    if not bool(citation_gate):
        return report
    checks = list(report.get("quality_checks") or [])
    resolved = _resolve_citation_report_path(
        source_path=source_path,
        citation_report_path=citation_report_path,
    )
    if resolved is None:
        checks.append(
            _report_quality_check(
                check_id="citation_report_presence",
                status=("fail" if citation_gate_required else "warn"),
                message=(
                    "Citation verification report is required but was not found."
                    if citation_gate_required
                    else "Citation verification report was not found; citation gate is advisory."
                ),
            )
        )
        report["quality_checks"] = checks
        report["quality_status"] = _quality_status_from_checks(checks)
        _append_markdown_gate(
            report,
            _citation_gate_markdown_block(
                status=report.get("quality_status", "warn"),
                details=(
                    "Citation verification report missing."
                    if citation_gate_required
                    else "Citation verification report missing (advisory mode)."
                ),
            ),
        )
        return report
    try:
        citation_summary = _load_citation_report_summary(resolved)
    except Exception as exc:
        checks.append(
            _report_quality_check(
                check_id="citation_report_parse",
                status="fail",
                message=f"Failed to parse citation verification report: {exc}",
            )
        )
        report["quality_checks"] = checks
        report["quality_status"] = _quality_status_from_checks(checks)
        _append_markdown_gate(
            report,
            _citation_gate_markdown_block(
                status=report.get("quality_status", "fail"),
                details=f"Failed to parse citation verification report: {exc}",
            ),
        )
        return report

    counts = dict(citation_summary.get("counts") or {})
    total = int(citation_summary.get("total") or 0)
    hallucinated = int(counts.get("hallucinated") or 0)
    suspicious = int(counts.get("suspicious") or 0)
    integrity_score = _safe_float(citation_summary.get("integrity_score"))
    suspicious_rate = (float(suspicious) / float(total)) if total > 0 else 0.0

    checks.append(
        _report_quality_check(
            check_id="citation_report_presence",
            status="pass",
            message=(
                f"Citation report loaded from {resolved} "
                f"(total={total}, hallucinated={hallucinated}, suspicious={suspicious})."
            ),
        )
    )
    checks.append(
        _report_quality_check(
            check_id="citation_hallucination_gate",
            status=(
                "fail" if hallucinated > int(citation_hallucinated_max) else "pass"
            ),
            message=(
                f"Hallucinated citations={hallucinated} exceeds threshold={citation_hallucinated_max}."
                if hallucinated > int(citation_hallucinated_max)
                else f"Hallucinated citations={hallucinated} within threshold={citation_hallucinated_max}."
            ),
        )
    )
    checks.append(
        _report_quality_check(
            check_id="citation_suspicious_rate",
            status=(
                "warn"
                if suspicious_rate > float(citation_suspicious_rate_warn)
                else "pass"
            ),
            message=(
                f"Suspicious citation rate={suspicious_rate:.3f} exceeds warning threshold={float(citation_suspicious_rate_warn):.3f}."
                if suspicious_rate > float(citation_suspicious_rate_warn)
                else f"Suspicious citation rate={suspicious_rate:.3f} within warning threshold={float(citation_suspicious_rate_warn):.3f}."
            ),
        )
    )
    if integrity_score is None:
        checks.append(
            _report_quality_check(
                check_id="citation_integrity_score",
                status="warn",
                message="Citation integrity score missing from report summary.",
            )
        )
    else:
        checks.append(
            _report_quality_check(
                check_id="citation_integrity_score",
                status=(
                    "warn"
                    if integrity_score < float(citation_integrity_min_warn)
                    else "pass"
                ),
                message=(
                    f"Citation integrity score={integrity_score:.3f} below warning threshold={float(citation_integrity_min_warn):.3f}."
                    if integrity_score < float(citation_integrity_min_warn)
                    else f"Citation integrity score={integrity_score:.3f} meets threshold={float(citation_integrity_min_warn):.3f}."
                ),
            )
        )

    report["citation_quality_gate"] = {
        "report_path": str(resolved),
        "summary": citation_summary,
        "thresholds": {
            "hallucinated_max": int(citation_hallucinated_max),
            "suspicious_rate_warn": float(citation_suspicious_rate_warn),
            "integrity_min_warn": float(citation_integrity_min_warn),
        },
    }
    report["quality_checks"] = checks
    report["quality_status"] = _quality_status_from_checks(checks)
    _append_markdown_gate(
        report,
        _citation_gate_markdown_block(
            status=report.get("quality_status", "warn"),
            details=(
                f"Loaded `{resolved}` "
                f"(total={total}, hallucinated={hallucinated}, suspicious={suspicious}, "
                f"integrity={float(integrity_score or 0.0):.3f})."
            ),
            thresholds=dict(report["citation_quality_gate"].get("thresholds") or {}),
        ),
    )
    return report


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        return [
            {
                str(k): ("" if v is None else str(v))
                for k, v in row.items()
                if k is not None
            }
            for row in reader
            if row
        ]


def _safe_float(value: object) -> Optional[float]:
    try:
        if value is None:
            return None
        text = str(value).strip()
        if not text:
            return None
        return float(text)
    except Exception:
        return None


def _safe_int(value: object) -> Optional[int]:
    parsed = _safe_float(value)
    return None if parsed is None else int(parsed)


def _format_ci(
    low: Optional[float], high: Optional[float], *, precision: int = 4
) -> str:
    if low is None or high is None:
        return "NA"
    return f"[{float(low):.{int(precision)}f}, {float(high):.{int(precision)}f}]"


def _percentile_interval(
    values: Sequence[float],
    *,
    alpha: float = 0.05,
) -> tuple[Optional[float], Optional[float]]:
    if not values:
        return (None, None)
    arr = np.asarray(list(values), dtype=float)
    if arr.size == 0:
        return (None, None)
    low_q = 100.0 * (float(alpha) / 2.0)
    high_q = 100.0 * (1.0 - float(alpha) / 2.0)
    return (float(np.percentile(arr, low_q)), float(np.percentile(arr, high_q)))


def _behavior_macro_map_from_probs(
    y_true: np.ndarray,
    probs: np.ndarray,
    *,
    num_classes: int,
) -> float:
    ap_scores: list[float] = []
    for idx in range(int(num_classes)):
        binary_true = (y_true == idx).astype(int)
        try:
            ap = metrics.average_precision_score(binary_true, probs[:, idx])
        except Exception:
            ap = 0.0
        if np.isnan(ap):
            ap = 0.0
        ap_scores.append(float(ap))
    return float(np.mean(ap_scores)) if ap_scores else 0.0


def _bootstrap_behavior_metric_intervals(
    *,
    y_true: Sequence[int],
    y_pred: Sequence[int],
    probs: Sequence[Sequence[float]] | None = None,
    n_bootstrap: int = 1000,
    seed: int = 0,
) -> dict[str, tuple[Optional[float], Optional[float]]]:
    true_arr = np.asarray(list(y_true), dtype=int)
    pred_arr = np.asarray(list(y_pred), dtype=int)
    if true_arr.size == 0 or pred_arr.size != true_arr.size:
        return {
            "accuracy": (None, None),
            "macro_f1": (None, None),
            "macro_map": (None, None),
        }
    prob_arr = None
    if probs is not None:
        prob_arr = np.asarray(list(probs), dtype=float)
        if prob_arr.ndim != 2 or prob_arr.shape[0] != true_arr.size:
            prob_arr = None

    rng = np.random.default_rng(int(seed))
    acc_samples: list[float] = []
    f1_samples: list[float] = []
    map_samples: list[float] = []
    n = int(true_arr.size)
    for _ in range(max(1, int(n_bootstrap))):
        sample_idx = rng.integers(0, n, size=n)
        y_t = true_arr[sample_idx]
        y_p = pred_arr[sample_idx]
        acc_samples.append(float(metrics.accuracy_score(y_t, y_p)))
        f1_samples.append(
            float(metrics.f1_score(y_t, y_p, average="macro", zero_division=0))
        )
        if prob_arr is not None:
            map_samples.append(
                _behavior_macro_map_from_probs(
                    y_t,
                    prob_arr[sample_idx, :],
                    num_classes=int(prob_arr.shape[1]),
                )
            )
    return {
        "accuracy": _percentile_interval(acc_samples),
        "macro_f1": _percentile_interval(f1_samples),
        "macro_map": _percentile_interval(map_samples),
    }


def _bootstrap_behavior_per_class_intervals(
    *,
    y_true: Sequence[int],
    y_pred: Sequence[int],
    probs: Sequence[Sequence[float]] | None,
    label_names: Sequence[str],
    n_bootstrap: int = 1000,
    seed: int = 0,
) -> dict[str, dict[str, tuple[Optional[float], Optional[float]]]]:
    true_arr = np.asarray(list(y_true), dtype=int)
    pred_arr = np.asarray(list(y_pred), dtype=int)
    labels = [str(name) for name in label_names]
    if true_arr.size == 0 or pred_arr.size != true_arr.size or not labels:
        return {}
    prob_arr = None
    if probs is not None:
        prob_arr = np.asarray(list(probs), dtype=float)
        if prob_arr.ndim != 2 or prob_arr.shape[0] != true_arr.size:
            prob_arr = None

    rng = np.random.default_rng(int(seed))
    label_indices = list(range(len(labels)))
    collectors: dict[str, dict[str, list[float]]] = {
        label: {"precision": [], "recall": [], "f1": [], "ap": []} for label in labels
    }
    n = int(true_arr.size)
    for _ in range(max(1, int(n_bootstrap))):
        sample_idx = rng.integers(0, n, size=n)
        y_t = true_arr[sample_idx]
        y_p = pred_arr[sample_idx]
        precision, recall, f1, _ = metrics.precision_recall_fscore_support(
            y_t,
            y_p,
            labels=label_indices,
            zero_division=0,
        )
        for idx, label in enumerate(labels):
            collectors[label]["precision"].append(float(precision[idx]))
            collectors[label]["recall"].append(float(recall[idx]))
            collectors[label]["f1"].append(float(f1[idx]))
            if prob_arr is not None and idx < prob_arr.shape[1]:
                binary_true = (y_t == idx).astype(int)
                try:
                    ap = metrics.average_precision_score(
                        binary_true, prob_arr[sample_idx, idx]
                    )
                except Exception:
                    ap = 0.0
                if np.isnan(ap):
                    ap = 0.0
                collectors[label]["ap"].append(float(ap))

    return {
        label: {
            "precision": _percentile_interval(collectors[label]["precision"]),
            "recall": _percentile_interval(collectors[label]["recall"]),
            "f1": _percentile_interval(collectors[label]["f1"]),
            "ap": _percentile_interval(collectors[label]["ap"]),
        }
        for label in labels
    }


def _read_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected YAML object in {path}")
    return payload


def _resolve_yolo_eval_type(
    *, metric_suffix: str, predictions: Sequence[dict[str, Any]]
) -> str:
    suffix = str(metric_suffix or "").strip().upper()
    if suffix == "P":
        return "keypoints"
    if suffix == "M":
        return "segm"
    if suffix == "B":
        return "bbox"
    if predictions:
        sample = predictions[0]
        if isinstance(sample, dict):
            if "keypoints" in sample:
                return "keypoints"
            if "segmentation" in sample:
                return "segm"
    return "bbox"


def _resolve_yolo_annotation_json(
    *,
    root: Path,
    split: str,
    eval_type: str,
) -> Optional[Path]:
    args_path = root / "args.yaml"
    if not args_path.exists():
        return None
    try:
        args_payload = _read_yaml(args_path)
    except Exception:
        return None
    data_value = args_payload.get("data")
    if not data_value:
        return None
    data_path = Path(str(data_value)).expanduser()
    if not data_path.is_absolute():
        data_path = (args_path.parent / data_path).resolve()
    if not data_path.exists():
        return None

    dataset_root: Optional[Path] = None
    if data_path.is_file():
        try:
            dataset_payload = _read_yaml(data_path)
        except Exception:
            dataset_payload = {}
        base_path = dataset_payload.get("path")
        if base_path:
            base = Path(str(base_path)).expanduser()
            dataset_root = (
                base if base.is_absolute() else (data_path.parent / base).resolve()
            )
        else:
            dataset_root = data_path.parent
    else:
        dataset_root = data_path
    if dataset_root is None:
        return None

    split_name = str(split or args_payload.get("split") or "val").strip() or "val"
    annotations_dir = dataset_root / "annotations"
    candidates = (
        [annotations_dir / f"person_keypoints_{split_name}.json"]
        if eval_type == "keypoints"
        else [annotations_dir / f"instances_{split_name}.json"]
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return None


def _build_bootstrap_coco_datasets(
    *,
    gt_payload: dict[str, Any],
    dt_payload: Sequence[dict[str, Any]],
    image_ids: Sequence[int],
) -> tuple[COCO, COCO]:
    assert COCO is not None
    gt_images = {
        int(item["id"]): item
        for item in gt_payload.get("images", [])
        if isinstance(item, dict) and item.get("id") is not None
    }
    gt_annotations: dict[int, list[dict[str, Any]]] = {}
    for ann in gt_payload.get("annotations", []):
        if not isinstance(ann, dict) or ann.get("image_id") is None:
            continue
        gt_annotations.setdefault(int(ann["image_id"]), []).append(ann)
    dt_annotations: dict[int, list[dict[str, Any]]] = {}
    for ann in dt_payload:
        if not isinstance(ann, dict) or ann.get("image_id") is None:
            continue
        dt_annotations.setdefault(int(ann["image_id"]), []).append(ann)

    images_out: list[dict[str, Any]] = []
    anns_out: list[dict[str, Any]] = []
    preds_out: list[dict[str, Any]] = []
    ann_id = 1
    for index, image_id in enumerate(image_ids, start=1):
        image = gt_images.get(int(image_id))
        if image is None:
            continue
        new_image = dict(image)
        new_image["id"] = index
        images_out.append(new_image)
        for ann in gt_annotations.get(int(image_id), []):
            new_ann = dict(ann)
            new_ann["id"] = ann_id
            new_ann["image_id"] = index
            anns_out.append(new_ann)
            ann_id += 1
        for pred in dt_annotations.get(int(image_id), []):
            new_pred = dict(pred)
            new_pred["image_id"] = index
            preds_out.append(new_pred)

    gt_dataset = {
        "info": gt_payload.get("info", {}),
        "licenses": gt_payload.get("licenses", []),
        "images": images_out,
        "annotations": anns_out,
        "categories": gt_payload.get("categories", []),
    }
    coco_gt = COCO()
    coco_gt.dataset = gt_dataset
    coco_gt.createIndex()
    coco_dt = coco_gt.loadRes(preds_out) if preds_out else coco_gt.loadRes([])
    return coco_gt, coco_dt


def _run_coco_eval_stats(
    *,
    annotation_json: Path,
    prediction_json: Path,
    eval_type: str,
    image_ids: Sequence[int] | None = None,
) -> list[float]:
    if COCO is None or COCOeval is None:
        raise RuntimeError("pycocotools is required for YOLO JSON evaluation.")
    coco_gt = COCO(str(annotation_json))
    with prediction_json.open("r", encoding="utf-8") as fh:
        predictions = json.load(fh)
    if not isinstance(predictions, list):
        raise ValueError(f"Expected a JSON array in {prediction_json}")
    coco_dt = coco_gt.loadRes(predictions)
    evaluator = COCOeval(coco_gt, coco_dt, eval_type)
    evaluator.params.imgIds = list(image_ids or coco_gt.getImgIds())
    evaluator.evaluate()
    evaluator.accumulate()
    evaluator.summarize()
    return [float(value) for value in evaluator.stats]


def _bootstrap_coco_metric_intervals(
    *,
    annotation_json: Path,
    prediction_json: Path,
    eval_type: str,
    n_bootstrap: int = 200,
    seed: int = 0,
) -> dict[str, tuple[Optional[float], Optional[float]]]:
    if COCO is None or COCOeval is None:
        return {}
    gt_payload = _read_json(annotation_json)
    dt_payload = json.loads(prediction_json.read_text(encoding="utf-8"))
    if not isinstance(dt_payload, list):
        return {}
    image_ids = [
        int(item["id"])
        for item in gt_payload.get("images", [])
        if isinstance(item, dict) and item.get("id") is not None
    ]
    if not image_ids:
        return {}
    rng = np.random.default_rng(int(seed))
    map50_samples: list[float] = []
    map50_95_samples: list[float] = []
    for _ in range(max(1, int(n_bootstrap))):
        sampled = rng.choice(image_ids, size=len(image_ids), replace=True).tolist()
        coco_gt, coco_dt = _build_bootstrap_coco_datasets(
            gt_payload=gt_payload,
            dt_payload=dt_payload,
            image_ids=sampled,
        )
        evaluator = COCOeval(coco_gt, coco_dt, eval_type)
        evaluator.params.imgIds = coco_gt.getImgIds()
        evaluator.evaluate()
        evaluator.accumulate()
        evaluator.summarize()
        map50_95_samples.append(float(evaluator.stats[0]))
        map50_samples.append(float(evaluator.stats[1]))
    return {
        "map50": _percentile_interval(map50_samples),
        "map50_95": _percentile_interval(map50_95_samples),
    }


def _find_first(root: Path, patterns: Iterable[str]) -> list[str]:
    found: list[str] = []
    seen: set[str] = set()
    for pattern in patterns:
        for path in sorted(root.rglob(pattern)):
            key = str(path.resolve())
            if key in seen or not path.is_file():
                continue
            seen.add(key)
            found.append(key)
    return found


def _detect_report_source(path: Path, explicit_family: str) -> tuple[str, Path]:
    family = str(explicit_family or "auto").strip().lower()
    if path.is_file():
        if family != "auto":
            return family, path
        if path.suffix.lower() == ".csv":
            return "yolo", path
        payload = _read_json(path)
        if "paper_table" in payload and "summary" in payload:
            return "dino_kpseg", path
        if "pck" in payload or "keypoints_visible_total" in payload:
            return "dino_kpseg", path
        if isinstance(payload.get("test_metrics"), dict):
            return "behavior_classifier", path
        return "generic_json", path

    if not path.is_dir():
        raise FileNotFoundError(f"Path not found: {path}")
    if family != "auto":
        return family, path

    metrics_json = path / "metrics.json"
    if metrics_json.exists():
        payload = _read_json(metrics_json)
        if isinstance(payload.get("test_metrics"), dict):
            return "behavior_classifier", path
    results_csv = path / "results.csv"
    if results_csv.exists():
        rows = _read_csv_rows(results_csv)
        if rows:
            row = rows[-1]
            keys = set(row.keys())
            if any("metrics/mAP50(P)" == key for key in keys) or any(
                key.startswith("metrics/mAP50(") for key in keys
            ):
                return "yolo", path
    return "generic_run", path


def _collect_run_artifacts(root: Path) -> list[str]:
    if not root.exists() or not root.is_dir():
        return []
    return _find_first(
        root,
        (
            "results.csv",
            "predictions.json",
            "metrics.json",
            "args.yaml",
            "results.png",
            "confusion_matrix*.png",
            "*_curve.png",
            "pr_curves.png",
            "paper_metrics*.json",
            "paper_metrics*.md",
            "paper_metrics*.csv",
            "paper_metrics*.tex",
        ),
    )


def _build_generic_metric_report(
    *,
    rows: Sequence[tuple[str, str, str]],
    csv_row: Dict[str, object],
    metadata: Dict[str, object],
    summary: Dict[str, object],
    artifacts: Sequence[str],
    per_class_rows: Sequence[dict[str, object]] = (),
    quality_checks: Sequence[dict[str, object]] = (),
) -> Dict[str, object]:
    table_markdown = _paper_markdown_table(rows=rows)
    table_latex = _paper_latex_table(rows=rows)
    table_csv = _paper_csv(row=csv_row)

    markdown_lines = [
        f"# Evaluation Report: {metadata.get('model', 'model')}",
        "",
        f"- Dataset: `{metadata.get('dataset', 'unknown')}`",
        f"- Split: `{metadata.get('split', 'unknown')}`",
        f"- Model family: `{metadata.get('model_family', 'unknown')}`",
        f"- Metrics source: `{metadata.get('source_path', '')}`",
        f"- Generated (UTC): `{metadata.get('generated_at_utc', '')}`",
        "",
        "## Primary Metrics",
        "",
        table_markdown,
        "",
        "## Reproducibility Checklist",
        "",
        "- Report generated from saved run artifacts rather than handwritten numbers.",
        "- Metric source path and artifact inventory are included below.",
        "- Report should cite dataset split, checkpoint/run identifier, and evaluation protocol.",
        "- Confidence intervals are shown only when the source metrics support them; otherwise `NA` is emitted.",
        "",
    ]
    if per_class_rows:
        markdown_lines.extend(
            [
                "## Per-Class Metrics",
                "",
                "| Class | Precision | Recall | F1 | AP | Support |",
                "|---|---:|---:|---:|---:|---:|",
            ]
        )
        for row in per_class_rows:
            markdown_lines.append(
                "| {label} | {precision} ({precision_ci95}) | {recall} ({recall_ci95}) | {f1} ({f1_ci95}) | {ap} ({ap_ci95}) | {support} |".format(
                    label=row.get("label", ""),
                    precision=row.get("precision", "NA"),
                    precision_ci95=row.get("precision_ci95", "NA"),
                    recall=row.get("recall", "NA"),
                    recall_ci95=row.get("recall_ci95", "NA"),
                    f1=row.get("f1", "NA"),
                    f1_ci95=row.get("f1_ci95", "NA"),
                    ap=row.get("ap", "NA"),
                    ap_ci95=row.get("ap_ci95", "NA"),
                    support=row.get("support", "NA"),
                )
            )
        markdown_lines.append("")
    markdown_lines.extend(
        [
            "## Quality Checks",
            "",
            _quality_checks_markdown(quality_checks),
            "",
            "## Artifacts",
            "",
        ]
    )
    if artifacts:
        markdown_lines.extend([f"- `{artifact}`" for artifact in artifacts])
    else:
        markdown_lines.append("- No extra artifacts discovered.")

    return {
        "metadata": metadata,
        "summary": summary,
        "paper_table": {
            "rows": [
                {"metric": metric, "value": value, "ci95": ci}
                for metric, value, ci in rows
            ],
            "markdown": table_markdown,
            "latex": table_latex,
            "csv": table_csv,
        },
        "report_markdown": "\n".join(markdown_lines).strip() + "\n",
        "artifacts": list(artifacts),
        "per_class": list(per_class_rows),
        "quality_status": _quality_status_from_checks(quality_checks),
        "quality_checks": list(quality_checks),
    }


def _build_dino_report(
    *,
    source_path: Path,
    dataset_name: str,
    model_name: str,
    split: str,
) -> Dict[str, object]:
    payload = _read_json(source_path)
    if "paper_table" in payload and "summary" in payload:
        report = dict(payload)
    else:
        report = build_paper_report(
            summary=payload,
            dataset_name=dataset_name,
            model_name=model_name,
            split=split,
        )
    report.setdefault("metadata", {})
    if isinstance(report["metadata"], dict):
        report["metadata"].update(
            {
                "dataset": dataset_name,
                "model": model_name,
                "split": split,
                "model_family": "dino_kpseg",
                "source_path": str(source_path),
            }
        )
    summary = report.get("summary") if isinstance(report.get("summary"), dict) else {}
    images_total = _safe_int(summary.get("images_total")) if summary else None
    images_used = _safe_int(summary.get("images_used")) if summary else None
    pck_rows = [
        row
        for row in (
            report.get("paper_table", {}).get("rows", [])
            if isinstance(report.get("paper_table"), dict)
            else []
        )
        if isinstance(row, dict) and str(row.get("metric") or "").startswith("PCK@")
    ]
    has_missing_pck_ci = any(str(row.get("ci95") or "NA") == "NA" for row in pck_rows)
    quality_checks: list[dict[str, object]] = []
    if images_total is not None and images_total > 0 and images_used is not None:
        if images_used < images_total:
            quality_checks.append(
                _report_quality_check(
                    check_id="dataset_coverage",
                    status="warn",
                    message=(
                        f"Only {images_used}/{images_total} images were evaluated. "
                        "Confirm filtering behavior before citing results."
                    ),
                )
            )
        else:
            quality_checks.append(
                _report_quality_check(
                    check_id="dataset_coverage",
                    status="pass",
                    message=f"All {images_used} available images were evaluated.",
                )
            )
    if pck_rows:
        quality_checks.append(
            _report_quality_check(
                check_id="pck_ci_coverage",
                status=("warn" if has_missing_pck_ci else "pass"),
                message=(
                    "Some PCK rows are missing CI values."
                    if has_missing_pck_ci
                    else "All PCK rows include Wilson 95% confidence intervals."
                ),
            )
        )
    report["quality_status"] = _quality_status_from_checks(quality_checks)
    report["quality_checks"] = quality_checks
    report["report_markdown"] = (
        f"# Evaluation Report: {model_name}\n\n"
        f"- Dataset: `{dataset_name}`\n"
        f"- Split: `{split}`\n"
        f"- Model family: `dino_kpseg`\n"
        f"- Metrics source: `{source_path}`\n\n"
        "## Primary Metrics\n\n"
        f"{report.get('paper_table', {}).get('markdown', '')}\n\n"
        "## Quality Checks\n\n"
        f"{_quality_checks_markdown(quality_checks)}\n"
    )
    return report


def _infer_yolo_target_suffix(row: dict[str, str]) -> str:
    for suffix in ("P", "M", "B"):
        if f"metrics/mAP50({suffix})" in row:
            return suffix
    for key in row:
        if key.startswith("metrics/mAP50(") and key.endswith(")"):
            return key.rsplit("(", 1)[-1].rstrip(")")
    return "B"


def _build_yolo_report(
    *,
    source_path: Path,
    dataset_name: str,
    model_name: str,
    split: str,
    bootstrap_samples: int = 200,
    bootstrap_seed: int = 0,
) -> Dict[str, object]:
    csv_path = source_path if source_path.is_file() else source_path / "results.csv"
    rows = _read_csv_rows(csv_path)
    if not rows:
        raise ValueError(f"No rows found in {csv_path}")
    final_row = rows[-1]
    suffix = _infer_yolo_target_suffix(final_row)
    score_key = f"metrics/mAP50-95({suffix})"
    best_row = max(
        rows, key=lambda row: _safe_float(row.get(score_key)) or float("-inf")
    )
    best_epoch = _safe_int(best_row.get("epoch"))
    final_epoch = _safe_int(final_row.get("epoch"))
    root = source_path if source_path.is_dir() else source_path.parent
    prediction_json = root / "predictions.json"
    annotation_json: Optional[Path] = None
    eval_type = "bbox"
    ci_map: dict[str, tuple[Optional[float], Optional[float]]] = {}
    if prediction_json.exists():
        try:
            prediction_payload = json.loads(prediction_json.read_text(encoding="utf-8"))
            if isinstance(prediction_payload, list):
                eval_type = _resolve_yolo_eval_type(
                    metric_suffix=suffix,
                    predictions=prediction_payload,
                )
                annotation_json = _resolve_yolo_annotation_json(
                    root=root,
                    split=split,
                    eval_type=eval_type,
                )
                if annotation_json is not None:
                    ci_map = _bootstrap_coco_metric_intervals(
                        annotation_json=annotation_json,
                        prediction_json=prediction_json,
                        eval_type=eval_type,
                        n_bootstrap=bootstrap_samples,
                        seed=bootstrap_seed,
                    )
        except Exception:
            ci_map = {}
    quality_checks: list[dict[str, object]] = []
    if prediction_json.exists():
        quality_checks.append(
            _report_quality_check(
                check_id="prediction_json_present",
                status="pass",
                message="predictions.json is present for downstream reproducibility.",
            )
        )
    else:
        quality_checks.append(
            _report_quality_check(
                check_id="prediction_json_present",
                status="warn",
                message=(
                    "predictions.json was not found; report cannot verify per-image "
                    "prediction outputs."
                ),
            )
        )
    if annotation_json is None:
        quality_checks.append(
            _report_quality_check(
                check_id="annotation_linkage",
                status="warn",
                message=(
                    "COCO annotation JSON could not be resolved from args.yaml/data "
                    "config; bootstrap CI remains unavailable."
                ),
            )
        )
    else:
        quality_checks.append(
            _report_quality_check(
                check_id="annotation_linkage",
                status="pass",
                message=f"Resolved annotation JSON at {annotation_json}.",
            )
        )
    ci_ready = all(
        ci_map.get(name, (None, None))[0] is not None for name in ("map50", "map50_95")
    )
    quality_checks.append(
        _report_quality_check(
            check_id="ci_coverage",
            status=("pass" if ci_ready else "warn"),
            message=(
                "mAP@50 and mAP@50-95 include bootstrap 95% CI values."
                if ci_ready
                else "Bootstrap CI values are missing for one or more primary mAP metrics."
            ),
        )
    )
    best_map = _safe_float(best_row.get(score_key))
    final_map = _safe_float(final_row.get(score_key))
    if best_map is not None and final_map is not None:
        drop = float(best_map) - float(final_map)
        quality_checks.append(
            _report_quality_check(
                check_id="best_final_gap",
                status=("warn" if drop > 0.03 else "pass"),
                message=(
                    f"Best-vs-final {score_key} gap is {drop:.4f}."
                    if drop > 0.03
                    else f"Best-vs-final {score_key} gap is {drop:.4f} (stable)."
                ),
            )
        )
    metric_rows = [
        (
            "Precision",
            _format_float(final_row.get(f"metrics/precision({suffix})")),
            "NA",
        ),
        ("Recall", _format_float(final_row.get(f"metrics/recall({suffix})")), "NA"),
        (
            "mAP@50",
            _format_float(final_row.get(f"metrics/mAP50({suffix})")),
            _format_ci(*ci_map.get("map50", (None, None))),
        ),
        (
            "mAP@50-95",
            _format_float(final_row.get(f"metrics/mAP50-95({suffix})")),
            _format_ci(*ci_map.get("map50_95", (None, None))),
        ),
        ("Best Epoch", ("" if best_epoch is None else str(best_epoch)), "NA"),
        ("Final Epoch", ("" if final_epoch is None else str(final_epoch)), "NA"),
    ]
    csv_row: Dict[str, object] = {
        "dataset": dataset_name,
        "split": split,
        "model": model_name,
        "model_family": "yolo",
        "metric_suffix": suffix,
        "precision": _safe_float(final_row.get(f"metrics/precision({suffix})")),
        "recall": _safe_float(final_row.get(f"metrics/recall({suffix})")),
        "map50": _safe_float(final_row.get(f"metrics/mAP50({suffix})")),
        "map50_95": _safe_float(final_row.get(f"metrics/mAP50-95({suffix})")),
        "map50_ci95_low": ci_map.get("map50", (None, None))[0],
        "map50_ci95_high": ci_map.get("map50", (None, None))[1],
        "map50_95_ci95_low": ci_map.get("map50_95", (None, None))[0],
        "map50_95_ci95_high": ci_map.get("map50_95", (None, None))[1],
        "best_epoch": best_epoch,
        "final_epoch": final_epoch,
    }
    metadata = {
        "dataset": dataset_name,
        "split": split,
        "model": model_name,
        "model_family": "yolo",
        "source_path": str(csv_path),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    if annotation_json is not None:
        metadata["annotation_json"] = str(annotation_json)
    if prediction_json.exists():
        metadata["prediction_json"] = str(prediction_json)
    summary = {
        "final_row": final_row,
        "best_row": best_row,
        "metric_suffix": suffix,
        "eval_type": eval_type,
        "confidence_intervals": {
            "map50": {
                "low": ci_map.get("map50", (None, None))[0],
                "high": ci_map.get("map50", (None, None))[1],
            },
            "map50_95": {
                "low": ci_map.get("map50_95", (None, None))[0],
                "high": ci_map.get("map50_95", (None, None))[1],
            },
        },
    }
    return _build_generic_metric_report(
        rows=metric_rows,
        csv_row=csv_row,
        metadata=metadata,
        summary=summary,
        artifacts=_collect_run_artifacts(root),
        quality_checks=quality_checks,
    )


def _build_behavior_report(
    *,
    source_path: Path,
    dataset_name: str,
    model_name: str,
    split: str,
) -> Dict[str, object]:
    json_path = source_path if source_path.is_file() else source_path / "metrics.json"
    payload = _read_json(json_path)
    test_metrics = (
        payload.get("test_metrics")
        if isinstance(payload.get("test_metrics"), dict)
        else payload
    )
    if not isinstance(test_metrics, dict):
        raise ValueError(f"No test_metrics object found in {json_path}")
    prediction_rows = (
        payload.get("predictions")
        if isinstance(payload.get("predictions"), list)
        else []
    )
    per_class = (
        test_metrics.get("per_class")
        if isinstance(test_metrics.get("per_class"), dict)
        else {}
    )
    per_class_ap = (
        test_metrics.get("per_class_ap")
        if isinstance(test_metrics.get("per_class_ap"), dict)
        else {}
    )
    label_names = (
        [str(name) for name in test_metrics.get("labels")]
        if isinstance(test_metrics.get("labels"), list)
        else [str(name) for name in per_class.keys()]
    )
    y_true: list[int] = []
    y_pred: list[int] = []
    prob_rows: list[list[float]] = []
    for row in prediction_rows:
        if not isinstance(row, dict):
            continue
        target_idx = _safe_int(row.get("target_index"))
        pred_idx = _safe_int(row.get("predicted_index"))
        probs = row.get("class_probabilities")
        if target_idx is None or pred_idx is None:
            continue
        y_true.append(int(target_idx))
        y_pred.append(int(pred_idx))
        if isinstance(probs, list):
            try:
                prob_rows.append([float(v) for v in probs])
            except Exception:
                prob_rows = []
    intervals = _bootstrap_behavior_metric_intervals(
        y_true=y_true,
        y_pred=y_pred,
        probs=(prob_rows if prob_rows else None),
    )
    per_class_intervals = _bootstrap_behavior_per_class_intervals(
        y_true=y_true,
        y_pred=y_pred,
        probs=(prob_rows if prob_rows else None),
        label_names=label_names,
    )
    rows = [
        ("Loss", _format_float(test_metrics.get("loss")), "NA"),
        (
            "Accuracy",
            _format_float(test_metrics.get("accuracy")),
            _format_ci(*intervals.get("accuracy", (None, None))),
        ),
        (
            "Macro F1",
            _format_float(test_metrics.get("macro_f1")),
            _format_ci(*intervals.get("macro_f1", (None, None))),
        ),
        (
            "Macro mAP",
            _format_float(test_metrics.get("macro_map")),
            _format_ci(*intervals.get("macro_map", (None, None))),
        ),
    ]
    csv_row: Dict[str, object] = {
        "dataset": dataset_name,
        "split": split,
        "model": model_name,
        "model_family": "behavior_classifier",
        "loss": _safe_float(test_metrics.get("loss")),
        "accuracy": _safe_float(test_metrics.get("accuracy")),
        "macro_f1": _safe_float(test_metrics.get("macro_f1")),
        "macro_map": _safe_float(test_metrics.get("macro_map")),
        "accuracy_ci95_low": intervals.get("accuracy", (None, None))[0],
        "accuracy_ci95_high": intervals.get("accuracy", (None, None))[1],
        "macro_f1_ci95_low": intervals.get("macro_f1", (None, None))[0],
        "macro_f1_ci95_high": intervals.get("macro_f1", (None, None))[1],
        "macro_map_ci95_low": intervals.get("macro_map", (None, None))[0],
        "macro_map_ci95_high": intervals.get("macro_map", (None, None))[1],
    }
    per_class_rows: list[dict[str, object]] = []
    for label, stats in per_class.items():
        if not isinstance(stats, dict):
            continue
        ci_map = per_class_intervals.get(str(label), {})
        per_class_rows.append(
            {
                "label": str(label),
                "precision": _format_float(stats.get("precision")),
                "precision_ci95": _format_ci(*ci_map.get("precision", (None, None))),
                "recall": _format_float(stats.get("recall")),
                "recall_ci95": _format_ci(*ci_map.get("recall", (None, None))),
                "f1": _format_float(stats.get("f1-score")),
                "f1_ci95": _format_ci(*ci_map.get("f1", (None, None))),
                "ap": _format_float(per_class_ap.get(label)),
                "ap_ci95": _format_ci(*ci_map.get("ap", (None, None))),
                "support": _safe_int(stats.get("support")) or "",
            }
        )
    metadata = {
        "dataset": dataset_name,
        "split": split,
        "model": model_name,
        "model_family": "behavior_classifier",
        "source_path": str(json_path),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    quality_checks: list[dict[str, object]] = []
    sample_count = len(y_true)
    quality_checks.append(
        _report_quality_check(
            check_id="prediction_sample_size",
            status=("warn" if sample_count < 30 else "pass"),
            message=(
                f"Only {sample_count} prediction rows were available; bootstrap CI may be unstable."
                if sample_count < 30
                else f"{sample_count} prediction rows available for bootstrap CI estimation."
            ),
        )
    )
    has_core_ci = all(
        intervals.get(key, (None, None))[0] is not None
        for key in ("accuracy", "macro_f1")
    )
    quality_checks.append(
        _report_quality_check(
            check_id="core_ci_coverage",
            status=("pass" if has_core_ci else "warn"),
            message=(
                "Accuracy and Macro F1 include 95% CI values."
                if has_core_ci
                else "One or more core classification metrics are missing CI values."
            ),
        )
    )
    has_probabilities = bool(prob_rows)
    quality_checks.append(
        _report_quality_check(
            check_id="probability_coverage",
            status=("pass" if has_probabilities else "warn"),
            message=(
                "Class probability rows are available for macro mAP/per-class AP checks."
                if has_probabilities
                else "class_probabilities are missing, so AP-oriented confidence intervals may be NA."
            ),
        )
    )
    supports = [
        _safe_int(stats.get("support"))
        for stats in per_class.values()
        if isinstance(stats, dict)
    ]
    supports = [int(v) for v in supports if v is not None]
    if supports:
        min_support = min(supports)
        quality_checks.append(
            _report_quality_check(
                check_id="class_support",
                status=("warn" if min_support < 5 else "pass"),
                message=(
                    f"At least one class has low support (min={min_support}); interpret per-class metrics carefully."
                    if min_support < 5
                    else f"Per-class support is adequate (minimum support={min_support})."
                ),
            )
        )
    root = source_path if source_path.is_dir() else source_path.parent
    return _build_generic_metric_report(
        rows=rows,
        csv_row=csv_row,
        metadata=metadata,
        summary={"test_metrics": test_metrics},
        artifacts=_collect_run_artifacts(root),
        per_class_rows=per_class_rows,
        quality_checks=quality_checks,
    )


def build_model_eval_report(
    *,
    source_path: Path,
    model_family: str = "auto",
    dataset_name: Optional[str] = None,
    model_name: Optional[str] = None,
    split: str = "test",
    bootstrap_samples: int = 200,
    bootstrap_seed: int = 0,
    citation_gate: bool = False,
    citation_gate_required: bool = False,
    citation_report_path: Optional[Path] = None,
    citation_hallucinated_max: int = 0,
    citation_suspicious_rate_warn: float = 0.20,
    citation_integrity_min_warn: float = 0.60,
) -> Dict[str, object]:
    family, resolved_source = _detect_report_source(
        Path(source_path).expanduser().resolve(), model_family
    )
    ds_name = str(dataset_name).strip() if dataset_name else resolved_source.stem
    mdl_name = str(model_name).strip() if model_name else resolved_source.stem
    split_name = str(split or "test").strip() or "test"

    if family == "dino_kpseg":
        report = _build_dino_report(
            source_path=resolved_source,
            dataset_name=ds_name,
            model_name=mdl_name,
            split=split_name,
        )
    elif family == "yolo":
        report = _build_yolo_report(
            source_path=resolved_source,
            dataset_name=ds_name,
            model_name=mdl_name,
            split=split_name,
            bootstrap_samples=bootstrap_samples,
            bootstrap_seed=bootstrap_seed,
        )
    elif family == "behavior_classifier":
        report = _build_behavior_report(
            source_path=resolved_source,
            dataset_name=ds_name,
            model_name=mdl_name,
            split=split_name,
        )
    else:
        raise ValueError(
            f"Unsupported or unrecognized evaluation source for model_family={family!r}"
        )
    return _apply_citation_quality_gate(
        report=report,
        source_path=resolved_source,
        citation_gate=citation_gate,
        citation_gate_required=citation_gate_required,
        citation_report_path=citation_report_path,
        citation_hallucinated_max=citation_hallucinated_max,
        citation_suspicious_rate_warn=citation_suspicious_rate_warn,
        citation_integrity_min_warn=citation_integrity_min_warn,
    )


def write_model_eval_report_files(
    *,
    report: Dict[str, object],
    report_dir: Path,
    base_name: str = "eval_report",
) -> Dict[str, str]:
    out_dir = Path(report_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = str(base_name or "eval_report").strip() or "eval_report"
    json_path = out_dir / f"{stem}.json"
    md_path = out_dir / f"{stem}.md"
    csv_path = out_dir / f"{stem}.csv"
    tex_path = out_dir / f"{stem}.tex"

    paper_table = report.get("paper_table") if isinstance(report, dict) else {}
    if not isinstance(paper_table, dict):
        paper_table = {}
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    md_path.write_text(str(report.get("report_markdown") or ""), encoding="utf-8")
    csv_path.write_text(str(paper_table.get("csv") or ""), encoding="utf-8")
    tex_path.write_text(str(paper_table.get("latex") or ""), encoding="utf-8")
    return {
        "json": str(json_path),
        "markdown": str(md_path),
        "csv": str(csv_path),
        "latex": str(tex_path),
    }


class AnnolidEvalReportTool(FunctionTool):
    def __init__(
        self,
        *,
        allowed_dir: Path | None = None,
        allowed_read_roots: Sequence[str | Path] | None = None,
    ) -> None:
        self._allowed_dir = (
            Path(allowed_dir).expanduser().resolve()
            if allowed_dir is not None
            else None
        )
        self._allowed_read_roots = _normalize_allowed_read_roots(
            self._allowed_dir,
            allowed_read_roots,
        )

    @property
    def name(self) -> str:
        return "annolid_eval_report"

    @property
    def description(self) -> str:
        return (
            "Read saved Annolid evaluation artifacts such as DinoKPSEG eval JSON, "
            "YOLO results.csv, or behavior-classifier metrics.json and generate a "
            "paper-ready evaluation report with reproducibility metadata."
        )

    @property
    def parameters(self) -> dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "path": {"type": "string", "minLength": 1},
                "model_family": {
                    "type": "string",
                    "enum": ["auto", "dino_kpseg", "yolo", "behavior_classifier"],
                },
                "dataset_name": {"type": "string"},
                "model_name": {"type": "string"},
                "split": {"type": "string"},
                "report_dir": {"type": "string"},
                "report_basename": {"type": "string"},
                "bootstrap_samples": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 2000,
                },
                "bootstrap_seed": {"type": "integer"},
                "citation_gate": {"type": "boolean", "default": False},
                "citation_gate_required": {"type": "boolean", "default": False},
                "citation_report_path": {"type": "string"},
                "citation_hallucinated_max": {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": 1000000,
                    "default": 0,
                },
                "citation_suspicious_rate_warn": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "default": 0.2,
                },
                "citation_integrity_min_warn": {
                    "type": "number",
                    "minimum": 0.0,
                    "maximum": 1.0,
                    "default": 0.6,
                },
                "allow_mutation": {"type": "boolean"},
            },
            "required": ["path"],
        }

    async def execute(
        self,
        path: str,
        model_family: str = "auto",
        dataset_name: str = "",
        model_name: str = "",
        split: str = "test",
        report_dir: str = "",
        report_basename: str = "eval_report",
        bootstrap_samples: int = 200,
        bootstrap_seed: int = 0,
        citation_gate: bool = False,
        citation_gate_required: bool = False,
        citation_report_path: str = "",
        citation_hallucinated_max: int = 0,
        citation_suspicious_rate_warn: float = 0.2,
        citation_integrity_min_warn: float = 0.6,
        allow_mutation: bool = False,
        **kwargs: Any,
    ) -> str:
        del kwargs
        try:
            source_path = _resolve_read_path(
                path,
                allowed_dir=self._allowed_dir,
                allowed_read_roots=self._allowed_read_roots,
            )
            report = build_model_eval_report(
                source_path=source_path,
                model_family=model_family,
                dataset_name=(dataset_name or None),
                model_name=(model_name or None),
                split=split,
                bootstrap_samples=bootstrap_samples,
                bootstrap_seed=bootstrap_seed,
                citation_gate=bool(citation_gate),
                citation_gate_required=bool(citation_gate_required),
                citation_report_path=(
                    _resolve_read_path(
                        citation_report_path,
                        allowed_dir=self._allowed_dir,
                        allowed_read_roots=self._allowed_read_roots,
                    )
                    if str(citation_report_path or "").strip()
                    else None
                ),
                citation_hallucinated_max=int(citation_hallucinated_max),
                citation_suspicious_rate_warn=float(citation_suspicious_rate_warn),
                citation_integrity_min_warn=float(citation_integrity_min_warn),
            )
            if str(report_dir or "").strip():
                if not allow_mutation:
                    return json.dumps(
                        {
                            "ok": False,
                            "error": (
                                "Writing report files modifies state. Retry with "
                                "allow_mutation=true only when that is intended."
                            ),
                            "path": str(source_path),
                        },
                        ensure_ascii=False,
                    )
                report["written_files"] = write_model_eval_report_files(
                    report=report,
                    report_dir=_resolve_write_path(
                        report_dir,
                        allowed_dir=self._allowed_dir,
                    ),
                    base_name=report_basename,
                )
            return json.dumps({"ok": True, "report": report}, ensure_ascii=False)
        except Exception as exc:
            return json.dumps(
                {
                    "ok": False,
                    "error": str(exc),
                    "path": str(path),
                },
                ensure_ascii=False,
            )


__all__ = [
    "AnnolidEvalReportTool",
    "build_model_eval_report",
    "write_model_eval_report_files",
]
