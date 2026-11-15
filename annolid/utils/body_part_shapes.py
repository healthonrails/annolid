"""Canonical LabelMe metadata helpers for intra-instance body-part shapes."""

from __future__ import annotations

import re
from typing import Any, Mapping

BODY_PART_SEMANTIC_TYPE = "body_part"

_COARSE_PART_PATTERNS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "head",
        (
            "nose",
            "snout",
            "rostrum",
            "head",
            "skull",
            "forehead",
            "crown",
            "beak",
            "bill",
            "jaw",
            "chin",
            "mouth",
            "lip",
            "eye",
            "pupil",
            "ear",
            "antenna",
            "proboscis",
            "cheek",
            "whisker",
        ),
    ),
    (
        "wings",
        ("wing",),
    ),
    (
        "limbs",
        (
            "paw",
            "foot",
            "feet",
            "hand",
            "wrist",
            "ankle",
            "elbow",
            "knee",
            "limb",
            "leg",
            "arm",
            "tarsus",
            "tibia",
            "femur",
            "claw",
            "toe",
            "finger",
            "fin",
        ),
    ),
    (
        "tail",
        (
            "tail",
            "caudal",
            "uropod",
            "fluke",
        ),
    ),
    (
        "torso",
        (
            "neck",
            "nape",
            "throat",
            "shoulder",
            "withers",
            "chest",
            "thorax",
            "breast",
            "back",
            "dorsal",
            "spine",
            "vertebra",
            "midbody",
            "center",
            "centroid",
            "body",
            "trunk",
            "torso",
            "rib",
            "flank",
            "side",
            "belly",
            "ventral",
            "abdomen",
            "waist",
            "hip",
            "pelvis",
            "rump",
            "base_of_tail",
        ),
    ),
)


def _get_field(shape: Mapping[str, Any] | object | None, key: str) -> Any:
    if shape is None:
        return None
    if isinstance(shape, Mapping):
        return shape.get(key)
    return getattr(shape, key, None)


def _get_flags(shape: Mapping[str, Any] | object | None) -> Mapping[str, Any]:
    raw = _get_field(shape, "flags")
    return raw if isinstance(raw, Mapping) else {}


def _get_other_data(shape: Mapping[str, Any] | object | None) -> Mapping[str, Any]:
    for key in ("other_data", "otherData"):
        raw = _get_field(shape, key)
        if isinstance(raw, Mapping):
            return raw
    return {}


def _clean_text(value: Any) -> str:
    if value is None or isinstance(value, bool):
        return ""
    return str(value).strip()


def is_body_part_shape_payload(shape: Mapping[str, Any] | object | None) -> bool:
    """Return True when a LabelMe shape dict or GUI Shape represents an intra-instance body part."""
    if shape is None:
        return False

    flags = _get_flags(shape)
    other = _get_other_data(shape)

    for container in (flags, other):
        semantic_type = _clean_text(container.get("semantic_type")).lower()
        if semantic_type == BODY_PART_SEMANTIC_TYPE:
            return True
        shape_category = _clean_text(container.get("shape_category")).lower()
        if shape_category == BODY_PART_SEMANTIC_TYPE:
            return True
        if bool(container.get("is_body_part")):
            return True
        if _clean_text(container.get("part_label")) or _clean_text(
            container.get("body_part")
        ):
            return True
        if _clean_text(container.get("parent_instance")):
            return True

    if bool(_get_field(shape, "is_body_part")):
        return True
    if _clean_text(_get_field(shape, "part_label")) or _clean_text(
        _get_field(shape, "parent_instance")
    ):
        return True

    semantic_top = _clean_text(_get_field(shape, "semantic_type")).lower()
    return semantic_top == BODY_PART_SEMANTIC_TYPE


def extract_body_part_parent_instance(
    shape: Mapping[str, Any] | object | None,
) -> str:
    """Extract the owning parent instance label from a body-part shape, if present."""
    if shape is None:
        return ""
    flags = _get_flags(shape)
    other = _get_other_data(shape)

    for key in ("parent_instance", "instance_label", "instance_name"):
        val = _clean_text(_get_field(shape, key))
        if val:
            return val
        for container in (flags, other):
            val = _clean_text(container.get(key))
            if val:
                return val

    raw_label = _clean_text(_get_field(shape, "label"))
    if ":" in raw_label:
        prefix, _rest = raw_label.split(":", 1)
        if prefix.strip():
            return prefix.strip()
    return ""


def extract_body_part_label(shape: Mapping[str, Any] | object | None) -> str:
    """Extract the canonical anatomical part name (without parent prefix) from a shape."""
    if shape is None:
        return ""
    flags = _get_flags(shape)
    other = _get_other_data(shape)

    for key in ("part_label", "body_part", "display_label"):
        val = _clean_text(_get_field(shape, key))
        if val:
            return val
        for container in (flags, other):
            val = _clean_text(container.get(key))
            if val:
                return val

    raw_label = _clean_text(_get_field(shape, "label"))
    parent = extract_body_part_parent_instance(shape)
    if ":" in raw_label:
        prefix, remainder = raw_label.split(":", 1)
        if remainder.strip() and (not parent or prefix.strip() == parent):
            return remainder.strip()
    if parent and raw_label.lower().startswith(f"{parent.lower()}_"):
        trimmed = raw_label[len(parent) + 1 :].strip()
        if trimmed:
            return trimmed
    return raw_label


def format_body_part_csv_instance_name(
    parent_instance: str,
    part_label: str,
) -> str:
    """Format a hierarchical ``<parent_instance>:<part_label>`` identifier for CSV export."""
    parent = _clean_text(parent_instance)
    part = _clean_text(part_label)
    if parent and part:
        if part.lower().startswith(f"{parent.lower()}:"):
            return part
        return f"{parent}:{part}"
    return part or parent


def build_body_part_flags(
    *,
    extra_flags: Mapping[str, Any] | None = None,
) -> dict[str, bool]:
    """Construct boolean-only LabelMe per-shape flags safe for Qt flag widgets."""
    flags: dict[str, bool] = {}
    if isinstance(extra_flags, Mapping):
        for key, val in extra_flags.items():
            if isinstance(val, bool) and str(key).strip():
                flags[str(key).strip()] = val
    flags["is_body_part"] = True
    return flags


def build_body_part_metadata(
    *,
    parent_instance: str,
    part_label: str,
    confidence: float | None = None,
    area_ratio: float | None = None,
    mode: str | None = None,
    extra_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Construct LabelMe top-level / ``other_data`` metadata for a body-part polygon."""
    meta: dict[str, Any] = {}
    if isinstance(extra_metadata, Mapping):
        meta.update(dict(extra_metadata))
    parent_clean = _clean_text(parent_instance)
    part_clean = _clean_text(part_label) or "part"
    meta["semantic_type"] = BODY_PART_SEMANTIC_TYPE
    meta["part_label"] = part_clean
    meta["display_label"] = part_clean
    if parent_clean:
        meta["parent_instance"] = parent_clean
        meta["instance_label"] = parent_clean
    if confidence is not None:
        meta["score"] = round(float(confidence), 4)
    if area_ratio is not None:
        meta["area_ratio"] = round(float(area_ratio), 4)
    if mode:
        meta["segmentation_mode"] = str(mode)
    return meta


def build_body_part_shape_dict(
    *,
    points: list[list[float]],
    parent_instance: str,
    part_label: str,
    group_id: int | str | None = None,
    label_override: str | None = None,
    confidence: float | None = None,
    area_ratio: float | None = None,
    mode: str | None = None,
    description: str = "InstanceBodyPartSeg",
) -> dict[str, Any]:
    """Build a complete LabelMe-compatible polygon shape dictionary for a body part."""
    part_clean = _clean_text(part_label) or "part"
    shape_label = _clean_text(label_override) or part_clean
    payload: dict[str, Any] = {
        "label": shape_label,
        "points": [[float(pt[0]), float(pt[1])] for pt in points],
        "group_id": group_id,
        "shape_type": "polygon",
        "flags": build_body_part_flags(),
        "visible": True,
        "description": description,
    }
    payload.update(
        build_body_part_metadata(
            parent_instance=parent_instance,
            part_label=part_clean,
            confidence=confidence,
            area_ratio=area_ratio,
            mode=mode,
        )
    )
    return payload


def _detect_laterality(tokens: set[str], normalized: str) -> str:
    if (
        "left" in tokens
        or normalized.startswith(("l_", "lh_", "lf_"))
        or normalized.endswith(("_l", "_left"))
    ):
        return "left"
    if (
        "right" in tokens
        or normalized.startswith(("r_", "rh_", "rf_"))
        or normalized.endswith(("_r", "_right"))
    ):
        return "right"
    return ""


def _detect_fore_hind(tokens: set[str], normalized: str) -> str:
    if tokens & {"fore", "front", "anterior", "upper", "hand", "wrist", "elbow", "arm"}:
        return "fore"
    if normalized.startswith(("lf_", "rf_", "fl_", "fr_")):
        return "fore"
    if tokens & {
        "hind",
        "rear",
        "back",
        "posterior",
        "lower",
        "foot",
        "ankle",
        "knee",
        "leg",
    }:
        return "hind"
    if normalized.startswith(("lh_", "rh_", "hl_", "hr_")):
        return "hind"
    return ""


def infer_part_from_keypoint_name(
    keypoint_label: str,
    *,
    granularity: str = "coarse",
) -> str:
    """Map an anatomical landmark/keypoint name to a body-part region name.

    Args:
        keypoint_label: Landmark name (e.g. ``"nose"``, ``"left_ear"``, ``"tail_base"``).
        granularity: ``"coarse"`` (head/torso/limbs/wings/tail), ``"fine"``
            (preserves left/right and fore/hind limb distinctions), or
            ``"per_keypoint"`` (one part region per keypoint).
    """
    clean = _clean_text(keypoint_label)
    if not clean:
        return "torso"
    if ":" in clean:
        clean = clean.rsplit(":", 1)[-1].strip() or clean

    mode = str(granularity or "coarse").strip().lower()
    if mode == "per_keypoint":
        return clean

    normalized = re.sub(r"[^a-z0-9]+", "_", clean.lower()).strip("_")
    tokens = {tok for tok in normalized.split("_") if tok}

    # Special case: "tail_base" / "base_tail" belongs to tail unless it's explicitly spine/rump.
    if "tail" in tokens:
        return "tail"

    matched_coarse = ""
    for part_name, keywords in _COARSE_PART_PATTERNS:
        for kw in keywords:
            if kw in tokens or (len(kw) >= 4 and kw in normalized):
                matched_coarse = part_name
                break
        if matched_coarse:
            break

    if not matched_coarse:
        return clean if mode == "fine" else "torso"

    if mode != "fine":
        return matched_coarse

    side = _detect_laterality(tokens, normalized)
    if matched_coarse == "wings":
        return f"{side}_wing" if side else "wings"
    if matched_coarse == "limbs":
        fh = _detect_fore_hind(tokens, normalized)
        limb_stem = f"{fh}limb" if fh else "limb"
        return f"{side}_{limb_stem}" if side else limb_stem
    return matched_coarse


def keypoint_axial_polarity(keypoint_label: str) -> float:
    """Return +1.0 for anterior landmarks, -1.0 for posterior landmarks, or 0.0 if neutral."""
    clean = _clean_text(keypoint_label)
    if ":" in clean:
        clean = clean.rsplit(":", 1)[-1].strip()
    normalized = re.sub(r"[^a-z0-9]+", "_", clean.lower()).strip("_")
    tokens = {tok for tok in normalized.split("_") if tok}
    if tokens & {
        "nose",
        "snout",
        "rostrum",
        "head",
        "beak",
        "bill",
        "jaw",
        "chin",
        "eye",
        "ear",
        "antenna",
        "proboscis",
        "forehead",
    }:
        return 1.0
    if "neck" in tokens or "nape" in tokens:
        return 0.5
    if tokens & {"tail", "caudal", "uropod", "fluke", "anus", "rump", "pelvis", "hip"}:
        return -1.0
    if _detect_fore_hind(tokens, normalized) == "fore":
        return 0.4
    if _detect_fore_hind(tokens, normalized) == "hind":
        return -0.5
    return 0.0
