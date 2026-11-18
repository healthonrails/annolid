"""Canonical LabelMe metadata helpers for intra-instance body-part shapes."""

from __future__ import annotations

import re
from typing import Any, Iterable, Mapping

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
            "forepaw",
            "hindpaw",
            "rearpaw",
            "frontpaw",
            "backpaw",
            "foreleg",
            "hindleg",
            "rearleg",
            "frontleg",
            "backleg",
            "forelimb",
            "hindlimb",
            "forearm",
            "upperarm",
            "forefoot",
            "hindfoot",
            "paw",
            "foot",
            "feet",
            "hand",
            "wrist",
            "ankle",
            "elbow",
            "knee",
            "thigh",
            "shank",
            "hock",
            "stifle",
            "carpus",
            "limb",
            "leg",
            "arm",
            "tarsus",
            "tibia",
            "femur",
            "manus",
            "pes",
            "digit",
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


def format_shape_csv_tracking_name(shape: Mapping[str, Any] | object | None) -> str:
    """Return a collision-free CSV ``instance_name`` for whole-instance or body-part shapes."""
    if shape is None:
        return ""
    if is_body_part_shape_payload(shape):
        parent = extract_body_part_parent_instance(shape)
        part = extract_body_part_label(shape)
        if not parent:
            gid = _get_field(shape, "group_id")
            if gid not in (None, "") and not isinstance(gid, bool):
                parent = str(gid).strip()
        base = format_body_part_csv_instance_name(parent, part)
        qidx = _get_field(shape, "query_index")
        if qidx is None:
            qidx = _get_other_data(shape).get("query_index")
        if qidx is None:
            qidx = _get_flags(shape).get("query_index")
        if qidx is not None and not isinstance(qidx, bool) and str(qidx).strip():
            return f"{base}#{str(qidx).strip()}"
        return base
    inst = _clean_text(
        _get_field(shape, "instance_label") or _get_field(shape, "instance_name")
    )
    lbl = _clean_text(_get_field(shape, "label"))
    return inst or lbl


def split_body_part_csv_instance_name(
    instance_name: Any,
    *,
    known_parents: set[str] | None = None,
) -> tuple[str, str]:
    """Split ``'<parent>:<part>'`` into ``(parent, part)``, or ``(name, '')`` for whole instances."""
    clean = _clean_text(instance_name)
    if ":" not in clean:
        return clean, ""
    prefix, suffix = clean.split(":", 1)
    parent = prefix.strip()
    part = suffix.strip()
    if not parent or not part:
        return clean, ""
    if known_parents is not None and parent not in known_parents:
        part_base = part.split("#", 1)[0].strip()
        is_synthetic_parent = bool(
            re.fullmatch(r"(?:instance_|group_)?\d+", parent, flags=re.IGNORECASE)
        )
        if not (is_synthetic_parent or is_anatomical_body_part_label(part_base)):
            return clean, ""
    return parent, part


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
    if tokens & {
        "fore",
        "front",
        "anterior",
        "upper",
        "hand",
        "wrist",
        "elbow",
        "arm",
        "carpus",
        "manus",
    } or any(
        tok.startswith(("fore", "front", "anterior", "upper")) and tok != "forehead"
        for tok in tokens
    ):
        return "fore"
    if normalized.startswith(("lf_", "rf_", "fl_", "fr_", "lfore", "rfore")):
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
        "thigh",
        "shank",
        "hock",
        "stifle",
        "tarsus",
        "tibia",
        "femur",
        "pes",
    } or any(
        tok.startswith(("hind", "rear", "back", "posterior", "lower")) for tok in tokens
    ):
        return "hind"
    if normalized.startswith(("lh_", "rh_", "hl_", "hr_", "lhind", "rhind")):
        return "hind"
    return ""


def classify_keypoint_junction_role(keypoint_label: str) -> str:
    """Return ``'torso_tail'`` or ``'head_torso'`` when a landmark sits on an inter-part hinge."""
    clean = _clean_text(keypoint_label)
    if not clean:
        return ""
    if ":" in clean:
        clean = clean.rsplit(":", 1)[-1].strip() or clean
    normalized = re.sub(r"[^a-z0-9]+", "_", clean.lower()).strip("_")
    tokens = {tok for tok in normalized.split("_") if tok}
    if (
        normalized
        in {
            "tail_base",
            "base_tail",
            "base_of_tail",
            "tail_root",
            "root_tail",
            "tailbase",
            "tailroot",
        }
        or ("tail" in tokens and tokens & {"base", "root", "origin", "start"})
        or tokens & {"rump", "sacrum"}
    ):
        return "torso_tail"
    if tokens & {"neck", "nape", "throat", "occiput", "cervical"}:
        return "head_torso"
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

    # Special case: "tail_base" / "base_tail" / "tailtip" belongs to tail unless explicitly spine/rump.
    if "tail" in tokens or any(tok.startswith("tail") for tok in tokens):
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
        if any(
            tok.endswith(("ear", "eye", "jaw", "lip")) and tok != "rear"
            for tok in tokens
        ):
            matched_coarse = "head"
        elif any(tok.endswith(("paw", "leg", "arm", "toe", "fin")) for tok in tokens):
            matched_coarse = "limbs"

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
    } or any(tok.endswith("ear") and tok != "rear" for tok in tokens):
        return 1.0
    if "neck" in tokens or "nape" in tokens:
        return 0.5
    if tokens & {
        "tail",
        "caudal",
        "uropod",
        "fluke",
        "anus",
        "rump",
        "pelvis",
        "hip",
    } or any(tok.startswith("tail") for tok in tokens):
        return -1.0
    if _detect_fore_hind(tokens, normalized) == "fore":
        return 0.4
    if _detect_fore_hind(tokens, normalized) == "hind":
        return -0.5
    return 0.0


def _tail_chain_rank(normalized: str, tokens: set[str]) -> float | None:
    if not ("tail" in tokens or any(t.startswith("tail") for t in tokens)):
        return None
    if tokens & {"base", "root", "start", "origin"} or normalized in {
        "tailbase",
        "tailroot",
    }:
        return 0.0
    if tokens & {"prox", "proximal", "upper"}:
        return 1.0
    if tokens & {"mid", "middle", "center"} or normalized == "tailmid":
        return 2.5
    if tokens & {"dist", "distal", "lower"}:
        return 3.5
    if tokens & {"tip", "end"} or normalized == "tailtip":
        return 5.0
    for tok in tokens:
        if tok.isdigit():
            return 0.5 + float(int(tok)) * 0.8
        m = re.match(r"^tail(\d+)$", tok)
        if m:
            return 0.5 + float(int(m.group(1))) * 0.8
    return 2.5


def _spine_chain_rank(normalized: str, tokens: set[str]) -> float | None:
    if _detect_laterality(tokens, normalized):
        return None
    if tokens & {"neck", "nape", "throat", "cervical", "occiput"}:
        return 0.0
    if tokens & {"withers", "thorax", "chest"} or normalized == "upper_back":
        return 1.0
    if "spine" in tokens or "vertebra" in tokens or "back" in tokens:
        for tok in tokens:
            if tok.isdigit():
                return 1.0 + float(int(tok)) * 0.8
        if "mid" in tokens or "center" in tokens:
            return 2.2
        if "lower" in tokens:
            return 3.2
        return 2.0
    if tokens & {"center", "centroid", "midbody"}:
        return 2.2
    if tokens & {"pelvis", "sacrum", "rump"}:
        return 4.0
    return None


def _limb_joint_rank(tokens: set[str]) -> float | None:
    if tokens & {"shoulder", "hip", "coxa"}:
        return 0.0
    if tokens & {"upperarm", "thigh", "femur"}:
        return 1.0
    if tokens & {"elbow", "knee", "stifle"}:
        return 2.0
    if tokens & {"forearm", "shank", "tibia"}:
        return 3.0
    if tokens & {"wrist", "ankle", "hock", "carpus", "tarsus"}:
        return 4.0
    if tokens & {
        "paw",
        "forepaw",
        "hindpaw",
        "frontpaw",
        "backpaw",
        "rearpaw",
        "foot",
        "forefoot",
        "hindfoot",
        "hand",
        "manus",
        "pes",
        "toe",
        "claw",
    }:
        return 5.0
    return None


def infer_intra_part_skeleton_edges(
    keypoint_names: Iterable[str],
) -> list[tuple[str, str]]:
    """Infer canonical anatomical bone edges among ``keypoint_names`` when no schema is supplied."""
    names = [str(k).strip() for k in keypoint_names if str(k).strip()]
    if len(names) < 2:
        return []

    parsed: list[tuple[str, str, set[str]]] = []
    for raw in names:
        leaf = raw.rsplit(":", 1)[-1].strip() if ":" in raw else raw
        norm = re.sub(r"[^a-z0-9]+", "_", leaf.lower()).strip("_")
        toks = {t for t in norm.split("_") if t}
        parsed.append((raw, norm, toks))

    edges: list[tuple[str, str]] = []
    seen_edges: set[tuple[str, str]] = set()

    def _add(a: str, b: str) -> None:
        if a == b:
            return
        pair = (a, b) if a < b else (b, a)
        if pair not in seen_edges:
            seen_edges.add(pair)
            edges.append((a, b))

    # 1. Tail chain (tail_base -> tail_1 -> tail_mid -> tail_tip).
    tail_items: list[tuple[float, str]] = []
    for raw, norm, toks in parsed:
        r = _tail_chain_rank(norm, toks)
        if r is not None:
            tail_items.append((r, raw))
    tail_items.sort(key=lambda item: (item[0], item[1]))
    for i in range(len(tail_items) - 1):
        _add(tail_items[i][1], tail_items[i + 1][1])

    # 2. Dorsal midline spine chain (neck -> spine_1 -> spine_2 -> pelvis).
    spine_items: list[tuple[float, str]] = []
    for raw, norm, toks in parsed:
        r = _spine_chain_rank(norm, toks)
        if r is not None:
            spine_items.append((r, raw))
    spine_items.sort(key=lambda item: (item[0], item[1]))
    for i in range(len(spine_items) - 1):
        _add(spine_items[i][1], spine_items[i + 1][1])

    # 3. Cranial chain (nose/snout -> eyes -> ears, or nose -> ears directly).
    rostral = [
        raw
        for raw, _n, toks in parsed
        if toks & {"nose", "snout", "rostrum", "beak", "bill", "proboscis"}
    ]
    eyes_by_side: dict[str, list[str]] = {"left": [], "right": [], "": []}
    ears_by_side: dict[str, list[str]] = {"left": [], "right": [], "": []}
    for raw, norm, toks in parsed:
        side = _detect_laterality(toks, norm)
        if toks & {"eye", "pupil"} or any(t.endswith("eye") for t in toks):
            eyes_by_side[side].append(raw)
        elif toks & {"ear"} or any(t.endswith("ear") and t != "rear" for t in toks):
            ears_by_side[side].append(raw)
    if rostral:
        r0 = rostral[0]
        for side in ("left", "right", ""):
            if eyes_by_side[side]:
                e0 = eyes_by_side[side][0]
                _add(r0, e0)
                for ear_k in ears_by_side[side]:
                    _add(e0, ear_k)
            else:
                for ear_k in ears_by_side[side]:
                    _add(r0, ear_k)

    # 4. Same-limb joint chains (e.g. left_elbow -> left_wrist -> left_forepaw).
    limb_groups: dict[tuple[str, str], list[tuple[float, str]]] = {}
    for raw, norm, toks in parsed:
        side = _detect_laterality(toks, norm)
        fh = _detect_fore_hind(toks, norm)
        jr = _limb_joint_rank(toks)
        if jr is not None and (side or fh):
            limb_groups.setdefault((side, fh), []).append((jr, raw))
    for grp in limb_groups.values():
        grp.sort(key=lambda item: (item[0], item[1]))
        for i in range(len(grp) - 1):
            if grp[i][0] < grp[i + 1][0]:
                _add(grp[i][1], grp[i + 1][1])

    return edges


_ANATOMICAL_POLYGON_PART_TOKENS: frozenset[str] = frozenset(
    {
        "head",
        "snout",
        "nose",
        "rostrum",
        "skull",
        "forehead",
        "ear",
        "ears",
        "eye",
        "eyes",
        "beak",
        "jaw",
        "chin",
        "mouth",
        "neck",
        "nape",
        "throat",
        "torso",
        "trunk",
        "thorax",
        "abdomen",
        "belly",
        "flank",
        "rump",
        "pelvis",
        "tail",
        "caudal",
        "limb",
        "limbs",
        "forelimb",
        "hindlimb",
        "forelimbs",
        "hindlimbs",
        "foreleg",
        "hindleg",
        "forepaw",
        "hindpaw",
        "paw",
        "paws",
        "leg",
        "legs",
        "arm",
        "arms",
        "hand",
        "hands",
        "foot",
        "feet",
        "wing",
        "wings",
        "fin",
        "fins",
        "implant",
        "headstage",
        "miniscope",
        "cannula",
        "fiber",
        "tether",
        "collar",
        "harness",
        "backpack",
        "jacket",
        "eartag",
        "transponder",
        "crest",
        "comb",
        "wattle",
        "horn",
        "horns",
        "antler",
        "antlers",
        "carapace",
        "plastron",
        "cephalothorax",
        "mandible",
        "elytra",
        "halteres",
        "ovipositor",
        "clasper",
        "gill",
        "gills",
        "siphon",
        "tentacle",
        "tentacles",
        "mantle",
    }
)

_SCENE_OR_APPARATUS_TOKENS: frozenset[str] = frozenset(
    {
        "arena",
        "floor",
        "chamber",
        "cage",
        "maze",
        "wall",
        "walls",
        "background",
        "table",
        "dish",
        "plate",
        "tank",
        "bedding",
        "nest",
        "shelter",
        "burrow",
        "teaball",
        "cup",
        "bottle",
        "spout",
        "lever",
        "port",
        "wheel",
        "feeder",
        "hopper",
        "dish",
        "tray",
        "".join(("trou", "gh")),
        "pellet",
        "pellets",
        "food",
        "water",
        "toy",
        "block",
        "object",
        "novel",
        "zone",
        "region",
    }
)


def is_scene_or_apparatus_label(label: Any) -> bool:
    """Return True when ``label`` names a static arena region or non-animal object."""
    raw = _clean_text(label)
    if not raw:
        return False
    tokens = {tok for tok in re.split(r"[^a-z0-9]+", raw.lower()) if tok}
    return bool(tokens & _SCENE_OR_APPARATUS_TOKENS)


def is_anatomical_body_part_label(label: Any) -> bool:
    """Return True when ``label`` names an anatomical or attached animal sub-part."""
    raw = _clean_text(label)
    if not raw:
        return False
    if ":" in raw:
        _prefix, suffix = raw.split(":", 1)
        part_tokens = {
            tok for tok in re.split(r"[^a-z0-9]+", suffix.strip().lower()) if tok
        }
        if part_tokens & _ANATOMICAL_POLYGON_PART_TOKENS:
            return True
    tokens = {tok for tok in re.split(r"[^a-z0-9]+", raw.lower()) if tok}
    return bool(tokens & _ANATOMICAL_POLYGON_PART_TOKENS)


def _extract_xy_points(shape: Mapping[str, Any] | object) -> list[tuple[float, float]]:
    pts_raw = _get_field(shape, "points") or []
    pts: list[tuple[float, float]] = []
    for pt in pts_raw:
        if hasattr(pt, "x") and hasattr(pt, "y"):
            pts.append((float(pt.x()), float(pt.y())))
        elif isinstance(pt, (list, tuple)) and len(pt) >= 2:
            pts.append((float(pt[0]), float(pt[1])))
    return pts


def _has_nested_custom_subpart_polygon(
    candidate_shapes: list[Mapping[str, Any] | object],
) -> bool:
    """Detect non-scene custom sub-part polygons strictly nested inside an animal polygon."""
    if len(candidate_shapes) < 2:
        return False
    try:
        from shapely import make_valid
        from shapely.geometry import Polygon
    except Exception:
        return False

    polys: list[tuple[str, str, Any, float]] = []
    for sh in candidate_shapes:
        lbl = _clean_text(_get_field(sh, "label"))
        if not lbl or is_scene_or_apparatus_label(lbl):
            continue
        pts = _extract_xy_points(sh)
        if len(pts) < 3:
            continue
        try:
            geom = make_valid(Polygon(pts))
        except Exception:
            continue
        area = float(geom.area)
        if area <= 0.0:
            continue
        gid = _get_field(sh, "group_id")
        gid_s = "" if gid is None or isinstance(gid, bool) else str(gid).strip()
        polys.append((lbl, gid_s, geom, area))

    if len(polys) < 2:
        return False
    for i, (c_lbl, c_gid, c_geom, c_area) in enumerate(polys):
        for j, (p_lbl, p_gid, p_geom, p_area) in enumerate(polys):
            if i == j or c_lbl.lower() == p_lbl.lower():
                continue
            if c_area > 0.75 * p_area:
                continue
            if c_gid and p_gid and c_gid != p_gid:
                continue
            if not p_geom.bounds or not c_geom.bounds:
                continue
            px0, py0, px1, py1 = p_geom.bounds
            cx0, cy0, cx1, cy1 = c_geom.bounds
            if cx1 < px0 or cx0 > px1 or cy1 < py0 or cy0 > py1:
                continue
            inter_area = float(c_geom.intersection(p_geom).area)
            if inter_area >= 0.90 * c_area:
                return True
    return False


def has_body_part_polygon_annotations(
    shapes: Iterable[Mapping[str, Any] | object] | None,
) -> bool:
    """Return True if ``shapes`` contain any body-part or nested animal sub-part polygon."""
    if shapes is None:
        return False
    from annolid.utils.zone_shapes import is_zone_shape_payload

    polygon_candidates: list[Mapping[str, Any] | object] = []
    for shape in shapes:
        if shape is None or is_zone_shape_payload(shape):
            continue
        shape_type = _clean_text(_get_field(shape, "shape_type") or "polygon").lower()
        if shape_type in {"point", "line", "linestrip"}:
            continue
        if is_body_part_shape_payload(shape):
            return True
        lbl = _clean_text(_get_field(shape, "label"))
        if is_anatomical_body_part_label(lbl):
            return True
        if ":" in lbl:
            prefix, suffix = lbl.split(":", 1)
            if (
                prefix.strip()
                and suffix.strip()
                and not is_scene_or_apparatus_label(prefix)
                and not is_scene_or_apparatus_label(suffix)
            ):
                return True
        polygon_candidates.append(shape)
    return _has_nested_custom_subpart_polygon(polygon_candidates)


def folder_has_body_part_annotations(
    results_folder: Any,
    *,
    max_probe_files: int = 8,
) -> bool:
    """Fast probe checking manual seed JSONs, sampled frame JSONs, and AnnotationStore for body parts."""
    from pathlib import Path
    import json

    if not results_folder:
        return False
    folder = Path(results_folder).expanduser()
    if not folder.is_dir():
        return False

    def _json_file_has_parts(jf: Path) -> bool:
        try:
            payload = json.loads(jf.read_text(encoding="utf-8"))
        except Exception:
            return False
        if not isinstance(payload, Mapping):
            return False
        shapes = payload.get("shapes")
        return isinstance(shapes, list) and has_body_part_polygon_annotations(shapes)

    checked_paths: set[Path] = set()
    # 1. Always check manual seed frames (*.png -> matching *.json) first so mid/late-video seeds
    #    (e.g. Frame 50 or Frame 500 after whole-animal tracking) are never missed.
    for png_path in sorted(folder.glob("*.png"))[:64]:
        seed_jf = png_path.with_suffix(".json")
        if seed_jf.is_file():
            checked_paths.add(seed_jf)
            if _json_file_has_parts(seed_jf):
                return True

    # 2. Probe evenly spaced loose JSON files across the folder (start, middle, end)
    all_jsons = [
        jf
        for jf in sorted(folder.glob("*.json"))
        if jf not in checked_paths and not jf.name.endswith("_tracking_stats.json")
    ]
    probe_budget = max(1, int(max_probe_files))
    if len(all_jsons) <= probe_budget:
        sampled_jsons = all_jsons
    else:
        step = max(1, (len(all_jsons) - 1) // max(1, probe_budget - 1))
        idxs = list(
            dict.fromkeys(
                [*(i * step for i in range(probe_budget - 1)), len(all_jsons) - 1]
            )
        )
        sampled_jsons = [all_jsons[min(i, len(all_jsons) - 1)] for i in idxs]
    for jf in sampled_jsons:
        if _json_file_has_parts(jf):
            return True

    # 3. Probe evenly spaced frames in AnnotationStore when present
    try:
        from annolid.utils.annotation_store import AnnotationStore

        store = AnnotationStore.for_frame_path(folder / f"{folder.name}_000000000.json")
        if store.store_path.is_file():
            frame_nums = sorted(int(f) for f in store.iter_frames())
            if len(frame_nums) <= probe_budget:
                sampled_frames = frame_nums
            else:
                step = max(1, (len(frame_nums) - 1) // max(1, probe_budget - 1))
                idxs = list(
                    dict.fromkeys(
                        [
                            *(i * step for i in range(probe_budget - 1)),
                            len(frame_nums) - 1,
                        ]
                    )
                )
                sampled_frames = [frame_nums[min(i, len(frame_nums) - 1)] for i in idxs]
            for frame_no in sampled_frames:
                rec = store.get_frame(int(frame_no))
                shapes = rec.get("shapes") if isinstance(rec, Mapping) else None
                if isinstance(shapes, list) and has_body_part_polygon_annotations(
                    shapes
                ):
                    return True
    except Exception:
        pass
    return False


class CutieSeedHierarchyPlan:
    """Resolved Cutie 2D seed index mask, tracking-key map, and hierarchy metadata."""

    __slots__ = (
        "mask",
        "label_name_to_value",
        "body_part_specs",
        "partitioned_parents",
        "parent_group_ids",
        "cluster_centroids",
        "overlaps",
    )

    def __init__(
        self,
        *,
        mask: Any,
        label_name_to_value: dict[str, int],
        body_part_specs: dict[str, dict[str, Any]],
        partitioned_parents: dict[str, list[str]],
        parent_group_ids: dict[str, Any],
        cluster_centroids: dict[str, tuple[float, float]],
        overlaps: list[Any],
    ) -> None:
        self.mask = mask
        self.label_name_to_value = label_name_to_value
        self.body_part_specs = body_part_specs
        self.partitioned_parents = partitioned_parents
        self.parent_group_ids = parent_group_ids
        self.cluster_centroids = cluster_centroids
        self.overlaps = overlaps


def _mask_centroid(mask: Any, fallback_hw: tuple[int, int]) -> tuple[float, float]:
    from annolid.utils.mask_geometry import compute_mask_centroid_and_area

    stats = compute_mask_centroid_and_area(mask)
    if stats is not None:
        return (float(stats[0]), float(stats[1]))
    return (0.5 * float(fallback_hw[1]), 0.5 * float(fallback_hw[0]))


def _cluster_ungrouped_bare_parts(
    bare_items: list[dict[str, Any]],
    image_hw: tuple[int, int],
    prior_cluster_centroids: Mapping[str, tuple[float, float]] | None = None,
) -> tuple[list[tuple[str, int]], dict[str, tuple[float, float]]]:
    """Group ungrouped duplicate bare anatomical polygons into per-animal clusters."""
    from collections import defaultdict
    import math
    import cv2
    import numpy as np
    from scipy.optimize import linear_sum_assignment

    by_part: dict[str, list[int]] = defaultdict(list)
    for idx, item in enumerate(bare_items):
        by_part[str(item["part_label"]).lower()].append(idx)

    max_dup = max((len(idxs) for idxs in by_part.values()), default=0)
    if max_dup <= 1:
        return [("", 0) for _ in bare_items], {}

    def _anchor_priority(part_key: str) -> tuple[int, int, float, str]:
        idxs = by_part[part_key]
        is_core = (
            2
            if part_key in {"torso", "body", "trunk", "thorax", "abdomen"}
            else (1 if part_key in {"head", "snout", "skull"} else 0)
        )
        mean_area = sum(float(bare_items[i]["area"]) for i in idxs) / max(1, len(idxs))
        return (len(idxs), is_core, mean_area, part_key)

    anchor_key = max(by_part.keys(), key=_anchor_priority)
    anchor_idxs = sorted(
        by_part[anchor_key],
        key=lambda i: (
            round(bare_items[i]["centroid"][1] / max(16.0, 0.05 * image_hw[0])),
            bare_items[i]["centroid"][0],
            bare_items[i]["centroid"][1],
        ),
    )
    num_clusters = len(anchor_idxs)
    clusters: list[dict[str, Any]] = [
        {
            "indices": [a_idx],
            "union_mask": bare_items[a_idx]["mask"].copy(),
        }
        for a_idx in anchor_idxs
    ]

    remaining_keys = sorted(
        (k for k in by_part.keys() if k != anchor_key),
        key=lambda k: (
            -sum(float(bare_items[i]["area"]) for i in by_part[k])
            / max(1, len(by_part[k])),
            k,
        ),
    )

    for part_key in remaining_keys:
        p_idxs = by_part[part_key]
        m_items = len(p_idxs)
        cost = np.zeros((m_items, num_clusters), dtype=np.float64)
        dist_maps: list[Any] = []
        cluster_cxy: list[tuple[float, float]] = []
        for cl in clusters:
            umask = cl["union_mask"]
            cluster_cxy.append(_mask_centroid(umask, image_hw))
            if np.any(umask):
                inv = (~umask).astype(np.uint8)
                dist_maps.append(cv2.distanceTransform(inv, cv2.DIST_L2, 3))
            else:
                dist_maps.append(None)

        for r_i, item_idx in enumerate(p_idxs):
            imask = bare_items[item_idx]["mask"]
            icx, icy = bare_items[item_idx]["centroid"]
            has_fg = bool(np.any(imask))
            for c_i in range(num_clusters):
                ccx, ccy = cluster_cxy[c_i]
                cdist = math.hypot(icx - ccx, icy - ccy)
                dmap = dist_maps[c_i]
                if has_fg and dmap is not None:
                    bdist = float(dmap[imask].min())
                else:
                    bdist = cdist
                cost[r_i, c_i] = bdist + 1e-3 * cdist

        row_ind, col_ind = linear_sum_assignment(cost)
        for r_i, c_i in zip(row_ind, col_ind):
            chosen_item_idx = p_idxs[int(r_i)]
            clusters[int(c_i)]["indices"].append(chosen_item_idx)
            clusters[int(c_i)]["union_mask"] |= bare_items[chosen_item_idx]["mask"]

    raw_centroids = [_mask_centroid(cl["union_mask"], image_hw) for cl in clusters]

    cluster_to_inst_num: dict[int, int] = {}
    prior_clean = {
        str(k): (float(v[0]), float(v[1]))
        for k, v in (prior_cluster_centroids or {}).items()
        if str(k).startswith("instance_")
    }
    if prior_clean:
        prior_keys = sorted(
            prior_clean.keys(),
            key=lambda k: (
                int(k.split("_", 1)[1]) if k.split("_", 1)[1].isdigit() else 9999,
                k,
            ),
        )
        p_cost = np.zeros((num_clusters, len(prior_keys)), dtype=np.float64)
        for c_i, (cx, cy) in enumerate(raw_centroids):
            for p_j, pk in enumerate(prior_keys):
                px, py = prior_clean[pk]
                p_cost[c_i, p_j] = math.hypot(cx - px, cy - py)
        r_ind, c_ind = linear_sum_assignment(p_cost)
        used_nums: set[int] = set()
        for c_i, p_j in zip(r_ind, c_ind):
            pk = prior_keys[int(p_j)]
            suffix = pk.split("_", 1)[1]
            num = int(suffix) if suffix.isdigit() else (int(p_j) + 1)
            cluster_to_inst_num[int(c_i)] = num
            used_nums.add(num)
        next_num = 1
        for c_i in sorted(
            range(num_clusters),
            key=lambda i: (
                round(raw_centroids[i][1] / max(16.0, 0.05 * image_hw[0])),
                raw_centroids[i][0],
                raw_centroids[i][1],
            ),
        ):
            if c_i not in cluster_to_inst_num:
                while next_num in used_nums:
                    next_num += 1
                cluster_to_inst_num[c_i] = next_num
                used_nums.add(next_num)
    else:
        ordered_c = sorted(
            range(num_clusters),
            key=lambda i: (
                round(raw_centroids[i][1] / max(16.0, 0.05 * image_hw[0])),
                raw_centroids[i][0],
                raw_centroids[i][1],
            ),
        )
        for rank, c_i in enumerate(ordered_c, start=1):
            cluster_to_inst_num[c_i] = rank

    assignments: list[tuple[str, int]] = [("", 0) for _ in bare_items]
    out_centroids: dict[str, tuple[float, float]] = {}
    for c_i, cl in enumerate(clusters):
        inst_num = cluster_to_inst_num[c_i]
        inst_name = f"instance_{inst_num}"
        out_centroids[inst_name] = raw_centroids[c_i]
        for item_idx in cl["indices"]:
            assignments[item_idx] = (inst_name, inst_num)

    return assignments, out_centroids


def prepare_cutie_seed_hierarchy(
    filtered_shapes: Iterable[Mapping[str, Any]],
    image_size: tuple[int, ...],
    *,
    prior_cluster_centroids: Mapping[str, tuple[float, float]] | None = None,
    existing_body_part_specs: dict[str, dict[str, Any]] | None = None,
    existing_partitioned_parents: dict[str, list[str]] | None = None,
    existing_parent_group_ids: dict[str, Any] | None = None,
) -> CutieSeedHierarchyPlan:
    """Build an exclusive 2D Cutie index mask while disambiguating multi-animal body parts."""
    from collections import defaultdict
    import numpy as np
    from annolid.annotation.polygon_constraints import make_instance_masks_exclusive
    from annolid.tracking.identity_continuity import find_cross_instance_mask_overlaps
    from annolid.utils.annotation_compat import shape_to_mask
    from annolid.utils.mask_geometry import tight_mask_bbox

    image_hw = (int(image_size[0]), int(image_size[1]))
    body_part_specs = (
        existing_body_part_specs if existing_body_part_specs is not None else {}
    )
    partitioned_parents = (
        existing_partitioned_parents if existing_partitioned_parents is not None else {}
    )
    parent_group_ids = (
        existing_parent_group_ids if existing_parent_group_ids is not None else {}
    )
    cluster_centroids: dict[str, tuple[float, float]] = dict(
        prior_cluster_centroids or {}
    )

    shape_list = [s for s in filtered_shapes if isinstance(s, Mapping)]
    parsed: list[dict[str, Any]] = []
    for shape in shape_list:
        raw_label = _clean_text(shape.get("label"))
        if not raw_label:
            continue
        smask = np.asarray(
            shape_to_mask(
                image_hw,
                shape.get("points") or [],
                shape_type=shape.get("shape_type", "polygon"),
            ),
            dtype=bool,
        )
        area = int(np.count_nonzero(smask))
        gid = shape.get("group_id")
        gid_str = "" if gid is None or isinstance(gid, bool) else str(gid).strip()
        has_colon_pair = ":" in raw_label and all(
            p.strip() for p in raw_label.split(":", 1)
        )
        explicit_part = is_body_part_shape_payload(shape) or has_colon_pair
        anatomical = is_anatomical_body_part_label(raw_label)
        parsed.append(
            {
                "shape": shape,
                "raw_label": raw_label,
                "mask": smask,
                "area": area,
                "gid": gid,
                "gid_str": gid_str,
                "explicit_part": explicit_part,
                "anatomical": anatomical,
                "centroid": _mask_centroid(smask, image_hw),
            }
        )

    label_name_to_value: dict[str, int] = {"_background_": 0}
    if not parsed:
        return CutieSeedHierarchyPlan(
            mask=None,
            label_name_to_value=label_name_to_value,
            body_part_specs=body_part_specs,
            partitioned_parents=partitioned_parents,
            parent_group_ids=parent_group_ids,
            cluster_centroids=cluster_centroids,
            overlaps=[],
        )

    frame_area = float(max(1, image_hw[0] * image_hw[1]))
    # Detect strictly nested custom non-scene sub-part polygons inside bounded animal parent polygons
    nested_custom_indices: set[int] = set()
    if len(parsed) >= 2:
        for i, c in enumerate(parsed):
            if (
                c["explicit_part"]
                or c["anatomical"]
                or c["area"] <= 0
                or is_scene_or_apparatus_label(c["raw_label"])
            ):
                continue
            for j, p in enumerate(parsed):
                if (
                    i == j
                    or p["explicit_part"]
                    or p["anatomical"]
                    or p["area"] <= 0
                    or float(p["area"]) >= 0.65 * frame_area
                    or is_scene_or_apparatus_label(p["raw_label"])
                ):
                    continue
                if c["raw_label"].lower() == p["raw_label"].lower():
                    continue
                if float(c["area"]) > 0.75 * float(p["area"]):
                    continue
                if c["gid_str"] and p["gid_str"] and c["gid_str"] != p["gid_str"]:
                    continue
                inter = int(np.count_nonzero(c["mask"] & p["mask"]))
                if float(inter) >= 0.90 * float(c["area"]):
                    nested_custom_indices.add(i)
                    break

    for idx_n in nested_custom_indices:
        parsed[idx_n]["inferred_nested"] = True

    any_body_part = any(
        p["explicit_part"] or p["anatomical"] or p.get("inferred_nested", False)
        for p in parsed
    )
    if not any_body_part:
        for p in sorted(parsed, key=lambda item: item["raw_label"]):
            lbl = p["raw_label"]
            if lbl not in label_name_to_value:
                label_name_to_value[lbl] = len(label_name_to_value)
        if len(label_name_to_value) <= 1:
            return CutieSeedHierarchyPlan(
                mask=None,
                label_name_to_value=label_name_to_value,
                body_part_specs=body_part_specs,
                partitioned_parents=partitioned_parents,
                parent_group_ids=parent_group_ids,
                cluster_centroids=cluster_centroids,
                overlaps=[],
            )
        instance_masks: dict[str, np.ndarray] = {}
        for p in parsed:
            lbl = p["raw_label"]
            if lbl in instance_masks:
                instance_masks[lbl] |= p["mask"]
            else:
                instance_masks[lbl] = p["mask"].copy()
        overlaps = find_cross_instance_mask_overlaps(instance_masks)
        ordered_labels = [
            lbl
            for lbl, val in sorted(label_name_to_value.items(), key=lambda x: x[1])
            if val and lbl in instance_masks
        ]
        exclusive_masks = make_instance_masks_exclusive(
            [instance_masks[lbl] for lbl in ordered_labels]
        )
        out_mask = np.zeros(image_hw, dtype=np.int32)
        for lbl, imask in zip(ordered_labels, exclusive_masks):
            bbox = tight_mask_bbox(imask)
            if bbox is None:
                continue
            x0, y0, x1, y1 = bbox
            out_mask[y0:y1, x0:x1][imask[y0:y1, x0:x1]] = int(label_name_to_value[lbl])
        return CutieSeedHierarchyPlan(
            mask=out_mask,
            label_name_to_value=label_name_to_value,
            body_part_specs=body_part_specs,
            partitioned_parents=partitioned_parents,
            parent_group_ids=parent_group_ids,
            cluster_centroids=cluster_centroids,
            overlaps=overlaps,
        )

    # Separate whole-instance parent polygons from body-part polygons
    parent_items = [
        p
        for p in parsed
        if not p["explicit_part"]
        and not p["anatomical"]
        and not p.get("inferred_nested", False)
    ]
    part_items = [
        p
        for p in parsed
        if p["explicit_part"] or p["anatomical"] or p.get("inferred_nested", False)
    ]

    # Merge multi-polygon parents that share (raw_label, non-empty group_id) before disambiguating
    merged_parent_items: list[dict[str, Any]] = []
    by_lbl_gid: dict[tuple[str, str], dict[str, Any]] = {}
    for p in parent_items:
        if p["gid_str"]:
            lg_key = (p["raw_label"], p["gid_str"])
            if lg_key in by_lbl_gid:
                acc = by_lbl_gid[lg_key]
                acc["mask"] = acc["mask"] | p["mask"]
                acc["area"] = int(np.count_nonzero(acc["mask"]))
                acc["centroid"] = _mask_centroid(acc["mask"], image_hw)
            else:
                clone = dict(p)
                clone["mask"] = p["mask"].copy()
                by_lbl_gid[lg_key] = clone
                merged_parent_items.append(clone)
        else:
            merged_parent_items.append(p)

    parent_lbl_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for p in merged_parent_items:
        parent_lbl_groups[p["raw_label"]].append(p)

    parent_masks_by_key: dict[str, np.ndarray] = {}
    parent_gid_str_by_key: dict[str, str] = {}
    gid_to_parent_key: dict[str, str] = {}
    resolved_entries: list[dict[str, Any]] = []

    for raw_lbl, group in parent_lbl_groups.items():
        if len(group) == 1:
            p = group[0]
            pkey = raw_lbl
            parent_masks_by_key[pkey] = p["mask"].copy()
            parent_gid_str_by_key[pkey] = p["gid_str"]
            if p["gid"] is not None:
                parent_group_ids[pkey] = p["gid"]
            if p["gid_str"]:
                gid_to_parent_key[p["gid_str"]] = pkey
            resolved_entries.append({"tracking_key": pkey, "mask": p["mask"]})
        else:
            ordered_group = sorted(
                group,
                key=lambda item: (
                    round(item["centroid"][1] / max(16.0, 0.05 * image_hw[0])),
                    item["centroid"][0],
                    item["centroid"][1],
                ),
            )
            for idx_g, p in enumerate(ordered_group, start=1):
                pkey = (
                    f"{raw_lbl}_{p['gid_str']}"
                    if p["gid_str"]
                    else f"{raw_lbl}_{idx_g}"
                )
                while pkey in parent_masks_by_key:
                    idx_g += 1
                    pkey = f"{raw_lbl}_{idx_g}"
                parent_masks_by_key[pkey] = p["mask"].copy()
                parent_gid_str_by_key[pkey] = p["gid_str"]
                eff_gid = p["gid"] if p["gid"] is not None else idx_g
                parent_group_ids[pkey] = eff_gid
                if p["gid_str"]:
                    gid_to_parent_key[p["gid_str"]] = pkey
                resolved_entries.append({"tracking_key": pkey, "mask": p["mask"]})

    # Resolve parent_instance & group_id for each body-part polygon (5-tier cascade)
    pending_parts: list[dict[str, Any]] = []
    ungrouped_bare_slots: list[int] = []

    for p in part_items:
        shape = p["shape"]
        raw_lbl = p["raw_label"]
        explicit_parent = extract_body_part_parent_instance(shape)
        part_lbl = extract_body_part_label(shape) or raw_lbl
        if not explicit_parent and ":" in raw_lbl:
            sp_parent, sp_part = split_body_part_csv_instance_name(raw_lbl)
            if sp_parent and sp_part:
                explicit_parent = sp_parent
                part_lbl = sp_part

        parent_inst = ""
        resolved_gid = p["gid"]

        if explicit_parent:
            parent_inst = explicit_parent
            if resolved_gid is None and parent_inst in parent_group_ids:
                resolved_gid = parent_group_ids[parent_inst]
        elif p["gid_str"] and p["gid_str"] in gid_to_parent_key:
            parent_inst = gid_to_parent_key[p["gid_str"]]
        elif parent_masks_by_key and p["area"] > 0:
            best_parent = ""
            best_inter = 0
            for pkey, pmask in parent_masks_by_key.items():
                pgid_s = parent_gid_str_by_key.get(pkey, "")
                if p["gid_str"] and pgid_s and p["gid_str"] != pgid_s:
                    continue
                inter = int(np.count_nonzero(p["mask"] & pmask))
                if inter > best_inter:
                    best_inter = inter
                    best_parent = pkey
            if best_parent and float(best_inter) >= 0.50 * float(p["area"]):
                parent_inst = best_parent
                if resolved_gid is None and best_parent in parent_group_ids:
                    resolved_gid = parent_group_ids[best_parent]
        if not parent_inst and p["gid_str"]:
            parent_inst = f"instance_{p['gid_str']}"

        slot_idx = len(pending_parts)
        pending_parts.append(
            {
                "raw_label": raw_lbl,
                "part_label": part_lbl,
                "parent_instance": parent_inst,
                "group_id": resolved_gid,
                "mask": p["mask"],
                "area": p["area"],
                "centroid": p["centroid"],
            }
        )
        if not parent_inst and not p["gid_str"]:
            ungrouped_bare_slots.append(slot_idx)

    if ungrouped_bare_slots:
        bare_subset = [pending_parts[s] for s in ungrouped_bare_slots]
        bare_assignments, updated_centroids = _cluster_ungrouped_bare_parts(
            bare_subset,
            image_hw,
            prior_cluster_centroids=cluster_centroids,
        )
        if updated_centroids:
            cluster_centroids.update(updated_centroids)
        for s_idx, (assigned_inst, assigned_gid) in zip(
            ungrouped_bare_slots, bare_assignments
        ):
            if assigned_inst:
                pending_parts[s_idx]["parent_instance"] = assigned_inst
                if pending_parts[s_idx]["group_id"] is None and assigned_gid > 0:
                    pending_parts[s_idx]["group_id"] = assigned_gid

    # Disambiguate bilateral / duplicate same-label parts on the same resolved parent (e.g. two "ear"s)
    by_base_key: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in pending_parts:
        parent_inst = item["parent_instance"]
        part_lbl = item["part_label"]
        base_key = (
            format_body_part_csv_instance_name(parent_inst, part_lbl)
            if parent_inst
            else item["raw_label"]
        )
        item["base_key"] = base_key
        by_base_key[base_key].append(item)

    for base_key, group in by_base_key.items():
        if len(group) == 1:
            item = group[0]
            tkey = base_key
            body_part_specs[tkey] = {
                "raw_label": item["raw_label"],
                "parent_instance": item["parent_instance"],
                "part_label": item["part_label"],
                "group_id": item["group_id"],
                "query_index": None,
            }
            resolved_entries.append({"tracking_key": tkey, "mask": item["mask"]})
        else:
            ordered_sub = sorted(
                group,
                key=lambda x: (
                    round(x["centroid"][1] / max(12.0, 0.04 * image_hw[0])),
                    x["centroid"][0],
                    x["centroid"][1],
                ),
            )
            for sub_idx, item in enumerate(ordered_sub, start=1):
                tkey = f"{base_key}#{sub_idx}"
                body_part_specs[tkey] = {
                    "raw_label": item["raw_label"],
                    "parent_instance": item["parent_instance"],
                    "part_label": item["part_label"],
                    "group_id": item["group_id"],
                    "query_index": sub_idx,
                }
                resolved_entries.append({"tracking_key": tkey, "mask": item["mask"]})

    for entry in sorted(resolved_entries, key=lambda x: x["tracking_key"]):
        tkey = entry["tracking_key"]
        if tkey not in label_name_to_value:
            label_name_to_value[tkey] = len(label_name_to_value)

    if len(label_name_to_value) <= 1:
        return CutieSeedHierarchyPlan(
            mask=None,
            label_name_to_value=label_name_to_value,
            body_part_specs=body_part_specs,
            partitioned_parents=partitioned_parents,
            parent_group_ids=parent_group_ids,
            cluster_centroids=cluster_centroids,
            overlaps=[],
        )

    instance_masks = {}
    for entry in resolved_entries:
        tkey = entry["tracking_key"]
        if tkey in instance_masks:
            instance_masks[tkey] |= entry["mask"]
        else:
            instance_masks[tkey] = entry["mask"].copy()

    overlaps = find_cross_instance_mask_overlaps(instance_masks)
    ordered_labels = [
        lbl
        for lbl, val in sorted(label_name_to_value.items(), key=lambda x: x[1])
        if val and lbl in instance_masks
    ]
    priorities = [
        (
            1.0 + 1.0 / (1.0 + float(np.count_nonzero(instance_masks[lbl])))
            if lbl in body_part_specs
            else 0.0
        )
        for lbl in ordered_labels
    ]
    exclusive_masks = make_instance_masks_exclusive(
        [instance_masks[lbl] for lbl in ordered_labels],
        scores=priorities,
    )

    kept_labels: list[str] = []
    kept_masks: list[Any] = []
    for lbl, ex_mask in zip(ordered_labels, exclusive_masks):
        if lbl not in body_part_specs and np.any(instance_masks[lbl]):
            child_keys = [
                ck
                for ck, spec in body_part_specs.items()
                if spec.get("parent_instance") == lbl and ck in instance_masks
            ]
            if child_keys:
                existing_children = partitioned_parents.get(lbl, [])
                partitioned_parents[lbl] = list(
                    dict.fromkeys([*existing_children, *child_keys])
                )
                parent_total = int(instance_masks[lbl].sum())
                rem_area = int(ex_mask.sum())
                if rem_area < max(32, int(0.25 * parent_total)):
                    continue
        kept_labels.append(lbl)
        kept_masks.append(ex_mask)

    rebuilt_map: dict[str, int] = {"_background_": 0}
    for idx, lbl in enumerate(kept_labels, start=1):
        rebuilt_map[lbl] = idx

    out_mask = np.zeros(image_hw, dtype=np.int32)
    for lbl, imask in zip(kept_labels, kept_masks):
        bbox = tight_mask_bbox(imask)
        if bbox is None:
            continue
        x0, y0, x1, y1 = bbox
        out_mask[y0:y1, x0:x1][imask[y0:y1, x0:x1]] = int(rebuilt_map[lbl])

    return CutieSeedHierarchyPlan(
        mask=out_mask,
        label_name_to_value=rebuilt_map,
        body_part_specs=body_part_specs,
        partitioned_parents=partitioned_parents,
        parent_group_ids=parent_group_ids,
        cluster_centroids=cluster_centroids,
        overlaps=overlaps,
    )


def apply_cutie_body_part_metadata(
    shape_obj: Any,
    tracking_key: str,
    body_part_specs: Mapping[str, Mapping[str, Any]] | None,
) -> None:
    """Attach canonical LabelMe body-part flags, group_id, and other_data onto ``shape_obj``."""
    if not body_part_specs:
        return
    spec = body_part_specs.get(tracking_key)
    if not spec:
        return
    raw_label = str(spec.get("raw_label") or tracking_key)
    part_lbl = str(spec.get("part_label") or raw_label)
    parent_inst = str(spec.get("parent_instance") or "")
    shape_obj.label = raw_label
    flags = dict(getattr(shape_obj, "flags", None) or {})
    flags["is_body_part"] = True
    shape_obj.flags = flags
    if spec.get("group_id") is not None:
        shape_obj.group_id = spec.get("group_id")
    other = dict(getattr(shape_obj, "other_data", None) or {})
    other["semantic_type"] = BODY_PART_SEMANTIC_TYPE
    other["part_label"] = part_lbl
    other["display_label"] = part_lbl
    if parent_inst:
        other["parent_instance"] = parent_inst
        other["instance_label"] = parent_inst
    if spec.get("query_index") is not None:
        other["query_index"] = int(spec["query_index"])
    shape_obj.other_data = other
